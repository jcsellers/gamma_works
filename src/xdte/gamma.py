"""Helpers for percentile-driven gamma sizing rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, cast


@dataclass(frozen=True)
class GammaRule:
    """Configuration describing how to map margins to gamma levels."""

    lo: float
    hi: float
    levels: tuple[float, ...]
    percentiles: tuple[tuple[float, float], ...] = ()
    kill_day_level: float | None = None
    target_percentile: float | None = None

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the rule."""

        payload: dict[str, object] = {
            "lo": float(self.lo),
            "hi": float(self.hi),
            "levels": [float(level) for level in self.levels],
        }
        if self.percentiles:
            payload["percentiles"] = [
                [float(percentile), float(score)]
                for percentile, score in self.percentiles
            ]
        if self.kill_day_level is not None:
            payload["kill_day_level"] = float(self.kill_day_level)
        if self.target_percentile is not None:
            payload["target_percentile"] = float(self.target_percentile)
        return payload

    def level_for_margin(self, margin: float | None) -> float | None:
        """Resolve the gamma level for the supplied (normalised) margin."""

        if not self.levels:
            return None
        if margin is None:
            return self.levels[-1]
        positive = max(margin, 0.0)
        ordered_levels = tuple(float(level) for level in self.levels)
        if len(ordered_levels) == 1:
            return ordered_levels[0]
        if len(ordered_levels) == 2:
            threshold = float(self.lo)
            return ordered_levels[0] if positive < threshold else ordered_levels[1]
        lower = float(self.lo)
        upper = float(self.hi)
        if positive < lower:
            return ordered_levels[0]
        if positive < upper:
            return ordered_levels[min(1, len(ordered_levels) - 1)]
        return ordered_levels[-1]


def rule_from_mapping(payload: Mapping[str, object]) -> GammaRule | None:
    """Normalise an arbitrary mapping into a :class:`GammaRule`."""

    lo_raw = payload.get("lo", 0.0)
    hi_raw = payload.get("hi", 1.0)
    try:
        lo = float(cast(Any, lo_raw))
        hi = float(cast(Any, hi_raw))
    except (TypeError, ValueError):
        return None

    levels_raw = payload.get("levels")
    levels: list[float] = []
    if isinstance(levels_raw, Sequence) and not isinstance(levels_raw, (str, bytes)):
        for entry in levels_raw:
            try:
                levels.append(float(cast(Any, entry)))
            except (TypeError, ValueError):
                continue
    if not levels:
        return None

    percentiles_payload = payload.get("percentiles")
    percentiles: list[tuple[float, float]] = []
    if isinstance(percentiles_payload, Sequence) and not isinstance(
        percentiles_payload, (str, bytes)
    ):
        for entry in percentiles_payload:
            if (
                isinstance(entry, Sequence)
                and len(entry) == 2
                and not isinstance(entry, (str, bytes))
            ):
                try:
                    percentile = float(cast(Any, entry[0]))
                    score = float(cast(Any, entry[1]))
                except (TypeError, ValueError):
                    continue
                percentiles.append((percentile, score))
    percentiles.sort(key=lambda item: item[0])

    kill_day_level_raw = payload.get("kill_day_level")
    kill_day_level: float | None
    try:
        kill_day_level = (
            None if kill_day_level_raw is None else float(cast(Any, kill_day_level_raw))
        )
    except (TypeError, ValueError):
        kill_day_level = None

    target_percentile_raw = payload.get("target_percentile")
    target_percentile: float | None
    try:
        target_percentile = (
            None
            if target_percentile_raw is None
            else float(cast(Any, target_percentile_raw))
        )
    except (TypeError, ValueError):
        target_percentile = None

    return GammaRule(
        lo=float(lo),
        hi=float(hi),
        levels=tuple(levels),
        percentiles=tuple(percentiles),
        kill_day_level=kill_day_level,
        target_percentile=target_percentile,
    )


def percentile_for_score(
    score: float | None, percentiles: Sequence[tuple[float, float]]
) -> float | None:
    """Estimate the percentile rank for ``score`` based on training percentiles."""

    if score is None or math.isnan(score):
        return None
    if not percentiles:
        return None
    ordered = sorted(percentiles, key=lambda item: (item[0], item[1]))
    first_percentile, first_score = ordered[0]
    last_percentile, last_score = ordered[-1]
    if score <= first_score:
        return float(first_percentile)
    if score >= last_score:
        return float(last_percentile)
    for index in range(len(ordered) - 1):
        lower_p, lower_score = ordered[index]
        upper_p, upper_score = ordered[index + 1]
        if math.isclose(lower_score, upper_score):
            continue
        if lower_score <= score <= upper_score or upper_score <= score <= lower_score:
            span = upper_score - lower_score
            if math.isclose(span, 0.0):
                continue
            position = (score - lower_score) / span
            return float(lower_p + position * (upper_p - lower_p))
    return float(last_percentile)


def normalised_margin(percentile: float | None, target: float | None) -> float | None:
    """Return the signed, normalised distance between ``percentile`` and ``target``."""

    if percentile is None or target is None:
        return None
    if percentile >= target:
        span = max(1.0 - target, 1e-9)
        return min((percentile - target) / span, 1.0)
    span = max(target, 1e-9)
    return -min((target - percentile) / span, 1.0)


def decile_margin(decile: int | None, *, total_deciles: int = 10) -> float | None:
    """Compute a normalised distance based on decile ranking."""

    if decile is None or total_deciles <= 0:
        return None
    if decile <= 1:
        return 1.0
    if decile >= total_deciles:
        return 0.0
    return max((total_deciles - decile) / float(total_deciles - 1), 0.0)


def compute_percentiles(
    values: Iterable[float], *, quantiles: Sequence[float] | None = None
) -> list[tuple[float, float]]:
    """Return score percentiles for the supplied values."""

    sample = [value for value in values if isinstance(value, (int, float))]
    if not sample:
        return []
    if quantiles is None:
        quantiles = tuple(step / 20.0 for step in range(21))
    results: list[tuple[float, float]] = []
    ordered = sorted(sample)
    for quantile in quantiles:
        quantile = float(min(max(quantile, 0.0), 1.0))
        position = quantile * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            value = float(ordered[int(position)])
        else:
            blend = position - lower
            value = float((1.0 - blend) * ordered[lower] + blend * ordered[upper])
        results.append((quantile, value))
    return results
