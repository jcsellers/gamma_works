"""Hybrid policy evaluation and persistence helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
from typing_extensions import TypeGuard

from xdte.config import Settings, get_settings
from xdte.gamma import (
    GammaRule,
    normalised_margin,
    percentile_for_score,
    rule_from_mapping,
)
from xdte.metrics import portfolio_metrics


@dataclass(frozen=True)
class HybridRow:
    """Single prediction row emitted by :mod:`xdte.model.apply`."""

    book: str
    open_date: date
    pnl: float
    prediction: float
    keep: bool
    kill_switch: bool = False


@dataclass(frozen=True)
class HybridPolicy:
    """Tail and gamma adjustments applied during hybrid evaluation."""

    tails: Mapping[str, tuple[float, float]]
    gammas: Mapping[str, float]
    gamma_rules: Mapping[str, GammaRule] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Mapping[str, object]]:
        """Return a JSON friendly representation of the policy."""

        return {
            "tails": {
                book: [float(lower), float(upper)]
                for book, (lower, upper) in self.tails.items()
            },
            "gammas": {book: float(value) for book, value in self.gammas.items()},
            "gamma_rules": {
                book: rule.to_mapping() for book, rule in self.gamma_rules.items()
            },
        }


@dataclass(frozen=True)
class HybridCandidateEvaluation:
    """Evaluation summary for a single candidate policy."""

    policy: HybridPolicy
    metrics: Mapping[str, float]
    accepted: bool
    reasons: Sequence[str]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON serialisable payload describing the evaluation."""

        return {
            "policy": self.policy.to_dict(),
            "metrics": _serialise_metrics(self.metrics),
            "accepted": bool(self.accepted),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class HybridSelectionResult:
    """Result returned by :func:`select_hybrid_policy`."""

    baseline_metrics: Mapping[str, float]
    evaluations: Sequence[HybridCandidateEvaluation]
    selected_policy: HybridPolicy
    accepted: bool
    output_path: Path


__all__ = [
    "HybridPolicy",
    "HybridCandidateEvaluation",
    "HybridSelectionResult",
    "select_hybrid_policy",
]


def _parse_open_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value).date()
    raise ValueError(f"Unsupported open_date value: {value!r}")


def _parse_float(value: object) -> float:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return not math.isclose(float(value), 0.0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y"}
    return False


def _load_hybrid_rows(path: Path) -> List[HybridRow]:
    if not path.exists():
        return []

    rows: List[HybridRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            book = str(raw.get("book", "")).strip()
            if not book:
                continue
            try:
                open_date = _parse_open_date(raw.get("open_date"))
            except ValueError:
                continue
            pnl_value = _parse_float(raw.get("pnl_strategy"))
            if math.isnan(pnl_value):
                pnl_value = _parse_float(raw.get("pnl"))
            if math.isnan(pnl_value):
                continue
            prediction = _parse_float(raw.get("prediction"))
            keep_flag = _parse_bool(raw.get("keep", False))
            kill_flag = _parse_bool(raw.get("L1_vvix_above_ema30", False))
            rows.append(
                HybridRow(
                    book=book,
                    open_date=open_date,
                    pnl=pnl_value,
                    prediction=prediction,
                    keep=keep_flag,
                    kill_switch=kill_flag,
                )
        return rows

    resolved = path.resolve()
    if resolved.is_dir():
        candidate_paths = [
            resolved / "hybrid_predictions.csv",
            resolved / "hybrid_input.csv",
        ]
    else:
        candidate_paths = [resolved]
        predictions_path = resolved.parent / "hybrid_predictions.csv"
        if predictions_path not in candidate_paths:
            candidate_paths.insert(0, predictions_path)

    for candidate in candidate_paths:
        rows = _load_from_file(candidate)
        if rows:
            return rows
    return []


def _serialise_metrics(metrics: Mapping[str, float]) -> Dict[str, float | None]:
    payload: Dict[str, float | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = None
        else:
            payload[key] = float(value)
    return payload


def _serialise_settings(settings: Settings) -> Dict[str, object]:
    return {
        "alpha": float(settings.ALPHA),
        "kill_switch_tails": {
            book: [float(lower), float(upper)]
            for book, (lower, upper) in settings.KILL_SWITCH_TAILS.items()
        },
        "kill_switch_gammas": {
            book: float(value) for book, value in settings.KILL_SWITCH_GAMMAS.items()
        },
    }


def _quantile(values: Sequence[float], q: float) -> float:
    """Return the *q* quantile using :func:`numpy.quantile` for interpolation."""

    if math.isnan(q):
        return math.nan
    bounded_q = min(max(q, 0.0), 1.0)
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan
    return float(np.quantile(array, bounded_q, method="linear"))


def _book_predictions(rows: Sequence[HybridRow]) -> Dict[str, List[float]]:
    predictions: Dict[str, List[float]] = {}
    for row in rows:
        if not row.keep:
            continue
        if math.isnan(row.prediction):
            continue
        predictions.setdefault(row.book, []).append(row.prediction)
    return predictions


def _check_safety_limits(policy: HybridPolicy, settings: Settings) -> List[str]:
    reasons: List[str] = []
    for book, (lower, upper) in policy.tails.items():
        if lower >= upper:
            reasons.append(f"{book}: tail lower bound must be < upper bound")
        limit_lower, limit_upper = settings.KILL_SWITCH_TAILS.get(book, (0.0, 1.0))
        if lower < limit_lower:
            reasons.append(
                f"{book}: tail lower {lower:.4f} below safety limit {limit_lower:.4f}"
            )
        if upper > limit_upper:
            reasons.append(
                f"{book}: tail upper {upper:.4f} above safety limit {limit_upper:.4f}"
            )
        if not 0.0 <= lower <= 1.0:
            reasons.append(f"{book}: tail lower {lower:.4f} outside [0, 1]")
        if not 0.0 <= upper <= 1.0:
            reasons.append(f"{book}: tail upper {upper:.4f} outside [0, 1]")

    for book, gamma in policy.gammas.items():
        limit_gamma = settings.KILL_SWITCH_GAMMAS.get(book, math.inf)
        if gamma > limit_gamma:
            reasons.append(
                f"{book}: gamma {gamma:.4f} exceeds safety limit {limit_gamma:.4f}"
            )
        if gamma <= 0.0:
            reasons.append(f"{book}: gamma {gamma:.4f} must be positive")

    for book, rule in policy.gamma_rules.items():
        if rule.lo < 0.0:
            reasons.append(f"{book}: gamma lo breakpoint {rule.lo:.4f} below 0")
        if rule.hi < rule.lo:
            reasons.append(
                f"{book}: gamma hi breakpoint {rule.hi:.4f} below lo {rule.lo:.4f}"
            )
        if any(level <= 0.0 for level in rule.levels):
            reasons.append(f"{book}: gamma levels must be positive")
        if (
            rule.target_percentile is not None
            and not 0.0 <= rule.target_percentile <= 1.0
        ):
            reasons.append(
                f"{book}: gamma target percentile {rule.target_percentile:.4f} outside [0, 1]"
            )

    return reasons


def _resolve_tail_thresholds(
    policy: HybridPolicy,
    book_predictions: Mapping[str, Sequence[float]],
) -> Dict[str, tuple[float, float] | None]:
    thresholds: Dict[str, tuple[float, float] | None] = {}
    for book, (lower, upper) in policy.tails.items():
        predictions = list(book_predictions.get(book, ()))
        if not predictions:
            thresholds[book] = None
            continue
        thresholds[book] = (
            _quantile(predictions, lower),
            _quantile(predictions, upper),
        )
    return thresholds


def _gamma_components(
    row: HybridRow,
    *,
    gammas: Mapping[str, float],
    gamma_rules: Mapping[str, GammaRule],
) -> tuple[
    float,
    float | None,
    float,
    float | None,
    float | None,
    float | None,
]:
    base_gamma = float(gammas.get(row.book, 1.0))
    rule = gamma_rules.get(row.book)
    percentile = None
    target_percentile = None
    margin = None
    rule_gamma = None
    if rule is not None:
        percentile = percentile_for_score(row.prediction, rule.percentiles)
        target_percentile = (
            rule.target_percentile if rule.target_percentile is not None else None
        )
        margin = normalised_margin(percentile, target_percentile)
        candidate_gamma = rule.level_for_margin(margin)
        if candidate_gamma is not None:
            rule_gamma = float(candidate_gamma)
    final_gamma = rule_gamma if rule_gamma is not None else base_gamma
    return (
        base_gamma,
        rule_gamma,
        final_gamma,
        percentile,
        target_percentile,
        margin,
    )


def _apply_tail_on_kill(
    row: HybridRow, thresholds: tuple[float, float]
) -> float | None:
    """Return the tail-only pnl for kill days.

    Mirrors the notebook's ``_apply_tail_on_kill`` helper: when VVIX kill
    conditions are active we only trade the prediction tails, shorting the
    lower tail and going long on the upper tail. When predictions fall
    inside the tail bounds the contribution is zero.
    """

    pnl_value = row.pnl
    if math.isnan(pnl_value):
        return None
    lower, upper = thresholds
    if math.isnan(lower) or math.isnan(upper):
        return None
    prediction = row.prediction
    if math.isnan(prediction):
        return None
    if prediction <= lower:
        return -pnl_value
    if prediction >= upper:
        return pnl_value
    return 0.0


def _apply_gamma_on_kill(
    row: HybridRow,
    *,
    gammas: Mapping[str, float] | None = None,
    gamma_rules: Mapping[str, GammaRule] | None = None,
) -> float | None:
    """Return the gamma-scaled pnl for kill days.

    Mirrors the notebook's gamma scaling: applies the final gamma (from rules or base gamma)
    to scale the PnL when VVIX kill conditions are active. If a gamma rule is present for the
    book, the rule's gamma is used; otherwise, the base gamma is applied. Returns None if
    PnL or gamma values are invalid (NaN or infinite).
    """
    pnl_value = row.pnl
    if math.isnan(pnl_value):
        return None
    _, _, final_gamma, _, _, _ = _gamma_components(
        row,
        gammas=gammas or {},
        gamma_rules=gamma_rules or {},
    )
    if math.isnan(final_gamma) or math.isinf(final_gamma):
        return None
    return pnl_value * final_gamma


def _aggregate_daily_pnl(
    rows: Sequence[HybridRow],
    *,
    tail_thresholds: Mapping[str, tuple[float, float] | None] | None = None,
    gammas: Mapping[str, float] | None = None,
    gamma_rules: Mapping[str, GammaRule] | None = None,
) -> List[float]:
    daily: MutableMapping[date, float] = {}
    tail_thresholds = tail_thresholds or {}
    gammas = gammas or {}
    gamma_rules = gamma_rules or {}

    for row in rows:
        if not row.keep:
            continue
        base_pnl = row.pnl
        if math.isnan(base_pnl):
            continue
        if row.kill_switch:
            thresholds = tail_thresholds.get(row.book)
            if thresholds is not None:
                pnl_value = _apply_tail_on_kill(row, thresholds)
            else:
                pnl_value = _apply_gamma_on_kill(
                    row,
                    gammas=gammas,
                    gamma_rules=gamma_rules,
                )
        else:
            pnl_value = base_pnl

        if pnl_value is None:
            continue
        if math.isnan(pnl_value) or math.isinf(pnl_value):
            continue
        daily[row.open_date] = daily.get(row.open_date, 0.0) + pnl_value

    ordered_dates = sorted(daily)
    return [daily[day] for day in ordered_dates]


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(str(key))
                seen.add(str(key))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised: dict[str, object] = {}
            for column in columns:
                value = row.get(column)
                if isinstance(value, datetime):
                    serialised[column] = value.date().isoformat()
                elif isinstance(value, date):
                    serialised[column] = value.isoformat()
                elif isinstance(value, float):
                    serialised[column] = float(value)
                else:
                    serialised[column] = value
            writer.writerow(serialised)


def _gamma_backtest_records(
    rows: Sequence[HybridRow], policy: HybridPolicy
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in rows:
        (
            base_gamma,
            rule_gamma,
            final_gamma,
            percentile,
            target_percentile,
            margin,
        ) = _gamma_components(
            row,
            gammas=policy.gammas,
            gamma_rules=policy.gamma_rules,
        )
        records.append(
            {
                "book": row.book,
                "open_date": row.open_date,
                "prediction": row.prediction,
                "pnl": row.pnl,
                "keep": row.keep,
                "baseline_gamma": base_gamma,
                "rule_gamma": rule_gamma,
                "final_gamma": final_gamma,
                "percentile": percentile,
                "target_percentile": target_percentile,
                "margin": margin,
            }
        )
    return records


def _normalise_policy(candidate: Mapping[str, object]) -> HybridPolicy:
    raw_tails = candidate.get("tails", {})
    tails: Dict[str, tuple[float, float]] = {}
    if isinstance(raw_tails, Mapping):
        for book, bounds in raw_tails.items():
            if not isinstance(book, str):
                continue
            if isinstance(bounds, Sequence) and len(bounds) == 2:
                try:
                    lower = float(bounds[0])
                    upper = float(bounds[1])
                except (TypeError, ValueError):
                    continue
                tails[book] = (lower, upper)

    raw_gammas = candidate.get("gammas", {})
    gammas: Dict[str, float] = {}
    if isinstance(raw_gammas, Mapping):
        for book, gamma in raw_gammas.items():
            if not isinstance(book, str):
                continue
            try:
                gammas[book] = float(gamma)
            except (TypeError, ValueError):
                continue

    raw_rules = candidate.get("gamma_rules", {})
    gamma_rules: Dict[str, GammaRule] = {}
    if isinstance(raw_rules, Mapping):
        for book, payload in raw_rules.items():
            if not isinstance(book, str):
                continue
            if not isinstance(payload, Mapping):
                continue
            rule = rule_from_mapping(payload)
            if rule is not None:
                gamma_rules[book] = rule

    return HybridPolicy(tails=tails, gammas=gammas, gamma_rules=gamma_rules)


def _metrics_constraints(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
) -> List[str]:
    reasons: List[str] = []
    baseline_edp = baseline.get("EDP")
    candidate_edp = metrics.get("EDP")
    if baseline_edp is not None and candidate_edp is not None:
        if math.isfinite(baseline_edp) and math.isfinite(candidate_edp):
            if candidate_edp < 0.95 * baseline_edp:
                reasons.append("EDP fell below 95% of the no-kill baseline")

    baseline_cvar = baseline.get("CVaR")
    candidate_cvar = metrics.get("CVaR")
    if baseline_cvar is not None and candidate_cvar is not None:
        if math.isfinite(baseline_cvar) and math.isfinite(candidate_cvar):
            if candidate_cvar < baseline_cvar:
                reasons.append("CVaR is worse than the no-kill baseline")

    return reasons


def _is_finite(value: float | None) -> TypeGuard[float]:
    return value is not None and math.isfinite(value)


def select_hybrid_policy(
    apply_dir: Path,
    candidates: Sequence[Mapping[str, object]] | Mapping[str, object],
    output_path: Path,
    *,
    settings: Settings | None = None,
    timestamp_factory: Callable[[], datetime] | None = None,
) -> HybridSelectionResult:
    """Evaluate candidate hybrid policies and persist an accepted policy.

    Parameters
    ----------
    apply_dir:
        Directory containing the frozen ``hybrid_input.csv`` payload that was
        produced by :mod:`xdte.model.apply`. The data is used to compute the
        baseline metrics and the per-book predictions required to evaluate the
        candidates.
    candidates:
        One or more hybrid policy candidates encoded as mappings. Each
        candidate may specify ``tails`` and ``gammas`` overrides; raw payloads
        are normalised before evaluation so sequences and mappings are both
        supported for convenience.
    output_path:
        Location where the JSON audit artefact should be written. Parent
        directories are created automatically when missing.
    settings:
        Optional settings overrides. When ``None`` (the default) the global
        :class:`~xdte.config.Settings` singleton is used.
    timestamp_factory:
        Optional callable that returns a timezone-aware :class:`datetime`
        instance. Primarily intended for tests so deterministic timestamps can
        be injected into the audit payload.

    Returns
    -------
    HybridSelectionResult
        Summary of the baseline metrics, per-candidate evaluations and the
        policy chosen for persistence.

    Raises
    ------
    OSError
        Propagated if the destination directory cannot be created or if the
        audit artefact cannot be written.
    """

    active_settings = settings or get_settings()
    resolved_apply = apply_dir.resolve()
    rows = _load_hybrid_rows(resolved_apply)
    baseline_series = _aggregate_daily_pnl(rows)
    baseline_metrics = portfolio_metrics(baseline_series, alpha=active_settings.ALPHA)

    book_predictions = _book_predictions(rows)

    candidate_sequence: Sequence[Mapping[str, object]]
    if isinstance(candidates, Mapping):
        candidate_sequence = [candidates]
    else:
        candidate_sequence = list(candidates)

    evaluations: List[HybridCandidateEvaluation] = []

    for candidate in candidate_sequence:
        policy = _normalise_policy(candidate)
        reasons = _check_safety_limits(policy, active_settings)
        thresholds = _resolve_tail_thresholds(policy, book_predictions)
        if any(value is None for value in thresholds.values()):
            for book, value in thresholds.items():
                if value is None:
                    reasons.append(
                        f"{book}: insufficient prediction data to compute tail thresholds"
                    )

        daily_values = _aggregate_daily_pnl(
            rows,
            tail_thresholds={
                book: value for book, value in thresholds.items() if value is not None
            },
            gammas=policy.gammas,
            gamma_rules=policy.gamma_rules,
        )
        metrics = portfolio_metrics(daily_values, alpha=active_settings.ALPHA)

        constraint_reasons = _metrics_constraints(metrics, baseline_metrics)
        reasons.extend(constraint_reasons)
        accepted = not reasons

        evaluations.append(
            HybridCandidateEvaluation(
                policy=policy,
                metrics=metrics,
                accepted=accepted,
                reasons=reasons,
            )
        )

    best: HybridCandidateEvaluation | None = None
    for evaluation in evaluations:
        if not evaluation.accepted:
            continue
        if best is None:
            best = evaluation
            continue
        current_pf = evaluation.metrics.get("PF")
        best_pf = best.metrics.get("PF")
        current_cvar = evaluation.metrics.get("CVaR")
        best_cvar = best.metrics.get("CVaR")
        if _is_finite(current_pf):
            current_pf_value = current_pf
            if not _is_finite(best_pf):
                best = evaluation
                continue
            best_pf_value = best_pf
            if current_pf_value > best_pf_value:
                best = evaluation
                continue
        if _is_finite(current_cvar):
            current_cvar_value = current_cvar
            if not _is_finite(best_cvar):
                best = evaluation
                continue
            best_cvar_value = best_cvar
            if current_cvar_value > best_cvar_value:
                best = evaluation

    selected_policy = best.policy if best is not None else HybridPolicy({}, {})
    accepted = best is not None

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp_factory or (lambda: datetime.now(timezone.utc))
    evaluation_time = timestamp()
    audit_payload = {
        "evaluated_at": evaluation_time.isoformat(),
        "baseline_metrics": _serialise_metrics(baseline_metrics),
        "candidates": [evaluation.to_dict() for evaluation in evaluations],
        "selected_policy": selected_policy.to_dict(),
        "accepted": accepted,
        "settings": _serialise_settings(active_settings),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2, sort_keys=True)

    gamma_records = _gamma_backtest_records(rows, selected_policy)
    if gamma_records:
        gamma_path = resolved_apply / "gamma_backtest.csv"
        _write_rows(gamma_path, gamma_records)

    return HybridSelectionResult(
        baseline_metrics=baseline_metrics,
        evaluations=evaluations,
        selected_policy=selected_policy,
        accepted=accepted,
        output_path=output_path,
    )
