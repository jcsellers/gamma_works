"""Aggregation helpers for live trading directives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, TypeVar

from xdte.config import Settings, get_settings
from xdte.gamma import (
    GammaRule,
    decile_margin,
    normalised_margin,
    percentile_for_score,
    rule_from_mapping,
)

from .decide import DecisionRecord, LiveDecisionResult

_DEFAULT_CONFIDENCE_MARGIN_SCALE = Settings().CONFIDENCE_MARGIN_SCALE


@dataclass(frozen=True)
class FinalAction:
    """Human-friendly rendering of the final directive for a (session, book)."""

    session: str
    book: str
    rule_action: str
    final_action: str
    size_gamma: float
    score: float | None
    margin_to_threshold: float | None
    kill_flag: bool
    policy_type: str
    thresholds: Mapping[str, float]
    decile: int | None
    reason_text: str
    rationale: Mapping[str, object]


def _normalise_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    return None


def _normalise_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _iter_mappings(payload: object) -> Iterable[Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        return []
    stack: list[Mapping[str, object]] = [payload]
    seen: set[int] = set()
    collected: list[Mapping[str, object]] = []
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        collected.append(current)
        for value in current.values():
            if isinstance(value, Mapping):
                stack.append(value)
    return collected


T = TypeVar("T")


def _collect_mapping_setting(
    sources: Sequence[Mapping[str, object]],
    key: str,
    converter: Callable[[object], T | None],
) -> dict[str, T]:
    collected: dict[str, T] = {}
    for source in sources:
        for mapping in _iter_mappings(source):
            candidate = mapping.get(key)
            if not isinstance(candidate, Mapping):
                continue
            for book, raw_value in candidate.items():
                if not isinstance(book, str):
                    continue
                converted = converter(raw_value)
                if converted is None:
                    continue
                collected.setdefault(book, converted)
    return collected


def _collect_scalar_setting(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> float | None:
    for source in sources:
        for mapping in _iter_mappings(source):
            value = mapping.get(key)
            numeric = _normalise_float(value)
            if numeric is not None:
                return numeric
    return None


def _convert_tail_bounds(raw_value: object) -> tuple[float, float] | None:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        if len(raw_value) == 2:
            lower = _normalise_float(raw_value[0])
            upper = _normalise_float(raw_value[1])
            if lower is not None and upper is not None:
                return (lower, upper)
    return None


def _convert_gamma(raw_value: object) -> float | None:
    return _normalise_float(raw_value)


def _collect_gamma_rules(
    sources: Sequence[Mapping[str, object]],
) -> dict[str, GammaRule]:
    rules: dict[str, GammaRule] = {}
    for source in sources:
        for mapping in _iter_mappings(source):
            payload = mapping.get("gamma_rules")
            if not isinstance(payload, Mapping):
                continue
            for book, raw_rule in payload.items():
                if not isinstance(book, str) or not isinstance(raw_rule, Mapping):
                    continue
                rule = rule_from_mapping(raw_rule)
                if rule is not None:
                    rules.setdefault(book, rule)
    return rules


def _collect_kill_days(sources: Sequence[Mapping[str, object]]) -> set[str]:
    flagged: set[str] = set()

    def _ingest(entry: object) -> None:
        if isinstance(entry, Mapping):
            for book, value in entry.items():
                if isinstance(book, str) and bool(value):
                    flagged.add(book)
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            for item in entry:
                if isinstance(item, str):
                    flagged.add(item)
        elif isinstance(entry, str):
            flagged.add(entry)

    for source in sources:
        for mapping in _iter_mappings(source):
            for key in ("kill_day", "kill_days"):
                if key in mapping:
                    _ingest(mapping[key])
    return flagged


def _extract_top_features(trade_payload: Mapping[str, object]) -> list[list[object]]:
    shap_payload = trade_payload.get("shap")
    if not isinstance(shap_payload, Mapping):
        return []
    candidates: object = shap_payload.get("contributions")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        candidates = shap_payload.get("top_contributors")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            return []
    rendered: list[list[object]] = []
    for entry in candidates:
        if not isinstance(entry, Mapping):
            continue
        feature = entry.get("feature")
        impact_value = entry.get("impact")
        if impact_value is None:
            impact_value = entry.get("mean_abs_shap")
        impact = _normalise_float(impact_value)
        if isinstance(feature, str) and impact is not None:
            rendered.append([feature, impact])
        if len(rendered) >= 3:
            break
    return rendered


def _compute_keep_fraction(trade_payload: Mapping[str, object]) -> float | None:
    folds = trade_payload.get("folds")
    if isinstance(folds, Sequence) and folds:
        votes = 0
        for fold in folds:
            if isinstance(fold, Mapping) and fold.get("keep"):
                votes += 1
        return votes / len(folds)
    keep_value = trade_payload.get("keep")
    if isinstance(keep_value, bool):
        return 1.0 if keep_value else 0.0
    return None


def _compute_confidence(
    keep_fraction: float | None,
    margin: float | None,
    *,
    margin_scale: float,
) -> float | None:
    """Return a composite confidence score in the range [0, 1].

    The score blends the model vote share (``keep_fraction``) with the absolute
    decision margin. The margin contribution follows an exponential saturation
    curve, ``1 - exp(-|margin| / 0.05)``, so confidence quickly approaches one
    once predictions clear the threshold by a few percentage points. Missing
    inputs are ignored; ``None`` is returned only when both inputs are
    unavailable. The computed value is exposed under ``why['confidence']`` in
    JSON outputs.
    """

    components: list[float] = []
    if keep_fraction is not None:
        components.append(min(max(keep_fraction, 0.0), 1.0))
    if margin is not None:
        scale = margin_scale if margin_scale > 0 else _DEFAULT_CONFIDENCE_MARGIN_SCALE
        margin_component = 1.0 - math.exp(-abs(margin) / scale)
        components.append(min(max(margin_component, 0.0), 1.0))
    if not components:
        return None
    return sum(components) / len(components)


def _build_reason_text(
    *,
    rule_action: str,
    final_action: str,
    margin: float | None,
    keep_fraction: float | None,
    tails_trigger: Mapping[str, object] | None,
    gamma_capped: bool,
    gamma_value: float,
    kill_day_trigger: bool,
    gamma_rule_value: float | None,
    gamma_rule_margin: float | None,
) -> str:
    parts: list[str] = []
    if kill_day_trigger:
        parts.append("Kill day override")
    if tails_trigger is not None:
        side = tails_trigger.get("side")
        bound = tails_trigger.get("bound")
        prediction = tails_trigger.get("prediction")
        if isinstance(side, str) and isinstance(bound, (int, float)):
            formatted = f"Tail {side} ({float(bound):0.3f})"
            if isinstance(prediction, (int, float)) and math.isfinite(
                float(prediction)
            ):
                formatted += f" vs {float(prediction):0.3f}"
            parts.append(formatted)
        else:
            parts.append("Tail guardrail")
    if gamma_capped:
        parts.append(f"Gamma capped to {gamma_value:0.2f}")
    if gamma_rule_value is not None:
        if gamma_rule_margin is not None:
            parts.append(
                f"Gamma rule {gamma_rule_value:0.2f} (margin {gamma_rule_margin:+0.2f})"
            )
        else:
            parts.append(f"Gamma rule {gamma_rule_value:0.2f}")
    if final_action != rule_action:
        parts.append(f"Rule {rule_action} ⇒ {final_action}")
    if margin is not None:
        parts.append(f"Margin {margin:+0.3f}")
    if keep_fraction is not None:
        parts.append(f"Keep votes {keep_fraction*100:0.0f}%")
    if not parts:
        parts.append(f"Rule {rule_action}")
    return "; ".join(parts)


def _gather_decision_payloads(
    decisions: Sequence[DecisionRecord],
) -> tuple[
    dict[tuple[str, str], Mapping[str, object]],
    dict[tuple[str, str], float],
    dict[str, tuple[float, float]],
]:
    trades: dict[tuple[str, str], Mapping[str, object]] = {}
    gammas: dict[tuple[str, str], float] = {}
    tails: dict[str, tuple[float, float]] = {}

    for record in decisions:
        key = (record.session, record.book)
        if record.action == "TRADE_SIGNAL" and isinstance(record.value, Mapping):
            trades[key] = record.value
        elif record.action == "ADJUST_GAMMA":
            gamma_value = _normalise_float(record.value)
            if gamma_value is not None:
                gammas[key] = gamma_value
        elif record.action == "SET_TAILS":
            bounds = _convert_tail_bounds(record.value)
            if bounds is not None:
                tails[record.book] = bounds
    return trades, gammas, tails


def compile_operational_directives(
    result: LiveDecisionResult,
    *,
    settings: Settings | None = None,
) -> list[FinalAction]:
    """Merge raw decision records into operator-friendly directives."""

    trade_signals, gamma_overrides, tails_from_decisions = _gather_decision_payloads(
        result.decisions
    )

    active_settings = settings or get_settings()

    policy_sources: list[Mapping[str, object]] = []
    if isinstance(result.policy, Mapping):
        policy_sources.append(result.policy)
    if isinstance(result.manifest, Mapping):
        policy_sources.append(result.manifest)

    kill_switch_tails = _collect_mapping_setting(
        policy_sources,
        "kill_switch_tails",
        _convert_tail_bounds,
    )
    if tails_from_decisions:
        kill_switch_tails = {**kill_switch_tails, **tails_from_decisions}

    kill_switch_gammas = _collect_mapping_setting(
        policy_sources,
        "kill_switch_gammas",
        _convert_gamma,
    )
    kill_days = _collect_kill_days(policy_sources)
    gamma_rules = _collect_gamma_rules(policy_sources)

    no_trade_margin = _collect_scalar_setting(policy_sources, "no_trade_margin") or 0.0
    keep_quorum = _collect_scalar_setting(policy_sources, "keep_quorum") or 0.0

    books = set(trade_signals.keys()) | set(gamma_overrides.keys())
    actions: list[FinalAction] = []

    for session, book in sorted(books):
        trade_payload = trade_signals.get((session, book))
        keep_flag = False
        prediction = None
        threshold = None
        margin = None
        keep_fraction = None
        top_features: list[list[object]] = []
        decile = None

        if trade_payload is not None:
            keep_value = trade_payload.get("keep")
            keep_flag = bool(keep_value)
            prediction = _normalise_float(trade_payload.get("prediction"))
            threshold = _normalise_float(trade_payload.get("threshold"))
            if prediction is not None and threshold is not None:
                margin = prediction - threshold
            keep_fraction = _compute_keep_fraction(trade_payload)
            top_features = _extract_top_features(trade_payload)
            decile = _normalise_int(trade_payload.get("decile"))

        rule_action = "KEEP_SHORT" if keep_flag else "SKIP"
        final_action = rule_action

        gamma_base = gamma_overrides.get((session, book), 1.0)
        gamma_value = gamma_base
        gamma_rule = gamma_rules.get(book)
        gamma_rule_margin = None
        gamma_rule_value = None
        prediction_percentile = None
        threshold_percentile = None
        if gamma_rule is not None:
            prediction_percentile = percentile_for_score(
                prediction, gamma_rule.percentiles
            )
            if gamma_rule.target_percentile is not None:
                threshold_percentile = gamma_rule.target_percentile
            else:
                threshold_percentile = percentile_for_score(
                    threshold, gamma_rule.percentiles
                )
            margin_from_percentile = normalised_margin(
                prediction_percentile, threshold_percentile
            )
            if margin_from_percentile is None:
                gamma_rule_margin = decile_margin(decile)
            else:
                gamma_rule_margin = margin_from_percentile
            gamma_rule_value = gamma_rule.level_for_margin(gamma_rule_margin)
            if gamma_rule_value is not None:
                gamma_value = gamma_rule_value
        kill_switch = False
        gamma_cap = kill_switch_gammas.get(book)
        gamma_capped = False
        if gamma_cap is not None and gamma_cap < gamma_value:
            gamma_value = gamma_cap
            gamma_capped = True
            kill_switch = True

        tails_bounds = kill_switch_tails.get(book)
        tails_trigger: dict[str, float | str] | None = None
        if tails_bounds is not None and prediction is not None:
            lower, upper = tails_bounds
            if prediction < lower:
                tails_trigger = {
                    "side": "low",
                    "bound": lower,
                    "prediction": prediction,
                }
            elif prediction > upper:
                tails_trigger = {
                    "side": "high",
                    "bound": upper,
                    "prediction": prediction,
                }
        if tails_trigger is not None:
            final_action = "SKIP"
            kill_switch = True

        kill_day_trigger = book in kill_days
        if kill_day_trigger:
            final_action = "SKIP"
            kill_switch = True
            if gamma_rule is not None:
                kill_cap = (
                    gamma_rule.kill_day_level
                    if gamma_rule.kill_day_level is not None
                    else (min(gamma_rule.levels) if gamma_rule.levels else gamma_value)
                )
                if kill_cap is not None and gamma_value > kill_cap:
                    gamma_value = kill_cap

        if (
            final_action != "SKIP"
            and margin is not None
            and abs(margin) < no_trade_margin
        ):
            if keep_fraction is not None and keep_fraction < keep_quorum:
                final_action = "SKIP"

        confidence = _compute_confidence(
            keep_fraction,
            margin,
            margin_scale=active_settings.CONFIDENCE_MARGIN_SCALE,
        )

        score = prediction
        margin_to_threshold = margin

        thresholds_map: dict[str, float] = {}
        if threshold is not None:
            thresholds_map["model_threshold"] = threshold
        if tails_bounds is not None:
            lower, upper = tails_bounds
            thresholds_map["tail_lower"] = lower
            thresholds_map["tail_upper"] = upper

        policy_type = "none"
        if kill_day_trigger:
            policy_type = "kill_day"
        elif kill_switch:
            if tails_trigger is not None:
                policy_type = "tails"
            elif gamma_capped:
                policy_type = "gamma"
        elif gamma_rule_value is not None:
            policy_type = "gamma_rule"

        reason_text = _build_reason_text(
            rule_action=rule_action,
            final_action=final_action,
            margin=margin,
            keep_fraction=keep_fraction,
            tails_trigger=tails_trigger,
            gamma_capped=gamma_capped,
            gamma_value=gamma_value,
            kill_day_trigger=kill_day_trigger,
            gamma_rule_value=gamma_rule_value,
            gamma_rule_margin=gamma_rule_margin,
        )

        rationale: MutableMapping[str, object] = {
            "keep": keep_flag,
            "prediction": prediction,
            "threshold": threshold,
            "margin": margin,
            "keep_fraction": keep_fraction,
            "kill_switch": kill_switch,
            "tails": tails_trigger,
            "confidence": confidence,
            "kill_flag": kill_switch,
            "policy_type": policy_type,
            "score": score,
            "margin_to_threshold": margin_to_threshold,
            "reason_text": reason_text,
            "gamma_base": gamma_base,
        }
        if prediction_percentile is not None:
            rationale["prediction_percentile"] = prediction_percentile
        if threshold_percentile is not None:
            rationale["threshold_percentile"] = threshold_percentile
        if gamma_rule_margin is not None:
            rationale["gamma_rule_margin"] = gamma_rule_margin
        if gamma_rule_value is not None:
            rationale["gamma_rule_value"] = gamma_rule_value
        if thresholds_map:
            rationale["thresholds"] = thresholds_map
        if decile is not None:
            rationale["decile"] = decile
        if gamma_capped:
            rationale["gamma_cap"] = gamma_cap
        if tails_bounds is not None:
            rationale.setdefault("tail_bounds", list(tails_bounds))
        if kill_day_trigger:
            rationale["kill_day"] = True
        if top_features:
            rationale["top_features"] = top_features

        actions.append(
            FinalAction(
                session=session,
                book=book,
                rule_action=rule_action,
                final_action=final_action,
                size_gamma=gamma_value,
                score=score,
                margin_to_threshold=margin_to_threshold,
                kill_flag=kill_switch,
                policy_type=policy_type,
                thresholds=thresholds_map,
                decile=decile,
                reason_text=reason_text,
                rationale=rationale,
            )
        )

    return actions


__all__ = ["FinalAction", "compile_operational_directives"]
