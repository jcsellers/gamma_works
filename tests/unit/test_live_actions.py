from __future__ import annotations

import math
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from xdte.cli import _format_actions_table
from xdte.config import Settings
from xdte.live.actions import compile_operational_directives
from xdte.live.decide import DecisionRecord, LiveDecisionResult, SessionDecision


def _make_result(
    *,
    decisions: list[DecisionRecord],
    policy: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
) -> LiveDecisionResult:
    session = SessionDecision(
        session="11:00",
        open_date=date(2024, 7, 1),
        features={},
        warnings=(),
    )
    return LiveDecisionResult(
        kit_dir=Path("/kit"),
        manifest=manifest or {},
        policy=policy or {},
        sessions=(session,),
        decisions=tuple(decisions),
        warnings=(),
        generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )


def test_compile_actions_flags_tail_violation() -> None:
    decision = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.99,
            "threshold": 0.50,
            "folds": [{"keep": True}, {"keep": True}],
        },
    )
    result = _make_result(
        decisions=[decision],
        policy={"settings": {"kill_switch_tails": {"BOOK": [0.05, 0.95]}}},
    )

    actions = compile_operational_directives(result)

    assert len(actions) == 1
    action = actions[0]
    assert action.rule_action == "KEEP_SHORT"
    assert action.final_action == "SKIP"
    assert action.kill_flag is True
    assert action.policy_type == "tails"
    tails = action.rationale["tails"]
    assert isinstance(tails, dict)
    assert tails["side"] == "high"
    assert action.reason_text


def test_compile_actions_caps_gamma() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.80,
            "threshold": 0.70,
            "folds": [{"keep": True}, {"keep": False}],
        },
    )
    gamma = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="ADJUST_GAMMA",
        value=1.2,
    )
    result = _make_result(
        decisions=[trade, gamma],
        policy={"settings": {"kill_switch_gammas": {"BOOK": 0.8}}},
    )

    actions = compile_operational_directives(result)

    action = actions[0]
    assert action.size_gamma == pytest.approx(0.8)
    assert action.kill_flag is True
    assert action.policy_type == "gamma"
    assert action.rationale["gamma_cap"] == pytest.approx(0.8)
    assert "Gamma capped" in action.reason_text


def test_compile_actions_applies_gamma_rules() -> None:
    rule_policy = {
        "gamma_rules": {
            "BOOK": {
                "lo": 0.2,
                "hi": 0.6,
                "levels": [0.5, 0.75, 1.0],
                "percentiles": [[0.0, 0.0], [1.0, 1.0]],
                "target_percentile": 0.8,
            }
        }
    }

    def _gamma_for_prediction(prediction: float) -> float:
        decision = DecisionRecord(
            session="11:00",
            book="BOOK",
            action="TRADE_SIGNAL",
            value={"keep": True, "prediction": prediction, "threshold": 0.70},
        )
        result = _make_result(decisions=[decision], policy=rule_policy)
        action = compile_operational_directives(result)[0]
        assert "Gamma rule" in action.reason_text
        assert action.policy_type == "gamma_rule"
        return action.size_gamma

    low = _gamma_for_prediction(0.8)
    mid = _gamma_for_prediction(0.9)
    high = _gamma_for_prediction(0.98)

    assert low == pytest.approx(0.5)
    assert mid == pytest.approx(0.75)
    assert high == pytest.approx(1.0)
    assert low < mid < high


def test_compile_actions_caps_gamma_rules_on_kill_day() -> None:
    decision = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={"keep": True, "prediction": 0.95, "threshold": 0.70},
    )
    policy = {
        "gamma_rules": {
            "BOOK": {
                "lo": 0.1,
                "hi": 0.3,
                "levels": [0.5, 0.75, 1.0],
                "percentiles": [[0.0, 0.0], [1.0, 1.0]],
                "target_percentile": 0.6,
                "kill_day_level": 0.4,
            }
        },
        "kill_day": {"BOOK": True},
    }
    result = _make_result(decisions=[decision], policy=policy)

    action = compile_operational_directives(result)[0]

    assert action.size_gamma == pytest.approx(0.4)
    assert action.policy_type == "kill_day"
    assert "Kill day" in action.reason_text


def test_compile_actions_respects_kill_day() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.82,
            "threshold": 0.70,
            "folds": [{"keep": True}, {"keep": True}],
        },
    )
    result = _make_result(
        decisions=[trade],
        policy={"kill_day": {"BOOK": True}},
    )

    actions = compile_operational_directives(result)

    action = actions[0]
    assert action.rule_action == "KEEP_SHORT"
    assert action.final_action == "SKIP"
    assert action.kill_flag is True
    assert action.policy_type == "kill_day"
    assert action.rationale["kill_day"] is True
    assert "Kill day" in action.reason_text


def test_compile_actions_enforces_no_trade_band() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.71,
            "threshold": 0.69,
            "folds": [
                {"keep": True},
                {"keep": False},
                {"keep": False},
            ],
        },
    )
    policy = {
        "settings": {
            "no_trade_margin": 0.05,
            "keep_quorum": 0.75,
        }
    }
    result = _make_result(decisions=[trade], policy=policy)

    actions = compile_operational_directives(result)

    action = actions[0]
    assert action.final_action == "SKIP"
    assert action.kill_flag is False
    assert action.margin_to_threshold == pytest.approx(0.02)
    assert action.rationale["keep_fraction"] == pytest.approx(1 / 3)


_DEFAULT_MARGIN_SCALE = Settings().CONFIDENCE_MARGIN_SCALE


def _expected_confidence(
    keep_fraction: float | None,
    margin: float | None,
    *,
    margin_scale: float = _DEFAULT_MARGIN_SCALE,
) -> float | None:
    components: list[float] = []
    if keep_fraction is not None:
        components.append(min(max(keep_fraction, 0.0), 1.0))
    if margin is not None:
        scale = margin_scale if margin_scale > 0 else _DEFAULT_MARGIN_SCALE
        components.append(min(max(1.0 - math.exp(-abs(margin) / scale), 0.0), 1.0))
    if not components:
        return None
    return sum(components) / len(components)


def test_compile_actions_calculates_confidence() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.78,
            "threshold": 0.65,
            "folds": [
                {"keep": True},
                {"keep": True},
                {"keep": False},
                {"keep": True},
            ],
        },
    )
    result = _make_result(decisions=[trade])

    actions = compile_operational_directives(result)

    confidence = actions[0].rationale["confidence"]
    expected = _expected_confidence(keep_fraction=3 / 4, margin=0.13)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert confidence == pytest.approx(expected)


def test_compile_actions_respects_configured_confidence_scale() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.78,
            "threshold": 0.65,
            "folds": [
                {"keep": True},
                {"keep": True},
                {"keep": False},
                {"keep": True},
            ],
        },
    )
    result = _make_result(decisions=[trade])
    custom_settings = Settings(CONFIDENCE_MARGIN_SCALE=0.1)

    actions = compile_operational_directives(result, settings=custom_settings)

    confidence = actions[0].rationale["confidence"]
    expected = _expected_confidence(
        keep_fraction=3 / 4,
        margin=0.13,
        margin_scale=custom_settings.CONFIDENCE_MARGIN_SCALE,
    )
    assert isinstance(confidence, float)
    assert confidence == pytest.approx(expected)


def test_compile_actions_confidence_handles_missing_votes() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "prediction": 0.57,
            "threshold": 0.52,
        },
    )
    result = _make_result(decisions=[trade])

    actions = compile_operational_directives(result)

    confidence = actions[0].rationale["confidence"]
    expected = _expected_confidence(keep_fraction=None, margin=0.05)
    assert isinstance(confidence, float)
    assert confidence == pytest.approx(expected)


def test_compile_actions_confidence_absent_without_inputs() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={},
    )
    result = _make_result(decisions=[trade])

    actions = compile_operational_directives(result)

    assert actions[0].rationale["confidence"] is None


def test_format_actions_table_includes_why_column() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.8,
            "threshold": 0.6,
            "folds": [{"keep": True}],
        },
    )
    result = _make_result(decisions=[trade])

    actions = compile_operational_directives(result)
    table = _format_actions_table(result, actions)

    header_line = next(
        line for line in table.splitlines() if "Rule" in line and "WHY" in line
    )
    assert "WHY" in header_line
    assert "Rule" in header_line
    assert actions[0].reason_text in table


def test_compile_actions_uses_shap_contributions() -> None:
    trade = DecisionRecord(
        session="11:00",
        book="BOOK",
        action="TRADE_SIGNAL",
        value={
            "keep": True,
            "prediction": 0.7,
            "threshold": 0.6,
            "folds": [],
            "shap": {
                "contributions": [
                    {"feature": "alpha", "mean_abs_shap": 0.3},
                    {"feature": "beta", "mean_abs_shap": 0.2},
                ]
            },
        },
    )
    result = _make_result(decisions=[trade])

    action = compile_operational_directives(result)[0]

    assert action.rationale["top_features"] == [["alpha", 0.3], ["beta", 0.2]]
