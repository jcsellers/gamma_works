from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from xdte.config import Settings
from xdte.gamma import GammaRule
from xdte.model.hybrid import (
    HybridPolicy,
    HybridRow,
    _aggregate_daily_pnl,
    _metrics_constraints,
    _quantile,
    select_hybrid_policy,
)


@pytest.mark.parametrize(
    ("metrics", "baseline", "expected"),
    [
        (
            {"EDP": 94.0, "CVaR": 1.5},
            {"EDP": 100.0, "CVaR": 1.0},
            ["EDP fell below 95% of the no-kill baseline"],
        ),
        (
            {"EDP": 100.0, "CVaR": -1.1},
            {"EDP": 100.0, "CVaR": -1.0},
            ["CVaR is worse than the no-kill baseline"],
        ),
        (
            {"EDP": 96.0, "CVaR": -0.4},
            {"EDP": 100.0, "CVaR": -0.5},
            [],
        ),
    ],
)
def test_metrics_constraints_reasons(
    metrics: dict[str, float], baseline: dict[str, float], expected: list[str]
) -> None:
    assert _metrics_constraints(metrics, baseline) == expected


def test_quantile_wraps_numpy_behaviour() -> None:
    assert math.isnan(_quantile([], 0.5))
    assert _quantile([1.0, 3.0], -0.5) == pytest.approx(1.0)
    assert _quantile([1.0, 3.0], 1.5) == pytest.approx(3.0)
    assert _quantile([0.0, 10.0], 0.75) == pytest.approx(7.5)
    assert math.isnan(_quantile([1.0, math.nan, 2.0], 0.5))


def _write_hybrid_input(path: Path) -> None:
    rows = [
        {
            "book": "BOOK_A",
            "open_date": "2024-01-01",
            "pnl": 100.0,
            "prediction": 0.90,
            "keep": "1",
        },
        {
            "book": "BOOK_A",
            "open_date": "2024-01-02",
            "pnl": -20.0,
            "prediction": 0.10,
            "keep": "1",
        },
        {
            "book": "BOOK_A",
            "open_date": "2024-01-03",
            "pnl": 30.0,
            "prediction": 0.50,
            "keep": "1",
        },
        {
            "book": "BOOK_B",
            "open_date": "2024-01-01",
            "pnl": 40.0,
            "prediction": 0.80,
            "keep": "1",
        },
        {
            "book": "BOOK_B",
            "open_date": "2024-01-02",
            "pnl": -10.0,
            "prediction": 0.20,
            "keep": "1",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) + ["pnl_strategy", "L1_vvix_above_ema30"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            payload = dict(row)
            payload["pnl_strategy"] = row["pnl"]
            payload["L1_vvix_above_ema30"] = bool(index % 2)
            writer.writerow(payload)


def test_select_hybrid_policy_enforces_kill_switch_limits(tmp_path: Path) -> None:
    apply_dir = tmp_path / "apply"
    apply_dir.mkdir()
    _write_hybrid_input(apply_dir / "hybrid_input.csv")

    settings = Settings(
        KILL_SWITCH_TAILS={"BOOK_A": (0.05, 0.95)},
        KILL_SWITCH_GAMMAS={"BOOK_B": 0.9},
        ALPHA=0.2,
    )

    candidate_invalid = {
        "tails": {"BOOK_A": [0.01, 0.97]},
        "gammas": {"BOOK_B": 0.95},
        "gamma_rules": {},
    }
    candidate_valid: dict[str, dict[str, object]] = {
        "tails": {},
        "gammas": {},
        "gamma_rules": {},
    }

    output_path = tmp_path / "hybrid" / "audit.json"
    result = select_hybrid_policy(
        apply_dir,
        [candidate_invalid, candidate_valid],
        output_path,
        settings=settings,
    )

    assert result.accepted
    assert result.selected_policy.to_dict() == candidate_valid
    assert any(
        any("BOOK_A" in reason for reason in evaluation.reasons)
        for evaluation in result.evaluations
        if evaluation.policy.to_dict() == candidate_invalid
    )

    payload = json.loads(output_path.read_text())
    assert payload["accepted"] is True
    assert payload["selected_policy"] == candidate_valid
    assert len(payload["candidates"]) == 2

    gamma_path = apply_dir / "gamma_backtest.csv"
    assert gamma_path.exists()
    with gamma_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows, "gamma backtest should contain rows"
    assert {"book", "open_date", "final_gamma", "baseline_gamma"} <= set(rows[0].keys())


def test_aggregate_daily_pnl_applies_gamma_rules() -> None:
    rows = [
        HybridRow(
            book="BOOK",
            open_date=date(2024, 1, 1),
            pnl=100.0,
            prediction=0.9,
            keep=True,
            kill_switch=True,
        )
    ]
    policy = HybridPolicy(
        tails={},
        gammas={"BOOK": 1.0},
        gamma_rules={
            "BOOK": GammaRule(
                lo=0.2,
                hi=0.6,
                levels=(0.5, 0.75, 1.0),
                percentiles=((0.0, 0.0), (1.0, 1.0)),
                target_percentile=0.8,
            )
        },
    )

    values = _aggregate_daily_pnl(
        rows,
        gammas=policy.gammas,
        gamma_rules=policy.gamma_rules,
    )

    assert values == pytest.approx([75.0])


def test_aggregate_daily_pnl_only_applies_tail_on_kill_days() -> None:
    rows = [
        HybridRow(
            book="TAIL",
            open_date=date(2024, 1, 1),
            pnl=50.0,
            prediction=0.05,
            keep=True,
            kill_switch=False,
        ),
        HybridRow(
            book="TAIL",
            open_date=date(2024, 1, 1),
            pnl=60.0,
            prediction=0.95,
            keep=True,
            kill_switch=True,
        ),
    ]
    values = _aggregate_daily_pnl(
        rows,
        tail_thresholds={"TAIL": (0.1, 0.9)},
    )

    assert values == pytest.approx([110.0])


def test_aggregate_daily_pnl_scales_gamma_only_when_kill_true() -> None:
    rows = [
        HybridRow(
            book="GAMMA",
            open_date=date(2024, 1, 1),
            pnl=50.0,
            prediction=0.5,
            keep=True,
            kill_switch=False,
        ),
        HybridRow(
            book="GAMMA",
            open_date=date(2024, 1, 2),
            pnl=40.0,
            prediction=0.5,
            keep=True,
            kill_switch=True,
        ),
    ]

    values = _aggregate_daily_pnl(rows, gammas={"GAMMA": 0.5})

    assert values == pytest.approx([50.0, 20.0])


def test_select_hybrid_policy_prefers_finite_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    apply_dir = tmp_path / "apply"
    apply_dir.mkdir()
    _write_hybrid_input(apply_dir / "hybrid_input.csv")

    settings = Settings()
    output_path = tmp_path / "hybrid" / "audit.json"

    before = datetime.now(timezone.utc)
    select_hybrid_policy(
        apply_dir,
        {"tails": {}, "gammas": {}, "gamma_rules": {}},
        output_path,
        settings=settings,
    )
    after = datetime.now(timezone.utc)

    payload = json.loads(output_path.read_text())
    evaluated_at = datetime.fromisoformat(payload["evaluated_at"])

    assert before <= evaluated_at <= after


def test_select_hybrid_policy_uses_custom_timestamp_factory(tmp_path: Path) -> None:
    apply_dir = tmp_path / "apply"
    apply_dir.mkdir()
    _write_hybrid_input(apply_dir / "hybrid_input.csv")

    settings = Settings()
    output_path = tmp_path / "hybrid" / "audit.json"
    expected = datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc)

    select_hybrid_policy(
        apply_dir,
        {"tails": {}, "gammas": {}, "gamma_rules": {}},
        output_path,
        settings=settings,
        timestamp_factory=lambda: expected,
    )

    payload = json.loads(output_path.read_text())
    assert payload["evaluated_at"] == expected.isoformat()
