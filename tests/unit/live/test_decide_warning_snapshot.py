from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

from tests.unit.test_live_decide import _make_kit
from xdte.live import decide as decide_module
from xdte.live.decide import run_live_decisions
from xdte.live.providers import build_stub_market_provider


class _FailingDailyProvider:
    def __call__(self, open_date: date) -> Mapping[str, object]:
        raise RuntimeError("daily feed unavailable")


def test_run_live_decisions_merges_snapshot_warnings(tmp_path: Path) -> None:
    open_date = date(2024, 1, 5)
    policy = {
        "tails": {
            "PUTS_0DTE_11": [0.1, 0.9],
            "CALLS_1DTE_1515": [0.2, 0.8],
        },
        "gammas": {
            "PUTS_0DTE_11": 0.95,
            "CALLS_0DTE_11": 1.05,
        },
    }
    predictions = {
        "CALLS_0DTE_11": 0.65,
        "PUTS_0DTE_11": 0.72,
        "CALLS_1DTE_1515": 0.48,
        "PUTS_1DTE_1515": 0.52,
    }
    thresholds = {
        "CALLS_0DTE_11": 0.6,
        "PUTS_0DTE_11": 0.55,
        "CALLS_1DTE_1515": 0.5,
        "PUTS_1DTE_1515": 0.5,
    }
    kit_dir = _make_kit(
        tmp_path,
        policy,
        predictions=predictions,
        thresholds=thresholds,
    )

    snapshot_path = kit_dir / decide_module._FROZEN_SNAPSHOT_FILENAME
    stored_warning = {
        "level": "WARNING",
        "message": "persisted warning from snapshot",
        "source": "previous_run",
        "timestamp": datetime(2024, 1, 4, tzinfo=timezone.utc).isoformat(),
    }
    snapshot_payload = {
        "market": {},
        "daily": None,
        "open_date": open_date.isoformat(),
        "generated_at": datetime(2024, 1, 4, tzinfo=timezone.utc).isoformat(),
        "decision_context_id": "test-context",
        "warnings": [stored_warning],
        "extra_daily_warnings": [],
    }
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    market_provider = build_stub_market_provider(
        open_date=open_date,
        eleven_payload={
            "spx_open": 4800.0,
            "spx_last": 4805.0,
            "intraday_move": 5.0,
            "vix": 15.2,
            "gap": 4.5,
            "movement": 5.0,
            "closing_vix": 15.0,
            "opening_vix": 14.8,
        },
        fifteen_payload={
            "spx_open": 4795.0,
            "spx_last": 4788.0,
            "intraday_move": -7.0,
            "vix": 15.6,
            "gap": 4.5,
            "movement": -7.0,
            "closing_vix": 15.0,
            "opening_vix": 14.8,
        },
    )

    result = run_live_decisions(
        kit_dir,
        market_provider=market_provider,
        daily_provider=_FailingDailyProvider(),
    )

    messages = [warning["message"] for warning in result.warnings]
    assert any(
        message.startswith("daily provider failed for") for message in messages
    ), "fresh warnings from the current run should be preserved"
    assert "persisted warning from snapshot" in messages
