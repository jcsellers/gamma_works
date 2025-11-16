"""Unit tests for helpers in :mod:`xdte.model.train`."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

import xdte.model.train as train_module

_DEF_KILL = train_module._KILL_SWITCH_COLUMN


def test_build_kill_switch_lookup_extracts_daily_flags() -> None:
    frame = pd.DataFrame(
        {
            "open_date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            _DEF_KILL: [True, 0, "true"],
        }
    )

    lookup = train_module._build_kill_switch_lookup(frame)

    assert lookup[date(2024, 1, 1)] is True
    assert lookup[date(2024, 1, 2)] is False
    assert lookup[date(2024, 1, 3)] is True


def test_build_prediction_rows_uses_lookup_for_missing_column() -> None:
    rows = [
        {"open_date": datetime(2024, 1, 1), "pnl": 1.5},
        {
            "open_date": datetime(2024, 1, 2),
            "pnl": 2.5,
            _DEF_KILL: False,
        },
    ]
    predictions = [0.5, 1.0]
    kill_lookup = {date(2024, 1, 1): True, date(2024, 1, 2): True}

    payload = train_module._build_prediction_rows(
        rows,
        predictions,
        book="CALLS_0DTE_11",
        kill_switch_lookup=kill_lookup,
    )

    assert payload[0][_DEF_KILL] is True
    assert payload[1][_DEF_KILL] is False
    assert payload[0]["score"] == predictions[0]
    assert payload[1]["score"] == predictions[1]
