from __future__ import annotations

import builtins
import importlib
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

# Import the module so the parity checks can exercise the private helpers.
from xdte.data import features
from xdte.data.features import (
    build_market_feature_frame,
    build_trade_panel,
    compute_daily_context_features,
    describe_feature_inputs,
    generate_feature_panel,
    infer_canonical_book,
    infer_market_session,
    market_records_to_frame,
    trade_records_to_frame,
)
from xdte.data.loaders import PortfolioRecord, read_portfolio_file


def _backtest_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "backtest" / name


def _load_records(*names: str) -> Sequence[PortfolioRecord]:
    records: list[PortfolioRecord] = []
    for name in names:
        records.extend(read_portfolio_file(_backtest_path(name)))
    return records


def _sample_daily_context() -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for offset in range(10):
        current = start + timedelta(days=offset)
        rows.append(
            {
                "open_date": current,
                "SPX_Close": 4700 + offset,
                "SPX_High": 4705 + offset,
                "SPX_Low": 4695 + offset,
                "VIX_Close": 15 + 0.1 * offset,
                "VIX3M_Close": 17 + 0.1 * offset,
                "VVIX_Close": 90 + (-1) ** offset * 2 + offset,
            }
        )
    frame = pd.DataFrame(rows)
    frame["open_date"] = pd.to_datetime(frame["open_date"])
    return frame


def test_features_builtin_registration_is_idempotent() -> None:
    original_present = hasattr(builtins, "features")
    original_value = getattr(builtins, "features", None)

    try:
        if original_present:
            delattr(builtins, "features")

        module = importlib.reload(features)
        assert getattr(builtins, "features") is module

        reloaded = importlib.reload(module)
        assert getattr(builtins, "features") is reloaded

        sentinel = object()
        setattr(builtins, "features", sentinel)
        importlib.reload(module)
        assert getattr(builtins, "features") is sentinel
    finally:
        if original_present:
            setattr(builtins, "features", original_value)
        if not original_present and hasattr(builtins, "features"):
            delattr(builtins, "features")


def test_feature_catalog_includes_kill_switch_inputs() -> None:
    catalogue = describe_feature_inputs()
    assert catalogue["gap"].source == "portfolio"
    assert catalogue["L1_vvix_above_ema20"].origin == "L1_vvix_close > L1_vvix_ema20"
    assert "pnl" in catalogue


def test_infer_canonical_book_uses_filename() -> None:
    record = next(iter(_load_records("portfolio_0dte_call_spreads_10pts.csv")))
    assert infer_canonical_book(record) == "CALLS_0DTE_11"


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("book_0dte_put_spread_15delta_20_pts_1100.csv", "11:00"),
        ("book_1dte_put_spread_15delta_20_pts_1515.csv", "15:15"),
        ("book_1dte_call_spread_20delta_20_pts_3pm.csv", "15:15"),
        ("11am_market_stats.csv", "11:00"),
        ("1515_market_stats.csv", "15:15"),
    ),
)
def test_infer_market_session_prefers_explicit_tokens(
    source: str, expected: str | None
) -> None:
    base = next(iter(_load_records("portfolio_0dte_call_spreads_10pts.csv")))
    record = replace(base, context={"source": source}, strategy=source)
    assert infer_market_session(record) == expected


def test_build_trade_panel_selects_largest_notional() -> None:
    records = list(_load_records("portfolio_0dte_call_spreads_10pts.csv"))
    first = records[0]
    larger = replace(first, premium=(first.premium or 0) * 2)
    trade_frame = trade_records_to_frame([first, larger])
    panel = build_trade_panel(trade_frame)
    assert isinstance(panel, pd.DataFrame)
    key = ("CALLS_0DTE_11", pd.Timestamp(date(2024, 1, 1)))
    row = panel.loc[key]
    assert row["pnl"] == pytest.approx(larger.pnl or 0.0)


def test_build_market_feature_frame_pivots_sessions() -> None:
    market_records = _load_records("11am_market_stats.csv", "1515_market_stats.csv")
    market_frame = build_market_feature_frame(market_records_to_frame(market_records))
    assert isinstance(market_frame, pd.DataFrame)
    jan_first = market_frame.loc[pd.Timestamp(date(2024, 1, 1))]
    assert jan_first["VIX_Entry_11"] == pytest.approx(16.5)
    assert set(market_frame.columns) >= {"VIX_Entry_11", "VIX_Entry_1515"}


def test_compute_daily_context_features_emits_kill_switch_flags() -> None:
    daily = _sample_daily_context()
    context = compute_daily_context_features(daily)
    assert isinstance(context, pd.DataFrame)
    assert {"L1_vvix_above_ema20", "L1_vvix_above_ema30"}.issubset(context.columns)
    bool_values = context["L1_vvix_above_ema20"].dropna()
    assert all(isinstance(value, (bool, np.bool_)) for value in bool_values)


def test_generate_feature_panel_is_deterministic() -> None:
    trades = _load_records(
        "portfolio_0dte_call_spreads_10pts.csv",
        "portfolio_0dte_put_spreads_10pts.csv",
        "portfolio_1dte_call_spreads_10pts.csv",
        "portfolio_1dte_put_spreads_10pts.csv",
    )
    market = _load_records("11am_market_stats.csv", "1515_market_stats.csv")
    daily = _sample_daily_context()

    panel_one = generate_feature_panel(
        trade_records_to_frame(trades),
        market_records_to_frame(market),
        daily,
    )
    panel_two = generate_feature_panel(
        trade_records_to_frame(trades),
        market_records_to_frame(market),
        daily,
    )

    pd.testing.assert_frame_equal(panel_one, panel_two)
    assert "pnl" in panel_one.columns
    pnl_values = panel_one["pnl"].tolist()
    assert all(isinstance(value, float) for value in pnl_values)
    assert "t0_VIX_change_from_close_11" in panel_one.columns
    assert tuple(panel_one.index.names) == ("book", "open_date")


def test_market_feature_frame_handles_empty_input() -> None:
    empty = build_market_feature_frame(market_records_to_frame([]))
    assert empty.empty
    assert list(empty.columns) == [
        "Intraday_Move_OpenToEntry_11",
        "Intraday_Move_OpenToEntry_1515",
        "VIX_Entry_11",
        "VIX_Entry_1515",
    ]


def _legacy_ewm(values: Sequence[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    result: list[float] = []
    previous: float | None = None
    for value in values:
        if math.isnan(value):
            result.append(math.nan)
            previous = None
            continue
        if previous is None:
            previous = value
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        result.append(previous)
    return result


def _legacy_rolling_max(
    values: Sequence[float], window: int, min_periods: int
) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = [v for v in values[start : index + 1] if not math.isnan(v)]
        if len(window_values) >= min_periods:
            result.append(max(window_values))
        else:
            result.append(math.nan)
    return result


def _legacy_rolling_std(
    values: Sequence[float], window: int, min_periods: int
) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = [v for v in values[start : index + 1] if not math.isnan(v)]
        if len(window_values) >= min_periods:
            mean = sum(window_values) / len(window_values)
            variance = sum((v - mean) ** 2 for v in window_values) / len(window_values)
            result.append(math.sqrt(variance))
        else:
            result.append(math.nan)
    return result


def _legacy_pct_change(values: Sequence[float]) -> list[float]:
    result: list[float] = [math.nan]
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        if math.isnan(previous) or previous == 0.0 or math.isnan(current):
            result.append(math.nan)
        else:
            result.append((current - previous) / previous)
    return result


def _legacy_rank_pct(values: Sequence[float]) -> list[float]:
    indexed = [(index, value) for index, value in enumerate(values)]
    valid = [item for item in indexed if not math.isnan(item[1])]
    total = len(values)
    if not valid:
        return [math.nan] * total
    valid.sort(key=lambda item: item[1])
    ranks: dict[int, float] = {}
    position = 0
    while position < len(valid):
        same_start = position
        current_value = valid[position][1]
        while position < len(valid) and valid[position][1] == current_value:
            position += 1
        same_end = position
        average_rank = (same_start + same_end - 1) / 2.0 + 1.0
        percentile = average_rank / total
        for index in range(same_start, same_end):
            ranks[valid[index][0]] = percentile
    return [ranks.get(index, math.nan) for index in range(total)]


def _legacy_shift(values: Sequence[float], periods: int = 1) -> list[float]:
    if periods <= 0:
        return list(values)
    return [math.nan] * periods + list(values[:-periods])


def _assert_float_lists_equal(
    actual: Sequence[float], expected: Sequence[float]
) -> None:
    assert len(actual) == len(expected)
    for observed, baseline in zip(actual, expected):
        if math.isnan(baseline):
            assert math.isnan(observed)
        else:
            assert observed == pytest.approx(baseline)


@pytest.mark.parametrize(
    ("values", "span"),
    [
        ([1.0, 2.0, 3.0, 4.0], 2),
        ([1.0, math.nan, 2.0, 5.0], 3),
        ([math.nan, 1.5, 2.5, math.nan, 3.5], 4),
        ([1.0, math.nan, 2.0, math.nan, 4.0, 8.0], 2),
    ],
)
def test_ewm_matches_legacy(values: Sequence[float], span: int) -> None:
    expected = _legacy_ewm(values, span)
    actual = features._ewm(values, span)
    _assert_float_lists_equal(actual, expected)


@pytest.mark.parametrize(
    ("values", "window", "min_periods"),
    [
        ([1.0, 2.0, 3.0, 4.0], 2, 1),
        ([math.nan, 5.0, 2.0, 7.0, 1.0], 3, 2),
        ([math.nan, math.nan, 1.0, 3.0, 2.0, 5.0], 3, 2),
    ],
)
def test_rolling_max_matches_legacy(
    values: Sequence[float], window: int, min_periods: int
) -> None:
    expected = _legacy_rolling_max(values, window, min_periods)
    actual = features._rolling_max(values, window, min_periods)
    _assert_float_lists_equal(actual, expected)


@pytest.mark.parametrize(
    ("values", "window", "min_periods"),
    [
        ([1.0, 2.0, 3.0, 4.0], 3, 2),
        ([math.nan, 5.0, 2.0, 7.0, 1.0], 2, 2),
        ([4.0, math.nan, 5.0, 6.0, math.nan, 7.0], 3, 2),
    ],
)
def test_rolling_std_matches_legacy(
    values: Sequence[float], window: int, min_periods: int
) -> None:
    expected = _legacy_rolling_std(values, window, min_periods)
    actual = features._rolling_std(values, window, min_periods)
    _assert_float_lists_equal(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 4.0, 8.0],
        [math.nan, 2.0, 4.0, 0.0, 3.0],
        [1.0, 0.0, -1.0, -1.0, 2.0],
    ],
)
def test_pct_change_matches_legacy(values: Sequence[float]) -> None:
    expected = _legacy_pct_change(values)
    actual = features._pct_change(values)
    _assert_float_lists_equal(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0, 4.0],
        [math.nan, 2.0, math.nan, 5.0, 5.0],
        [math.nan, 2.0, 2.0, 1.0, 3.0],
    ],
)
def test_rank_pct_matches_legacy(values: Sequence[float]) -> None:
    expected = _legacy_rank_pct(values)
    actual = features._rank_pct(values)
    _assert_float_lists_equal(actual, expected)


@pytest.mark.parametrize(
    ("values", "periods"),
    [
        ([1.0, 2.0, 3.0, 4.0], 1),
        ([math.nan, 2.0, 4.0, 6.0], 2),
        ([1.0, 2.0, 3.0], 0),
        ([1.0, math.nan, 3.0, 5.0], 3),
    ],
)
def test_shift_matches_legacy(values: Sequence[float], periods: int) -> None:
    expected = _legacy_shift(values, periods)
    actual = features._shift(values, periods)
    _assert_float_lists_equal(actual, expected)


def test_helper_functions_fixed_sample_matches_legacy() -> None:
    values = [math.nan, 1.0, 2.5, math.nan, 0.5, 1.5]
    span = 3
    window = 3
    min_periods = 2
    shift_periods = 2

    expected_ewm = [math.nan, 1.0, 1.75, math.nan, 0.5, 1.0]
    expected_rolling_max = [math.nan, math.nan, 2.5, 2.5, 2.5, 1.5]
    expected_rolling_std = [math.nan, math.nan, 0.75, 0.75, 1.0, 0.5]
    expected_pct_change = [math.nan, math.nan, 1.5, math.nan, math.nan, 2.0]
    expected_rank_pct = [
        math.nan,
        1.0 / 3.0,
        2.0 / 3.0,
        math.nan,
        1.0 / 6.0,
        0.5,
    ]
    expected_shift = [math.nan, math.nan, math.nan, 1.0, 2.5, math.nan]

    _assert_float_lists_equal(features._ewm(values, span), expected_ewm)
    _assert_float_lists_equal(_legacy_ewm(values, span), expected_ewm)

    _assert_float_lists_equal(
        features._rolling_max(values, window, min_periods), expected_rolling_max
    )
    _assert_float_lists_equal(
        _legacy_rolling_max(values, window, min_periods), expected_rolling_max
    )

    _assert_float_lists_equal(
        features._rolling_std(values, window, min_periods), expected_rolling_std
    )
    _assert_float_lists_equal(
        _legacy_rolling_std(values, window, min_periods), expected_rolling_std
    )

    _assert_float_lists_equal(features._pct_change(values), expected_pct_change)
    _assert_float_lists_equal(_legacy_pct_change(values), expected_pct_change)

    _assert_float_lists_equal(features._rank_pct(values), expected_rank_pct)
    _assert_float_lists_equal(_legacy_rank_pct(values), expected_rank_pct)

    _assert_float_lists_equal(features._shift(values, shift_periods), expected_shift)
    _assert_float_lists_equal(_legacy_shift(values, shift_periods), expected_shift)
