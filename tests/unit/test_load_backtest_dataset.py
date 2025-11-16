from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from xdte.model.train import load_backtest_dataset

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "backtest_book_mode"

TRADE_COMPARE_COLUMNS = [
    "book",
    "strategy",
    "open_timestamp",
    "open_date",
    "notional",
    "pnl",
    "gap",
    "movement",
    "opening_vix",
    "closing_vix",
]

MARKET_COMPARE_COLUMNS = [
    "strategy",
    "open_timestamp",
    "open_date",
    "session",
    "opening_vix",
    "closing_vix",
    "movement",
]


def _load_normalised_frames(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_frame, market_frame, _ = load_backtest_dataset(directory)
    normalised_trade = trade_frame.sort_values(
        ["book", "strategy", "open_timestamp"], ignore_index=True
    )
    normalised_market = market_frame.sort_values(
        ["strategy", "open_timestamp"], ignore_index=True
    )
    return normalised_trade, normalised_market


def _assert_trade_equivalence(legacy: pd.DataFrame, book_mode: pd.DataFrame) -> None:
    legacy_view = legacy.loc[:, TRADE_COMPARE_COLUMNS].reset_index(drop=True)
    book_view = book_mode.loc[:, TRADE_COMPARE_COLUMNS].reset_index(drop=True)
    assert_frame_equal(legacy_view, book_view)


def _assert_market_equivalence(legacy: pd.DataFrame, book_mode: pd.DataFrame) -> None:
    legacy_view = legacy.loc[:, MARKET_COMPARE_COLUMNS].reset_index(drop=True)
    book_view = book_mode.loc[:, MARKET_COMPARE_COLUMNS].reset_index(drop=True)
    assert_frame_equal(legacy_view, book_view)


def test_load_backtest_dataset_single_book_equivalence() -> None:
    legacy_dir = FIXTURES / "legacy_single"
    book_dir = FIXTURES / "book_single"

    legacy_trade, legacy_market = _load_normalised_frames(legacy_dir)
    book_trade, book_market = _load_normalised_frames(book_dir)

    _assert_trade_equivalence(legacy_trade, book_trade)
    _assert_market_equivalence(legacy_market, book_market)


def test_load_backtest_dataset_put_call_equivalence() -> None:
    legacy_dir = FIXTURES / "legacy_combo"
    book_dir = FIXTURES / "book_combo"

    legacy_trade, legacy_market = _load_normalised_frames(legacy_dir)
    book_trade, book_market = _load_normalised_frames(book_dir)

    _assert_trade_equivalence(legacy_trade, book_trade)
    _assert_market_equivalence(legacy_market, book_market)


def test_load_backtest_dataset_uses_trade_rows_when_market_file_empty() -> None:
    baseline_dir = FIXTURES / "book_single"
    empty_market_dir = FIXTURES / "book_single_with_empty_market"

    baseline_trade, baseline_market = _load_normalised_frames(baseline_dir)
    trade_frame, market_frame = _load_normalised_frames(empty_market_dir)

    _assert_trade_equivalence(baseline_trade, trade_frame)
    _assert_market_equivalence(baseline_market, market_frame)


def test_load_backtest_dataset_raises_when_daily_context_missing(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "missing_daily"
    shutil.copytree(FIXTURES / "book_single", bundle_dir)
    (bundle_dir / "daily_context.csv").unlink()

    with pytest.raises(
        ValueError, match="daily_context.csv does not cover trade range"
    ):
        load_backtest_dataset(bundle_dir)


def test_load_backtest_dataset_raises_when_daily_context_incomplete(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "incomplete_daily"
    shutil.copytree(FIXTURES / "book_single", bundle_dir)
    daily_path = bundle_dir / "daily_context.csv"
    rows = daily_path.read_text(encoding="utf-8").splitlines()
    daily_path.write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="daily_context.csv does not cover trade range"
    ):
        load_backtest_dataset(bundle_dir)
