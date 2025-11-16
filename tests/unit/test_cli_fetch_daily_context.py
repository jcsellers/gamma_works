from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from xdte.cli import app

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "backtest_book_mode"
    / "book_single"
)

runner = CliRunner()


def _build_daily_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-10-02", periods=200, freq="B")
    base = pd.Series(range(len(dates)), index=dates, dtype=float)
    data = {
        ("Close", "^GSPC"): 4000 + base,
        ("Open", "^GSPC"): 3995 + base,
        ("High", "^GSPC"): 4005 + base,
        ("Low", "^GSPC"): 3990 + base,
        ("Close", "^VIX"): 15 + base / 100.0,
        ("Close", "^VIX3M"): 17 + base / 100.0,
        ("Close", "^VVIX"): 70 + base / 50.0,
    }
    columns = pd.MultiIndex.from_tuples(list(data.keys()))
    frame = pd.DataFrame(
        {key: value.to_numpy() for key, value in data.items()}, index=dates
    )
    frame.columns = columns
    return frame


class _StubDailyProvider:
    def __init__(self, **_: object) -> None:
        self._frame = _build_daily_frame()
        self._warnings = ("cache refreshed",)

    def to_frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._warnings


def test_fetch_daily_context_generates_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(FIXTURES, bundle_dir)
    daily_path = bundle_dir / "daily_context.csv"
    daily_path.unlink()

    monkeypatch.setattr("xdte.cli.YFinanceDailyContextProvider", _StubDailyProvider)

    result = runner.invoke(app, ["fetch-daily-context", "--data-dir", str(bundle_dir)])

    assert result.exit_code == 0
    assert "daily_context.csv updated" in result.output

    frame = pd.read_csv(daily_path)
    assert {"2024-01-02", "2024-01-03"}.issubset(set(frame["open_date"]))
    assert "L1_TS" in frame.columns
    assert "L1_vvix_ema20" in frame.columns
