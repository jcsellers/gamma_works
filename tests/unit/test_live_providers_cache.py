from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from xdte.live.providers import YFinanceDailyContextProvider, YFinanceMarketDataProvider

_EASTERN = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _patch_parquet_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame, path: Path, *args: object, **kwargs: object
    ) -> None:
        self.to_pickle(path)

    def _fake_read_parquet(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        return pd.read_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=False)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet, raising=False)


def _build_daily_frame(start: datetime, periods: int) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "^GSPC"),
            ("High", "^GSPC"),
            ("Low", "^GSPC"),
            ("Close", "^GSPC"),
            ("Open", "^VIX"),
            ("Close", "^VIX"),
            ("Close", "^VIX3M"),
            ("Close", "^VVIX"),
        ]
    )
    data = [
        [
            4800.0 + day,
            4805.0 + day,
            4795.0 + day,
            4802.0 + day,
            16.0 + day,
            15.5 + day,
            18.0 + day,
            90.0 + day,
        ]
        for day in range(periods)
    ]
    return pd.DataFrame(data, index=index, columns=columns)


def _build_intraday_frame(
    start: datetime, periods: int, freq: str = "5min"
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz=_EASTERN)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "^GSPC"),
            ("Close", "^GSPC"),
            ("Open", "^VIX"),
            ("Close", "^VIX"),
        ]
    )
    data = [
        [
            4800.0 + idx,
            4801.0 + idx,
            16.0 + idx * 0.1,
            15.8 + idx * 0.1,
        ]
        for idx in range(periods)
    ]
    return pd.DataFrame(data, index=index, columns=columns)


def test_daily_provider_uses_cached_frame_when_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_start = datetime.now(tz=_EASTERN) + timedelta(days=1)
    initial_frame = _build_daily_frame(base_start, periods=1)
    calls: list[dict[str, object]] = []

    def initial_download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return initial_frame

    monkeypatch.setattr(
        "xdte.live.providers.yf", SimpleNamespace(download=initial_download)
    )
    first = YFinanceDailyContextProvider(
        cache_dir=tmp_path,
        cache_freshness=timedelta(days=1),
    )
    assert len(calls) == 1

    def fail_download(**_kwargs: object) -> pd.DataFrame:  # pragma: no cover - safety
        raise AssertionError("download should not be called when cache is fresh")

    monkeypatch.setattr(
        "xdte.live.providers.yf", SimpleNamespace(download=fail_download)
    )
    second = YFinanceDailyContextProvider(
        cache_dir=tmp_path,
        cache_freshness=timedelta(days=1),
    )
    pd.testing.assert_frame_equal(second._data, first._data)


def test_daily_provider_incremental_refresh_appends_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_frame = _build_daily_frame(datetime(2024, 7, 1), periods=2)
    incremental_frame = _build_daily_frame(datetime(2024, 7, 3), periods=1)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        if "start" in kwargs:
            return incremental_frame
        return initial_frame

    monkeypatch.setattr("xdte.live.providers.yf", SimpleNamespace(download=download))
    _ = YFinanceDailyContextProvider(
        cache_dir=tmp_path,
        cache_freshness=timedelta(days=1),
    )
    assert len(calls) == 1
    refresher = YFinanceDailyContextProvider(
        cache_dir=tmp_path,
        cache_freshness=timedelta(seconds=0),
    )
    assert len(calls) == 2
    assert calls[1]["start"] == "2024-07-03"
    assert refresher._data.index[-1] == pd.Timestamp("2024-07-03")
    assert len(refresher._data) == 3


def test_market_provider_stale_cache_triggers_full_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_dt = datetime(2024, 7, 1, 9, 30, tzinfo=_EASTERN)
    initial_frame = _build_intraday_frame(start_dt, periods=3)
    replacement_frame = _build_intraday_frame(start_dt, periods=4)
    calls: list[dict[str, object]] = []

    def download_sequence(**kwargs: object) -> pd.DataFrame:
        call_index = len(calls)
        calls.append(kwargs)
        if call_index == 0:
            return initial_frame
        return replacement_frame

    monkeypatch.setattr(
        "xdte.live.providers.yf", SimpleNamespace(download=download_sequence)
    )
    clock_time = start_dt + timedelta(minutes=2)
    first = YFinanceMarketDataProvider(
        cache_dir=tmp_path,
        cache_freshness=timedelta(days=1),
        clock=lambda: clock_time,
    )
    assert len(calls) == 1

    monkeypatch.setattr(
        "xdte.live.providers.yf", SimpleNamespace(download=download_sequence)
    )
    refreshed = YFinanceMarketDataProvider(
        cache_dir=tmp_path,
        cache_freshness=timedelta(seconds=0),
        clock=lambda: clock_time,
    )
    assert len(calls) == 2
    assert "start" not in calls[1]
    pd.testing.assert_frame_equal(refreshed._raw_frame, replacement_frame)
    assert first._raw_frame.equals(initial_frame)
