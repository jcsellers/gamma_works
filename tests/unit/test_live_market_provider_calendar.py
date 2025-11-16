from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pytest import MonkeyPatch

from xdte.live.providers import YFinanceMarketDataProvider

_EASTERN = ZoneInfo("America/New_York")


def _intraday_frame(open_date: date) -> pd.DataFrame:
    index = pd.date_range(
        start=datetime(
            open_date.year,
            open_date.month,
            open_date.day,
            9,
            30,
            tzinfo=_EASTERN,
        ),
        periods=6,
        freq="15min",
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "^GSPC"),
            ("Close", "^GSPC"),
            ("Open", "^VIX"),
            ("Close", "^VIX"),
        ]
    )
    data = [
        [4800.0, 4800.0, 16.0, 16.0],
        [4800.0, 4802.0, 16.0, 15.8],
        [4800.0, 4805.0, 16.0, 15.6],
        [4800.0, 4806.0, 16.0, 15.4],
        [4800.0, 4807.0, 16.0, 15.2],
        [4800.0, 4808.0, 16.0, 15.0],
    ]
    frame = pd.DataFrame(data, index=index, columns=columns)
    return frame


def test_provider_emits_calendar_warning_on_weekend() -> None:
    frame = _intraday_frame(date(2024, 7, 5))
    provider = YFinanceMarketDataProvider(
        frame=frame,
        clock=lambda: datetime(2024, 7, 6, 10, 0, tzinfo=_EASTERN),
    )

    payload = provider("11:00")
    assert payload["open_date"] == date(2024, 7, 5)
    warning_text = "Trading calendar resolved open_date 2024-07-05"
    assert any(warning_text in warning for warning in provider.warnings)


def test_provider_calls_resolve_open_date_for_each_session(
    monkeypatch: MonkeyPatch,
) -> None:
    frame = _intraday_frame(date(2024, 7, 5))
    calls: list[datetime] = []

    def _fake_resolve(now: datetime) -> date:
        calls.append(now)
        return date(2024, 7, 5)

    monkeypatch.setattr("xdte.live.providers.resolve_open_date", _fake_resolve)
    provider = YFinanceMarketDataProvider(
        frame=frame,
        clock=lambda: datetime(2024, 7, 5, 12, 0, tzinfo=_EASTERN),
    )

    provider("11:00")
    provider("15:15")

    assert len(calls) == 2
