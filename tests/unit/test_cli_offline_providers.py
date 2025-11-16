"""Tests covering the offline provider helpers exposed by :mod:`xdte.cli`."""

from __future__ import annotations

from datetime import date
from typing import Callable

import pytest

from xdte.cli import _offline_daily_provider, _offline_market_provider
from xdte.config import Settings


@pytest.fixture()
def settings() -> Settings:
    return Settings.load()


def test_offline_daily_provider_handles_builders_without_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    captured_open_dates: list[date] = []

    def fake_daily_builder(open_date: date) -> Callable[[date], dict[str, object]]:
        captured_open_dates.append(open_date)

        def provider(request_date: date) -> dict[str, object]:
            return {
                "open_date": request_date.isoformat(),
                "seed": open_date.isoformat(),
            }

        return provider

    monkeypatch.setattr("xdte.cli._build_offline_daily", fake_daily_builder)

    provider = _offline_daily_provider(date(2024, 7, 1), settings)
    payload = provider(date(2024, 7, 2))

    assert payload == {"open_date": "2024-07-02", "seed": "2024-07-01"}
    assert captured_open_dates == [date(2024, 7, 1)]


def test_offline_daily_provider_propagates_other_type_errors(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def bad_builder(open_date: date, *, settings: Settings) -> object:
        raise TypeError("daily boom")

    monkeypatch.setattr("xdte.cli._build_offline_daily", bad_builder)

    with pytest.raises(TypeError, match="daily boom"):
        _offline_daily_provider(date(2024, 7, 1), settings)


def test_offline_market_provider_handles_builders_without_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    captured_open_dates: list[date] = []

    def fake_market_builder(open_date: date) -> Callable[[str], dict[str, object]]:
        captured_open_dates.append(open_date)

        def provider(session: str) -> dict[str, object]:
            return {"session": session, "seed": open_date.isoformat()}

        return provider

    monkeypatch.setattr("xdte.cli._build_offline_market", fake_market_builder)

    provider = _offline_market_provider(date(2024, 7, 1), settings)
    payload = provider("11:00")

    assert payload == {"session": "11:00", "seed": "2024-07-01"}
    assert captured_open_dates == [date(2024, 7, 1)]


def test_offline_market_provider_propagates_other_type_errors(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def bad_builder(open_date: date, *, settings: Settings) -> object:
        raise TypeError("bad builder")

    monkeypatch.setattr("xdte.cli._build_offline_market", bad_builder)

    with pytest.raises(TypeError, match="bad builder"):
        _offline_market_provider(date(2024, 7, 1), settings)
