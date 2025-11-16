"""Trading calendar utilities for live providers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

_EASTERN_TZ = ZoneInfo("America/New_York")


def _normalise_timestamp(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(tz=_EASTERN_TZ)
    if value.tzinfo is None:
        return value.replace(tzinfo=_EASTERN_TZ)
    return value.astimezone(_EASTERN_TZ)


def _observed(holiday: date) -> date:
    if holiday.weekday() == 5:  # Saturday observed Friday
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:  # Sunday observed Monday
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    current = date(year, month, 1)
    count = 0
    while True:
        if current.weekday() == weekday:
            count += 1
            if count == occurrence:
                return current
        current += timedelta(days=1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def _holiday_map(year: int) -> set[date]:
    easter = _easter_sunday(year)
    good_friday = easter - timedelta(days=2)
    holidays = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        good_friday,
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),  # Juneteenth
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving Day
        _observed(date(year, 12, 25)),  # Christmas Day
    }
    return holidays


def _is_holiday(day: date) -> bool:
    return day in _holiday_map(day.year)


def _is_trading_day(day: date) -> bool:
    if day.weekday() >= 5:
        return False
    if _is_holiday(day):
        return False
    return True


def resolve_open_date(now: datetime | None = None) -> date:
    """Map *now* to the corresponding trading date."""

    normalised = _normalise_timestamp(now)
    session_date = normalised.date()
    if _is_trading_day(session_date):
        return session_date

    current = session_date - timedelta(days=1)
    while not _is_trading_day(current):
        current -= timedelta(days=1)
    return current


__all__ = ["resolve_open_date"]
