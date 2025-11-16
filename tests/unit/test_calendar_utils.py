from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from xdte.live.calendar import resolve_open_date

_EASTERN = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    "timestamp, expected",
    [
        (datetime(2024, 7, 1, 13, 30, tzinfo=_EASTERN), date(2024, 7, 1)),
        (datetime(2024, 7, 6, 10, 0, tzinfo=_EASTERN), date(2024, 7, 5)),
        (datetime(2024, 7, 4, 12, 0, tzinfo=_EASTERN), date(2024, 7, 3)),
        (datetime(2024, 7, 1, 8, 0, tzinfo=_EASTERN), date(2024, 7, 1)),
    ],
)
def test_resolve_open_date(timestamp: datetime, expected: date) -> None:
    assert resolve_open_date(timestamp) == expected
