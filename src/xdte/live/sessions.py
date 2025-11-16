"""Shared session metadata for XDTE live decisioning."""

from __future__ import annotations

from datetime import time
from typing import Final, Mapping

SESSION_ELEVEN_AM: Final[str] = "11:00"
SESSION_FIFTEEN_FIFTEEN: Final[str] = "15:15"

SUFFIX_ELEVEN_AM: Final[str] = "11"
SUFFIX_FIFTEEN_FIFTEEN: Final[str] = "1515"

SESSION_TO_SUFFIX: Final[Mapping[str, str]] = {
    SESSION_ELEVEN_AM: SUFFIX_ELEVEN_AM,
    SESSION_FIFTEEN_FIFTEEN: SUFFIX_FIFTEEN_FIFTEEN,
}

SUFFIX_TO_SESSION: Final[Mapping[str, str]] = {
    suffix: session for session, suffix in SESSION_TO_SUFFIX.items()
}

SESSION_TIMES: Final[Mapping[str, time]] = {
    SESSION_ELEVEN_AM: time(11, 0),
    SESSION_FIFTEEN_FIFTEEN: time(15, 15),
}

REQUIRED_SESSIONS: Final[tuple[str, ...]] = (
    SESSION_ELEVEN_AM,
    SESSION_FIFTEEN_FIFTEEN,
)

__all__ = [
    "REQUIRED_SESSIONS",
    "SESSION_ELEVEN_AM",
    "SESSION_FIFTEEN_FIFTEEN",
    "SUFFIX_ELEVEN_AM",
    "SUFFIX_FIFTEEN_FIFTEEN",
    "SESSION_TIMES",
    "SESSION_TO_SUFFIX",
    "SUFFIX_TO_SESSION",
]
