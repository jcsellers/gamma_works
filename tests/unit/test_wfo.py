"""Unit tests for the walk-forward splitter implementation."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest

# Property-based tests depend on Hypothesis; skip gracefully when unavailable.
pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from xdte.config import Settings
from xdte.metrics import _prepare, cvar, profit_factor
from xdte.model.wfo import WalkForwardSplitter, walk_forward_splits


def _generate_dates(start: date, count: int) -> list[date]:
    return [start + timedelta(days=offset) for offset in range(count)]


def test_walk_forward_standard_history() -> None:
    """The splitter reproduces the legacy fold pattern for ample data."""

    dates = _generate_dates(date(2024, 1, 1), 90)
    splitter = WalkForwardSplitter(settings=Settings())
    splits = splitter.splits(dates)

    assert len(splits) == 8  # legacy behaviour keeps the first short train fold
    first_train, first_test = splits[0]
    assert len(first_train) == 10
    assert len(first_test) == 10
    last_train, last_test = splits[-1]
    assert last_train[-1] < last_test[0]
    assert all(earlier < later for earlier, later in zip(last_test, last_test[1:]))


def test_walk_forward_short_history_triggers_fallback() -> None:
    """Short histories fall back to chunked splits instead of failing silently."""

    dates = _generate_dates(date(2024, 1, 1), 41)
    splitter = WalkForwardSplitter(settings=Settings())
    splits = splitter.splits(dates)

    assert splits  # fallback produces valid folds
    test_lengths = {len(test) for _, test in splits}
    assert 5 in test_lengths  # chunked fallback yields wider evaluation windows


def test_walk_forward_is_deterministic() -> None:
    """Repeated invocations with the same data yield identical splits."""

    dates = _generate_dates(date(2024, 1, 1), 60)
    first = walk_forward_splits(dates, settings=Settings())
    second = walk_forward_splits(dates, settings=Settings())

    assert len(first) == len(second)
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        assert train_a == train_b
        assert test_a == test_b


def test_walk_forward_accepts_non_datetime_inputs() -> None:
    """The splitter converts plain integers without touching external data."""

    values = list(range(50))
    splits = walk_forward_splits(values, settings=Settings())

    assert splits
    train, test = splits[0]
    assert isinstance(train[0], datetime)
    assert isinstance(test[0], datetime)


_date_lists = st.lists(
    st.dates(min_value=date(2000, 1, 1), max_value=date(2035, 12, 31)),
    min_size=1,
    max_size=90,
)


def _normalize_dates(dates: list[date]) -> list[date]:
    return sorted(set(dates))


@given(_date_lists)
def test_walk_forward_property_deterministic(dates: list[date]) -> None:
    """Repeated invocations with arbitrary histories are deterministic."""

    normalized = _normalize_dates(dates)

    first = walk_forward_splits(normalized, settings=Settings())
    second = walk_forward_splits(normalized, settings=Settings())

    assert len(first) == len(second)
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        assert train_a == train_b
        assert test_a == test_b


@given(_date_lists)
def test_walk_forward_preserves_monotonicity(dates: list[date]) -> None:
    """Each generated fold yields strictly increasing training and test windows."""

    normalized = _normalize_dates(dates)

    splitter = WalkForwardSplitter(settings=Settings())
    splits = splitter.splits(normalized)
    for train, test in splits:
        assert train
        assert test
        assert list(train) == sorted(train)
        assert list(test) == sorted(test)
        assert train[-1] < test[0]


_numericish = st.one_of(
    st.integers(-(10**6), 10**6),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.none(),
    st.text(min_size=0, max_size=10),
)


@given(st.lists(_numericish, max_size=60))
def test_profit_factor_handles_random_numeric_inputs(values: list[object]) -> None:
    """``profit_factor`` never raises and always returns a float or NaN."""

    result = profit_factor(values)

    assert isinstance(result, float)
    assert math.isnan(result) or result >= 0.0


@given(st.lists(_numericish, max_size=60), st.floats(min_value=1e-6, max_value=1.0))
def test_cvar_handles_random_numeric_inputs(values: list[object], alpha: float) -> None:
    """``cvar`` gracefully handles arbitrary numeric-ish sequences."""

    result = cvar(values, alpha=alpha)

    assert isinstance(result, float)

    prepared = _prepare(values)
    if not prepared:
        assert math.isnan(result)
        return

    minimum = min(prepared)
    maximum = max(prepared)
    assert minimum <= result <= maximum
