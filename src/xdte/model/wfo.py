"""Walk-forward optimisation helpers for XDTE models.

The legacy notebooks implemented walk-forward evaluation inline.  This module
provides a reusable splitter that mirrors that behaviour while exposing the
fold count via :class:`xdte.config.Settings`.  The implementation retains the
two-phase splitting strategy: we first attempt evenly sized folds and fall back
to chunked splits when short histories would otherwise produce no evaluation
windows.  Documenting this behaviour is crucial because the fallback guards
against regressions when working with sparse synthetic data during tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Sequence, Tuple

from xdte.config import Settings, get_settings

DateTuple = Tuple[datetime, ...]
Split = Tuple[DateTuple, DateTuple]

__all__ = ["WalkForwardSplitter", "walk_forward_splits"]


@dataclass
class WalkForwardSplitter:
    """Generate walk-forward splits that mirror the legacy notebook logic.

    Parameters
    ----------
    min_train_size:
        Minimum number of unique training days required for a fold.
    min_test_size:
        Minimum number of unique test days required for a fold.
    settings:
        Optional :class:`~xdte.config.Settings` instance providing defaults.
    n_folds:
        Optional override for the number of folds.  When omitted the value from
        ``settings`` is used.

    Notes
    -----
    The splitter first attempts to build evenly sized folds based on the
    configured number of splits.  If no folds satisfy the minimum train/test
    constraints—common with short histories—it falls back to a chunked approach
    that mimics ``numpy.array_split``.  The fallback ensures synthetic and
    partially populated datasets still produce deterministic outputs instead of
    silently skipping evaluation.
    """

    min_train_size: int = 20
    min_test_size: int = 5
    settings: Settings | None = None
    n_folds: int | None = None

    def __post_init__(self) -> None:
        if self.settings is None:
            self.settings = get_settings()
        if self.n_folds is None:
            self.n_folds = self.settings.N_FOLDS
        if self.n_folds < 0:
            msg = "Number of folds must be non-negative"
            raise ValueError(msg)

    def __call__(self, dates: Sequence[object]) -> List[Split]:
        """Delegate to :meth:`splits` for call-style usage."""

        return self.splits(dates)

    def splits(self, dates: Sequence[object]) -> List[Split]:
        """Return walk-forward train/test splits for the provided dates.

        The input dates are de-duplicated, converted to timezone-naive Python
        ``datetime`` objects, and sorted to maintain deterministic ordering.  The
        method first constructs evenly spaced folds.  When those folds do not
        satisfy the minimum evaluation-window length the method falls back to
        chunked splits.  The primary windowed phase mirrors the legacy notebook
        implementation by only requiring a minimum number of test days.  The
        fallback phase enforces the stricter train/test thresholds so short
        histories retain deterministic (possibly empty) results rather than
        crashing or touching any on-disk datasets.
        """

        unique_dates = self._prepare_dates(dates)
        primary = self._windowed_splits(unique_dates)
        if primary:
            return primary
        return self._chunked_splits(unique_dates)

    def _prepare_dates(self, dates: Sequence[object]) -> DateTuple:
        seen = {
            value
            for value in (self._coerce_datetime(item) for item in dates)
            if value is not None
        }
        return tuple(sorted(seen))

    def _windowed_splits(self, unique_dates: DateTuple) -> List[Split]:
        n_dates = len(unique_dates)
        if n_dates == 0:
            return []

        folds = self.n_folds if self.n_folds is not None else 0
        step = max(1, n_dates // (folds + 1)) if (folds + 1) > 0 else n_dates
        splits: List[Split] = []
        for idx in range(1, folds + 1):
            cut = idx * step
            train = unique_dates[:cut]
            test = unique_dates[cut : min(cut + step, n_dates)]
            if self._is_valid(train, test, require_train=False):
                splits.append((train, test))
        return splits

    def _chunked_splits(self, unique_dates: DateTuple) -> List[Split]:
        n_dates = len(unique_dates)
        if n_dates == 0:
            return []

        divisor = (self.n_folds if self.n_folds is not None else 0) + 1
        if divisor <= 0:
            chunk_count = n_dates
        else:
            chunk_count = min(n_dates, divisor)

        raw_chunks = self._split_into_chunks(unique_dates, chunk_count)
        splits: List[Split] = []
        for idx in range(1, len(raw_chunks)):
            train = tuple(value for chunk in raw_chunks[:idx] for value in chunk)
            test = raw_chunks[idx]
            if self._is_valid(train, test):
                splits.append((train, test))
        return splits

    def _split_into_chunks(
        self, unique_dates: DateTuple, chunk_count: int
    ) -> List[DateTuple]:
        if chunk_count <= 0:
            return []
        base, remainder = divmod(len(unique_dates), chunk_count)
        sizes = [
            base + 1 if index < remainder else base for index in range(chunk_count)
        ]
        chunks: List[DateTuple] = []
        position = 0
        for size in sizes:
            if size <= 0:
                continue
            chunk = unique_dates[position : position + size]
            if chunk:
                chunks.append(chunk)
            position += size
        return chunks

    @staticmethod
    def _coerce_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc).replace(
                    tzinfo=None
                )
            except (OverflowError, OSError, ValueError):
                return None
        return None

    def _is_valid(
        self, train: DateTuple, test: DateTuple, *, require_train: bool = True
    ) -> bool:
        if len(test) < self.min_test_size:
            return False
        if require_train and len(train) < self.min_train_size:
            return False
        return True


def walk_forward_splits(
    dates: Sequence[object],
    *,
    settings: Settings | None = None,
    n_folds: int | None = None,
    min_train_size: int = 20,
    min_test_size: int = 5,
) -> List[Split]:
    """Convenience wrapper returning walk-forward splits.

    The helper mirrors :func:`WalkForwardSplitter.splits`, exposing the same
    fallback semantics for short histories.  When the primary evenly spaced
    folds do not satisfy the minimum evaluation-window length the fallback
    ensures a chunked, deterministic split is returned.  The implementation
    relies solely on in-memory inputs, keeping the module free of any
    data-directory dependencies.
    """

    splitter = WalkForwardSplitter(
        min_train_size=min_train_size,
        min_test_size=min_test_size,
        settings=settings,
        n_folds=n_folds,
    )
    return splitter.splits(dates)
