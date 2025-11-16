"""LightGBM wrapper with a deterministic fallback implementation."""

from __future__ import annotations

import importlib
import importlib.util
import math
from typing import TYPE_CHECKING, List, Sequence, Type


class _FallbackLGBMRegressor:
    def __init__(self, *, alpha: float = 0.05, **_: object) -> None:
        self.alpha = float(alpha)
        self._prediction = 0.0

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> "_FallbackLGBMRegressor":
        del X
        self._prediction = _quantile(
            tuple(float(value) for value in y), 1.0 - self.alpha
        )
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        return [self._prediction for _ in X]


def _quantile(values: Sequence[float], q: float) -> float:
    numeric: List[float] = [value for value in values if not math.isnan(value)]
    if not numeric:
        return 0.0
    ordered = sorted(numeric)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return (1 - weight) * ordered[lower] + weight * ordered[upper]


def _load_runtime_regressor() -> Type[object] | None:
    spec = importlib.util.find_spec("lightgbm")
    if spec is None:
        return None
    module = importlib.import_module("lightgbm")
    regressor = getattr(module, "LGBMRegressor", None)
    if isinstance(regressor, type):
        return regressor
    return None


_RUNTIME_REGRESSOR = _load_runtime_regressor()

if TYPE_CHECKING:

    class LGBMRegressor(_FallbackLGBMRegressor): ...

elif _RUNTIME_REGRESSOR is not None:
    LGBMRegressor = _RUNTIME_REGRESSOR  # type: ignore[assignment]
else:
    LGBMRegressor = _FallbackLGBMRegressor

__all__ = ["LGBMRegressor"]
