from __future__ import annotations

import math

from xdte.model import _lgbm


def test_fallback_regressor_learns_quantile_prediction() -> None:
    regressor = _lgbm._FallbackLGBMRegressor(alpha=0.25)
    fitted = regressor.fit([[0.0], [1.0], [2.0]], [-10.0, 5.0, 10.0])

    assert fitted is regressor
    assert regressor.predict([[1.0], [2.0]]) == [7.5, 7.5]


def test_lgbm_quantile_handles_bounds() -> None:
    assert _lgbm._quantile([math.nan], 0.5) == 0.0
    assert _lgbm._quantile([1.0, 3.0], -0.5) == 1.0
    assert _lgbm._quantile([1.0, 3.0], 1.5) == 3.0
