from __future__ import annotations

import math

from xdte.gamma import GammaRule


def test_gamma_rule_level_monotonicity() -> None:
    rule = GammaRule(lo=0.2, hi=0.6, levels=(0.4, 0.8, 1.2))
    margins = [-0.5, -0.1, 0.0, 0.1, 0.3, 0.6, 0.9]
    levels = [rule.level_for_margin(value) for value in margins]
    assert all(level is not None for level in levels)
    numeric_levels = [float(level) for level in levels if level is not None]
    for left, right in zip(numeric_levels, numeric_levels[1:]):
        assert left <= right + 1e-12

    none_level = rule.level_for_margin(None)
    assert math.isclose(none_level or 0.0, rule.levels[-1])
