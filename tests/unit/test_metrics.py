"""Unit tests for the metrics utilities."""

from __future__ import annotations

import math
from typing import Any, Mapping, cast

import pytest

from xdte import metrics as metrics_module
from xdte.metrics import (
    cvar,
    portfolio_metrics,
    portfolio_metrics_extended,
    portfolio_summary,
    profit_factor,
)


def test_mean_returns_nan_for_empty_inputs() -> None:
    """The internal mean helper should return NaN for empty inputs."""

    assert math.isnan(metrics_module._mean([]))


@pytest.mark.parametrize(
    "values",
    [[], [1.0], [2.0, 2.0, 2.0]],
)
def test_stddev_returns_nan_for_degenerate_inputs(values: list[float]) -> None:
    """The internal stddev helper should return NaN for degenerate inputs."""

    assert math.isnan(metrics_module._stddev(values))


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], math.nan),
        ([math.nan, math.nan], math.nan),
        ([1.0, 2.0, 3.0], 9999.0),
        ([0.0, 1.0, 2.0], 9999.0),
        ([-1.0, -2.0, -3.0], 0.0),
        ([1.0, -0.5, -0.25], 1.0 / 0.75),
        ([1.0, math.nan, -1.0], 1.0),
        ({"a": 1.0, "b": -1.0}, 1.0),
    ],
)
def test_profit_factor(values: Any, expected: float) -> None:
    """Verify profit factor handles gains, losses, and NaNs."""
    result = profit_factor(values)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "values, alpha, expected",
    [
        ([], 0.05, math.nan),
        ([math.nan], 0.05, math.nan),
        ([-2.0, -1.0, 3.0, 4.0], 0.5, -1.5),
        ([-5.0, -2.0, -1.0, 4.0], 0.25, -5.0),
        ([1.0, 2.0, 3.0], 1.0, 2.0),
    ],
)
def test_cvar(values: Any, alpha: float, expected: float) -> None:
    """Check CVaR across empty, mixed, and all-positive inputs."""
    result = cvar(values, alpha=alpha)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_cvar_invalid_alpha() -> None:
    """Reject invalid alpha parameters."""
    with pytest.raises(ValueError):
        cvar([1.0, 2.0], alpha=0.0)


def test_portfolio_metrics_bundle() -> None:
    """Ensure metric bundle includes all expected keys and values."""
    pnl = [1.0, -0.5, math.nan, -0.5]
    metrics = portfolio_metrics(pnl)
    assert set(metrics) == {
        "EDP",
        "PF",
        "CVaR",
        "Sharpe",
        "MaxDrawdown",
        "WinRate",
    }
    assert metrics["EDP"] == pytest.approx(0.0)
    assert metrics["PF"] == pytest.approx(1.0)
    assert metrics["CVaR"] == pytest.approx(-0.5)
    assert metrics["Sharpe"] == pytest.approx(0.0)
    assert metrics["MaxDrawdown"] == pytest.approx(1.0)
    assert metrics["WinRate"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_portfolio_summary(alpha: float) -> None:
    """Summarise multiple portfolios with and without pandas."""
    inputs = {
        "gains": [1.0, 2.0, 3.0],
        "mix": [1.0, -1.0, 0.0, math.nan],
        "empty": [],
    }
    summary = portfolio_summary(inputs, alpha=alpha)

    if hasattr(summary, "loc"):
        df = cast(Any, summary)
        columns = list(df.columns)
        expected_cvar_name = "CVaR95" if alpha == 0.05 else "CVaR"
        assert columns == [
            "EDP",
            "PF",
            expected_cvar_name,
            "Sharpe",
            "MaxDrawdown",
            "WinRate",
        ]
        mix_row = df.loc["mix"]
        assert mix_row["EDP"] == pytest.approx(0.0)
        assert mix_row["PF"] == pytest.approx(1.0)
        assert math.isnan(df.loc["empty", columns[-1]])
    else:
        mapping_summary = cast(
            Mapping[str, Mapping[str, float]],
            summary,
        )
        assert set(mapping_summary) == {"gains", "mix", "empty"}
        expected_cvar_name = "CVaR95" if alpha == 0.05 else "CVaR"
        mix_row = mapping_summary["mix"]
        assert mix_row["EDP"] == pytest.approx(0.0)
        assert mix_row["PF"] == pytest.approx(1.0)
        assert expected_cvar_name in mix_row
        assert "Sharpe" in mix_row
        assert "MaxDrawdown" in mix_row
        assert "WinRate" in mix_row
        empty_row = mapping_summary["empty"]
        assert math.isnan(empty_row[expected_cvar_name])


def test_portfolio_metrics_extended_expected_values() -> None:
    """Extended metrics return deterministic values for a mixed series."""

    pnl = [-3.0, -1.0, 0.0, 2.0, 4.0]
    metrics = portfolio_metrics_extended(pnl, alpha=0.4, top_k_losses=2)

    assert metrics["EDP"] == pytest.approx(0.4)
    assert metrics["PF"] == pytest.approx(1.5)
    assert metrics["CVaR"] == pytest.approx(-2.0)
    assert metrics["Sharpe"] == pytest.approx(0.1480466420)
    assert metrics["Sortino"] == pytest.approx(0.1788854382)
    assert metrics["Omega1.0"] == pytest.approx(0.5714285714)
    assert metrics["MaxDrawdown"] == pytest.approx(4.0)
    assert metrics["WinRate"] == pytest.approx(0.4)
    assert metrics["Skewness"] == pytest.approx(0.1825232573)
    assert metrics["Kurtosis"] == pytest.approx(-0.6811784575)
    assert metrics["Sharpe"] == pytest.approx(0.148046642, rel=1e-6)
    assert metrics["Sortino"] == pytest.approx(0.178885438, rel=1e-6)
    assert metrics["Omega1.0"] == pytest.approx(4.0 / 7.0)
    assert metrics["MaxDrawdown"] == pytest.approx(4.0)
    assert metrics["WinRate"] == pytest.approx(2.0 / 5.0)
    assert metrics["Skewness"] == pytest.approx(0.122440323, rel=1e-6)
    assert metrics["Kurtosis"] == pytest.approx(-1.170294614, rel=1e-6)
    assert metrics["TailHitRate"] == pytest.approx(0.4)
    assert metrics["MeanGain"] == pytest.approx(3.0)
    assert metrics["MeanLoss"] == pytest.approx(-2.0)
    assert metrics["TopKLossConcentration"] == pytest.approx(1.0)


def test_sortino_non_negative_for_non_negative_returns() -> None:
    """The Sortino ratio should be non-negative when there are no losses."""

    pnl = [0.1, 0.2, 0.05]
    metrics = portfolio_metrics_extended(pnl)

    assert metrics["Sortino"] >= 0.0
    sortino = metrics["Sortino"]
    assert not math.isnan(sortino)
    assert sortino >= 0.0


def test_skewness_sign_reflects_tail_direction() -> None:
    """Positive and negative tails should produce matching skew signs."""

    positive_tail = [-1.0, 0.0, 0.5, 3.0]
    negative_tail = [-3.0, -0.5, 0.0, 1.0]

    positive_skew = portfolio_metrics_extended(positive_tail)["Skewness"]
    negative_skew = portfolio_metrics_extended(negative_tail)["Skewness"]

    assert positive_skew > 0.0
    assert negative_skew < 0.0
    assert positive_skew == pytest.approx(-negative_skew)
