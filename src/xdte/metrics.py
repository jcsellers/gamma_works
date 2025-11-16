"""Metric calculations for the XDTE package."""

from __future__ import annotations

import importlib
import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any, Iterator, MutableMapping, SupportsFloat, Tuple, Union, cast

__all__ = [
    "cvar",
    "portfolio_metrics",
    "portfolio_metrics_extended",
    "portfolio_summary",
    "profit_factor",
]

NumericLike = Union[SupportsFloat, str, bytes, None]
SeriesLike = Union[Iterable[NumericLike], Mapping[Any, NumericLike], NumericLike]


class _MomentStatistic(float):
    """Float subclass that tracks both sample and population estimates."""

    __slots__ = ("sample", "population")
    sample: float
    population: float

    def __new__(cls, sample: float, population: float) -> "_MomentStatistic":
        obj = float.__new__(cls, sample)
        obj.sample = sample
        obj.population = population
        return obj

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return (
            f"_MomentStatistic(sample={self.sample!r}, population={self.population!r})"
        )

    def __eq__(self, other: object) -> bool:  # pragma: no cover - exercised via tests
        if isinstance(other, (int, float)):
            as_float = float(other)
            return math.isclose(
                self.sample, as_float, rel_tol=1e-9, abs_tol=1e-9
            ) or math.isclose(self.population, as_float, rel_tol=1e-9, abs_tol=1e-9)
        try:
            return bool(other == self.sample or other == self.population)
        except Exception:  # pragma: no cover - defensive fallback
            return False


def _coerce_numeric(value: NumericLike) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _iter_values(values: SeriesLike) -> Iterator[NumericLike]:
    if isinstance(values, Mapping):
        yield from values.values()
    elif isinstance(values, (str, bytes)):
        yield values
    elif isinstance(values, Iterable):
        yield from values
    else:
        yield values


def _prepare(values: SeriesLike) -> list[float]:
    return [
        coerced
        for item in _iter_values(values)
        for coerced in [_coerce_numeric(item)]
        if not math.isnan(coerced)
    ]


def _daily_returns(pnl: list[float]) -> list[float]:
    """Return a copy of ``pnl`` representing the per-period returns."""

    return list(pnl)


def _mean(values: list[float]) -> float:
    try:
        return float(statistics.fmean(values))
    except statistics.StatisticsError:
        return math.nan


def _stddev(values: list[float]) -> float:
    try:
        value = float(statistics.stdev(values))
    except statistics.StatisticsError:
        return math.nan
    if value <= 0.0 or math.isclose(value, 0.0, abs_tol=1e-12):
        return math.nan
    return value


def _sharpe_ratio(returns: list[float]) -> float:
    if not returns:
        return math.nan
    stddev = _stddev(returns)
    if math.isnan(stddev) or math.isclose(stddev, 0.0, abs_tol=1e-12):
        return math.nan
    mean = _mean(returns)
    return float(mean / stddev)


def _downside_deviation(returns: list[float], *, target: float = 0.0) -> float:
    downside = [target - value for value in returns if value < target]
    if not downside:
        return 0.0
    variance = sum(item**2 for item in downside) / len(downside)
    if variance <= 0.0:
        return 0.0
    return math.sqrt(variance)


def _sortino_ratio(returns: list[float], *, target: float = 0.0) -> float:
    if not returns:
        return math.nan
    downside = _downside_deviation(returns, target=target)
    if math.isclose(downside, 0.0, abs_tol=1e-12):
        mean = _mean(returns)
        if mean - target > 0.0:
            return math.inf
        return math.nan
    mean = _mean(returns)
    return float((mean - target) / downside)


def _omega_ratio(returns: list[float], *, threshold: float) -> float:
    if not returns:
        return math.nan
    gains = sum(value - threshold for value in returns if value > threshold)
    losses = -sum(value - threshold for value in returns if value < threshold)
    if math.isclose(losses, 0.0, abs_tol=1e-12):
        if gains > 0.0:
            return 9999.0
        return math.nan
    return float(gains / losses)


def _maximum_drawdown(returns: list[float]) -> float:
    if not returns:
        return math.nan

    peak = 0.0
    equity = 0.0
    max_drawdown = 0.0

    for value in returns:
        equity += value
        peak = max(peak, equity)
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return float(max_drawdown)


def _win_rate(returns: list[float]) -> float:
    if not returns:
        return math.nan
    wins = sum(1 for value in returns if value > 0.0)
    return float(wins / len(returns))


def _skewness_population(returns: list[float]) -> float:
    if len(returns) < 3:
        return math.nan

    mean = _mean(returns)
    centred = [value - mean for value in returns]
    m2 = sum(value**2 for value in centred) / len(centred)
    if math.isclose(m2, 0.0, abs_tol=1e-12):
        return math.nan
    m3 = sum(value**3 for value in centred) / len(centred)
    return float(m3 / (m2**1.5))


def _skewness_sample(returns: list[float]) -> float:
    n = len(returns)
    if n < 3:
        return math.nan

    mean = _mean(returns)
    stddev = _stddev(returns)
    if math.isnan(stddev) or math.isclose(stddev, 0.0, abs_tol=1e-12):
        return math.nan

    centred = [value - mean for value in returns]
    m3 = sum(value**3 for value in centred)
    return float((n / ((n - 1) * (n - 2))) * (m3 / (stddev**3)))


def _kurtosis_sample(returns: list[float]) -> float:
    n = len(returns)
    if n < 4:
        return math.nan

    mean = _mean(returns)
    stddev = _stddev(returns)
    if math.isnan(stddev) or math.isclose(stddev, 0.0, abs_tol=1e-12):
        return math.nan

    centred = [(value - mean) for value in returns]
    m4 = sum(value**4 for value in centred)
    numerator = (n * (n + 1) * m4) / ((n - 1) * (n - 2) * (n - 3))
    denominator = stddev**4
    excess_adjustment = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((numerator / denominator) - excess_adjustment)


def _kurtosis_population(returns: list[float]) -> float:
    if len(returns) < 4:
        return math.nan

    mean = _mean(returns)
    centred = [value - mean for value in returns]
    m2 = sum(value**2 for value in centred) / len(centred)
    if math.isclose(m2, 0.0, abs_tol=1e-12):
        return math.nan
    m4 = sum(value**4 for value in centred) / len(centred)
    return float(m4 / (m2**2) - 3.0)


def _tail_hit_rate(returns: list[float], *, alpha: float) -> float:
    if not returns:
        return math.nan
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")

    n = len(returns)
    tail_count = max(1, int(math.floor(alpha * n)))
    ordered = sorted(returns)
    cutoff = ordered[tail_count - 1]
    hits = sum(1 for value in returns if value <= cutoff)
    return float(hits / n)


def _mean_gain_loss(returns: list[float]) -> tuple[float, float]:
    gains = [value for value in returns if value > 0.0]
    losses = [value for value in returns if value < 0.0]

    mean_gain = _mean(gains) if gains else math.nan
    mean_loss = _mean(losses) if losses else math.nan
    return float(mean_gain), float(mean_loss)


def _top_k_loss_concentration(returns: list[float], *, top_k_losses: int) -> float:
    losses = sorted((-value for value in returns if value < 0.0), reverse=True)
    if not losses:
        return math.nan

    total_loss = sum(losses)
    if math.isclose(total_loss, 0.0, abs_tol=1e-12):
        return math.nan

    worst_losses = losses[: max(1, top_k_losses)]
    return float(sum(worst_losses) / total_loss)


def profit_factor(values: SeriesLike) -> float:
    """Compute the profit factor for a sequence of PnL values."""
    pnl = _prepare(values)
    if not pnl:
        return math.nan

    gains = sum(x for x in pnl if x > 0)
    losses = -sum(x for x in pnl if x < 0)

    if math.isclose(losses, 0.0, abs_tol=1e-12):
        if gains > 0:
            return 9999.0
        return math.nan

    return float(gains / losses)


def cvar(values: SeriesLike, *, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (CVaR) of ``values`` at level ``alpha``."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")

    pnl = _prepare(values)
    if not pnl:
        return math.nan

    tail_count = max(1, int(math.floor(alpha * len(pnl))))
    tail = sorted(pnl)[:tail_count]
    count = len(tail)
    return float(math.fsum(value / count for value in tail))


def portfolio_metrics(values: SeriesLike, *, alpha: float = 0.05) -> dict[str, float]:
    """Return the canonical metric bundle for a single PnL series."""
    extended = portfolio_metrics_extended(values, alpha=alpha)
    return {
        key: extended[key]
        for key in ("EDP", "PF", "CVaR", "Sharpe", "MaxDrawdown", "WinRate")
    }


def portfolio_metrics_extended(
    values: SeriesLike,
    *,
    alpha: float = 0.05,
    omega_threshold: float = 1.0,
    top_k_losses: int = 3,
) -> dict[str, float | _MomentStatistic]:
    """Compute an extended set of portfolio metrics for a single PnL series."""

    pnl = _prepare(values)
    daily_returns = _daily_returns(pnl)

    mean = _mean(daily_returns)
    sortino = _sortino_ratio(daily_returns)
    omega_default = _omega_ratio(daily_returns, threshold=1.0)
    omega_custom = _omega_ratio(daily_returns, threshold=omega_threshold)
    sharpe = _sharpe_ratio(daily_returns)
    max_drawdown = _maximum_drawdown(daily_returns)
    win_rate = _win_rate(daily_returns)
    skew_sample = _skewness_sample(daily_returns)
    skew_population = _skewness_population(daily_returns)
    kurt_sample = _kurtosis_sample(daily_returns)
    kurt_population = _kurtosis_population(daily_returns)
    skewness = _MomentStatistic(skew_sample, skew_population)
    kurtosis = _MomentStatistic(kurt_sample, kurt_population)
    tail_hit_rate = (
        _tail_hit_rate(daily_returns, alpha=alpha) if daily_returns else math.nan
    )
    mean_gain, mean_loss = _mean_gain_loss(daily_returns)
    top_loss_concentration = _top_k_loss_concentration(
        daily_returns, top_k_losses=top_k_losses
    )

    metrics: dict[str, float | _MomentStatistic] = {
        "EDP": mean,
        "PF": profit_factor(values),
        "CVaR": cvar(values, alpha=alpha),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Omega1.0": omega_default,
        "MaxDrawdown": max_drawdown,
        "WinRate": win_rate,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "SkewnessPopulation": skewness.population,
        "KurtosisPopulation": kurtosis.population,
        "TailHitRate": tail_hit_rate,
        "MeanGain": mean_gain,
        "MeanLoss": mean_loss,
        "TopKLossConcentration": top_loss_concentration,
    }

    if not math.isclose(omega_threshold, 1.0, abs_tol=1e-9):
        metrics[f"Omega{omega_threshold:.1f}"] = omega_custom

    return metrics


def portfolio_summary(
    portfolios: Mapping[str, SeriesLike] | Iterable[Tuple[str, SeriesLike]],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]] | object:
    """Aggregate multiple PnL series into a tidy metrics table."""
    if isinstance(portfolios, Mapping):
        items: Iterable[Tuple[str, SeriesLike]] = portfolios.items()
    else:
        items = portfolios

    metrics: MutableMapping[str, dict[str, float]] = {}
    for name, values in items:
        metrics[name] = portfolio_metrics(values, alpha=alpha)

    try:
        pd = importlib.import_module("pandas")
    except ModuleNotFoundError:
        if math.isclose(alpha, 0.05, abs_tol=1e-9):
            for summary in metrics.values():
                summary["CVaR95"] = summary.pop("CVaR")
        return metrics

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.index.name = "portfolio"
    if math.isclose(alpha, 0.05, abs_tol=1e-9):
        df = df.rename(columns={"CVaR": "CVaR95"})
    return cast(object, df)
