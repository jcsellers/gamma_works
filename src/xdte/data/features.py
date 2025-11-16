"""Feature engineering helpers for the XDTE project."""

from __future__ import annotations

import builtins
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, cast

import pandas as pd

from xdte.data.loaders import PortfolioRecord
from xdte.live.sessions import SESSION_TO_SUFFIX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureInput:
    """Description of a single engineered feature."""

    name: str
    source: str
    origin: str
    description: str


NOTEBOOK_FEATURE_INPUTS: Dict[str, FeatureInput] = {
    "gap": FeatureInput(
        name="gap",
        source="portfolio",
        origin="PortfolioRecord.gap",
        description="Overnight move captured per trade; lifted directly from the loader normalisation.",
    ),
    "movement": FeatureInput(
        name="movement",
        source="portfolio",
        origin="PortfolioRecord.movement",
        description="Intraday movement from open to close for the underlying instrument.",
    ),
    "opening_vix": FeatureInput(
        name="opening_vix",
        source="portfolio",
        origin="PortfolioRecord.opening_vix",
        description="VIX level observed when the trade opened (as recorded in the trade CSV).",
    ),
    "closing_vix": FeatureInput(
        name="closing_vix",
        source="portfolio",
        origin="PortfolioRecord.closing_vix",
        description="VIX level observed when the trade closed.",
    ),
    "pnl": FeatureInput(
        name="pnl",
        source="portfolio",
        origin="PortfolioRecord.pnl",
        description="Realised profit/loss per trade – the primary series used for winsorisation and rule discovery.",
    ),
    "VIX_Entry_11": FeatureInput(
        name="VIX_Entry_11",
        source="market",
        origin="Market stats 11:00 session → Opening VIX",
        description="Session level VIX reading injected at decision time for the 11:00 books.",
    ),
    "VIX_Entry_1515": FeatureInput(
        name="VIX_Entry_1515",
        source="market",
        origin="Market stats 15:15 session → Opening VIX",
        description="Session level VIX reading used for 15:15 deployment windows.",
    ),
    "Intraday_Move_OpenToEntry_11": FeatureInput(
        name="Intraday_Move_OpenToEntry_11",
        source="market",
        origin="Market stats 11:00 session → Movement",
        description="Intraday move between the market open and the 11:00 entry snapshot.",
    ),
    "Intraday_Move_OpenToEntry_1515": FeatureInput(
        name="Intraday_Move_OpenToEntry_1515",
        source="market",
        origin="Market stats 15:15 session → Movement",
        description="Intraday move between the market open and the 15:15 decision point.",
    ),
    "L1_TS": FeatureInput(
        name="L1_TS",
        source="daily",
        origin="Lagged term-structure ratio (VIX_Close / VIX3M_Close)",
        description="Lagged daily term-structure efficiency ratio mirroring the exploratory notebooks.",
    ),
    "L1_VIX_Close": FeatureInput(
        name="L1_VIX_Close",
        source="daily",
        origin="Lagged VIX close",
        description="Previous session VIX close used for t=0 delta features and rule discovery baselines.",
    ),
    "L1_VIX_pct": FeatureInput(
        name="L1_VIX_pct",
        source="daily",
        origin="Lagged VIX percentile rank",
        description="Lagged percentile rank of the VIX close; canonical live feature name.",
    ),
    "L1_vvix_close": FeatureInput(
        name="L1_vvix_close",
        source="daily",
        origin="Lagged VVIX close",
        description="Previous day VVIX level used for kill-switch toggles.",
    ),
    "L1_vvix_pct": FeatureInput(
        name="L1_vvix_pct",
        source="daily",
        origin="Lagged VVIX percentile rank",
        description="Lagged percentile rank of VVIX for tail monitoring.",
    ),
    "L1_vvix_ema20": FeatureInput(
        name="L1_vvix_ema20",
        source="daily",
        origin="Lagged 20-day EMA of VVIX",
        description="Smoothed VVIX signal used to create kill-switch flags.",
    ),
    "L1_vvix_ema30": FeatureInput(
        name="L1_vvix_ema30",
        source="daily",
        origin="Lagged 30-day EMA of VVIX",
        description="Longer-term VVIX smoothing for the EMA30 kill-switch.",
    ),
    "L1_SPX_ATR_Pct": FeatureInput(
        name="L1_SPX_ATR_Pct",
        source="daily",
        origin="Lagged ATR percentage",
        description="Lagged 14-period ATR (high/low) expressed as a percentage of the SPX close.",
    ),
    "L1_SPX_Drawdown_Pct": FeatureInput(
        name="L1_SPX_Drawdown_Pct",
        source="daily",
        origin="Lagged drawdown percentage",
        description="Lagged rolling 252-day drawdown used as a stress proxy.",
    ),
    "L1_rv5": FeatureInput(
        name="L1_rv5",
        source="daily",
        origin="Lagged 5-day realised volatility",
        description="Lagged five-day realised volatility annualised to match the research notebooks.",
    ),
    "L1_rv20": FeatureInput(
        name="L1_rv20",
        source="daily",
        origin="Lagged 20-day realised volatility",
        description="Lagged twenty-day realised volatility used for the longer-horizon signal.",
    ),
    "DoW": FeatureInput(
        name="DoW",
        source="daily",
        origin="Calendar weekday (0=Mon)",
        description="Weekday indicator employed for calendar seasonality checks.",
    ),
    "L1_vvix_above_ema20": FeatureInput(
        name="L1_vvix_above_ema20",
        source="derived",
        origin="L1_vvix_close > L1_vvix_ema20",
        description="Boolean kill-switch toggle mirroring the EMA20 guardrail from the notebooks.",
    ),
    "L1_vvix_above_ema30": FeatureInput(
        name="L1_vvix_above_ema30",
        source="derived",
        origin="L1_vvix_close > L1_vvix_ema30",
        description="Boolean kill-switch toggle aligned with the EMA30 constraint.",
    ),
    "t0_VIX_change_from_close_11": FeatureInput(
        name="t0_VIX_change_from_close_11",
        source="derived",
        origin="VIX_Entry_11 - L1_VIX_Close",
        description="Delta between the 11:00 VIX snapshot and the prior close used in the call book features.",
    ),
    "t0_VIX_change_from_close_15": FeatureInput(
        name="t0_VIX_change_from_close_15",
        source="derived",
        origin="VIX_Entry_1515 - L1_VIX_Close",
        description="Delta between the 15:15 VIX snapshot and the prior close supporting the afternoon strategies.",
    ),
}

# Sourced from ``original_code.txt`` Cell 0 (lines 369-388) to keep the
# package in lockstep with the reference Colab feature masks. Do not change
# without updating the notebook reference and associated tests.
BOOK_FEATS_CONFIG: dict[str, list[str]] = {
    "PUTS_0DTE_11": [
        "Intraday_Move_OpenToEntry_11",
        "VIX_Entry_11",
        "L1_TS",
        "L1_vvix_pct",
        "L1_SPX_ATR_Pct",
        "L1_SPX_Drawdown_Pct",
        "DoW",
        "L1_rv20",
        "L1_vvix_above_ema20",
    ],
    "PUTS_1DTE_1515": [
        "Intraday_Move_OpenToEntry_1515",
        "VIX_Entry_1515",
        "t0_VIX_change_from_close_15",
        "L1_TS",
        "L1_vvix_pct",
        "L1_SPX_ATR_Pct",
        "L1_SPX_Drawdown_Pct",
        "DoW",
        "L1_rv20",
        "L1_vvix_above_ema20",
    ],
    "CALLS_0DTE_11": [
        "Intraday_Move_OpenToEntry_11",
        "VIX_Entry_11",
        "t0_VIX_change_from_close_11",
        "L1_TS",
        "L1_SPX_ATR_Pct",
        "L1_VIX_pct",
        "DoW",
    ],
    "CALLS_1DTE_1515": [
        "Intraday_Move_OpenToEntry_1515",
        "VIX_Entry_1515",
        "t0_VIX_change_from_close_15",
        "L1_TS",
        "L1_SPX_ATR_Pct",
        "L1_VIX_pct",
        "DoW",
    ],
}


def describe_feature_inputs() -> Dict[str, FeatureInput]:
    return dict(NOTEBOOK_FEATURE_INPUTS)


def infer_canonical_book(record: PortfolioRecord | Mapping[str, object]) -> str:
    if isinstance(record, Mapping):
        context = record.get("context")
        if isinstance(context, Mapping):
            source = str(context.get("source", ""))
        else:
            source = str(record.get("source", ""))
        strategy = str(record.get("strategy", ""))
        probe = (source or strategy).lower()
    else:
        probe = record.context.get("source", record.strategy).lower()

    source = probe
    if "0dte" in source:
        if "put" in source:
            return "PUTS_0DTE_11"
        if "call" in source:
            return "CALLS_0DTE_11"
    if "1dte" in source:
        if "put" in source:
            return "PUTS_1DTE_1515"
        if "call" in source:
            return "CALLS_1DTE_1515"
    return "UNKNOWN"


_SESSION_PATTERN_GROUPS: Tuple[
    Tuple[str, Tuple[re.Pattern[str], ...]],
    ...,
] = (
    (
        "15:15",
        (
            re.compile(r"(?<!\d)1515(?!\d)"),
            re.compile(r"(?<!\d)15:15(?=$|[^0-9a-z])"),
            re.compile(r"(?<!\d)3(?:[:]?15)?(?:\s|[-_])*pm(?=$|[^0-9a-z])"),
        ),
    ),
    (
        "11:00",
        (
            re.compile(r"(?<!\d)1100(?!\d)"),
            re.compile(r"(?<!\d)11:00(?=$|[^0-9a-z])"),
            re.compile(r"(?<!\d)11(?:[:]?00)?(?:\s|[-_])*am(?=$|[^0-9a-z])"),
        ),
    ),
)


def _detect_session_from_source(source: str) -> str | None:
    probe = source.lower()
    for session, patterns in _SESSION_PATTERN_GROUPS:
        for pattern in patterns:
            if pattern.search(probe):
                return session
    return None


def infer_market_session(record: PortfolioRecord | Mapping[str, object]) -> str | None:
    if isinstance(record, Mapping):
        context = record.get("context")
        if isinstance(context, Mapping):
            source = str(context.get("source", ""))
        else:
            source = str(record.get("source", ""))
    else:
        source = record.context.get("source", "")
    return _detect_session_from_source(source)


def _to_float(value: float | None) -> float:
    return float(value) if value is not None else math.nan


def _extract_numeric(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TypeError(f"Unable to parse numeric value for {key!r}") from exc
    raise TypeError(f"Unsupported numeric type for {key!r}: {type(value)!r}")


def _normalise_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported date value: {value!r}")


def _as_float_series(values: Sequence[float] | pd.Series) -> pd.Series:
    """Return a ``float64`` series for helper computations."""

    return pd.Series(values, dtype="float64")


def _ewm(values: Sequence[float] | pd.Series, span: int) -> List[float]:
    series = _as_float_series(values)
    if series.empty:
        return []

    mask = series.notna()
    if not mask.any():
        return cast(List[float], series.tolist())

    grouped = (~mask).cumsum()
    transformed = series.groupby(grouped).transform(
        lambda group: group.ewm(span=span, adjust=False).mean()
    )
    result = transformed.where(mask)
    return cast(List[float], result.tolist())


def _rolling_max(values: Sequence[float], window: int, min_periods: int) -> List[float]:
    series = _as_float_series(values)
    rolled = series.rolling(window=window, min_periods=min_periods).max()
    return cast(List[float], rolled.tolist())


def _rolling_std(values: Sequence[float], window: int, min_periods: int) -> List[float]:
    series = _as_float_series(values)
    rolled = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return cast(List[float], rolled.tolist())


def _pct_change(values: Sequence[float]) -> List[float]:
    series = _as_float_series(values)
    previous = series.shift(1)
    change = series.pct_change(fill_method=None)
    change = change.where(previous != 0.0)
    change = change.replace([math.inf, -math.inf], math.nan)
    return cast(List[float], change.tolist())


def _rank_pct(values: Sequence[float]) -> List[float]:
    series = _as_float_series(values)
    if series.empty:
        return []
    ranks = series.rank(method="average", na_option="keep")
    percentiles = ranks / float(len(series))
    return cast(List[float], percentiles.tolist())


def _shift(values: Sequence[float], periods: int = 1) -> List[float]:
    series = _as_float_series(values)
    if periods <= 0:
        shifted = series.copy()
    else:
        shifted = series.shift(periods)
    return cast(List[float], shifted.tolist())


def _safe_bool(lhs: float, rhs: float) -> bool:
    if math.isnan(lhs) or math.isnan(rhs):
        return False
    return lhs > rhs


_TRADE_FRAME_COLUMNS = [
    "strategy",
    "source",
    "book",
    "open_timestamp",
    "open_date",
    "notional",
    "pnl",
    "gap",
    "movement",
    "opening_vix",
    "closing_vix",
    "opening_price",
    "closing_price",
    "avg_closing_cost",
    "premium",
    "contracts",
    "funds_at_close",
    "margin_requirement",
    "opening_short_long_ratio",
    "closing_short_long_ratio",
    "max_profit",
    "max_loss",
]


_MARKET_FRAME_COLUMNS = [
    "strategy",
    "source",
    "open_timestamp",
    "open_date",
    "session",
    "opening_vix",
    "closing_vix",
    "movement",
]


def _records_to_frame(
    records: Sequence[PortfolioRecord],
    columns: Sequence[str],
    row_builder: Callable[[PortfolioRecord], Mapping[str, object]],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(columns))
    rows = [dict(row_builder(record)) for record in records]
    return pd.DataFrame(rows, columns=list(columns))


def trade_records_to_frame(records: Sequence[PortfolioRecord]) -> pd.DataFrame:
    def _row(record: PortfolioRecord) -> Mapping[str, object]:
        open_ts = pd.Timestamp(record.open_timestamp)
        return {
            "strategy": record.strategy,
            "source": record.context.get("source", record.strategy),
            "book": infer_canonical_book(record),
            "open_timestamp": open_ts,
            "open_date": open_ts.normalize(),
            "notional": float(record.notional),
            "pnl": _to_float(record.pnl),
            "gap": _to_float(record.gap),
            "movement": _to_float(record.movement),
            "opening_vix": _to_float(record.opening_vix),
            "closing_vix": _to_float(record.closing_vix),
            "opening_price": _to_float(record.opening_price),
            "closing_price": _to_float(record.closing_price),
            "avg_closing_cost": _to_float(record.avg_closing_cost),
            "premium": _to_float(record.premium),
            "contracts": _to_float(record.contracts),
            "funds_at_close": _to_float(record.funds_at_close),
            "margin_requirement": _to_float(record.margin_requirement),
            "opening_short_long_ratio": _to_float(record.opening_short_long_ratio),
            "closing_short_long_ratio": _to_float(record.closing_short_long_ratio),
            "max_profit": _to_float(record.max_profit),
            "max_loss": _to_float(record.max_loss),
        }

    return _records_to_frame(records, _TRADE_FRAME_COLUMNS, _row)


def market_records_to_frame(records: Sequence[PortfolioRecord]) -> pd.DataFrame:
    def _row(record: PortfolioRecord) -> Mapping[str, object]:
        open_ts = pd.Timestamp(record.open_timestamp)
        return {
            "strategy": record.strategy,
            "source": record.context.get("source", record.strategy),
            "open_timestamp": open_ts,
            "open_date": open_ts.normalize(),
            "session": infer_market_session(record),
            "opening_vix": _to_float(record.opening_vix),
            "closing_vix": _to_float(record.closing_vix),
            "movement": _to_float(record.movement),
        }

    return _records_to_frame(records, _MARKET_FRAME_COLUMNS, _row)


def _infer_book_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    source = frame.get("source", pd.Series(index=frame.index, dtype="object"))
    strategy = frame.get("strategy", pd.Series(index=frame.index, dtype="object"))
    source_str = source.astype(str).where(source.notna(), "")
    strategy_str = strategy.astype(str).where(strategy.notna(), "")
    probe = source_str.where(source_str.astype(bool), strategy_str).str.lower()
    inferred = pd.Series("UNKNOWN", index=frame.index, dtype="object")
    mask_0dte = probe.str.contains("0dte", na=False)
    mask_1dte = probe.str.contains("1dte", na=False)
    mask_put = probe.str.contains("put", na=False)
    mask_call = probe.str.contains("call", na=False)
    inferred.loc[mask_0dte & mask_put] = "PUTS_0DTE_11"
    inferred.loc[mask_0dte & mask_call] = "CALLS_0DTE_11"
    inferred.loc[mask_1dte & mask_put] = "PUTS_1DTE_1515"
    inferred.loc[mask_1dte & mask_call] = "CALLS_1DTE_1515"
    return inferred


def _infer_session_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    source = frame.get("source", pd.Series(index=frame.index, dtype="object"))
    source_str = source.astype(str).where(source.notna(), "")
    inferred = source_str.map(_detect_session_from_source)
    return inferred.astype("object")


def build_trade_panel(
    trades: pd.DataFrame,
    *,
    book_mapper: Callable[[Mapping[str, object]], str] | None = None,
) -> pd.DataFrame:
    if trades.empty:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["book", "open_date"])
        columns = [
            "pnl",
            "gap",
            "movement",
            "opening_vix",
            "closing_vix",
        ]
        return pd.DataFrame(columns=columns, index=empty_index)

    frame = trades.copy()
    if book_mapper is not None:
        rows = frame.to_dict("records")
        frame["book"] = pd.Series(
            (book_mapper(row) for row in rows), index=frame.index, dtype="object"
        )
    elif "book" not in frame:
        frame["book"] = _infer_book_series(frame)
    else:
        frame["book"] = frame["book"].astype(str)

    frame["book"] = frame["book"].fillna("UNKNOWN").astype(str)

    if "open_date" in frame:
        frame["open_date"] = pd.to_datetime(frame["open_date"])
    elif "open_timestamp" in frame:
        frame["open_date"] = pd.to_datetime(frame["open_timestamp"])
    else:
        raise KeyError(
            "build_trade_panel requires an 'open_date' or 'open_timestamp' column"
        )
    frame["open_date"] = frame["open_date"].dt.normalize()
    frame = frame.dropna(subset=["open_date"])

    numeric_columns = [
        "notional",
        "pnl",
        "gap",
        "movement",
        "opening_vix",
        "closing_vix",
        "opening_price",
        "closing_price",
        "avg_closing_cost",
        "premium",
        "contracts",
        "funds_at_close",
        "margin_requirement",
        "opening_short_long_ratio",
        "closing_short_long_ratio",
        "max_profit",
        "max_loss",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.Series(math.nan, index=frame.index, dtype="float64")

    frame = frame.dropna(subset=["notional"])
    if frame.empty:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["book", "open_date"])
        columns = [
            "pnl",
            "gap",
            "movement",
            "opening_vix",
            "closing_vix",
        ]
        return pd.DataFrame(columns=columns, index=empty_index)

    selector = frame.groupby(["book", "open_date"], sort=False)["notional"].idxmax()
    selected = frame.loc[
        selector,
        [
            "book",
            "open_date",
            "pnl",
            "gap",
            "movement",
            "opening_vix",
            "closing_vix",
            "opening_price",
            "closing_price",
            "avg_closing_cost",
            "premium",
            "contracts",
            "funds_at_close",
            "margin_requirement",
            "opening_short_long_ratio",
            "closing_short_long_ratio",
            "max_profit",
            "max_loss",
        ],
    ]
    selected = selected.sort_values(["book", "open_date"]).set_index(
        ["book", "open_date"]
    )
    return selected


def build_market_feature_frame(
    market: pd.DataFrame,
    *,
    session_suffix_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    suffix_map = session_suffix_map or dict(SESSION_TO_SUFFIX)
    if market.empty:
        index = pd.DatetimeIndex([], name="open_date")
        columns = [
            "Intraday_Move_OpenToEntry_11",
            "Intraday_Move_OpenToEntry_1515",
            "VIX_Entry_11",
            "VIX_Entry_1515",
        ]
        return pd.DataFrame(columns=columns, index=index)

    frame = market.copy()
    if "session" not in frame:
        frame["session"] = _infer_session_series(frame)

    if "open_date" in frame:
        frame["open_date"] = pd.to_datetime(frame["open_date"])
    elif "open_timestamp" in frame:
        frame["open_date"] = pd.to_datetime(frame["open_timestamp"])
    else:
        raise KeyError(
            "build_market_feature_frame requires an 'open_date' or 'open_timestamp' column"
        )
    frame["open_date"] = frame["open_date"].dt.normalize()

    frame["opening_vix"] = pd.to_numeric(frame.get("opening_vix"), errors="coerce")
    frame["closing_vix"] = pd.to_numeric(frame.get("closing_vix"), errors="coerce")
    frame["movement"] = pd.to_numeric(frame.get("movement"), errors="coerce")
    frame["vix_value"] = frame["opening_vix"].where(
        frame["opening_vix"].notna(), frame["closing_vix"]
    )

    valid_sessions = set(suffix_map.keys())
    frame = frame[frame["session"].isin(valid_sessions)]
    frame = frame.dropna(subset=["open_date"])
    frame = frame.dropna(subset=["vix_value", "movement"])

    if frame.empty:
        index = pd.DatetimeIndex([], name="open_date")
        columns = [
            "Intraday_Move_OpenToEntry_11",
            "Intraday_Move_OpenToEntry_1515",
            "VIX_Entry_11",
            "VIX_Entry_1515",
        ]
        return pd.DataFrame(columns=columns, index=index)

    aggregated = (
        frame.groupby(["open_date", "session"], sort=True)
        .agg(vix_value=("vix_value", "mean"), movement=("movement", "mean"))
        .reset_index()
    )

    index = pd.DatetimeIndex(sorted(aggregated["open_date"].unique()), name="open_date")
    result = pd.DataFrame(index=index)
    for session, suffix in suffix_map.items():
        session_stats = (
            aggregated[aggregated["session"] == session]
            .set_index("open_date")
            .reindex(index)
        )
        if session_stats.empty:
            result[f"VIX_Entry_{suffix}"] = pd.Series(
                math.nan, index=result.index, dtype="float64"
            )
            result[f"Intraday_Move_OpenToEntry_{suffix}"] = pd.Series(
                math.nan, index=result.index, dtype="float64"
            )
        else:
            result[f"VIX_Entry_{suffix}"] = session_stats["vix_value"]
            result[f"Intraday_Move_OpenToEntry_{suffix}"] = session_stats["movement"]

    ordered_columns = [
        "Intraday_Move_OpenToEntry_11",
        "Intraday_Move_OpenToEntry_1515",
        "VIX_Entry_11",
        "VIX_Entry_1515",
    ]
    for column in ordered_columns:
        if column not in result:
            result[column] = pd.Series(math.nan, index=result.index, dtype="float64")
    return result[ordered_columns]


def compute_daily_context_features(
    daily_rows: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "DoW",
        "L1_TS",
        "L1_VIX_Close",
        "L1_VIX_pct",
        "L1_vvix_close",
        "L1_vvix_pct",
        "L1_vvix_ema20",
        "L1_vvix_ema30",
        "L1_SPX_ATR_Pct",
        "L1_SPX_Drawdown_Pct",
        "L1_rv5",
        "L1_rv20",
        "L1_vvix_above_ema20",
        "L1_vvix_above_ema30",
    ]
    if daily_rows.empty or "open_date" not in daily_rows:
        index = pd.DatetimeIndex([], name="open_date")
        return pd.DataFrame(columns=columns, index=index)

    frame = daily_rows.copy()
    frame["open_date"] = frame["open_date"].map(_normalise_date)
    frame = frame.sort_values("open_date")
    frame["open_date"] = pd.to_datetime(frame["open_date"])
    frame = frame.set_index("open_date")
    frame.index.name = "open_date"

    def _numeric_column(name: str) -> pd.Series:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(math.nan, index=frame.index, dtype="float64")

    high = _numeric_column("SPX_High")
    low = _numeric_column("SPX_Low")
    close = _numeric_column("SPX_Close")
    vix_close = _numeric_column("VIX_Close")
    vix3m_close = _numeric_column("VIX3M_Close")
    vvix_close = _numeric_column("VVIX_Close")

    atr_input = ((high - low) / close).where(close != 0.0)
    atr_pct = pd.Series(_ewm(atr_input, span=14), index=frame.index)

    drawdown_denominator = close.rolling(window=252, min_periods=20).max()
    drawdown = ((close - drawdown_denominator) / drawdown_denominator).where(
        (drawdown_denominator.notna()) & (drawdown_denominator != 0.0)
    )

    ts_eff = (vix_close / vix3m_close).where(vix3m_close.notna() & (vix3m_close != 0.0))

    if len(frame.index) == 0:
        vix_pct = pd.Series(dtype="float64", index=frame.index)
        vvix_pct = pd.Series(dtype="float64", index=frame.index)
    else:
        vix_pct = vix_close.rank(method="average", na_option="keep") / float(
            len(frame.index)
        )
        vvix_pct = vvix_close.rank(method="average", na_option="keep") / float(
            len(frame.index)
        )

    vvix_ema20 = pd.Series(_ewm(vvix_close, span=20), index=frame.index)
    vvix_ema30 = pd.Series(_ewm(vvix_close, span=30), index=frame.index)

    returns = close.pct_change(fill_method=None)
    returns = returns.where(close.shift(1) != 0.0)
    returns = returns.replace([math.inf, -math.inf], math.nan)
    annualisation = math.sqrt(252.0)
    rv5 = returns.rolling(window=5, min_periods=2).std(ddof=0) * annualisation
    rv20 = returns.rolling(window=20, min_periods=5).std(ddof=0) * annualisation

    result = pd.DataFrame(index=frame.index)
    result["DoW"] = result.index.dayofweek
    result["L1_TS"] = ts_eff.shift(1)
    result["L1_VIX_Close"] = vix_close.shift(1)
    result["L1_VIX_pct"] = vix_pct.shift(1)
    result["L1_vvix_close"] = vvix_close.shift(1)
    result["L1_vvix_pct"] = vvix_pct.shift(1)
    result["L1_vvix_ema20"] = vvix_ema20.shift(1)
    result["L1_vvix_ema30"] = vvix_ema30.shift(1)
    result["L1_SPX_ATR_Pct"] = atr_pct.shift(1)
    result["L1_SPX_Drawdown_Pct"] = drawdown.shift(1)
    result["L1_rv5"] = rv5.shift(1)
    result["L1_rv20"] = rv20.shift(1)

    lhs_vvix = result["L1_vvix_close"]
    ema20 = result["L1_vvix_ema20"]
    ema30 = result["L1_vvix_ema30"]
    result["L1_vvix_above_ema20"] = (
        lhs_vvix.gt(ema20) & lhs_vvix.notna() & ema20.notna()
    )
    result["L1_vvix_above_ema30"] = (
        lhs_vvix.gt(ema30) & lhs_vvix.notna() & ema30.notna()
    )

    return result


def write_daily_context_csv(
    data_dir: Path,
    frame: pd.DataFrame,
    *,
    warnings: Iterable[str] | None = None,
) -> Path:
    """Normalise ``open_date`` and persist the engineered daily features."""

    resolved = data_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    output_path = resolved / "daily_context.csv"

    working = frame.copy()
    if "open_date" not in working:
        if working.index.name == "open_date":
            working = working.reset_index()
        else:
            raise ValueError("daily context frame must include an open_date column")

    working["open_date"] = pd.to_datetime(working["open_date"]).dt.date
    working = working.sort_values("open_date")
    working["open_date"] = working["open_date"].map(lambda value: value.isoformat())

    ordered_columns = ["open_date"] + [
        column for column in working.columns if column != "open_date"
    ]
    working.to_csv(output_path, index=False, columns=ordered_columns)

    if warnings:
        for message in warnings:
            logger.warning("Daily context provider warning: %s", message)

    return output_path


def generate_feature_panel(
    trades: pd.DataFrame,
    market_stats: pd.DataFrame,
    daily_context: pd.DataFrame,
    *,
    book_mapper: Callable[[Mapping[str, object]], str] | None = None,
) -> pd.DataFrame:
    trade_panel = build_trade_panel(trades, book_mapper=book_mapper)
    market_panel = build_market_feature_frame(market_stats)
    has_daily_columns = [
        column for column in daily_context.columns if column != "open_date"
    ]
    if has_daily_columns:
        daily_panel = compute_daily_context_features(daily_context)
        daily_df = daily_panel.reset_index()
    else:
        daily_df = pd.DataFrame(columns=["open_date"])

    trade_df = trade_panel.reset_index()
    market_df = market_panel.reset_index()

    panel = trade_df.merge(market_df, on="open_date", how="left")
    panel = panel.merge(daily_df, on="open_date", how="left")

    if "VIX_Entry_11" in panel and "L1_VIX_Close" in panel:
        panel["t0_VIX_change_from_close_11"] = (
            panel["VIX_Entry_11"] - panel["L1_VIX_Close"]
        )
    else:
        panel["t0_VIX_change_from_close_11"] = pd.Series(
            math.nan, index=panel.index, dtype="float64"
        )
    if "VIX_Entry_1515" in panel and "L1_VIX_Close" in panel:
        panel["t0_VIX_change_from_close_15"] = (
            panel["VIX_Entry_1515"] - panel["L1_VIX_Close"]
        )
    else:
        panel["t0_VIX_change_from_close_15"] = pd.Series(
            math.nan, index=panel.index, dtype="float64"
        )

    panel = panel.sort_values(["book", "open_date"])
    panel = panel.set_index(["book", "open_date"])
    return panel


# This shim registers the feature helpers into ``builtins`` so exploratory
# notebooks can reference ``features`` without verbose imports. Production code
# and tests should import ``xdte.data.features`` directly; the builtins alias is
# purely for ergonomics and not required for core execution.
def _register_features_module() -> None:
    """Expose the features module under the ``features`` alias."""

    module: ModuleType = sys.modules[__name__]
    if not hasattr(builtins, "features"):
        setattr(builtins, "features", module)


_register_features_module()


__all__ = [
    "FeatureInput",
    "NOTEBOOK_FEATURE_INPUTS",
    "BOOK_FEATS_CONFIG",
    "build_market_feature_frame",
    "build_trade_panel",
    "compute_daily_context_features",
    "write_daily_context_csv",
    "describe_feature_inputs",
    "generate_feature_panel",
    "infer_canonical_book",
    "infer_market_session",
    "market_records_to_frame",
    "trade_records_to_frame",
]
