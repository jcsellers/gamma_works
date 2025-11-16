"""Dataset loading scaffolding for the XDTE project.

The functions in this module intentionally mirror the helpers used during
notebook exploration so that experimentation code can migrate into the
package with minimal churn. They operate on lightweight data structures and
avoid hard dependencies on heavy-weight data processing libraries. Once the
production data contracts are finalised this module will grow validation and
normalisation logic around those schemas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Dict, Hashable, Mapping, Sequence, Tuple

import pandas as pd

# NOTE: These helpers intentionally avoid third-party dependencies to keep the
# early scaffolding lightweight. They focus on the behaviour that was proven
# useful during notebook exploration and are small enough to be easily replaced
# once richer infrastructure is available.


@dataclass(frozen=True)
class OptionLeg:
    """Structured representation of a single option leg."""

    expiry: date
    strike: float
    option_type: str
    position: str
    price: float


@dataclass(frozen=True)
class PortfolioRecord:
    """Normalised representation of a single portfolio row."""

    open_timestamp: datetime
    close_timestamp: datetime
    premium: float | None
    pnl: float | None
    strategy: str
    tenor: str | None
    open_session: str
    close_session: str
    legs: Tuple[OptionLeg, ...]
    opening_price: float | None
    closing_price: float | None
    avg_closing_cost: float | None
    contracts: float | None
    funds_at_close: float | None
    margin_requirement: float | None
    opening_short_long_ratio: float | None
    closing_short_long_ratio: float | None
    opening_vix: float | None
    closing_vix: float | None
    gap: float | None
    movement: float | None
    max_profit: float | None
    max_loss: float | None
    close_reason: str
    context: Mapping[str, str]
    raw: Mapping[str, str]

    @property
    def notional(self) -> float:
        """Return the absolute notional value of the trade."""

        premium = self.premium if self.premium is not None else 0.0
        contracts = self.contracts if self.contracts is not None else 1.0
        return abs(premium * contracts)


TIMESTAMP_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%m-%d-%Y %H:%M",
)


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "Date Opened",
    "Time Opened",
    "Date Closed",
    "Time Closed",
    "Premium",
    "P/L",
    "Legs",
    "Strategy",
)


NUMERIC_COLUMN_MAP: Mapping[str, str] = {
    "Opening Price": "opening_price",
    "Closing Price": "closing_price",
    "Avg. Closing Cost": "avg_closing_cost",
    "Premium": "premium",
    "P/L": "pnl",
    "No. of Contracts": "contracts",
    "Funds at Close": "funds_at_close",
    "Margin Req.": "margin_requirement",
    "Opening Short/Long Ratio": "opening_short_long_ratio",
    "Closing Short/Long Ratio": "closing_short_long_ratio",
    "Opening VIX": "opening_vix",
    "Closing VIX": "closing_vix",
    "Gap": "gap",
    "Movement": "movement",
    "Max Profit": "max_profit",
    "Max Loss": "max_loss",
}


LEG_PATTERN = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3})\s+(?P<year>\d{1,2})\s+"
    r"(?P<strike>-?\d+(?:\.\d+)?)\s+(?P<option>[CP])\s+"
    r"(?P<position>[A-Z]{3})\s+(?P<price>-?\d+(?:\.\d+)?)$"
)


TENOR_PATTERN = re.compile(r"(?P<tenor>\d+dte)", re.IGNORECASE)


SESSION_WINDOWS: Tuple[Tuple[str, time, time], ...] = (
    ("pre", time(4, 0), time(9, 30)),
    ("regular", time(9, 30), time(16, 0)),
    ("post", time(16, 0), time(20, 0)),
)


def parse_timestamp(value: str) -> datetime:
    """Parse a timestamp string using the formats observed in the notebooks."""

    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised timestamp format: {value!r}")


def _parse_leg(value: str) -> OptionLeg:
    match = LEG_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unrecognised leg format: {value!r}")

    day = match.group("day")
    month = match.group("month")
    year = match.group("year")

    if len(year) == 1:
        year = year.zfill(2)

    try:
        expiry = datetime.strptime(
            f"{day} {month} {year}",
            "%d %b %y",
        ).date()
    except ValueError as exc:
        raise ValueError(f"Unable to parse leg expiry: {value!r}") from exc

    try:
        strike = float(match.group("strike"))
        price = float(match.group("price"))
    except ValueError as exc:
        raise ValueError(f"Unable to parse leg numeric values: {value!r}") from exc

    option_type = match.group("option").upper()
    position = match.group("position").upper()

    return OptionLeg(
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        position=position,
        price=price,
    )


def _parse_legs(value: str) -> Tuple[OptionLeg, ...]:
    if not value:
        return tuple()
    separators = ("|", ";")
    parts = [value]
    for separator in separators:
        if separator in value:
            parts = [part.strip() for part in value.split(separator) if part.strip()]
            break
    else:
        parts = [value.strip()]

    return tuple(_parse_leg(part) for part in parts if part)


def tag_trading_session(timestamp: datetime) -> str:
    """Return a rough trading-session tag for a timestamp."""

    # The implementation follows the same logic the exploratory notebooks used:
    # * 04:00 – 09:30   -> ``"pre"``
    # * 09:30 – 16:00   -> ``"regular"``
    # * 16:00 – 20:00   -> ``"post"``
    # * otherwise       -> ``"overnight"``
    ts_time = timestamp.time()
    for label, start, end in SESSION_WINDOWS:
        if start <= ts_time < end:
            return label
    return "overnight"


def _infer_tenor_from_path(path: Path) -> str | None:
    match = TENOR_PATTERN.search(path.stem)
    if match is None:
        return None
    return match.group("tenor").lower()


def _build_context(source: Path, tenor: str | None) -> Mapping[str, str]:
    context: Dict[str, str] = {"source": source.name}
    if tenor is not None:
        context["tenor"] = tenor
    return context


def read_portfolio_file(
    path: Path | str, *, encoding: str = "utf-8-sig"
) -> Sequence[PortfolioRecord]:
    """Load a portfolio CSV file into normalised records.

    TODO: Integrate schema validation once the production data model is finalised.
    """

    resolved_path = Path(path)

    preview = pd.read_csv(
        resolved_path,
        nrows=0,
        encoding=encoding,
        keep_default_na=False,
    )

    original_by_trimmed: Dict[str, str] = {}
    trimmed_columns = []
    for column in preview.columns:
        trimmed = column.strip() if isinstance(column, str) else column
        trimmed_columns.append(trimmed)
        if trimmed not in original_by_trimmed and isinstance(column, str):
            original_by_trimmed[trimmed] = column
        elif trimmed not in original_by_trimmed:
            original_by_trimmed[trimmed] = str(column)
    preview.columns = trimmed_columns

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in preview.columns
    ]
    if missing_columns:
        missing_list = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"Portfolio file is missing required column {missing_list}")

    date_columns = {"Date Opened", "Time Opened", "Date Closed", "Time Closed"}

    dtype_map = {
        original: str
        for trimmed, original in original_by_trimmed.items()
        if trimmed not in date_columns
    }

    read_kwargs = {
        "encoding": encoding,
        "dtype": dtype_map,
        "keep_default_na": False,
    }

    dataframe = pd.read_csv(
        resolved_path,
        **read_kwargs,
    )

    dataframe.rename(
        columns=lambda value: value.strip() if isinstance(value, str) else value,
        inplace=True,
    )

    dataframe["open_timestamp"] = [
        parse_timestamp(f"{date_part} {time_part}")
        for date_part, time_part in zip(
            dataframe["Date Opened"], dataframe["Time Opened"]
        )
    ]
    dataframe["close_timestamp"] = [
        parse_timestamp(f"{date_part} {time_part}")
        for date_part, time_part in zip(
            dataframe["Date Closed"], dataframe["Time Closed"]
        )
    ]

    string_columns = dataframe.select_dtypes(include="object").columns
    dataframe[string_columns] = dataframe[string_columns].apply(
        lambda col: col.str.strip()
    )

    tenor = _infer_tenor_from_path(resolved_path)
    context = _build_context(resolved_path, tenor)

    numeric_source = dataframe.reindex(columns=NUMERIC_COLUMN_MAP.keys(), fill_value="")
    numeric_frame = (
        numeric_source.replace("", pd.NA)
        .apply(pd.to_numeric, errors="raise")
        .rename(columns=NUMERIC_COLUMN_MAP)
    )
    numeric_records = [
        {
            column: (None if pd.isna(value) else float(value))
            for column, value in row.items()
        }
        for row in numeric_frame.to_dict("records")
    ]

    legs_values = dataframe["Legs"].apply(_parse_legs)
    strategy_values = dataframe["Strategy"]
    if "Reason For Close" in dataframe.columns:
        close_reason_values = dataframe["Reason For Close"]
    else:
        close_reason_values = pd.Series(
            ["" for _ in range(len(dataframe))], index=dataframe.index
        )

    raw_columns = [
        column
        for column in dataframe.columns
        if column not in {"open_timestamp", "close_timestamp"}
    ]
    raw_records = dataframe.loc[:, raw_columns].to_dict("records")

    open_sessions = dataframe["open_timestamp"].apply(tag_trading_session)
    close_sessions = dataframe["close_timestamp"].apply(tag_trading_session)

    records = []
    for position, (index, numeric_values) in enumerate(
        zip(dataframe.index, numeric_records)
    ):
        open_timestamp_value = dataframe.at[index, "open_timestamp"]
        close_timestamp_value = dataframe.at[index, "close_timestamp"]
        if hasattr(open_timestamp_value, "to_pydatetime"):
            open_timestamp = open_timestamp_value.to_pydatetime()
        else:
            open_timestamp = open_timestamp_value
        if hasattr(close_timestamp_value, "to_pydatetime"):
            close_timestamp = close_timestamp_value.to_pydatetime()
        else:
            close_timestamp = close_timestamp_value

        record = PortfolioRecord(
            open_timestamp=open_timestamp,
            close_timestamp=close_timestamp,
            premium=numeric_values["premium"],
            pnl=numeric_values["pnl"],
            strategy=strategy_values.iat[position] if len(strategy_values) else "",
            tenor=tenor,
            open_session=open_sessions.iat[position],
            close_session=close_sessions.iat[position],
            legs=legs_values.iat[position] if len(legs_values) else tuple(),
            opening_price=numeric_values["opening_price"],
            closing_price=numeric_values["closing_price"],
            avg_closing_cost=numeric_values["avg_closing_cost"],
            contracts=numeric_values["contracts"],
            funds_at_close=numeric_values["funds_at_close"],
            margin_requirement=numeric_values["margin_requirement"],
            opening_short_long_ratio=numeric_values["opening_short_long_ratio"],
            closing_short_long_ratio=numeric_values["closing_short_long_ratio"],
            opening_vix=numeric_values["opening_vix"],
            closing_vix=numeric_values["closing_vix"],
            gap=numeric_values["gap"],
            movement=numeric_values["movement"],
            max_profit=numeric_values["max_profit"],
            max_loss=numeric_values["max_loss"],
            close_reason=(
                close_reason_values.iat[position] if len(close_reason_values) else ""
            ),
            context=context,
            raw={key: value for key, value in raw_records[position].items() if key},
        )
        records.append(record)

    return records


def compute_representative_trades(
    records: Sequence[PortfolioRecord],
    *,
    key: Callable[[PortfolioRecord], Hashable] | None = None,
    score: Callable[[PortfolioRecord], float] | None = None,
) -> Dict[Hashable, PortfolioRecord]:
    """Return the most representative trade per grouping key."""

    # The representative trade matches the notebooks' heuristic of selecting the
    # trade with the largest absolute notional value within the group.
    if key is None:

        def default_key(record: PortfolioRecord) -> Tuple[str, str]:
            strategy = record.strategy or "unknown"
            tenor = record.tenor or "unspecified"
            return (strategy, tenor)

        key = default_key
    if score is None:

        def default_score(record: PortfolioRecord) -> float:
            return record.notional

        score = default_score

    selections: Dict[Hashable, PortfolioRecord] = {}
    best_scores: Dict[Hashable, float] = {}
    for record in records:
        group = key(record)
        record_score = score(record)
        if group not in selections or record_score > best_scores[group]:
            selections[group] = record
            best_scores[group] = record_score
    return selections


__all__ = [
    "OptionLeg",
    "PortfolioRecord",
    "compute_representative_trades",
    "parse_timestamp",
    "read_portfolio_file",
    "tag_trading_session",
]
