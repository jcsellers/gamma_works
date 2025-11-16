# Backtest Data Schema Reference

This document captures the agreed contract for the frozen CSV extracts that
feed the XDTE pipeline. It enumerates the raw columns provided by the data
vendor, clarifies the loader-generated fields introduced during the refactor,
and documents the filename conventions that determine strategy routing.

## Portfolio Trade Files (`portfolio_*.csv`)

The loader expects every portfolio file to be UTF-8 (optionally with BOM)
containing a header row. Columns are trimmed for whitespace before use. Unless
otherwise noted, empty strings are treated as missing values and converted to
`null` (`None`) by the loader.

| Column | Type | Units / Domain | Nullability | Notes |
| --- | --- | --- | --- | --- |
| `Date Opened` | Date string (`YYYY-MM-DD`) | Calendar date | Required | Combined with `Time Opened` to form the trade open timestamp. |
| `Time Opened` | Time string (`HH:MM[:SS]`) | 24h clock | Required | Combined with `Date Opened`; also used to tag the trading session. |
| `Opening Price` | Float | Underlying index level (points) | Optional | Loader maps to `opening_price`. Typical values track the SPX spot level.【F:data/backtest_data/original/portfolio_0dte_call_spreads_10pts.csv†L2-L11】 |
| `Legs` | String | Option legs in "<day> <Mon> <YY> <strike> <C/P> <STO/BTO> <price>" format separated by `|` or `;` | Required | Parsed into structured `OptionLeg` entries. Empty strings raise validation errors. |
| `Premium` | Float | USD per spread (credit positive, debit negative) | Optional | Loader maps to `premium`. Empty cells become `null`. |
| `Closing Price` | Float | Underlying index level (points) | Optional | Loader maps to `closing_price`. |
| `Date Closed` | Date string | Calendar date | Required | Combined with `Time Closed` to form the close timestamp. |
| `Time Closed` | Time string (`HH:MM[:SS]`) | 24h clock | Required | Combined with `Date Closed`; participates in session tagging. |
| `Avg. Closing Cost` | Float | USD per spread | Optional | Loader maps to `avg_closing_cost`. |
| `Reason For Close` | String | Free-text reason | Optional | Loader surfaces as `close_reason` (empty string when missing). |
| `P/L` | Float | USD per spread (positive = profit) | Optional | Loader maps to `pnl`. |
| `No. of Contracts` | Float | Count of option spreads/contracts | Optional | Loader maps to `contracts`. Non-integers are preserved. |
| `Funds at Close` | Float | Account equity (USD) | Optional | Loader maps to `funds_at_close`. Negative balances are retained. |
| `Margin Req.` | Float | Margin requirement (USD) | Optional | Loader maps to `margin_requirement`. |
| `Strategy` | String | Strategy descriptor (e.g. `0dte_call_spread_10delta_10_pts_1100`) | Optional | Loader retains verbatim for downstream grouping. Some market-stat exports leave this blank.【F:data/backtest_data/original/11am_market_stats.csv†L2-L10】 |
| `Opening Short/Long Ratio` | Float | Ratio | Optional | Loader maps to `opening_short_long_ratio`. Missing cells yield `null`. |
| `Closing Short/Long Ratio` | Float | Ratio | Optional | Loader maps to `closing_short_long_ratio`. |
| `Opening VIX` | Float | VIX index level at open | Optional | Loader maps to `opening_vix`. Present in session summary files.【F:data/backtest_data/original/11am_market_stats.csv†L2-L10】 |
| `Closing VIX` | Float | VIX index level at close | Optional | Loader maps to `closing_vix`. |
| `Gap` | Float | Underlying overnight move (% points) | Optional | Loader maps to `gap`. Values reflect the research notebook’s calculations.【F:data/backtest_data/original/portfolio_0dte_call_spreads_10pts.csv†L2-L11】 |
| `Movement` | Float | Intraday move open→close (% points) | Optional | Loader maps to `movement`. |
| `Max Profit` | Float | USD per spread | Optional | Loader maps to `max_profit`. |
| `Max Loss` | Float | USD per spread (negative) | Optional | Loader maps to `max_loss`. |

### Structured legs

Each `Legs` entry is parsed into an immutable `OptionLeg` dataclass with the
following fields: `expiry` (`date`), `strike` (`float`, USD), `option_type`
(`"C"` or `"P"`), `position` (`"STO"`, `"BTO"`, etc.), and `price` (`float`,
USD). Malformed rows raise a loader error; blank leg cells produce an empty
collection.【F:src/xdte/data/loaders.py†L24-L116】【F:src/xdte/data/loaders.py†L184-L248】

### Normalised trade record

`read_portfolio_file` emits a `PortfolioRecord` with the raw CSV fields and
additional derived attributes used by the pipeline:

| Field | Type | Source | Notes |
| --- | --- | --- | --- |
| `open_timestamp` | `datetime` | `Date Opened` + `Time Opened` | Parsed via multiple legacy formats for notebook parity.【F:src/xdte/data/loaders.py†L92-L148】【F:src/xdte/data/loaders.py†L182-L236】 |
| `close_timestamp` | `datetime` | `Date Closed` + `Time Closed` | Same parsing rules as open. |
| `tenor` | `str | None` | File stem regex (`\d+dte`) | Captures tenor such as `0dte`, `1dte`, `10dte`; missing when filename lacks a match.【F:src/xdte/data/loaders.py†L150-L205】 |
| `open_session` | `str` | Derived from `open_timestamp` | Buckets into `pre`, `regular`, `post`, or `overnight` windows used for book routing.【F:src/xdte/data/loaders.py†L108-L176】 |
| `close_session` | `str` | Derived from `close_timestamp` | Same session taxonomy as open. |
| `legs` | `tuple[OptionLeg, ...]` | Parsed `Legs` field | Preserves leg order for risk analytics. |
| Monetary/ratio fields | `float | None` | Numeric columns | Loader coerces blanks to `None`; invalid strings raise errors. |
| `close_reason` | `str` | `Reason For Close` | Empty string if column absent. |
| `context` | `Mapping[str, str]` | Loader metadata | Always includes `source` (filename); adds `tenor` when inferred.【F:src/xdte/data/loaders.py†L150-L236】 |
| `raw` | `dict[str, str]` | Original row | Whitespace-stripped snapshot for auditing. |
| `notional` | `float` property | Derived | Absolute `premium × contracts` convenience accessor used for representative trade selection.【F:src/xdte/data/loaders.py†L55-L73】【F:src/xdte/data/loaders.py†L236-L308】 |

## Market Statistic Files (`*market_stats*.csv`)

The market-stat archives reuse the same column layout as the portfolio files
with the following nuances:

- The `Strategy` column is intentionally left blank; only session-level context
  is required for feature engineering.【F:data/backtest_data/original/11am_market_stats.csv†L2-L10】
- `Opening VIX`, `Closing VIX`, `Gap`, and `Movement` columns are always
  populated because these files originate from the daily market summary export.
- Quantity-oriented fields (`No. of Contracts`, `Funds at Close`, `Margin Req.`)
  retain the vendor values even though they are not used downstream; missing
  cells are treated identically to the portfolio files.

The loader applies identical parsing, nullability, and validation rules when a
market-stat file is ingested via `read_portfolio_file`, ensuring parity between
trade-level and summary sources.【F:src/xdte/data/loaders.py†L182-L308】

## Filename Conventions and Strategy Routing

### Portfolio filenames

Portfolio CSVs follow a descriptive stem that encodes the tenor, option side,
and deployment window. The loader stores the stem in the `context['source']`
field and infers tenor from the first `<digits>dte` token, which becomes
`PortfolioRecord.tenor`.

| Filename pattern | Example | Interpretation | Pipeline strategy |
| --- | --- | --- | --- |
| `portfolio_0dte_call_spreads_*.csv` | `portfolio_0dte_call_spreads_10pts.csv` | Same-day call credit spreads entered during the 11:00 ET sweep | `CALLS_0DTE_11` (call side, 11:00 session).【F:data/backtest_data/original/portfolio_0dte_call_spreads_10pts.csv†L2-L11】【F:src/xdte/data/loaders.py†L108-L205】 |
| `portfolio_0dte_put_spreads_*.csv` | `portfolio_0dte_put_spreads_10pts.csv` | Same-day put credit spreads launched at 11:00 ET | `PUTS_0DTE_11`. |
| `portfolio_1dte_call_spreads_*.csv` | `portfolio_1dte_call_spreads_10pts.csv` | Trades opened 15:15 ET for next-day expiry call spreads | `CALLS_1DTE_1515` (call side, 15:15 session).【F:data/backtest_data/original/portfolio_1dte_call_spreads_10pts.csv†L2-L11】 |
| `portfolio_1dte_put_spreads_*.csv` | `portfolio_1dte_put_spreads_10pts.csv` | Trades opened 15:15 ET for next-day expiry put spreads | `PUTS_1DTE_1515`.【F:data/backtest_data/original/portfolio_1dte_put_spreads_10pts.csv†L2-L10】 |
| `portfolio_calls_10dte_*` | `portfolio_calls_10dte_1535_mon.csv` | Long-dated call entries (10 DTE) with weekday/time suffixes | Routed manually; loader exposes tenor `10dte` for downstream feature toggles.【F:data/backtest_data/original/portfolio_calls_10dte_1535_mon.csv†L2-L10】 |
| `portfolio_put_10dte_*` | `portfolio_put_10dte_1515_wed.csv` | Long-dated put entries (10 DTE) with session cues | Routed manually (tenor `10dte`). |
| `portfolio_strangles.csv` | `portfolio_strangles.csv` | Multi-leg neutral book used for exploratory analysis | Not mapped to a production strategy; remains in the "UNKNOWN" bucket unless manually assigned.【F:data/backtest_data/original/portfolio_strangles.csv†L2-L10】【F:src/xdte/data/loaders.py†L200-L236】 |

When portfolio rows are ingested, the combination of the inferred session
(`open_session`) and the leg side determines the canonical strategy identifiers
used throughout the pipeline (`PUTS_0DTE_11`, `CALLS_0DTE_11`, `PUTS_1DTE_1515`,
`CALLS_1DTE_1515`). This mirrors the original notebook logic preserved in the
refactored loader.【F:src/xdte/data/loaders.py†L108-L248】【F:original_code.txt†L153-L212】

### Market-stat filenames

Market-stat files adhere to `(<clock>_|)<session>_market_stats.csv` naming. The
clock component signals the decision window captured inside the file:

| Filename | Session inferred | Relevant strategies |
| --- | --- | --- |
| `11am_market_stats.csv` | 11:00 ET | `PUTS_0DTE_11`, `CALLS_0DTE_11` (used for t=0 features at 11:00).【F:data/backtest_data/original/11am_market_stats.csv†L2-L10】【F:original_code.txt†L204-L268】 |
| `1515_market_stats.csv` | 15:15 ET | `PUTS_1DTE_1515`, `CALLS_1DTE_1515` (t=0 features for the afternoon sweep). |

Pipelines pivot these files by session to produce the `VIX_Entry_*` and
`Intraday_Move_OpenToEntry_*` features that seed daily feature engineering,
maintaining parity with the legacy notebooks.【F:original_code.txt†L228-L306】

## Sign-off Process

This draft should be circulated to the strategy, data, and engineering leads for
review. Once stakeholders confirm the contract, the link added to the planning
backlog (see `docs/PLAN_AND_TICKETS.md`) becomes the canonical reference for
future contributors.
