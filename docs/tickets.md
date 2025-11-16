Ticket 1 – XDTE-BOOKS-001: Support “book-only” backtest bundles in loader

Goal
Allow load_backtest_dataset to generate both trade and market frames when the backtest directory contains only book_*.csv files (no market_stats files), while keeping the legacy bundle behavior unchanged.

Scope

src/xdte/model/train.py – load_backtest_dataset and _categorise_inputs 

source_dump

src/xdte/data/features.py – ensure infer_market_session behavior is correct for book filenames. 

source_dump

Work

Update load_backtest_dataset:

After populating trades and market:

for trade_file in trade_files:
    trades.extend(read_portfolio_file(trade_file))
for market_file in market_files:
    market.extend(read_portfolio_file(market_file))

if not market_files:
    # No explicit market_stats; reuse trades as market source
    market = list(trades)


Do not change the existing _categorise_inputs logic; we still want *_market_stats_*.csv to be treated as market-only when present.

Confirm that infer_market_session(record) correctly resolves sessions from context["source"]:

For filenames like book_0dte_call_spread_20delta_20_pts_1100.csv:

Contains "0dte" + "call" → book CALLS_0DTE_11. 

source_dump

Contains "1100" → _infer_session_series / infer_market_session map to "11:00". 

source_dump

For 15:15 books, you’ll use …_1515.csv in the name, which already matches if "1515" in source … return "15:15". 

source_dump

No changes to PortfolioRecord or read_portfolio_file – the new “book” CSVs already match the required schema. 

source_dump

Acceptance

For an “old-style” bundle (with portfolio_*.csv + *_market_stats_*.csv + daily context):

trade_frame, market_frame, daily_frame are identical to current behavior (regression tests pass).

For a “book-only” directory containing:

book_0dte_put_spread_15delta_20_pts_1100.csv

book_0dte_call_spread_20delta_20_pts_1100.csv

plus a valid daily_context.csv / daily_context.json:

trade_frame has one row per (book, open_date) with pnl/gap/movement/opening_vix/closing_vix. 

source_dump

market_frame has non-NaN VIX_Entry_11 and Intraday_Move_OpenToEntry_11 for those dates (computed from the same PortfolioRecords via market_records_to_frame). 

source_dump

Ticket 2 – XDTE-BOOKS-002: Parity tests for book-only vs legacy layout

Goal
Prove that using book_* files with embedded market stats is behaviorally equivalent (for a single configuration) to using separate portfolio_*.csv + market_stats_*.csv files.

Scope

tests/unit/test_load_backtest_dataset.py (new)

Optionally small fixture CSVs under tests/fixtures/backtest_book_mode/.

Work

Construct a tiny synthetic dataset:

Legacy layout:

portfolio_0dte_put_spreads_15delta_20pts_1100.csv

market_stats_0dte_put_spreads_15delta_20pts_1100.csv

Book layout:

book_0dte_put_spread_15delta_20_pts_1100.csv
(same rows as portfolio+market merged; matches your attached examples).

Write tests that:

Load both layouts via load_backtest_dataset(...).

Compare:

trade_frame equality (or at least pnl, gap, movement, opening_vix, closing_vix for each (book, date)).

market_frame equality (VIX_Entry_11, Intraday_Move_OpenToEntry_11 for each date).

Do the same for a PUT+CALL pair (0DTE) to ensure multiple book_* files co-exist correctly.

Acceptance

Tests confirm that for the same underlying rows, legacy vs book-only layouts produce the same trade and market feature frames.

Changes don’t break existing golden backtest tests.

Ticket 3 – XDTE-DOC-BOOKS-001: Document book-only experimental bundles

Goal
Make the “single-book / few-book experiment” path first-class in the docs so you (and Codex) know exactly how to build these inputs.

Scope

USER_GUIDE.md – sections 2 & 3 (“Data inputs”, “Training artefacts”). 

USER_GUIDE

Optional: short note in DESIGN.md or DATA_SCHEMA.md.

Work

In Data inputs (USER_GUIDE.md):

Add a subsection “Book-only experimental bundles” describing:

New file pattern: book_0dte_put_spread_15delta_20_pts_1100.csv, etc.

Requirements:

Must include the standard portfolio columns (Date Opened, Time Opened, Date Closed, Time Closed, Premium, P/L, Legs, Strategy can be blank). 

source_dump

Should include Opening VIX, Closing VIX, Gap, Movement (as in the attached examples).

The filename must encode:

DTE (0dte, 1dte)

Side (put/call)

Session (1100, 1515, 3pm, etc.) so infer_canonical_book and infer_market_session can infer book and session correctly. 

source_dump

Explain that when no *_market_stats_*.csv files are present, the loader will automatically reuse the book_* rows to construct the market feature frame.

In Training artefacts:

Add a short note under the xdte train section:

For quick experiments on a single configuration (e.g., 20-point 15Δ puts and 20Δ calls), you may populate --backtest-dir with only book_*.csv files and a daily context file. The training pipeline will treat these as both trade and session-level market inputs and will only train books inferred from the filenames (e.g. PUTS_0DTE_11, CALLS_0DTE_11).

ticket 4:
as a part of CI or when book mode is run by codex, outputs from the dev folder should be put into their own dev outputs folder


Acceptance

A new user can read USER_GUIDE.md and understand exactly how to build a minimal “book-only” bundle for 20-point spreads without separate market_stats files.

The doc clearly states that daily_context is still required (unless you later decide to derive it from book data).

Optional (later) tickets

If, after this, you want to go one click further:

XDTE-BOOKS-EXTRA-001: Add a small CLI helper xdte prepare-book-bundle that merges an existing portfolio_*/market_stats_* bundle into book_* CSVs for a given book, using your preferred naming convention. (Nice-to-have; not required for your immediate goal.)