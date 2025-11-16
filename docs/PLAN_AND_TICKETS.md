0. Ground Rules

Goal: Implement the logic from xdte_gbm_factory_prod.ipynb inside the xdte package.

Definition of Done = Colab Parity: same backtest dataset, package metrics ≈ notebook metrics (within tolerances).

Golden artifacts:

Do not edit tests/golden_* or tests/golden_manifest.json until EPIC 5.

Existing golden files are not trusted for correctness; they are only scaffolding.

Small, focused changes:

Work in small units (one ticket / epic step per PR), not giant rewrites.

Do not redesign the CLI or directory structure.

Notebook is the reference:

When behavior conflicts between notebook and package, notebook wins.

Use the migration map (docs/migration_map.md) to see which notebook cell maps to which module.

EPIC 2 – Data & Features (DO THIS FIRST)
T2.1 – Daily context acquisition (CLI + loader validation)

Files:

src/xdte/live/providers.py (re-use YFinanceDailyContextProvider)

src/xdte/data/features.py (new helper)

src/xdte/model/train.py

xdte/cli.py (new CLI command)

Requirements:

Add CLI command:

xdte fetch-daily-context --data-dir <PATH>


Behavior:

Load trades from <PATH> using existing portfolio loader.

Compute min_date and max_date of trade open dates.

Use YFinanceDailyContextProvider to:

Download raw daily prices for SPX, VIX, VVIX, VXV over [min_date - 90d, max_date].

Compute the same daily features as the notebook did (ATR, TS, RVs, VVIX EMAs, L1_*, DoW).

Write daily_context.csv into <PATH>, one row per trading day.

In load_backtest_dataset(data_dir: Path):

Load daily_context.csv if present.

Compute trade date range (min_trade_date, max_trade_date).

If daily_context does not fully cover that range, raise:

ValueError("daily_context.csv does not cover trade range [...]. Run xdte fetch-daily-context --data-dir ....")

Do not silently run yfinance inside train by default.

T2.2 – Per-book feature masking (BOOK_FEATS_CONFIG)

Files:

src/xdte/config.py or src/xdte/model/train.py (BOOK_FEATS_CONFIG)

src/xdte/model/train.py (train_models, optional _filter_feature_columns_for_book)

Requirements:

Add BOOK_FEATS_CONFIG: dict[str, list[str]] with exact feature lists from the notebook for:

PUTS_0DTE_11

PUTS_1DTE_1515

CALLS_0DTE_11

CALLS_1DTE_1515

In train_models loop:

for book in selected_books:
    book_rows = [r for r in feature_rows if r["book"] == book]
    if not book_rows:
        continue

    if book in BOOK_FEATS_CONFIG:
        feature_columns = BOOK_FEATS_CONFIG[book]
    else:
        feature_columns = _prepare_features(book_rows)  # fallback only

    feature_columns = _filter_feature_columns_for_book(book, feature_columns)
    ...


Implement _filter_feature_columns_for_book(book, cols):

Infer session from book name:

Books ending in _0DTE_11 → session "11"

Books ending in _1DTE_1515 → session "1515"

For columns starting with:

VIX_Entry_

Intraday_Move_OpenToEntry_

t0_VIX_change_from_close_
drop any that do not end in the correct session suffix.

Leave all non-session-specific columns untouched.

This prevents 15:15 features from appearing in 11:00 models and vice versa.

T2.3 – Notebook loader parity (PHASE 2 / OPTIONAL)

Do not do this until EPIC 2 & 3 are stable.
Only implement if remaining differences are traced to loader/aggregation.

EPIC 3 – Model lifecycle (DISCOVERY & APPLY)
T3.1 – Export per-fold predictions

Files:

src/xdte/model/train.py

Requirements:

After each fold model is trained:

Build train_predictions.csv and test_predictions.csv for that fold and book:

Columns: open_date, book, pnl, score, L1_vvix_above_ema30 (joined from daily context).

Save under:

discovery/{book}/fold_{k}/train_predictions.csv

discovery/{book}/fold_{k}/test_predictions.csv

These are the inputs to discovery/apply.

T3.2 – Rewrite discover_rules (CELL B)

Files:

src/xdte/model/discovery.py

Requirements:

Port _winsor helper from notebook (use np.nanpercentile).

PUTS (books containing "PUTS"):

Read all test_predictions.csv for that book.

Winsorize OOS PnL.

For each keep_target in the PUT grid from notebook:

Compute CVaR on kept OOS PnL.

Compare vs baseline CVaR; enforce improvement constraint.

Choose best keep_target.

CALLS (books containing "CALLS"):

For each fold:

Use train_predictions.csv to compute decile edges (_edges_from_scores).

Use test_predictions.csv to compute per-decile OOS PnL (long & short).

Aggregate over folds and select long_deciles and short_deciles as in notebook.

Save rules to discovery/discovered_rules.json:

{
  "PUTS_0DTE_11": { "keep_target": 0.7 },
  "CALLS_0DTE_11": { "long_deciles": [0,1], "short_deciles": [9] },
  ...
}

T3.3 – Rewrite apply_models (CELL C PATCH)

Files:

src/xdte/model/apply.py

Requirements:

Load discovery/discovered_rules.json.

For each book, fold:

Load train_predictions.csv, test_predictions.csv.

PUTS:

On train:

Using keep_target, evaluate CVaR on both “top” and “bottom” side.

Choose better side (long/short).

On test:

Apply chosen side + keep_target to test scores, compute strategy PnL.

CALLS:

On test:

Use decile edges from train + long_deciles / short_deciles from rules.

Map trades to long/short/skip and compute PnL.

Aggregate across folds into apply/hybrid_input.csv:

Columns: open_date, book, pnl_strategy, L1_vvix_above_ema30 (needed for hybrid).

T3.4 – NaN handling

Remove any .fillna or imputation in the training matrix pipeline (_build_training_matrix etc.).

LightGBM must see np.nan directly.

Add a unit test that confirms np.nan in features is passed through to the model.

EPIC 4 – Hybrid & live
T4.1 – Kill switch (CELL D)

Files:

src/xdte/model/hybrid.py

Requirements:

Port kill-switch helpers from notebook (e.g., _apply_tail_on_kill, _apply_gamma_on_kill).

_aggregate_daily_pnl must:

Read L1_vvix_above_ema30 per day from hybrid_input.csv.

If kill flag is False: use pnl_strategy as-is.

If True: apply tail/gamma logic exactly as CELL D.

T4.2 – Live kit export (CELL G)

Files:

src/xdte/model/export.py

Requirements:

For each book:

Train a full model on all data (same features, hyperparams as folds).

Score all historical rows.

For PUTS: compute live thresholds (taus/side) as in CELL G.

For CALLS: compute decile edges and final long/short deciles.

Export:

live_kit/models/{book}_model.pkl – full model.

live_kit/policy.json – per-book thresholds/deciles.

live_kit/hybrid_policy.json – tuned hybrid policy.

live_kit/features.json – BOOK_FEATS_CONFIG used.

T4.3 – Live decision logic

Files:

src/xdte/live/decide.py

Requirements:

run_live_decisions must:

Load features.json, policy.json, hybrid_policy.json.

Build live feature row(s) per book using features.json.

Load single {book}_model.pkl and score.

Apply PUTS/CALLS rules from policy.json.

Apply kill-switch/hybrid logic using live L1_vvix_above_ema30.

Output the same JSON shape as current xdte decide, but based on this logic.

T4.4 – xdte retune (Phase 2)

Implement monthly retune wrapper from final notebook cell.

Only modify hybrid_policy_best.json if new policy strictly improves PF/CVaR and respects EDP floor.

EPIC 5 – Parity & golden
T5.1 – Colab parity test

Files:

tests/test_colab_parity.py

Reference metrics JSON produced by the notebook (place under tests/data/notebook_metrics.json).

Requirements:

Run the full package pipeline on the canonical backtest dataset.

Load tests/data/notebook_metrics.json.

For each book and portfolio (ALL):

Compare no_kill and hybrid: EDP, PF, CVaR, N trades.

Assert differences within agreed tolerances (EDP ±3%, PF ±0.03, CVaR ±200, etc.).

T5.2 – Rebaseline golden

Only after T5.1 passes:

Delete old tests/golden_manifest.json and golden artefacts.

Run pipeline again, snapshot new golden bundle and manifest.

Update tests/test_golden_parity.py to compare against this new baseline.

T5.3 – Component unit tests

Add small, focused tests for:

Daily feature generation.

discover_rules on synthetic data.

PF/CVaR helpers.

EPIC 6 – UX & observability (post-fix)

Logging: --verbose shows per-book rows/features/winsorization/folds.

mkdir -p: all CLIs create output directories if missing.

Run summary: write run_summary.json + run_summary.md with portfolio + per-book metrics, rules, and top features.

Debug: --debug on xdte train dumps _debug_feature_panel.csv (or similar) to the artifact dir.

maintain progress in PROGRESS_REPORT.md
