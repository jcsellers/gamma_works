# XDTE End-to-End Architecture

This document summarises how configuration, data ingestion, modelling, hybrid
selection, and live execution interact inside the XDTE package. Use it together
with the module-focused notes in [`DESIGN.md`](DESIGN.md) and delivery backlog in
[`PLAN_AND_TICKETS.md`](PLAN_AND_TICKETS.md) when onboarding new contributors.

## Pipeline quick reference

| Stage | Entry points | Key modules | Outputs |
| --- | --- | --- | --- |
| Configure | Environment variables (`XDTE_*`), CLI flags, config stubs | [`xdte.config.Settings`](../src/xdte/config.py), [`xdte.cli`](../src/xdte/cli.py) | Normalised `Settings` instance shared across commands |
| Load data | Backtest bundle CSVs and live providers | [`xdte.data.loaders`](../src/xdte/data/loaders.py), [`xdte.data.features`](../src/xdte/data/features.py), [`xdte.data.validation`](../src/xdte/data/validation.py) | Typed records, feature panels, schema checks |
| Train | `xdte train` | [`xdte.model.train`](../src/xdte/model/train.py), [`xdte.model.wfo`](../src/xdte/model/wfo.py) | Fold models, metrics, persisted training artefacts |
| Discover | `xdte train` (post-training) | [`xdte.model.discovery`](../src/xdte/model/discovery.py) | Winsor limits, keep thresholds, discovery metadata |
| Apply | `xdte train` (scoring) | [`xdte.model.apply`](../src/xdte/model/apply.py), [`xdte.metrics`](../src/xdte/metrics.py) | Apply scores, evaluation tables, fold summaries |
| Hybrid selection | `xdte tune` | [`xdte.model.hybrid`](../src/xdte/model/hybrid.py), [`xdte.gamma`](../src/xdte/gamma.py) | Candidate audit logs, accepted policy JSON, gamma backtests |
| Export | `xdte export-kit` | [`xdte.model.export`](../src/xdte/model/export.py), [`xdte.persistence`](../src/xdte/persistence.py) | Live kit directory with manifest and digests |
| Live execution | `xdte decide` | [`xdte.live.decide`](../src/xdte/live/decide.py), [`xdte.live.providers`](../src/xdte/live/providers.py) | Operational decisions, guardrail alerts |

## Configuration surface

The pipeline is driven by the immutable [`Settings`](../src/xdte/config.py) data
class. Defaults such as fold counts, alpha, winsorisation probability, kill
switch tails/gammas, and offline stub payloads live in the module so they remain
deterministic across environments. Overrides flow in through three channels:

1. **Environment variables** – any `XDTE_<FIELD>` variable is parsed via
   `_FIELD_CASTERS` so typed values (floats, integers, JSON mappings) can be
   supplied in CI or container images.
2. **CLI arguments** – `xdte.config.Settings.add_cli_arguments` registers
   command-line flags for all overrideable fields. The top-level CLI
   ([`xdte.cli`](../src/xdte/cli.py)) converts parsed arguments into a
   `Settings` instance that is threaded through every command handler.
3. **Offline stubs** – daily and market context defaults are stored as JSON
   payloads inside the package (`offline_daily_stub.json`,
   `offline_market_stub.json`). They guarantee reproducible tests when live
   providers are unavailable.

Because `Settings` is frozen, modules receive consistent configuration snapshots
that can be hashed and logged alongside artefacts.

## Data ingestion and feature assembly

The ingestion layer transforms raw portfolio snapshots into modelling-ready
records:

- [`loaders`](../src/xdte/data/loaders.py) parses CSV exports into
  `PortfolioRecord` objects, normalising timestamps, leg groupings, and
  notional calculations without pulling in heavyweight dataframe libraries.
- [`validation`](../src/xdte/data/validation.py) runs schema assertions and
  missing-column checks, surfacing actionable error messages whenever upstream
  data contracts drift.
- [`xdte.data.features`](../src/xdte/data/features.py) defines deterministic feature
  generators that operate directly on pandas `DataFrame` inputs to keep the
  modelling pipeline consistent across training, apply, and live decision paths.

  *Notebook convenience shim:* importing this module registers a `features`
  alias in Python's `builtins` so exploratory notebooks can reference the
  helpers without boilerplate. Production services and tests should import the
  module explicitly to avoid depending on the implicit alias.

### Feature assembly stages

| Stage | Inputs | Output shape | Downstream usage |
| --- | --- | --- | --- |
| `trade_records_to_frame` | Sequence of `PortfolioRecord` items parsed from trade CSVs. Each record carries strategy identifiers, context metadata, and numeric trade fields. | pandas `DataFrame` with columns `strategy`, `source`, `book`, `open_timestamp`, `open_date`, `notional`, `pnl`, `gap`, `movement`, `opening_vix`, `closing_vix`. Index is a default `RangeIndex`. | Called by [`xdte.model.train.load_backtest_dataset`](../src/xdte/model/train.py) before building book-level panels. The resulting frame feeds `build_trade_panel` during training and its columns flow into the fold CSV exports consumed by apply runs. |
| `build_trade_panel` | Trade `DataFrame` (either from `trade_records_to_frame` or pre-loaded CSV). Requires `book` and `open_date` columns; accepts optional `book_mapper` hook. | Multi-indexed `DataFrame` keyed by `(book, open_date)` with columns `pnl`, `gap`, `movement`, `opening_vix`, `closing_vix`. Rows are filtered to the maximum-notional trade per book/date. | Serves as the base of the model design matrix when `generate_feature_panel` is called in training. Fold-level `train.csv` and `test.csv` exports inherit this schema for apply scoring and live parity. |
| `market_records_to_frame` | Sequence of `PortfolioRecord` market snapshots generated from market CSV exports. | pandas `DataFrame` with columns `strategy`, `source`, `open_timestamp`, `open_date`, `session`, `opening_vix`, `closing_vix`, `movement`. Index is a default `RangeIndex`. | Invoked during training dataset loading to normalise market session data before `build_market_feature_frame`, keeping the aggregated features consistent with apply and live providers. |
| `build_market_feature_frame` | Market `DataFrame` with session-labelled rows (output of `market_records_to_frame` or live providers). | pandas `DataFrame` indexed by `open_date` containing columns `Intraday_Move_OpenToEntry_11`, `Intraday_Move_OpenToEntry_1515`, `VIX_Entry_11`, `VIX_Entry_1515`. | Produces per-session aggregates merged into the trade panel inside `generate_feature_panel`. The resulting columns propagate into fold CSV exports and are mirrored by [`xdte.live.decide`](../src/xdte/live/decide.py). |
| `compute_daily_context_features` | pandas `DataFrame` with an `open_date` column and lagged daily indicators (from JSON manifest or live providers). | pandas `DataFrame` indexed by `open_date` with engineered lagged columns such as `L1_VIX_Close`, `L1_vvix_pct`, `L1_SPX_Drawdown_Pct`, and boolean VVIX trend markers. | Used during training to attach daily context to each book/date combination. Those features persist into fold CSV exports for apply scoring, and live execution reconstructs them through provider payloads. |
| `generate_feature_panel` | Trade panel, market feature frame, and daily context frame. Accepts optional `book_mapper` hook. | pandas `DataFrame` indexed by `(book, open_date)` that joins the trade, market, and daily tables and derives additional deltas (`t0_VIX_change_from_close_11`, `t0_VIX_change_from_close_15`). | Central design matrix for model training (`xdte.model.train`). Exposed via `TrainResult.feature_frame` and encoded in fold metadata so apply scoring and golden parity tests reuse the same schema. Live kits include this structure so `xdte.live.decide` can map provider data onto the trained feature layout. |

Training runs load raw CSV inputs and normalise them via the first three stages
before assembling a per-book trade panel. `generate_feature_panel` then merges
those tables with daily context features, exposing the indexed panel via
`TrainResult.feature_frame`. During apply scoring the CLI reads the fold-level
`test.csv` exports produced in training, guaranteeing predictions are computed
against the identical feature columns. Exported live kits bundle the panel
schema and discovery metadata so `xdte decide` can map provider payloads into
the same columns before invoking the trained models.

During live execution, [`xdte.live.providers`](../src/xdte/live/providers.py)
implements the provider protocols to source market and daily context data using
`yfinance`, keeping the interface aligned with offline stubs used during tests.

## Training, discovery, and apply lifecycle

`xdte train` orchestrates three coupled phases inside
[`xdte.model`](../src/xdte/model/):

1. **Training** – [`train`](../src/xdte/model/train.py) joins portfolio records
   with generated feature panels, then fits fold-specific LightGBM regressors
   via the custom wrapper in [`_lgbm`](../src/xdte/model/_lgbm.py). Walk-forward
   splits are sourced from [`wfo`](../src/xdte/model/wfo.py) to prevent leakage.
   Persisted outputs include model binaries, fold metrics, and audit tables.
2. **Discovery** – [`discovery`](../src/xdte/model/discovery.py) calculates
   winsorisation thresholds, `keep_threshold` deciles, and supporting metadata
   (e.g. quantile summaries) which are written to CSV/JSON for reuse.
3. **Apply** – [`apply`](../src/xdte/model/apply.py) reloads trained models,
   applies discovery thresholds, and emits fold-level predictions plus metrics
   from [`xdte.metrics`](../src/xdte/metrics.py). The resulting tables fuel hybrid
   policy evaluation and regression tests.

Each phase deposits artefacts into a structured directory (`artifacts/`,
`discovery/`, `apply/`) so later steps can reuse persisted results without
recomputing upstream stages.

## Hybrid policy selection

Hybrid tuning is triggered via `xdte tune`. The command loads apply outputs and
candidate payloads to evaluate policy mixes:

- [`xdte.model.hybrid`](../src/xdte/model/hybrid.py) computes per-book quantiles
  and acceptance rationales while recording audit logs for rejected candidates.
- [`xdte.gamma`](../src/xdte/gamma.py) contains helper utilities for translating
  percentile rules into gamma scaling factors when dynamic sizing is enabled.

Accepted policies are written to `hybrid/policy.json`, accompanied by
diagnostics like `apply/gamma_backtest.csv` so reviewers can inspect the impact
before exporting.

## Explainability artefacts

`xdte train` emits explainability tables per book in addition to models and
metrics. [`xdte.model.train`](../src/xdte/model/train.py) aggregates fold-level
SHAP outputs into `shap_summary_{book}.csv` files capturing mean absolute SHAP
values and expected values, and records gain-based backups in
`feature_importance_{book}.csv`. When `xdte export-kit` runs, the exporter copies
those CSVs into the live kit under `models/` and lists them in the manifest so
downstream tooling can verify their presence. During live operation
[`xdte.live.decide`](../src/xdte/live/decide.py) loads the summaries from the kit
and attaches SHAP contributions, top contributors, and expected values to the
JSON payload returned by `xdte decide`.

## Export and live execution

`xdte export-kit` assembles a deterministic Live Kit by copying all relevant
artefacts and generating manifests:

- [`xdte.model.export`](../src/xdte/model/export.py) collates trained models,
  discovery metadata, apply outputs, and the tuned policy into a versioned
  directory. It produces manifest files and SHA256 digests, relying on
  [`xdte.persistence`](../src/xdte/persistence.py) for filesystem safety.

Operators run `xdte decide` against a Live Kit to obtain real-time decisions:

- [`xdte.live.decide`](../src/xdte/live/decide.py) reconstructs feature
  snapshots, applies kill-switch rules derived from `Settings`, and emits
  `DecisionRecord` objects along with guardrail warnings when data gaps appear.
- [`xdte.live.providers`](../src/xdte/live/providers.py) supplies default data
  providers backed by `yfinance` while supporting dependency injection for
  proprietary feeds in production.

## External dependencies

The runtime leans on a focused dependency set captured in `pyproject.toml`:

- **LightGBM** for gradient boosted regression.
- **NumPy** and **pandas** for efficient numerical and tabular operations.
- **SHAP** for explainability tooling referenced during model review.
- **Click** for the CLI command surface.
- **yfinance** for default market data retrieval during live execution.

Development tooling (`black`, `ruff`, `isort`, `mypy`, `pytest`, `sphinx`, etc.)
is codified under the `dev` extra to keep formatting, linting, typing, and
documentation builds consistent across contributors.

