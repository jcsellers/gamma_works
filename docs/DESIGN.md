# XDTE Architecture Overview

This document captures how the XDTE package organises its training and live decision
pipeline. Each section references the modules under `src/xdte/` that own the
behaviour so engineers can navigate the codebase quickly.

## Data loaders and feature preparation

The ingestion layer lives under [`xdte.data`](../src/xdte/data/):

- [`loaders.py`](../src/xdte/data/loaders.py) normalises raw CSV exports into
  typed `PortfolioRecord` rows and handles option leg parsing, timestamp
  coercion, and derived trade attributes such as `notional` and session
  windows. These helpers intentionally avoid third-party dependencies so they
  can be used inside notebooks and production code without drift.
- [`validation.py`](../src/xdte/data/validation.py) provides reusable checks for
  missing columns and schema mismatches. The routines are called by the CLI when
  new backtest bundles arrive and should be extended whenever the portfolio or
  market data contracts change.
- [`features.py`](../src/xdte/data/features.py) tracks engineered feature
  definitions and exposes `generate_feature_panel` for deterministic feature
  assembly. All stages operate on pandas `DataFrame` inputs and persist the
  merged panel used by training, apply scoring, and live parity checks.

> **Legacy note:** The historical `FeatureTable` shim has been removed. Pandas
> `DataFrame` usage is now canonical, and parity is enforced by unit tests plus
> the golden parity suite so contributors can trust identical behaviour across
> code paths.

Together these modules deliver three distinct data sources to the modelling
pipeline: trade-level portfolio rows (`portfolio`), session market snapshots
(`market`), and lagged daily context (`daily`).

When operators stage experimental `book_*.csv` bundles without separate market
snapshots, `load_backtest_dataset` reuses the combined rows to seed both the
trade and market frames while inferring book/session metadata from the
filenames.【F:src/xdte/model/train.py†L180-L204】【F:src/xdte/data/features.py†L188-L229】
See [User Guide §2](USER_GUIDE.md#book-only-experimental-bundles) for the
expected file layout and required columns.

### Notebook convenience shims

Exploratory notebooks historically imported the feature helpers via a
`features` alias injected into Python's `builtins`. The registration lives in
[`xdte.data.features`](../src/xdte/data/features.py) and runs on import for
ergonomics only. Production code and unit tests should continue to import the
module explicitly (`import xdte.data.features as features`) so behaviour does
not depend on the implicit alias.

## Model lifecycle

Model training and inference reside inside [`xdte.model`](../src/xdte/model/):

- [`train.py`](../src/xdte/model/train.py) orchestrates dataset preparation,
  LightGBM training, and fold artefact persistence via the custom
  [`LGBMRegressor`](../src/xdte/model/_lgbm.py) wrapper. It consumes
  `PortfolioRecord` objects, joins feature panels, and writes tidy CSV outputs
  for downstream discovery and apply stages. The `TrainResult` aggregates the
  persisted metadata for later reuse.
- [`wfo.py`](../src/xdte/model/wfo.py) defines `WalkForwardSplitter`, the
  deterministic time-ordered splitting utility that enforces no-leakage folds.
  Training seeds and fold counts are sourced from `xdte.config.Settings` so the
  pipeline remains reproducible.
- [`discovery.py`](../src/xdte/model/discovery.py) reads per-fold training
  predictions to compute winsorisation limits and `keep_threshold` deciles. The
  resulting metadata is persisted in CSV/JSON form and referenced during scoring
  to avoid recomputing thresholds on test data.
- [`apply.py`](../src/xdte/model/apply.py) loads persisted models, applies the
  discovery thresholds, and emits fold-level predictions with evaluation
  metrics produced by [`xdte.metrics.portfolio_metrics`](../src/xdte/metrics.py).
  The output directory contains tidy CSVs that power hybrid policy selection.

The CLI commands exposed in [`cli.py`](../src/xdte/cli.py) wire these modules
into a repeatable training → discovery → apply workflow, with configuration
passed through `xdte.config.Settings`.

## Hybrid selection and export kits

Hybrid policy evaluation and export packaging are implemented in dedicated
modules:

- [`hybrid.py`](../src/xdte/model/hybrid.py) evaluates candidate tail and gamma
  adjustments over apply predictions. `select_hybrid_policy` parses persisted
  prediction rows, computes per-book quantiles, and records why candidates were
  accepted or rejected. The results are serialised to JSON so the tuned policy
  can be versioned.
- [`export.py`](../src/xdte/model/export.py) assembles a live kit by copying the
  trained fold artefacts, discovery metadata, and chosen hybrid policy into a
  versioned directory. The module produces deterministic manifests and SHA256
  digests so downstream automation can verify integrity.

These modules ensure the modelling outputs can be tuned safely and bundled for
hand-off to live execution systems.

## Explainability artefacts

Training captures per-book explainability outputs alongside the usual model and
metric bundles. [`xdte.model.train`](../src/xdte/model/train.py) aggregates SHAP
values across folds to produce two CSV families for every book:

- `shap_summary_{book}.csv` – sorted mean absolute SHAP contributions with the
  averaged expected value used during review.
- `feature_importance_{book}.csv` – the gain-based fallback for columns that do
  not receive SHAP coverage.

The export pipeline ([`xdte.model.export`](../src/xdte/model/export.py)) copies
these explainability files into the live kit under the `models/` directory and
records each path in the manifest (`models/shap_summary_{book}.csv` and
`models/feature_importance_{book}.csv`). During live inference,
[`xdte.live.decide`](../src/xdte/live/decide.py) loads the bundled summaries so
`xdte decide` can surface SHAP contributions, top contributors, and expected
values without recomputing explainability at runtime.

## Live decision components

Live execution consumes the exported kit via [`xdte.live`](../src/xdte/live/):

- [`decide.py`](../src/xdte/live/decide.py) loads the live kit, reconstructs
  feature snapshots for each trading session, and combines fold predictions with
  kill-switch thresholds to emit actionable `DecisionRecord` objects. The module
  also surfaces guardrail warnings when required market inputs are missing.
- [`providers.py`](../src/xdte/live/providers.py) supplies default implementations
  for market and daily context data using `yfinance`. The provider interfaces are
  declared as protocols in `decide.py` so they can be swapped out during testing
  or when proprietary data feeds come online.

The live layer keeps inference deterministic by reading from the exported kit
and restricting side effects to provider call-outs, allowing production systems
to audit every decision.
