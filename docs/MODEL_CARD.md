# XDTE Model Card

The model card summarises the current XDTE option strategy configuration and
highlights key assumptions, evaluation practices, and outstanding risks. The
implementation lives inside the [`xdte`](../src/xdte/) package.

## Data sources

| Source | Module | Description |
| --- | --- | --- |
| Portfolio trades | [`xdte.data.loaders`](../src/xdte/data/loaders.py) | CSV exports of historical option books converted into `PortfolioRecord` rows with normalised timestamps, leg metadata, and realised PnL. |
| Market sessions | [`xdte.data.features`](../src/xdte/data/features.py) | Intraday market snapshots (e.g. VIX, SPX moves) injected per decision window for the 11:00 and 15:15 books. |
| Daily context | [`xdte.data.features`](../src/xdte/data/features.py) | Lagged volatility, drawdown, and realised volatility indicators aligned to each open date to enforce notebook parity. |
| Discovery outputs | [`xdte.model.discovery`](../src/xdte/model/discovery.py) | Winsorised predictions and decile thresholds derived from fold-train scores. |
| Hybrid policy | [`xdte.model.hybrid`](../src/xdte/model/hybrid.py) | Tail/gamma adjustments tuned on fold-test predictions prior to export. |

All data consumed during backtesting is immutable under `data/` and must not be
rewritten outside of golden snapshot refreshes (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)).

## Modelling assumptions

- **Learner** – Fold models are trained with a custom LightGBM wrapper exposed
  via [`xdte.model._lgbm.LGBMRegressor`](../src/xdte/model/_lgbm.py). Gradient
  boosting was selected for its ability to handle heterogeneous features without
  extensive scaling.
- **Walk-forward evaluation** – [`xdte.model.wfo.WalkForwardSplitter`](../src/xdte/model/wfo.py)
  enforces chronological folds and yields deterministic train/test splits tied to
  `Settings.N_FOLDS`. This prevents leakage and keeps evaluation aligned with
  production deployment order.
- **Winsorisation** – [`xdte.model.discovery`](../src/xdte/model/discovery.py) applies
  per-book winsor limits (`Settings.WINSOR_P`) to trim extreme PnL values before
  computing keep thresholds. The resulting metadata is stored alongside the
  training artefacts.
- **Hybrid guardrails** – [`xdte.model.hybrid.select_hybrid_policy`](../src/xdte/model/hybrid.py)
  evaluates candidate tail bands and gamma overrides against the apply scores and
  rejects policies that degrade baseline metrics or violate configured safety
  rails (`Settings.KILL_SWITCH_*`).
- **Live data alignment** – [`xdte.live.decide`](../src/xdte/live/decide.py) rebuilds
  the exact feature set expected by the models and validates live market inputs
  through the provider protocols defined in [`xdte.live.providers`](../src/xdte/live/providers.py).

## Evaluation metrics

The canonical evaluation bundle is defined in
[`xdte.metrics.portfolio_metrics`](../src/xdte/metrics.py). Each fold-test or
hybrid evaluation reports:

- **EDP** – Expected daily PnL (`mean` of the scored series).
- **PF** – Profit factor (`profit_factor` ratio of gains vs losses).
- **CVaR95** – Conditional value at risk at 5% (`cvar` with `alpha=0.05`).

During hybrid tuning the metrics are computed per book and aggregated to ensure
policy changes improve risk-adjusted returns.

## Explainability

Training runs write per-book SHAP summaries and fallback gain importances to the
artefact directory. Each book receives `shap_summary_{book}.csv` capturing the
mean absolute SHAP contribution and averaged expected value for every feature,
plus `feature_importance_{book}.csv` with LightGBM gain scores when SHAP data is
unavailable. The exporter copies those filenames into the live kit manifest as
`models/shap_summary_{book}.csv` and `models/feature_importance_{book}.csv` so
`xdte decide` can surface precomputed contributions, top contributors, and
expected values during live decisions without recomputing SHAP on demand.

## Known risks and mitigation

- **Data drift** – If the upstream trade exports change column names or
  semantics, [`xdte.data.validation`](../src/xdte/data/validation.py) must be
  updated and the golden regression snapshot refreshed.
- **Provider outages** – Live decisions depend on the default yfinance providers
  (`YFinanceDailyContextProvider`, `YFinanceMarketDataProvider`). Production
  deployments should wrap these in retry and caching layers or swap in firm-owned
  data feeds.
- **Hyperparameter coupling** – Fold counts, winsor percentiles, and alpha levels
  live in [`xdte.config.Settings`](../src/xdte/config.py). Adjusting one without
  coordinating the others can invalidate discovery thresholds; always rerun the
  full training and parity suite after changes.
- **Golden parity regression** – The suite in
  [`tests/test_golden_parity.py`](../tests/test_golden_parity.py) enforces parity
  with `tests/golden/`. Any intentional change to modelling or data must include a
  regenerated golden snapshot using `tools/update_golden_manifest.py`.
