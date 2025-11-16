# XDTE User Guide

This guide walks operators through preparing data, running the XDTE workflow,
customising configuration, and troubleshooting common issues. It complements the
architectural notes in [`DESIGN.md`](DESIGN.md) and the governance artefacts in
[`MODEL_CARD.md`](MODEL_CARD.md).

## 1. Prerequisites

### Python environment

1. Install Python 3.11 or newer.
2. Create an isolated environment and install the package with development
   extras:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

### Data inputs

| Asset            | Location                               | Notes |
| ---------------- | -------------------------------------- | ----- |
| Backtest bundle  | `data/backtest_data/original/`         | Frozen CSVs referenced by the regression suite. |
| Manifest         | `tests/fixtures/backtest/manifest.json`| SHA-256 digests used for validation. |
| Policy candidates| User-supplied JSON payload.            | See [Hybrid tuning](#4-hybrid-tuning). |

Keep the backtest bundle immutable. Regenerate it only through the controlled
process documented in `docs/README.md`.

## 2. Verifying the backtest bundle

Before training, validate that the local bundle matches the approved manifest:

```bash
xdte validate-backtest --data-dir data/backtest_data/original --manifest tests/fixtures/backtest/manifest.json
```

Point `--data-dir` at whichever frozen bundle you wish to validate and
`--manifest` at the corresponding digest file. Both options accept relative or
absolute paths, allowing you to keep manifests and datasets in separate
locations when needed.

The command checks directory contents against recorded digests and raises a
CLI error when files are missing or altered.【F:src/xdte/cli.py†L992-L1020】

### Book-only experimental bundles

For quick experiments you can stage a directory that only contains
`book_*.csv` files alongside the manifest. Each CSV combines the trade and
market statistics for a single configuration, so keep the naming scheme
descriptive—`book_0dte_put_spread_15delta_20_pts_1100.csv` and
`book_0dte_call_spread_20delta_20_pts_1515.csv` are typical examples. The
loader expects every row to carry the standard trade metrics plus market
context:

* `pnl`, `gap`, and `movement` so the training pipeline can build the book
  panels.
* `opening_vix` and `closing_vix` so VIX-dependent features stay available for
  market aggregation and live replay.

Although the bundle omits legacy `*_market_stats_*.csv` files, a daily context
file is now optional. When `daily_context.csv`/`daily_context.json` is missing,
`load_backtest_dataset` returns an empty daily frame and the feature generator
continues with the book data only, so you can still run lightweight
experiments.【F:src/xdte/model/train.py†L143-L210】 Run
`xdte validate-backtest` on these experimental bundles to confirm that all
required files and columns are present before training.

## 3. Training artefacts

Run the `train` command to produce fold models, discovery thresholds, and apply
scores:

```bash
xdte train --backtest-dir data/backtest_data/original --out-dir artifacts/2025-01
```

### Customising input and output locations

`--backtest-dir` can point at any directory containing the frozen backtest
bundle. Relative paths resolve against the working directory, while absolute
paths allow you to stage datasets elsewhere (for example on a mounted volume).
Similarly, `--out-dir` controls where training, discovery, and apply outputs are
written. The CLI creates `artifacts/`, `discovery/`, and `apply/` subdirectories
under the location you provide, so choose an empty or dedicated directory when
running new experiments.

Outputs are organised beneath the chosen `--out-dir`:

- `artifacts/` – persisted model artefacts returned by `train_models`.【F:src/xdte/cli.py†L452-L487】
- `discovery/` – metrics, thresholds, and keep/drop flags emitted by `discover_rules`.【F:src/xdte/cli.py†L470-L485】
- `apply/` – hold-out scores prepared for hybrid selection via `apply_models`.【F:src/xdte/cli.py†L470-L485】

On success the CLI reports the number of trained fold models.【F:src/xdte/cli.py†L488-L504】

### Training with book-only bundles

`xdte train` accepts the experimental book-only layout described earlier.
When `load_backtest_dataset` scans a directory that lacks
`*_market_stats_*.csv` files it automatically reuses the `book_*.csv` rows to
populate both the trade panel and the market feature frame, inferring the book
and session identifiers directly from the filenames.【F:src/xdte/model/train.py†L180-L204】【F:src/xdte/data/features.py†L188-L229】
Dropping the daily context file simply removes the lagged indicators from the
feature panel; training still proceeds with the books encoded in those
filenames, which makes this layout ideal for small-scale experiments or custom
spreads without regenerating the full legacy bundle.【F:src/xdte/model/train.py†L143-L210】

## 4. Hybrid tuning

With training outputs in place, evaluate candidate hybrid policies:

```bash
xdte tune --artifacts-dir artifacts/2025-01 --candidates policy_candidates.json
```

The candidates file may contain either:

- A mapping with a `candidates` field whose value is a sequence of candidate
  payloads; or
- A top-level sequence of candidate mappings.

Any other structure raises a CLI error before tuning starts.【F:src/xdte/cli.py†L110-L154】

### Sizing controls

Hybrid policies always understand static sizing via the `gammas` mapping. When
`gamma_rules` is omitted or empty the live decision engine honours only the
static overrides, so deployments remain unchanged unless you opt into dynamic
behaviour.【F:src/xdte/model/hybrid.py†L32-L61】【F:src/xdte/live/actions.py†L390-L417】

To enable dynamic sizing for a book, add a rule describing the percentile
thresholds that map score margins to gamma levels:

```json
{
  "gamma_rules": {
    "PUTS_1DTE_1515": {
      "lo": 0.1,
      "hi": 0.3,
      "levels": [0.5, 0.75, 1.0],
      "percentiles": [[0.0, -2.5], [0.5, 0.0], [1.0, 2.0]],
      "target_percentile": 0.7,
      "kill_day_level": 0.75
    }
  }
}
```

* `levels` lists the gamma values in ascending aggressiveness.
* `lo`/`hi` bound the margin breakpoints between the levels.
* `percentiles` persist the discovery quantiles used to translate predictions
  into percentile ranks.【F:src/xdte/model/hybrid.py†L334-L403】【F:src/xdte/gamma.py†L40-L131】
* `target_percentile` anchors the neutral point for the margin calculation, and
  `kill_day_level` caps gamma on kill days if present.【F:src/xdte/gamma.py†L164-L185】【F:src/xdte/live/actions.py†L418-L457】

Removing the rule or clearing `gamma_rules` returns the policy to static sizing.

Unless `--out` is provided, the accepted policy is written to
`<artifacts-dir>/hybrid/policy.json`. The CLI reports whether a candidate was
accepted.【F:src/xdte/cli.py†L529-L575】 The tuner also writes
`<artifacts-dir>/apply/gamma_backtest.csv`, a day-level audit of the baseline
and dynamic gamma choices for the selected policy so you can verify monotonic
behaviour before exporting.【F:src/xdte/model/hybrid.py†L324-L403】【F:src/xdte/model/hybrid.py†L592-L595】

## 5. Exporting a Live Kit

After accepting a hybrid policy, create a distributable kit:

```bash
xdte export-kit --artifacts-dir artifacts/2025-01 --out artifacts/2025-01/live_kit --version 2025-01
```

The exporter pulls together training artefacts, discovery outputs, and the
selected policy, then writes a manifest and digest file in the target directory
for downstream verification.【F:src/xdte/cli.py†L576-L622】

## 6. Generating live decisions

Feed an exported kit into `xdte decide` to produce operational decisions:

```bash
xdte decide --kit-dir artifacts/2025-01/live_kit --json
```

By default the command prints a table summarising sessions, features, and
decisions; use `--json` for a structured payload. The CLI also surfaces any
warnings generated during the run.【F:src/xdte/cli.py†L623-L704】【F:src/xdte/cli.py†L705-L744】

Explainability values in the JSON (expected value, global/top contributors) are
loaded from the precomputed `shap_summary_{book}.csv` and
`feature_importance_{book}.csv` files bundled in the live kit so decisions stay
deterministic and low-latency.

### Data providers

You can supply bespoke inputs or rely on offline stubs:

- `--market-json PATH` – mapping keyed by session (`"11:00"`, `"15:15"`, …)
  with market features for each decision window.【F:src/xdte/cli.py†L827-L838】
- `--daily-json PATH` – mapping keyed by ISO open date (or a flat mapping) with
  lagged daily indicators.【F:src/xdte/cli.py†L840-L853】
- `--offline-market` / `--offline-daily` – internal stubs useful for dry runs.
  These options are mutually exclusive with the JSON providers and each other
  as documented by CLI usage guards.【F:src/xdte/cli.py†L855-L872】

### Output structure

`--json` emits a payload containing the kit manifest, policy, sessions, and
per-book decisions with feature values normalised for JSON compatibility.
Without `--json`, the CLI formats a human-readable report and prints the kit
location consumed during the run.【F:src/xdte/cli.py†L954-L989】

## 7. Configuration overrides

All commands accept the shared settings options listed below. Overrides can be
provided via CLI flags or environment variables prefixed with `XDTE_`.

| Setting            | CLI flag             | Environment variable  | Default |
| ------------------ | --------------------| ----------------------| ------- |
| Walk-forward folds | `--n-folds`         | `XDTE_N_FOLDS`        | 8       |
| CVaR alpha         | `--alpha`           | `XDTE_ALPHA`          | 0.05    |
| Winsor percentile  | `--winsor-p`        | `XDTE_WINSOR_P`       | 0.01    |
| Kill-switch tails  | `--kill-switch-tails`| `XDTE_KILL_SWITCH_TAILS`| `{"PUTS_0DTE_11": [0.045, 0.961], …}` |
| Kill-switch gammas | `--kill-switch-gammas`| `XDTE_KILL_SWITCH_GAMMAS`| `{"PUTS_1DTE_1515": 1.0, …}` |
| Training seed      | `--model-seed`      | `XDTE_MODEL_SEED`     | 42      |
| Tuning seed        | `--tuner-seed`      | `XDTE_TUNER_SEED`     | 42      |

Environment values are parsed using the helper functions in
`xdte.config.Settings`, while CLI overrides are processed through `argparse`
before being forwarded to each command.【F:src/xdte/config.py†L12-L120】【F:src/xdte/config.py†L122-L177】

### Caching and freeze controls

- `--cache-dir` (or the `XDTE_CACHE_DIR` environment variable) stores market and
  daily provider payloads on disk so repeated runs reuse downloads when
  available.【F:src/xdte/cli.py†L720-L768】【F:src/xdte/cli.py†L809-L844】
- `--freeze-now/--no-freeze-now` toggles whether the first successful run freezes
  the captured provider payloads for subsequent invocations. Disable freezing
  when live refreshes are required on each execution.【F:src/xdte/cli.py†L720-L768】【F:src/xdte/cli.py†L845-L908】

## 8. End-to-end checklist

1. Validate backtest bundle integrity.
2. Run `xdte train` to refresh artefacts.
3. Review apply outputs and craft hybrid candidates.
4. Execute `xdte tune` and confirm the accepted policy.
5. Export a Live Kit with `xdte export-kit`.
6. Generate live decisions via `xdte decide`.
7. Commit artefacts and documentation updates as required.

## 9. Troubleshooting

| Symptom | Likely cause | Resolution |
| ------- | ------------ | ---------- |
| `click.ClickException: Market JSON must be a mapping keyed by session` | JSON file structure is incorrect. | Ensure top-level keys match decision sessions (for example `"11:00"`).【F:src/xdte/cli.py†L827-L838】 |
| `Candidates file must contain a mapping or sequence payload` | Candidate payload not formatted as mapping/sequence. | Wrap candidates inside a list or a mapping with a `candidates` field.【F:src/xdte/cli.py†L110-L154】 |
| `Parent directory for --out does not exist` | Target directory missing when exporting or tuning. | Create the parent directories before running the command.【F:src/xdte/cli.py†L325-L348】 |
| `click.UsageError: Cannot use --offline-market when a market provider is already set` | Conflicting CLI options on `decide`. | Choose either offline stubs or JSON providers, not both.【F:src/xdte/cli.py†L855-L872】 |

## 10. Further reading

- [`docs/DESIGN.md`](DESIGN.md) – Module responsibilities and integration points.
- [`docs/MODEL_CARD.md`](MODEL_CARD.md) – Strategy assumptions and evaluation.
- [`docs/README.md`](README.md) – Documentation map and parity refresh steps.

With these resources and the workflow above you can operate the XDTE selector
end-to-end, from validating historical data to producing live decisions.
