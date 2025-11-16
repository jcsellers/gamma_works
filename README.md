# XDTE

XDTE is an options strategy toolkit that packages the training, tuning, and
live-decision pipelines for the selector described in
[`docs/PLAN_AND_TICKETS.md`](docs/PLAN_AND_TICKETS.md). The repository includes
the Python package, operational runbooks, and governance documentation required
to refresh the strategy or run it in production.

## Quick start

1. Create an isolated environment and install the package together with the
   development tooling:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. Install the Git hooks and run the standard quality checks. The
   configuration in `.pre-commit-config.yaml` wires the formatter, linter,
   type-checker, security scanner, and test suite:

   ```bash
   pre-commit install
   pre-commit run --all-files
   pre-commit run mypy --all-files
   pre-commit run bandit --all-files
   pytest
   ```

   Re-run the hooks until they report no further changes.

3. Explore the CLI entry points exposed by the `xdte` console script. The core
   workflow is:

   ```bash
   xdte validate-backtest
   xdte train --backtest-dir data/backtest_data/original --out-dir artifacts/2025-01
   xdte tune --artifacts-dir artifacts/2025-01 --candidates policy_candidates.json
   xdte export-kit --artifacts-dir artifacts/2025-01 --out artifacts/2025-01/live_kit --version 2025-01
   xdte decide --kit-dir artifacts/2025-01/live_kit --json
   ```

   Each command accepts overrides for the shared settings documented in
   [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Guardrails you must preserve
Before touching data or model logic, review the guardrails documented in the plan:

- **Determinism:** Shared seeds originate from `xdte.config.Settings` and must be respected across
  `xdte.model.wfo`, `xdte.model.train`, and downstream pipelines.
- **No leakage:** Keep thresholds and decile edges learned strictly on fold-train scores inside
  `xdte.model.discovery`; `xdte.model.apply` only reads persisted parameters.
- **Golden parity:** The `tests/test_golden_parity.py` regression suite compares artifacts with
  `tests/golden/2025-11-10/` and blocks merges when tolerances are exceeded.

Consult the "Golden Artifacts" rules in [`AGENTS.md`](AGENTS.md) before modifying datasets,
snapshots, or manifests referenced by these guardrails.

See the full backlog and architectural map in [`docs/PLAN_AND_TICKETS.md`](docs/PLAN_AND_TICKETS.md)
alongside the detailed module map in [`docs/DESIGN.md`](docs/DESIGN.md) and the strategy summary in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the remaining guardrails, risks, and outstanding data
dependencies. Additional documentation links are curated in [`docs/README.md`](docs/README.md).

## Command line interface

The `xdte` console script coordinates the end-to-end workflow. Every command
accepts overrides for `Settings` fields (for example `--alpha`, `--winsor-p`, or
`--model-seed`) which are forwarded to the underlying modules. Key entry points:

### Validate the frozen backtest bundle

```
xdte validate-backtest --data-dir data/backtest_data/original --manifest tests/fixtures/backtest/manifest.json
```

Use this before retraining to confirm that the local CSV bundle matches the
recorded SHA-256 digests.

### Train models, discovery rules, and apply scores

```
xdte train --backtest-dir data/backtest_data/original --out-dir artifacts/2025-01 --books ALPHA --books BETA --alpha 0.2
```

The command loads the frozen backtest, optionally restricts training to the
supplied books, then materialises:

- `artifacts/` – persisted training fold artefacts.
- `discovery/` – winsorisation thresholds and keep rules.
- `apply/` – scored test rows prepared for hybrid tuning.

### Tune the hybrid policy

```
xdte tune --artifacts-dir artifacts/2025-01 --candidates policy_candidates.json
```

Candidate payloads may be a top-level sequence or a mapping with a
`candidates` key. Each entry can include static tails and gammas together with
optional `gamma_rules` for dynamic sizing. Accepted policies are written to
`<artifacts-dir>/hybrid/policy.json` unless `--out` points at an alternative
file. The tuner refreshes `<artifacts-dir>/apply/gamma_backtest.csv` to capture
the static-versus-dynamic audit for the selected policy.

### Export a Live Kit

```
xdte export-kit --artifacts-dir artifacts/2025-01 --out artifacts/2025-01/live_kit --version 2025-01
```

The exporter bundles the training artefacts, discovery metadata, tuned policy,
and a manifest with digests into a versioned directory ready for live use.

### Generate live decisions

```
xdte decide --kit-dir artifacts/2025-01/live_kit --json --market-json market_snapshot.json --daily-json daily_context.json
```

`decide` loads the exported kit and emits operational actions. JSON output is
optional; omit `--json` for human-readable tables. Supply `--market-json` and
`--daily-json` for bespoke inputs or opt into the internal stubs via
`--offline-market` / `--offline-daily`. Use `--cache-dir` when yfinance downloads
should be persisted between runs and `--no-freeze-now` to refresh live data on
repeat invocations.

## Repository Layout

- `src/xdte/`: Python package scaffolding for the application modules.
- `docs/`: Documentation placeholder for detailed architecture and process guides.
- `pyproject.toml`: Packaging metadata and tool configuration.
- `.pre-commit-config.yaml`: Local automation for formatters and linters.

> **Note**
> The legacy `xdte_ml_selector` prototype has been retired. All new work should
> target the `xdte` package, which contains the canonical training, tuning, and
> decisioning pipeline.

## License

MIT License. See `LICENSE` for details.
