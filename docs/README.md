# XDTE documentation hub

This directory gathers the operational guides, architecture notes, and
governance collateral for the XDTE selector. The recommended reading order is:

1. [`PLAN_AND_TICKETS.md`](PLAN_AND_TICKETS.md) – programme roadmap, delivery
   milestones, and non-negotiable guardrails.
2. [`DESIGN.md`](DESIGN.md) and [`architecture.md`](architecture.md) – module
   ownership, data flow, and pipeline topology.
3. [`USER_GUIDE.md`](USER_GUIDE.md) – day-to-day operator workflow from data
   validation through live decisioning.
4. [`ops.md`](ops.md) – runbook for executing `xdte decide` in production
   environments and reacting to warnings or failures.

Additional references:

- [`MODEL_CARD.md`](MODEL_CARD.md) – strategy assumptions, evaluation metrics,
  and risk disclosures.
- [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) and
  [`REVIEW_2025-09-23.md`](REVIEW_2025-09-23.md) – historical programme status
  and audit artefacts.
- [`DATA_SCHEMA.md`](DATA_SCHEMA.md) – canonical schema for the frozen backtest
  bundle and derived artefacts.

## Operational workflow snapshot

The CLI orchestrates four core phases:

1. **Validate** – `xdte validate-backtest --data-dir data/backtest_data/original`
   compares the on-disk bundle with the recorded manifest before retraining.
2. **Train** – `xdte train --backtest-dir data/backtest_data/original --out-dir artifacts/YYYYMM`
   materialises model artefacts, discovery metadata, and apply scores under the
   supplied directory.
3. **Tune** – `xdte tune --artifacts-dir artifacts/YYYYMM --candidates candidates.json`
   evaluates hybrid policy payloads and writes the accepted policy to the
   `hybrid/` subtree.
4. **Export and decide** – `xdte export-kit` packages a Live Kit that can be fed
   into `xdte decide` for live action generation. Use `--json`, `--market-json`,
   and `--daily-json` to integrate with automation or provide custom data.

All commands expose overrides for `Settings` parameters so fold counts, alpha
levels, and seeds can be adjusted without editing source files. Refer to
[`USER_GUIDE.md`](USER_GUIDE.md) for a deep dive into each stage.

## Refreshing the golden parity snapshot

`tests/test_golden_parity.py` compares end-to-end outputs against the frozen
manifest. When model logic or datasets change:

1. Run the sanctioned pipeline against the approved backtest bundle:

   ```bash
   PYTHONPATH=src python -m xdte.cli train --backtest-dir data/backtest_data/original --out-dir artifacts/new_run
   PYTHONPATH=src python -m xdte.cli tune --artifacts-dir artifacts/new_run --candidates artifacts/new_run/policy_candidates.json
   PYTHONPATH=src python -m xdte.cli export-kit --artifacts-dir artifacts/new_run --out artifacts/new_run/live_kit --version new_run
   ```

   Use the production candidate payload when invoking `xdte tune` so the
   accepted policy matches the controls recorded in
   `tests/golden_manifest.json`.

2. Update the golden directory and manifest:

   ```bash
   PYTHONPATH=src python tools/update_golden_manifest.py artifacts/new_run tests/golden_artifacts --label golden-YYYY-MM-DD
  ```

   The helper copies the sanctioned artefacts into
   `tests/golden_artifacts/`, refreshes digests, and syncs the expanded apply
   bundle (WHY reports, fold stability, extended metrics) used by the parity
   suite.

   When investigating a narrow regression, add `--books PUTS_1DTE_1515 CALLS_0DTE`
   (or any subset of book identifiers) to regenerate metrics, manifests, and
   digests only for the specified books. Omitting the flag keeps the legacy
   full-book behaviour.

3. Commit the refreshed snapshot with the corresponding code changes. Document
   any material modelling impact in [`MODEL_CARD.md`](MODEL_CARD.md).

## Building the API reference

The Sphinx project under `docs/sphinx/` generates the API reference directly
from modules in `src/xdte` and runs `sphinx-apidoc` automatically so stub files
stay current.

```bash
make html
```

Rendered HTML is written to `docs/_build/html/`. Open `index.html` to browse the
site locally.
