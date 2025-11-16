# Contributing to XDTE

Thanks for supporting the XDTE selector. This guide covers the local
environment, required quality gates, and the governance rules around immutable
data and golden parity updates.

## Environment setup

1. Create a virtual environment and install the tooling extras:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. Set `PYTHONPATH=src` when invoking modules directly (the `xdte` console script
   configures this automatically).
3. Optional: install the `pre-commit` hooks (`pre-commit install`) to run the
   linters before each commit.

## Quality gates

All pull requests must pass the repository quality gates before submission:

```bash
ruff check .
black .
isort .
mypy --strict
pytest

Targeted artefact guards live in:

```bash
pytest tests/test_cli.py::test_decide_reports_dual_sessions_and_why tests/unit/test_apply.py::test_portfolio_metrics_extended_report tests/unit/test_gamma.py
```

Run them when touching the CLI presentation layer, apply reports, or gamma
mapping logic to catch schema regressions early.
```

These commands match the defaults in [`pyproject.toml`](pyproject.toml) and the
instructions in [`docs/README.md`](docs/README.md). Run them locally prior to
pushing to avoid CI failures. The `ruff` configuration blocks `print`
statements in library code (rule `T201`) so rely on the logging utilities when
surfacing information to users. The CLI enforces structured logging, which is
covered by the accompanying regression tests.

## Data immutability rules

- Treat the datasets under [`data/`](data/) as read-only fixtures. Only refresh
  them when coordinated golden parity updates are planned.
- Never overwrite artefacts under [`tests/golden/`](tests/golden/) manually. Use
  the documented update scripts to regenerate new snapshots.
- If you need exploratory data, generate it under a separate path (e.g.
  `scratch/`) and add it to `.gitignore`.

These rules align with the "Golden Artifacts" guardrails in [`AGENTS.md`](AGENTS.md);
review that section before touching any immutable datasets or manifests.

## Golden update process

1. Execute the sanctioned training pipeline against the approved backtest bundle
   (see [`docs/README.md`](docs/README.md) for the canonical commands).
2. Run [`tools/update_golden_manifest.py`](tools/update_golden_manifest.py) to
   copy the refreshed artefacts into `tests/golden_artifacts/` and regenerate
   `tests/golden_manifest.json`. Use `--books BOOK_A BOOK_B` when you need to
   refresh only a subset of books during investigations; omit it for the
   default full snapshot. The helper syncs the full apply bundle, including
   `calls_edge_*`, `fold_stability.csv`, and `portfolio_metrics_extended.csv`.
3. Commit the regenerated snapshot together with the code changes and updated
   documentation (including notes in [`docs/PLAN_AND_TICKETS.md`](docs/PLAN_AND_TICKETS.md)
   if the roadmap shifts).
4. Ensure the full quality gate suite still passes before opening the pull
   request.

By following this process we keep backtests reproducible, history auditable, and
upgrades predictable.
