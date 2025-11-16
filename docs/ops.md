# Operations runbook: `xdte decide`

This runbook captures how to execute the production decision command, interpret
its outputs, and respond to operational incidents.

## Prerequisites

- A validated Live Kit exported via `xdte export-kit` and stored in a versioned
  directory such as `artifacts/2025-01/live_kit`.
- Python environment with the XDTE package installed so the `xdte` console
  script is available.【F:src/xdte/cli.py†L728-L808】
- Access to orchestrator logs or stdout capture to review warnings.

## Running the decision command

Invoke the command via the installed console script or the module entry point
when developing locally:

```bash
# Installed console script
xdte decide --kit-dir artifacts/2025-01/live_kit

# Module invocation for local development
PYTHONPATH=src python -m xdte.cli decide --kit-dir artifacts/2025-01/live_kit
```

Useful options:

- `--json/--no-json` toggles between human-readable tables and JSON
  output.【F:src/xdte/cli.py†L749-L755】【F:src/xdte/cli.py†L975-L989】
- `--market-json` / `--daily-json` inject bespoke payloads for market and daily
  features. The files must contain mappings keyed by session or ISO date
  respectively.【F:src/xdte/cli.py†L827-L853】
- `--offline-market` / `--offline-daily` swap in the internal stubs. These flags
  are mutually exclusive with the JSON providers and with each other to prevent
  configuration drift.【F:src/xdte/cli.py†L855-L872】
- `--cache-dir` caches yfinance downloads for reuse, and
  `--freeze-now/--no-freeze-now` controls whether the first run freezes provider
  payloads for subsequent executions.【F:src/xdte/cli.py†L720-L768】【F:src/xdte/cli.py†L809-L908】

The command exits with status code `0` on success. Click raises a
`ClickException` on failures, returning exit code `1` and surfacing the message
to stderr.【F:src/xdte/cli.py†L874-L984】【F:src/xdte/cli.py†L992-L1020】

## Reading the outputs

### Tabular output (default)

Without `--json`, the CLI prints a formatted table describing the selected
actions. Warnings are emitted on stderr ahead of the table so orchestrators can
surface them prominently.【F:src/xdte/cli.py†L975-L989】【F:src/xdte/formatting.py†L12-L90】

- Confirm the banner lines show the correct kit directory and timestamp.
- Review the per-book actions together with the rationale to confirm guardrails
  remain intact.
- Capture any warnings in the operational log.

### JSON output (`--json`)

When JSON mode is enabled the CLI emits a structured payload containing the kit
metadata, decision context ID, session snapshots, final actions, and any frozen
provider payloads.【F:src/xdte/cli.py†L942-L989】 Persist the payload or stream it
into downstream automation as required.

## Responding to warnings

Warnings indicate metric drift or other soft anomalies but do not stop the
command. They appear in stderr (and within the JSON payload under the `warnings`
field) while the exit status remains `0`.

1. Capture the warning message and rationale from the logs.【F:src/xdte/cli.py†L975-L989】
2. Compare the reported metrics with the thresholds in the golden manifest.
3. Annotate the run log with the decision and justification. Escalate to the
   modelling team if the warning signals potential guardrail violations.

## Responding to failures

Failures occur when the kit cannot be loaded, providers misbehave, or execution
raises unexpected exceptions. Click wraps these errors and exits with status
code `1`.

1. Capture the exception message emitted by Click (stdout/stderr).
2. Validate that the kit directory contains the manifest, policy, and model
   artefacts referenced by the manifest.【F:src/xdte/cli.py†L809-L872】
3. Re-run the command with `--verbose` to enable INFO/DEBUG logging for deeper
   diagnostics.【F:src/xdte/cli.py†L448-L475】
4. Escalate sustained issues with context: failing kit version, cached provider
   payloads, and the previous successful run ID.

Update this runbook whenever the CLI contract or operational controls change.
