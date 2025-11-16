1. Scope & Ground Rules

The package (src/xdte/…) is the target implementation.

The goal of the migration is Colab Parity: the package must reproduce the notebook’s behavior on a frozen backtest dataset within documented tolerances.


=======
## Reference implementations:
The notebooks folder contains reference implementations of the code that we are trying to build, use those as specifications. 

Agents must:
Prefer small, focused changes over large rewrites.

Preserve the existing CLI surface and directory structure unless explicitly told to change them.

Keep all public behavior changes (CLI, outputs, JSON schema) clearly documented in code comments and tickets.

Agents must not:

Invent new strategies or change the high-level behavior without an explicit ticket.

Edit golden or reference artifacts without explicit instructions.

2. Development Workflow Expectations
When implementing or modifying logic:

Locate the reference      - Use docs/migration_map.md to map notebook sections to package modules.    - If a notebook section is ambiguous, do not guess; leave a note and escalate.

Change one concept at a time      - Example: one PR for feature masking, another for discovery, another for hybrid, etc.    - Each change must have a clear, single responsibility.

Always run tests for the scope you touch      - At minimum: relevant tests/unit/… and any affected integration tests.    - Do not skip tests to “get a green CI”.

Preserve logging and UX      - New logs should be concise and informative.    - Do not spam debug output by default; use flags (--verbose, --debug).

3. Colab Parity Tests & Agent Behavior (Migration Phase)
During the migration, Colab parity is the primary correctness gate. The notebook is the reference implementation.

A dedicated test (e.g. tests/test_colab_parity.py) compares the package’s metrics against notebook reference metrics on a frozen backtest dataset.

Colab parity has the following semantics:

Reference metrics (e.g. tests/data/notebook_metrics.json) are treated as read-only.

For each book and for the aggregate portfolio:   - No-kill and Hybrid metrics (EDP, PF, CVaR, trade counts) must match the notebook within the tolerances defined in the project docs (e.g. PLAN_AND_TICKETS.md).

If the parity test fails, the default assumption is that the package implementation is wrong, not the notebook or reference metrics.

When Colab parity tests fail:

Treat failures as implementation defects or regressions in the package.

Investigate and fix the data loading, feature construction, discovery, apply, hybrid, or export logic.

Do not:   - Edit the reference metrics file (e.g. tests/data/notebook_metrics.json).   - Loosen test tolerances.   - Skip, comment out, or delete the parity test.

Once Colab parity is achieved and validated by a human maintainer:

The package’s outputs on the canonical dataset are snapshotted as the new golden bundle.

Golden parity (see below) becomes the main regression guard.

Colab parity remains as a historical reference and must not be modified without an explicit ticket.

4. Golden Artifacts
Golden artifacts cover immutable datasets, manifests, and regression fixtures that encode trading parity after the migration is complete.

They live primarily under:

data/

tests/golden*/

Documentation manifests referenced by governance guides.

Examples:

tests/golden_manifest.json

Golden live kit bundles and metric files

Any associated digests and hash manifests

Golden artifacts represent a known-good package behavior; they must only change via the official refresh procedure.

5. Golden Parity Tests & Agent Behavior
The tests/test_golden_parity.py suite is the primary behavioral guardrail after golden has been re-baselined against the notebook. It verifies that new code does not materially degrade performance relative to the published golden bundle.

Golden parity has the following semantics:

Golden metrics are treated as floors/ceilings with explicit tolerances, not as exact replay targets.

For metrics where higher is better (e.g. EDP, PF, Sharpe, WinRate), a run passes if the new value is greater than or equal to the golden value minus the configured tolerance.

For metrics where lower is better (e.g. CVaR, max drawdown), a run passes if the new value is less than or equal to the golden value plus the configured tolerance.

Per-book metrics are treated as floors with a per-book tolerance; small drift is allowed, large regressions are not.

When golden parity tests fail
If any test in tests/test_golden_parity.py fails:

Treat the failure as a code regression or pipeline defect, not as a problem with the golden bundle.

Investigate and fix the implementation (or revert the change) until parity passes.

Agents must not attempt to resolve the failure by:

Editing golden manifests or hashes.

Relaxing tolerances or changing acceptance logic.

Skipping, commenting out, or deleting tests.

Golden updates are human-controlled
Only human maintainers are allowed to move the golden baseline:

Automated agents must not run golden-refresh tooling or commit new golden artifacts unless a ticket explicitly instructs them to do so and documents human approval.

When in doubt about whether a change implies a new golden snapshot, halt and escalate to the maintainers listed in CONTRIBUTING.md.

6. Forbidden Edits
Agents must not:

Hand-edit CSV, JSON, or pickle files under data/ or tests/golden*/.

Modify manifest hashes (*_manifest.json) or snapshot metadata in documentation unless the official refresh procedure has already produced the replacements.

Rewrite historical benchmark outputs referenced by the plan and ticket documentation.

Modify tests/test_golden_parity.py or any other golden regression test harness unless a ticket explicitly instructs you to do so.

Modify tests/test_colab_parity.py or any reference metrics files (e.g. tests/data/notebook_metrics.json) unless a ticket explicitly instructs you to do so.

Modify this file (AGENTS.md) unless explicitly instructed by a human maintainer.

If a change accidentally touches a forbidden area, see “Required remediation steps” below.

7. Required Remediation Steps for Golden / Reference Artifacts
If a change inadvertently touches a golden or reference artifact:

Revert the file(s) to the previous committed state immediately.

Re-run the sanctioned pipeline described in the docs (e.g. tools/update_golden_manifest.py and the backtest scripts referenced in docs/README.md) to regenerate fresh outputs — only if a human maintainer has confirmed a refresh is intended.

Include regenerated artifacts and updated manifests in the same commit as the code changes.

Add a short note in docs/PLAN_AND_TICKETS.md (or equivalent) describing:    - What changed.    - Why a refresh was needed.    - How the new behavior was validated.

Automated agents must not perform these steps without explicit instructions.

8. Human-Only Update Procedures
Golden artifact and reference metric refreshes require human approval:

A human reviewer must sign off before merging any change that rewrites:   - Golden artifacts (tests/golden*/, tests/golden_manifest.json).   - Reference metrics (tests/data/notebook_metrics.json, etc.).

Pull requests that modify these files must:   - Explain the triggering event (e.g. new strategy version, extended backtest range).   - Provide validation evidence (e.g. updated metrics, parity checks).

Automated agents must not approve, merge, or self-approve changes that rewrite golden or reference artifacts without documented human confirmation.

When in doubt, pause and escalate to the maintainers listed in CONTRIBUTING.md.

9. Coding & Testing Practices (Strategy Logic)
These practices are specific to the modeling and strategy code under src/xdte/.

9.1 Feature & data rules
Do not introduce new features into the training or live pipelines without:   - Updating the appropriate feature config(s) (e.g. BOOK_FEATS_CONFIG).   - Adding tests and documentation.

Per-book feature masks (e.g. BOOK_FEATS_CONFIG) are mandatory:   - 0DTE 11:00 books must not see 15:15 session columns.   - 1DTE 15:15 books must not see 11:00 session columns.

Daily context:   - Training relies on daily_context.csv that fully covers the backtest date range.   - If training needs new daily features, extend the daily context generator and loader accordingly.

9.2 NaN handling
Do not impute training features (no .fillna, no median/mean imputation).

LGBM must be configured to handle np.nan natively. This was T3.4-FIX in the plan and is a non-negotiable part of the strategy.

9.3 Unit Testing
Any new or ported helper function (e.g., from notebook [SECTION 2] to src/xdte/data/loaders.py or logic for src/xdte/model/discovery.py) must be accompanied by a new unit test in tests/unit/.

This reinforces T5.3-FIX as an ongoing development practice. Do not just port the code; port the confidence in the code.

9.4 Code Style & Formatting
All new or modified Python code must pass ruff and black checks as defined in .pre-commit-config.yaml.

Agents are expected to run these formatters before finalizing their changes.
- When in doubt, pause and escalate to the user
