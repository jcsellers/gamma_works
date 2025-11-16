"""End-to-end regression guard comparing pipeline outputs to the golden run."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pytest

from xdte.config import Settings
from xdte.model.apply import BOOKS, apply_models
from xdte.model.discovery import discover_rules
from xdte.model.export import build_live_kit
from xdte.model.hybrid import HybridSelectionResult, select_hybrid_policy
from xdte.model.train import train_models

MANIFEST_PATH = Path(__file__).with_name("golden_manifest.json")
BACKTEST_DIR = Path("data/backtest_data/original").resolve()

HIGHER_IS_BETTER_METRICS = {"cvar", "edp", "pf", "sharpe", "winrate"}
LOWER_IS_BETTER_METRICS = {"maxdrawdown"}


@pytest.fixture(scope="module")
def golden_manifest() -> Mapping[str, object]:
    """Load and memoize the frozen golden manifest."""

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload


def _coerce_keep(value: str | None) -> bool:
    if value is None:
        return False
    return value in {"1", "1.0", "true", "True"}


def _compute_per_book_edp(hybrid_input: Path) -> dict[str, float]:
    """Aggregate the no-kill daily pnl by book and compute EDP."""

    frame = pd.read_csv(hybrid_input)
    if "book" in frame.columns:
        daily_totals: dict[str, list[float]] = defaultdict(list)
        rows_by_day: dict[datetime.date, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for _, row in frame.iterrows():
            keep_value = row.get("keep")
            keep_token = str(keep_value) if keep_value is not None else None
            if not _coerce_keep(keep_token):
                continue
            book = str(row.get("book", "")).strip()
            if not book:
                continue
            open_date_raw = row.get("open_date")
            if pd.isna(open_date_raw):
                continue
            open_date = datetime.fromisoformat(str(open_date_raw)).date()
            pnl_value = float(row.get("pnl", 0.0) or 0.0)
            rows_by_day[open_date][book] += pnl_value

        for books in rows_by_day.values():
            for book, pnl in books.items():
                daily_totals[book].append(float(pnl))

        return {
            book: (sum(values) / len(values) if values else 0.0)
            for book, values in daily_totals.items()
        }

    per_book: dict[str, float] = {}
    for book in BOOKS:
        if book not in frame.columns:
            continue
        series = pd.to_numeric(frame[book], errors="coerce").dropna()
        if series.empty:
            continue
        non_zero = series != 0
        if not non_zero.any():
            continue
        per_book[book] = float(series[non_zero].mean())
    return per_book


def _compute_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _infer_book_from_relative(path: str) -> str | None:
    relative = Path(path)
    parts = relative.parts
    if not parts:
        return None
    if parts[0] == "discovery" and len(parts) >= 2:
        candidate = parts[1]
        if candidate not in {"manifest.json", "rules_ci.json"}:
            return candidate
        return None
    if parts[0] == "models":
        name = relative.name
        for prefix in ("feature_importance_", "shap_summary_"):
            if name.startswith(prefix) and name.endswith(".csv"):
                book = name[len(prefix) : -4]
                return book or None
    return None


def _run_pipeline(
    manifest: Mapping[str, object],
    *,
    tmp_path: Path,
    books: Sequence[str] | None = None,
) -> tuple[Path, HybridSelectionResult]:
    """Execute the training → tuning → export pipeline for comparison."""

    settings = Settings()
    run_root = tmp_path / "run"
    artifacts_dir = run_root / "artifacts"
    discovery_dir = run_root / "discovery"
    apply_dir = run_root / "apply"
    hybrid_dir = run_root / "hybrid"
    kit_dir = run_root / "live_kit"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    discovery_dir.mkdir(parents=True, exist_ok=True)
    apply_dir.mkdir(parents=True, exist_ok=True)
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    kit_dir.mkdir(parents=True, exist_ok=True)

    train_models(BACKTEST_DIR, artifacts_dir, settings=settings, books=books)
    discover_rules(
        artifacts_dir,
        discovery_dir,
        settings=settings,
        books=books,
    )
    apply_models(artifacts_dir, discovery_dir, apply_dir, settings=settings)

    policy_path = hybrid_dir / "policy.json"
    candidate = manifest["hybrid_policy"]
    tuning_result = select_hybrid_policy(
        apply_dir,
        candidate,
        policy_path,
        settings=settings,
        timestamp_factory=lambda: datetime(1970, 1, 1, 0, 0, 42, tzinfo=timezone.utc),
    )

    kit_version = str(manifest.get("meta", {}).get("kit_version", "test"))

    build_live_kit(
        artifacts_dir,
        discovery_dir,
        policy_path,
        kit_dir,
        settings=settings,
        version=kit_version,
        books=books,
    )

    return run_root, tuning_result


def _assert_metrics(
    manifest_metrics: Mapping[str, object],
    result: HybridSelectionResult,
    per_book: Mapping[str, float],
    *,
    tolerances: Mapping[str, float],
    per_book_tolerance: float,
) -> None:
    baseline = result.baseline_metrics
    candidate = result.evaluations[0].metrics if result.evaluations else {}

    expected_no_kill = manifest_metrics["no_kill"]
    expected_hybrid = manifest_metrics["hybrid"]

    def _assert_expected_metrics(
        *,
        kind: str,
        expected_metrics: Mapping[str, object],
        actual_metrics: Mapping[str, object],
    ) -> None:
        for key, expected in expected_metrics.items():
            actual = actual_metrics.get(key)
            assert actual is not None, f"{kind} metric {key} missing"
            _assert_metric_direction(
                kind=kind,
                key=key,
                actual=actual,
                expected=expected,
                tolerances=tolerances,
            )

    _assert_expected_metrics(
        kind="No-kill", expected_metrics=expected_no_kill, actual_metrics=baseline
    )
    _assert_expected_metrics(
        kind="Hybrid", expected_metrics=expected_hybrid, actual_metrics=candidate
    )

    expected_per_book = manifest_metrics.get("per_book_no_kill_edp", {})
    _assert_per_book_edp_floors(
        expected_per_book,
        per_book,
        tolerance=per_book_tolerance,
    )


def _assert_metric_direction(
    *,
    kind: str,
    key: str,
    actual: object,
    expected: object,
    tolerances: Mapping[str, float],
) -> None:
    key_lower = key.lower()
    tolerance = float(tolerances[f"{key_lower}_abs"])
    actual_value = float(actual)
    expected_value = float(expected)
    diff = actual_value - expected_value

    if key_lower in HIGHER_IS_BETTER_METRICS:
        if actual_value < expected_value - tolerance:
            raise AssertionError(
                f"{kind} metric {key} drifted by {diff} (actual={actual_value}, expected={expected_value})"
            )
    elif key_lower in LOWER_IS_BETTER_METRICS:
        if actual_value > expected_value + tolerance:
            raise AssertionError(
                f"{kind} metric {key} drifted by {diff} (actual={actual_value}, expected={expected_value})"
            )
    else:
        if abs(diff) > tolerance:
            raise AssertionError(
                f"{kind} metric {key} drifted by {diff} (actual={actual_value}, expected={expected_value})"
            )


def _assert_per_book_edp_floors(
    expected: Mapping[str, object],
    actual: Mapping[str, float],
    *,
    tolerance: float,
) -> None:
    for book, expected_value in expected.items():
        observed = actual.get(book)
        assert observed is not None, f"Per-book EDP missing for {book}"
        actual_value = float(observed)
        baseline_value = float(expected_value)
        floor = baseline_value - tolerance
        if actual_value < floor:
            raise AssertionError(
                f"Per-book EDP for {book} dropped below floor {floor} "
                f"(actual={actual_value}, expected={baseline_value})"
            )


def _assert_kit_manifest(
    manifest_path: Path, expected_entries: Mapping[str, str]
) -> Mapping[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping), "Kit manifest must decode to a mapping"
    files_section = payload.get("files")
    assert isinstance(files_section, Sequence), "Kit manifest missing files list"

    actual_entries: dict[str, str] = {}
    for item in files_section:
        assert isinstance(item, Mapping), "Kit manifest entry must be a mapping"
        path = item.get("path")
        digest = item.get("sha256")
        assert isinstance(path, str), "Kit manifest entry missing relative path"
        assert isinstance(digest, str), "Kit manifest entry missing digest"
        actual_entries[path] = digest

    for relative, expected_digest in expected_entries.items():
        actual = actual_entries.get(relative)
        assert (
            actual == expected_digest
        ), f"Kit manifest missing expected entry for {relative}"

    return actual_entries


def _assert_digest_manifest(
    manifest_path: Path,
    *,
    expected_digest: str,
    expected_entries: Mapping[str, str],
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping), "Digest manifest must decode to a mapping"
    files_section = payload.get("files")
    assert isinstance(files_section, Mapping), "Digest manifest missing files map"

    # Validate that the digest manifest is self-consistent (all referenced files exist
    # and have the recorded digests)
    base_dir = manifest_path.parent

    for relative, digest in files_section.items():
        assert isinstance(relative, str), "Digest manifest entry missing relative path"
        assert isinstance(digest, str), "Digest manifest entry missing digest"
        artifact = base_dir / relative
        assert (
            artifact.exists()
        ), f"Digest manifest references missing artifact: {relative}"
        actual = _compute_sha256(artifact)
        assert actual == digest, f"Digest mismatch for {relative}: {actual} != {digest}"

    # Validate that all expected entries are present with correct digests
    for relative, expected_digest in expected_entries.items():
        actual = files_section.get(relative)
        assert (
            actual == expected_digest
        ), f"Digest manifest missing expected entry for {relative}"


def _assert_digests(
    manifest_files: Sequence[Mapping[str, object]],
    run_root: Path,
    *,
    hash_required: bool,
) -> None:
    if not hash_required:
        return

    expected_manifest_entries: dict[str, str] = {}
    for entry in manifest_files:
        relative = entry.get("relative")
        expected_digest = entry.get("sha256")
        assert isinstance(relative, str), "Manifest entry missing relative path"
        assert isinstance(expected_digest, str), "Manifest entry missing digest"
        if (
            relative.startswith("live_kit/")
            and not relative.endswith("digests.json")
            and Path(relative).name != "manifest.json"
        ):
            expected_manifest_entries[
                Path(relative).relative_to("live_kit").as_posix()
            ] = expected_digest

    kit_entries: dict[str, str] = {}
    digests_target: Path | None = None
    digests_expected_digest: str | None = None
    for entry in manifest_files:
        relative = entry.get("relative")
        expected_digest = entry.get("sha256")
        assert isinstance(relative, str), "Manifest entry missing relative path"
        assert isinstance(expected_digest, str), "Manifest entry missing digest"
        target = run_root / relative
        assert target.exists(), f"Expected artifact missing: {target}"

        if relative.endswith("digests.json"):
            digests_target = target
            digests_expected_digest = expected_digest
            continue

        if relative == "live_kit/manifest.json":
            manifest_entries = _assert_kit_manifest(target, expected_manifest_entries)
            manifest_digest = _compute_sha256(target)
            assert manifest_digest == expected_digest, (
                "Digest mismatch for live_kit/manifest.json: "
                f"{manifest_digest} != {expected_digest}"
            )
            kit_entries = dict(manifest_entries)
            kit_entries["manifest.json"] = manifest_digest
            continue

        digest = _compute_sha256(target)
        assert (
            digest == expected_digest
        ), f"Digest mismatch for {relative}: {digest} != {expected_digest}"

        if relative.startswith("live_kit/"):
            kit_entries[Path(relative).relative_to("live_kit").as_posix()] = (
                expected_digest
            )

    if digests_target is not None and digests_expected_digest is not None:
        _assert_digest_manifest(
            digests_target,
            expected_digest=digests_expected_digest,
            expected_entries=kit_entries,
        )


def test_pipeline_matches_golden_snapshot(
    tmp_path: Path, golden_manifest: Mapping[str, object]
) -> None:
    """End-to-end regression guard comparing the pipeline with the golden run."""

    run_root, tuning_result = _run_pipeline(golden_manifest, tmp_path=tmp_path)

    hybrid_input = run_root / "apply" / "hybrid_input.csv"
    per_book = _compute_per_book_edp(hybrid_input)

    metrics = golden_manifest["metrics"]
    tolerances = golden_manifest["acceptance"]["tolerances"]
    per_book_tol = float(golden_manifest["acceptance"]["per_book_edp_abs"])
    _assert_metrics(
        metrics,
        tuning_result,
        per_book,
        tolerances=tolerances,
        per_book_tolerance=per_book_tol,
    )

    artifacts_section = golden_manifest["artifacts"]
    files = artifacts_section["files"]
    _assert_digests(
        files,
        run_root,
        hash_required=bool(golden_manifest["acceptance"]["hash_required"]),
    )


def test_partial_snapshot_respects_filtered_books(
    tmp_path: Path, golden_manifest: Mapping[str, object]
) -> None:
    config_section = golden_manifest.get("config", {})
    available_books = list(map(str, config_section.get("books", ())))
    assert available_books, "Golden manifest must declare available books"
    subset = tuple(available_books[: min(2, len(available_books))])
    assert subset, "Expected at least one book for the filtered snapshot"

    run_root, _ = _run_pipeline(golden_manifest, tmp_path=tmp_path, books=subset)

    hybrid_input = run_root / "apply" / "hybrid_input.csv"
    per_book = _compute_per_book_edp(hybrid_input)
    assert set(per_book) <= set(subset)

    per_book_metrics = golden_manifest["metrics"].get("per_book_no_kill_edp", {})
    assert isinstance(per_book_metrics, Mapping)
    expected_per_book = {
        book: float(per_book_metrics[book])
        for book in subset
        if book in per_book_metrics
    }
    assert set(expected_per_book) == set(subset)

    per_book_tol = float(golden_manifest["acceptance"]["per_book_edp_abs"])
    _assert_per_book_edp_floors(
        expected_per_book,
        per_book,
        tolerance=per_book_tol,
    )

    kit_dir = run_root / "live_kit"
    manifest_path = kit_dir / "manifest.json"
    digests_path = kit_dir / "digests.json"
    live_payload_path = kit_dir / "live_kit.json"
    models_manifest_path = kit_dir / "models" / "manifest.json"
    discovery_manifest_path = kit_dir / "discovery" / "manifest.json"
    rules_ci_path = kit_dir / "discovery" / "rules_ci.json"

    assert manifest_path.exists(), "Filtered kit manifest missing"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_section = manifest_payload.get("files")
    assert isinstance(files_section, Sequence)
    manifest_entries: dict[str, str] = {}
    for entry in files_section:
        assert isinstance(entry, Mapping)
        path_value = entry.get("path")
        digest_value = entry.get("sha256")
        assert isinstance(path_value, str)
        assert isinstance(digest_value, str)
        manifest_entries[path_value] = digest_value
        book = _infer_book_from_relative(path_value)
        if book is not None:
            assert book in subset, f"Unexpected book {book} in manifest"

    for book in subset:
        assert any(
            _infer_book_from_relative(path) == book for path in manifest_entries
        ), f"Manifest missing entries for {book}"

    assert digests_path.exists(), "Filtered digests manifest missing"
    _assert_digest_manifest(
        digests_path,
        expected_digest=_compute_sha256(digests_path),
        expected_entries=manifest_entries,
    )

    digest_payload = json.loads(digests_path.read_text(encoding="utf-8"))
    digest_files = digest_payload.get("files")
    assert isinstance(digest_files, Mapping)
    for relative, digest in digest_files.items():
        assert isinstance(relative, str)
        assert isinstance(digest, str)
        book = _infer_book_from_relative(relative)
        if book is not None:
            assert book in subset, f"Unexpected book {book} in digests"
    for book in subset:
        assert any(
            _infer_book_from_relative(relative) == book for relative in digest_files
        ), f"Digests missing entries for {book}"

    assert live_payload_path.exists(), "Filtered live kit payload missing"
    live_payload = json.loads(live_payload_path.read_text(encoding="utf-8"))
    books_payload = live_payload.get("books")
    assert isinstance(books_payload, Mapping)
    assert set(map(str, books_payload)) == set(subset)

    if models_manifest_path.exists():
        models_manifest = json.loads(models_manifest_path.read_text(encoding="utf-8"))
        models_books = models_manifest.get("books", {})
        assert isinstance(models_books, Mapping)
        assert set(map(str, models_books)) == set(subset)

    if discovery_manifest_path.exists():
        discovery_manifest = json.loads(
            discovery_manifest_path.read_text(encoding="utf-8")
        )
        discovery_books = discovery_manifest.get("books", {})
        assert isinstance(discovery_books, Mapping)
        assert set(map(str, discovery_books)) == set(subset)

    if rules_ci_path.exists():
        rules_payload = json.loads(rules_ci_path.read_text(encoding="utf-8"))
        if isinstance(rules_payload, Mapping):
            assert set(map(str, rules_payload)) <= set(subset)
