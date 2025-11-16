"""Utility for refreshing the golden parity snapshot and manifest.

Supports optional book filtering so operators can regenerate parity assets for
selected books without touching the rest of the snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

from xdte.config import Settings
from xdte.metrics import portfolio_metrics

GOLDEN_FILES: tuple[str, ...] = (
    "apply/hybrid_input.csv",
    "apply/gamma_backtest.csv",
    "apply/calls_edge_report.csv",
    "apply/calls_edge_attribution.json",
    "apply/fold_stability.csv",
    "apply/portfolio_metrics_extended.csv",
    "artifacts/train/manifest.json",
    "discovery/manifest.json",
    "hybrid/policy.json",
    "live_kit/digests.json",
    "live_kit/discovery/manifest.json",
    "live_kit/hybrid/policy.json",
    "live_kit/live_kit.json",
    "live_kit/manifest.json",
    "live_kit/models/manifest.json",
)


def _copy_selected_files(source: Path, destination: Path) -> None:
    for relative in GOLDEN_FILES:
        origin = source / relative
        target = destination / relative
        if not origin.exists():
            raise FileNotFoundError(f"Expected artifact missing: {origin}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)


def _load_policy(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        raise TypeError("Hybrid policy payload must be a mapping")
    return payload


def _normalise_book_filter(books: Collection[str] | None) -> set[str] | None:
    if books is None:
        return None
    return {str(book) for book in books if str(book).strip()}


def _discover_books_in_directory(directory: Path) -> list[str]:
    """
    Discover book identifiers from the given directory.

    The function prefers subdirectories (excluding hidden ones, i.e., those starting with ".")
    as book identifiers. If no such subdirectories are found, it falls back to using the
    stems of non-hidden files in the directory. Hidden entries are ignored in both cases.

    Args:
        directory: Path to the directory to search for books.

    Returns:
        A list of book identifiers inferred from subdirectory names or, if none are found,
        from the stems of non-hidden files.
    """
    resolved = directory.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Book directory does not exist: {directory}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Book directory must be a directory: {directory}")

    directories = [
        entry.name
        for entry in sorted(resolved.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    if directories:
        return directories

    books: list[str] = []
    for entry in sorted(resolved.iterdir()):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        stem = entry.stem.strip()
        if stem:
            books.append(stem)
    return books


def _merge_book_sources(*sources: Sequence[str] | None) -> list[str] | None:
    """
    Merge multiple sequences of book identifiers into a single list.

    - Deduplicates identifiers while preserving their original order of appearance.
    - Trims whitespace from each identifier and filters out empty strings.
    - Returns None if all sources are empty or None.

    Args:
        *sources: Sequences of book identifiers (or None).

    Returns:
        A list of unique, non-empty, trimmed book identifiers, or None if no valid identifiers are found.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not source:
            continue
        for book in source:
            cleaned = str(book).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            merged.append(cleaned)
    return merged or None


def _compute_daily_metrics(
    hybrid_input: Path,
    *,
    settings: Settings,
    books: Collection[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    requested_books: list[str] | None = None
    if books is not None:
        requested_books = [str(book).strip() for book in books if str(book).strip()]
    allowed_books = _normalise_book_filter(requested_books)
    books_seen: set[str] = set()
    rows_by_day: dict[date, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    with hybrid_input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keep_flag = row.get("keep") in {"1", "1.0", "true", "True"}
            if not keep_flag:
                continue
            book = str(row.get("book", "")).strip()
            if not book:
                continue
            if allowed_books is not None and book not in allowed_books:
                continue
            books_seen.add(book)
            open_date_raw = row.get("open_date")
            if not open_date_raw:
                continue
            open_date = datetime.fromisoformat(open_date_raw).date()
            pnl_value = float(row.get("pnl", 0.0) or 0.0)
            rows_by_day[open_date][book] += pnl_value

    per_book_edp: dict[str, list[float]] = defaultdict(list)
    daily_totals: list[float] = []
    for book_pnls in rows_by_day.values():
        day_total = 0.0
        for book, pnl in book_pnls.items():
            per_book_edp[book].append(float(pnl))
            day_total += float(pnl)
        daily_totals.append(day_total)

    portfolio_stats = portfolio_metrics(daily_totals, alpha=settings.ALPHA)
    book_edp = {
        book: (sum(values) / len(values) if values else 0.0)
        for book, values in per_book_edp.items()
    }
    if allowed_books is not None and requested_books is not None:
        ordered_books: Iterable[str] = (
            book for book in requested_books if book in books_seen
        )
    else:
        ordered_books = sorted(books_seen)
    return portfolio_stats, book_edp, list(ordered_books)


def _compute_digests(directory: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel_repo = path.relative_to(Path.cwd()).as_posix()
        rel_root = path.relative_to(directory).as_posix()
        digest = sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "name": path.name,
                "path": rel_repo,
                "relative": rel_root,
                "sha256": digest,
            }
        )
    return files


def _filter_books_mapping(
    mapping: Mapping[str, object] | None, books: Collection[str] | None
) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        return {}
    allowed = _normalise_book_filter(books)
    if allowed is None:
        return dict(mapping)
    return {book: payload for book, payload in mapping.items() if book in allowed}


def refresh_golden_snapshot(
    source: Path,
    destination: Path,
    manifest_path: Path,
    *,
    run_label: str,
    books: Collection[str] | None = None,
) -> None:
    settings = Settings()
    requested_books: list[str] | None = None
    if books is not None:
        requested_books = [str(book).strip() for book in books if str(book).strip()]
    _copy_selected_files(source, destination)

    policy_payload = _load_policy(destination / "hybrid/policy.json")
    discovery_manifest = json.loads(
        (destination / "discovery/manifest.json").read_text(encoding="utf-8")
    )
    kit_manifest = json.loads(
        (destination / "live_kit/manifest.json").read_text(encoding="utf-8")
    )
    hybrid_metrics: dict[str, float] = {}
    candidates_payload = policy_payload.get("candidates")
    if (
        isinstance(candidates_payload, Sequence)
        and not isinstance(candidates_payload, str)
        and candidates_payload
    ):
        first_candidate = candidates_payload[0]
        if isinstance(first_candidate, Mapping):
            metrics_payload = first_candidate.get("metrics")
            if isinstance(metrics_payload, Mapping):
                hybrid_metrics = {
                    str(metric): float(value)
                    for metric, value in metrics_payload.items()
                }
    selected_policy = policy_payload.get("selected_policy", {})

    portfolio_stats, per_book_edp, observed_books = _compute_daily_metrics(
        destination / "apply/hybrid_input.csv",
        settings=settings,
        books=requested_books,
    )
    if requested_books:
        per_book_edp = {
            book: per_book_edp.get(book, 0.0)
            for book in observed_books
            if book in per_book_edp
        }
    baseline_metrics = policy_payload.get("baseline_metrics")
    if isinstance(baseline_metrics, Mapping):
        portfolio_stats = {
            str(metric): float(value) for metric, value in baseline_metrics.items()
        }

    manifest = {
        "meta": {
            "run_label": run_label,
            "created_utc": policy_payload.get("evaluated_at"),
            "repo_commit": "local-dev",
            "env": {
                "python": ">=3.10",
                "numpy": ">=1.26",
                "pandas": ">=2.0",
                "lightgbm": ">=4.0",
                "optuna": ">=3.5",
            },
            "seed": settings.MODEL_SEED,
            "kit_version": kit_manifest.get("version"),
        },
        "config": {
            "n_folds": settings.N_FOLDS,
            "alpha": settings.ALPHA,
            "winsor_p": settings.WINSOR_P,
            "books": requested_books,
        },
        "rules": _filter_books_mapping(
            discovery_manifest.get("books"), requested_books
        ),
        "hybrid_policy": selected_policy,
        "metrics": {
            "no_kill": portfolio_stats,
            "hybrid": hybrid_metrics,
            "per_book_no_kill_edp": per_book_edp,
        },
        "artifacts": {
            "root": destination.relative_to(Path.cwd()).as_posix(),
            "files": _compute_digests(destination),
        },
        "acceptance": {
            "tolerances": {
                "edp_abs": 2.0,
                "pf_abs": 0.03,
                "cvar_abs": 150.0,
                "sharpe_abs": 0.5,
                "maxdrawdown_abs": 150.0,
                "winrate_abs": 0.1,
            },
            "per_book_edp_abs": 3.0,
            "hash_required": True,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_run", type=Path, help="Directory containing pipeline outputs"
    )
    parser.add_argument(
        "golden_dir", type=Path, help="Directory where golden files should live"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/golden_manifest.json"),
        help="Path to the manifest JSON file",
    )
    parser.add_argument(
        "--label", default="golden-local", help="Run label written into the manifest"
    )
    parser.add_argument(
        "--books",
        nargs="+",
        help=(
            "Optional list of books to include when computing metrics and manifest "
            "entries"
        ),
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        help=(
            "Directory containing per-book artifact folders or files. "
            "All discovered book names are added to the --books filter."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_run.resolve()
    destination = args.golden_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    directory_books: list[str] | None = None
    if args.books_dir is not None:
        directory_books = _discover_books_in_directory(args.books_dir)
    refresh_golden_snapshot(
        source,
        destination,
        args.manifest.resolve(),
        run_label=args.label,
        books=_merge_book_sources(args.books, directory_books),
    )


if __name__ == "__main__":
    main()
