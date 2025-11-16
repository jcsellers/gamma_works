"""Utilities for validating frozen backtest datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


class BacktestDataValidationError(ValueError):
    """Raised when the backtest dataset fails immutability validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - ensure consistent message format
        return self.message


def _compute_sha256(path: Path) -> str:
    """Compute a stable SHA-256 digest for frozen artifacts.

    Windows users with ``core.autocrlf=true`` may end up with ``\r\n`` line
    endings even though the canonical artifacts are stored with ``\n``.  To keep
    the validator portable we normalize CSV files to LF before hashing so that
    ``xdte validate-backtest`` succeeds regardless of the local newline style.
    Other file types are still hashed byte-for-byte to avoid masking real
    corruption.
    """

    hasher = hashlib.sha256()
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline=None) as handle:
            for text_chunk in iter(lambda: handle.read(1024 * 1024), ""):
                if not text_chunk:
                    break
                hasher.update(text_chunk.encode("utf-8"))
    else:
        with path.open("rb") as handle:
            for binary_chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not binary_chunk:
                    break
                hasher.update(binary_chunk)
    return hasher.hexdigest()


def _load_sources(manifest_path: Path) -> Mapping[str, Mapping[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise BacktestDataValidationError("Manifest is missing a 'sources' mapping")
    return sources


def _unexpected_files(data_dir: Path, expected: Iterable[str]) -> list[str]:
    expected_set = set(expected)
    unexpected = [
        path.name
        for path in sorted(data_dir.glob("*.csv"))
        if path.name not in expected_set
    ]
    return unexpected


def validate_backtest_directory(data_dir: Path, manifest_path: Path) -> None:
    """Validate that ``data_dir`` matches the digests declared in the manifest."""

    sources = _load_sources(manifest_path)
    missing: list[str] = []
    mismatched: list[str] = []

    for name, metadata in sources.items():
        file_path = data_dir / name
        expected_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        if expected_digest is None or not isinstance(expected_digest, str):
            raise BacktestDataValidationError(
                f"Manifest entry for {name} is missing a sha256 digest"
            )
        if not file_path.exists():
            missing.append(name)
            continue
        actual_digest = _compute_sha256(file_path)
        if actual_digest != expected_digest:
            mismatched.append(name)

    unexpected = _unexpected_files(data_dir, sources.keys())

    if missing or mismatched or unexpected:
        parts: list[str] = []
        if missing:
            parts.append("missing: " + ", ".join(sorted(missing)))
        if mismatched:
            parts.append("mismatched: " + ", ".join(sorted(mismatched)))
        if unexpected:
            parts.append("unexpected: " + ", ".join(unexpected))
        detail = "; ".join(parts)
        raise BacktestDataValidationError(f"Backtest data failed validation; {detail}")


__all__ = [
    "BacktestDataValidationError",
    "validate_backtest_directory",
]
