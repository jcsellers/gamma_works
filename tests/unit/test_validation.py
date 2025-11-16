from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, TypedDict, cast

import pytest

from xdte.data.validation import (
    BacktestDataValidationError,
    validate_backtest_directory,
)


class _SourceEntry(TypedDict):
    sha256: str
    size: int


class _ManifestData(TypedDict):
    fixtures: Dict[str, Dict[str, object]]
    sources: Dict[str, _SourceEntry]


@pytest.fixture(scope="module")
def backtest_manifest() -> Tuple[Path, _ManifestData]:
    manifest_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "backtest" / "manifest.json"
    )
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = cast(_ManifestData, manifest_raw)
    return manifest_path, manifest


@pytest.fixture()
def backtest_bundle(
    tmp_path: Path, backtest_manifest: Tuple[Path, _ManifestData]
) -> Tuple[Path, Path, _ManifestData]:
    manifest_path, manifest = backtest_manifest
    data_dir = tmp_path / "bundle"
    data_dir.mkdir()
    source_dir = Path("data/backtest_data/original")
    for name in manifest["sources"].keys():
        (data_dir / name).write_bytes((source_dir / name).read_bytes())
    return data_dir, manifest_path, manifest


def test_validate_backtest_directory_accepts_clean_bundle(
    backtest_bundle: Tuple[Path, Path, _ManifestData],
) -> None:
    data_dir, manifest_path, _ = backtest_bundle
    validate_backtest_directory(data_dir, manifest_path)


def test_validate_backtest_directory_detects_missing_file(
    backtest_bundle: Tuple[Path, Path, _ManifestData],
) -> None:
    data_dir, manifest_path, manifest = backtest_bundle
    missing_name = next(iter(manifest["sources"]))
    (data_dir / missing_name).unlink()

    with pytest.raises(BacktestDataValidationError, match="missing"):
        validate_backtest_directory(data_dir, manifest_path)


def test_validate_backtest_directory_detects_mismatched_file(
    backtest_bundle: Tuple[Path, Path, _ManifestData],
) -> None:
    data_dir, manifest_path, manifest = backtest_bundle
    mismatched_name = next(iter(manifest["sources"]))
    (data_dir / mismatched_name).write_bytes(b"corrupted data")

    with pytest.raises(BacktestDataValidationError, match="mismatched"):
        validate_backtest_directory(data_dir, manifest_path)


def test_validate_backtest_directory_allows_crlf_backups(
    backtest_bundle: Tuple[Path, Path, _ManifestData],
) -> None:
    data_dir, manifest_path, manifest = backtest_bundle
    sample_name = next(iter(manifest["sources"]))
    sample_path = data_dir / sample_name
    original = sample_path.read_text(encoding="utf-8")
    sample_path.write_text(original.replace("\n", "\r\n"), encoding="utf-8", newline="")

    validate_backtest_directory(data_dir, manifest_path)
