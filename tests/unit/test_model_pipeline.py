from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import xdte.model.train as train_module
from xdte.config import Settings
from xdte.data.loaders import PortfolioRecord, read_portfolio_file
from xdte.model.apply import ApplyResult, apply_models
from xdte.model.discovery import discover_rules
from xdte.model.train import train_models

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "backtest"
TRADE_FIXTURES = [
    "portfolio_0dte_call_spreads_10pts.csv",
    "portfolio_0dte_put_spreads_10pts.csv",
]
MARKET_FIXTURES = ["11am_market_stats.csv", "1515_market_stats.csv"]

# Matches ``original_code.txt`` Cell 0 (lines 369-388) feature declarations.
EXPECTED_BOOK_FEATS_CONFIG = {
    "PUTS_0DTE_11": [
        "Intraday_Move_OpenToEntry_11",
        "VIX_Entry_11",
        "L1_TS",
        "L1_vvix_pct",
        "L1_SPX_ATR_Pct",
        "L1_SPX_Drawdown_Pct",
        "DoW",
        "L1_rv20",
        "L1_vvix_above_ema20",
    ],
    "PUTS_1DTE_1515": [
        "Intraday_Move_OpenToEntry_1515",
        "VIX_Entry_1515",
        "t0_VIX_change_from_close_15",
        "L1_TS",
        "L1_vvix_pct",
        "L1_SPX_ATR_Pct",
        "L1_SPX_Drawdown_Pct",
        "DoW",
        "L1_rv20",
        "L1_vvix_above_ema20",
    ],
    "CALLS_0DTE_11": [
        "Intraday_Move_OpenToEntry_11",
        "VIX_Entry_11",
        "t0_VIX_change_from_close_11",
        "L1_TS",
        "L1_SPX_ATR_Pct",
        "L1_VIX_pct",
        "DoW",
    ],
    "CALLS_1DTE_1515": [
        "Intraday_Move_OpenToEntry_1515",
        "VIX_Entry_1515",
        "t0_VIX_change_from_close_15",
        "L1_TS",
        "L1_SPX_ATR_Pct",
        "L1_VIX_pct",
        "DoW",
    ],
}


def test_book_feature_config_matches_notebook() -> None:
    """Ensure the package config never drifts from the Colab reference."""

    assert train_module.BOOK_FEATS_CONFIG == EXPECTED_BOOK_FEATS_CONFIG


def _prepare_daily_context(
    directory: Path, trade_fixture_names: Sequence[str] | None = None
) -> None:
    selected_trades = list(trade_fixture_names or TRADE_FIXTURES)
    records: list[PortfolioRecord] = []
    for name in selected_trades:
        records.extend(read_portfolio_file(FIXTURE_DIR / name))
    dates = sorted({record.open_timestamp.date() for record in records})
    rows = []
    for idx, open_date in enumerate(dates):
        rows.append(
            {
                "open_date": open_date.isoformat(),
                "SPX_Close": 4700 + idx,
                "SPX_High": 4701 + idx,
                "SPX_Low": 4699 + idx,
                "VIX_Close": 15.0 + idx,
                "VIX3M_Close": 18.0 + idx,
                "VVIX_Close": 90.0 + idx,
            }
        )
    fieldnames = [
        "open_date",
        "SPX_Close",
        "SPX_High",
        "SPX_Low",
        "VIX_Close",
        "VIX3M_Close",
        "VVIX_Close",
    ]
    with (directory / "daily_context.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_fixtures(
    destination: Path,
    *,
    trade_fixture_names: Sequence[str] | None = None,
    market_fixture_names: Sequence[str] | None = None,
) -> None:
    selected_trades = list(trade_fixture_names or TRADE_FIXTURES)
    selected_markets = list(market_fixture_names or MARKET_FIXTURES)
    destination.mkdir(parents=True, exist_ok=True)
    for name in selected_trades + selected_markets:
        shutil.copy(FIXTURE_DIR / name, destination / name)
    _prepare_daily_context(destination, trade_fixture_names=selected_trades)


@pytest.fixture
def sparse_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dataset_dir = tmp_path_factory.mktemp("sparse_dataset")
    _copy_fixtures(dataset_dir)

    call_path = dataset_dir / TRADE_FIXTURES[0]
    lines = call_path.read_text(encoding="utf-8").splitlines()
    if len(lines) > 2:
        truncated = lines[:2]
        call_path.write_text("\n".join(truncated) + "\n", encoding="utf-8")

    return dataset_dir


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _copy_fixtures(dataset_dir)

    artifact_dir = tmp_path / "artifacts"
    settings = Settings(N_FOLDS=2, MODEL_SEED=5, WINSOR_P=0.1, ALPHA=0.2)

    train_result = train_models(
        dataset_dir,
        artifact_dir,
        settings=settings,
        books=("CALLS_0DTE_11", "PUTS_0DTE_11"),
        min_train_size=1,
        min_test_size=1,
    )

    assert train_result.artifacts
    for artifact in train_result.artifacts:
        assert artifact.model_path.exists()
        assert artifact.metadata_path.exists()
        assert artifact.train_path.exists()
        assert artifact.test_path.exists()
        assert artifact.discovery_dir.exists()
        assert artifact.train_predictions_path.exists()
        assert artifact.test_predictions_path.exists()
        assert artifact.legacy_train_predictions_path is not None
        assert artifact.legacy_train_predictions_path.exists()
        required_columns = {
            "book",
            "open_date",
            "pnl",
            "score",
            "L1_vvix_above_ema30",
        }
        for prediction_path in (
            artifact.train_predictions_path,
            artifact.test_predictions_path,
        ):
            with prediction_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                assert required_columns.issubset(columns)

    discovery_dir = tmp_path / "discovery"
    discovery_result = discover_rules(artifact_dir, discovery_dir, settings=settings)
    assert discovery_result.folds
    for discovery_fold in discovery_result.folds:
        assert discovery_fold.output_path.exists()
        metadata_path = discovery_fold.output_path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["keep_threshold"] >= metadata["winsor_limits"][0]
        assert metadata["winsor_limits"][1] >= metadata["winsor_limits"][0]
        train_metadata_path = (
            artifact_dir
            / "train"
            / discovery_fold.book
            / f"fold_{discovery_fold.fold:02d}"
            / "metadata.json"
        )
        train_metadata = json.loads(train_metadata_path.read_text())
        assert metadata.get("feature_columns") == train_metadata["feature_columns"]

    daily_context_path = dataset_dir / "daily_context.csv"
    shutil.copy(daily_context_path, artifact_dir / "daily_context.csv")

    apply_dir = tmp_path / "apply"
    apply_result = apply_models(
        artifact_dir, discovery_dir, apply_dir, settings=settings
    )
    assert isinstance(apply_result, ApplyResult)
    assert not apply_result.folds
    assert apply_result.daily_pnl is not None
    assert not apply_result.daily_pnl.index.empty


def test_train_models_warns_for_sparse_book(
    tmp_path: Path, sparse_dataset_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    artifact_dir = tmp_path / "artifacts"
    settings = Settings(N_FOLDS=2, MODEL_SEED=5, WINSOR_P=0.1, ALPHA=0.2)

    caplog.set_level(logging.WARNING, logger="xdte.model.train")

    train_result = train_models(
        sparse_dataset_dir,
        artifact_dir,
        settings=settings,
        books=("CALLS_0DTE_11", "PUTS_0DTE_11"),
        min_train_size=1,
        min_test_size=1,
    )

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == "xdte.model.train"
        and "CALLS_0DTE_11" in record.getMessage()
    ]

    assert warnings
    assert any(artifact.book == "PUTS_0DTE_11" for artifact in train_result.artifacts)


def test_train_models_use_configured_features_and_filter_sessions(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset_configured"
    artifact_dir = tmp_path / "artifacts_configured"
    trade_fixture_names = [
        "portfolio_0dte_call_spreads_10pts.csv",
        "portfolio_0dte_put_spreads_10pts.csv",
        "portfolio_1dte_call_spreads_10pts.csv",
        "portfolio_1dte_put_spreads_10pts.csv",
    ]
    _copy_fixtures(
        dataset_dir,
        trade_fixture_names=trade_fixture_names,
        market_fixture_names=MARKET_FIXTURES,
    )

    for name in trade_fixture_names:
        path = dataset_dir / name
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 26:
            truncated = lines[:26]
            path.write_text("\n".join(truncated) + "\n", encoding="utf-8")

    settings = Settings(N_FOLDS=1, MODEL_SEED=7, WINSOR_P=0.1, ALPHA=0.2)
    books = (
        "CALLS_0DTE_11",
        "PUTS_0DTE_11",
        "CALLS_1DTE_1515",
        "PUTS_1DTE_1515",
    )
    train_result = train_models(
        dataset_dir,
        artifact_dir,
        settings=settings,
        books=books,
        min_train_size=1,
        min_test_size=1,
    )

    assert "VIX_Entry_1515" in train_result.feature_frame.columns
    assert "VIX_Entry_11" in train_result.feature_frame.columns

    metadata_by_book: dict[str, dict[str, object]] = {}
    for artifact in train_result.artifacts:
        payload = json.loads(artifact.metadata_path.read_text())
        metadata_by_book.setdefault(payload["book"], payload)

    manifest_path = artifact_dir / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_books = manifest["books"]

    for book in books:
        assert book in metadata_by_book
        actual_columns = metadata_by_book[book]["feature_columns"]
        assert isinstance(actual_columns, list)
        available = set(train_result.feature_frame.columns)
        expected_config = [
            column
            for column in train_module.BOOK_FEATS_CONFIG[book]
            if column in available
        ]
        expected_columns = train_module._filter_feature_columns_for_book(
            book, expected_config
        )
        assert actual_columns == expected_columns
        assert manifest_books[book]["feature_columns"] == expected_columns

    assert "VIX_Entry_1515" not in metadata_by_book["CALLS_0DTE_11"]["feature_columns"]
    assert "VIX_Entry_11" not in metadata_by_book["CALLS_1DTE_1515"]["feature_columns"]


def test_train_models_use_fallback_features_for_unknown_book(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset_unknown"
    artifact_dir = tmp_path / "artifacts_unknown"
    trade_fixture_names = ["portfolio_strangles.csv"]
    _copy_fixtures(
        dataset_dir,
        trade_fixture_names=trade_fixture_names,
        market_fixture_names=MARKET_FIXTURES,
    )

    settings = Settings(N_FOLDS=1, MODEL_SEED=3, WINSOR_P=0.1, ALPHA=0.2)
    train_result = train_models(
        dataset_dir,
        artifact_dir,
        settings=settings,
        books=("UNKNOWN",),
        min_train_size=1,
        min_test_size=1,
    )

    assert train_result.artifacts
    raw_frame = train_result.feature_frame.reset_index()
    book_frame = raw_frame[raw_frame["book"] == "UNKNOWN"].copy()
    fallback_columns = train_module._prepare_features(book_frame)
    legacy_priority = list(getattr(train_module, "_LEGACY_FEATURE_PRIORITY", ()))
    legacy_columns = [
        column for column in legacy_priority if column in fallback_columns
    ]
    ordered = legacy_columns + fallback_columns
    expected_columns = train_module._filter_feature_columns_for_book(
        "UNKNOWN", list(dict.fromkeys(ordered))
    )
    assert expected_columns

    metadata_columns = {
        artifact.book: json.loads(artifact.metadata_path.read_text())["feature_columns"]
        for artifact in train_result.artifacts
    }
    assert metadata_columns == {"UNKNOWN": expected_columns}

    manifest_path = artifact_dir / "train" / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    assert manifest_payload["books"]["UNKNOWN"]["feature_columns"] == expected_columns


@pytest.mark.skipif(train_module.shap is None, reason="SHAP is not installed")
def test_train_models_emits_shap_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _copy_fixtures(dataset_dir)

    artifact_dir = tmp_path / "artifacts"
    settings = Settings(N_FOLDS=2, MODEL_SEED=5, WINSOR_P=0.1, ALPHA=0.2)
    book = "CALLS_0DTE_11"

    def fake_shap_summary(
        model: object, matrix: Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, int, float | None] | None:
        del model
        if not matrix or not matrix[0]:
            return None
        feature_count = len(matrix[0])
        fold_count = len(matrix)
        values = np.linspace(feature_count, 1.0, feature_count, dtype=float)
        return values * float(fold_count), fold_count, 0.5

    train_module._compute_global_shap_summary  # ensure attribute exists for mypy

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            train_module,
            "_compute_global_shap_summary",
            fake_shap_summary,
            raising=False,
        )
        train_models(
            dataset_dir,
            artifact_dir,
            settings=settings,
            books=(book,),
            min_train_size=1,
            min_test_size=1,
        )

    feature_path = artifact_dir / f"feature_importance_{book}.csv"
    shap_path = artifact_dir / f"shap_summary_{book}.csv"

    assert feature_path.exists()
    assert shap_path.exists()

    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows
    assert [int(row["rank"]) for row in rows] == list(range(1, len(rows) + 1))
    assert {row["method"] for row in rows} == {"shap"}

    feature_values = [float(row["importance"]) for row in rows]
    expected_values = list(np.linspace(len(rows), 1.0, len(rows), dtype=float))
    assert feature_values == expected_values

    with shap_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        shap_rows = list(reader)

    assert shap_rows
    summary_values = [float(row["mean_abs_shap"]) for row in shap_rows]
    assert summary_values == expected_values
    assert {float(row["expected_value"]) for row in shap_rows} == {0.5}


def test_train_models_emits_gain_importance_when_shap_missing(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _copy_fixtures(dataset_dir)

    artifact_dir = tmp_path / "artifacts"
    settings = Settings(N_FOLDS=2, MODEL_SEED=5, WINSOR_P=0.1, ALPHA=0.2)
    book = "CALLS_0DTE_11"

    def fake_shap_summary(
        model: object, matrix: Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, int, float | None] | None:
        del model, matrix
        return None

    def fake_gain_importance(model: object) -> np.ndarray | None:
        del model
        return np.linspace(5.0, 1.0, 5, dtype=float)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            train_module,
            "_compute_global_shap_summary",
            fake_shap_summary,
            raising=False,
        )
        monkeypatch.setattr(
            train_module,
            "_extract_gain_importance",
            fake_gain_importance,
            raising=False,
        )
        train_models(
            dataset_dir,
            artifact_dir,
            settings=settings,
            books=(book,),
            min_train_size=1,
            min_test_size=1,
        )

    feature_path = artifact_dir / f"feature_importance_{book}.csv"
    shap_path = artifact_dir / f"shap_summary_{book}.csv"

    assert feature_path.exists()
    assert not shap_path.exists()

    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows
    assert {row["method"] for row in rows} == {"gain"}
    assert [int(row["rank"]) for row in rows] == list(range(1, len(rows) + 1))
    feature_values = [float(row["importance"]) for row in rows]
    expected_values = list(np.linspace(len(rows), 1.0, len(rows), dtype=float))
    assert feature_values == expected_values
