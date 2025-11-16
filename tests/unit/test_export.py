from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from xdte.config import Settings
from xdte.data.features import BOOK_FEATS_CONFIG
from xdte.data.loaders import PortfolioRecord, read_portfolio_file
from xdte.model.discovery import discover_rules
from xdte.model.export import build_live_kit
from xdte.model.hybrid import select_hybrid_policy
from xdte.model.train import train_models

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "backtest"
TRADE_FIXTURES = [
    "portfolio_0dte_call_spreads_10pts.csv",
    "portfolio_0dte_put_spreads_10pts.csv",
]
MARKET_FIXTURES = ["11am_market_stats.csv"]


def _prepare_daily_context(directory: Path) -> None:
    records: list[PortfolioRecord] = []
    for name in TRADE_FIXTURES:
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


def _copy_fixtures(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in TRADE_FIXTURES + MARKET_FIXTURES:
        shutil.copy(FIXTURE_DIR / name, destination / name)
    _prepare_daily_context(destination)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def test_build_live_kit_emits_manifest_and_digests(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _copy_fixtures(dataset_dir)

    artifact_dir = tmp_path / "artifacts"
    discovery_dir = tmp_path / "discovery"
    apply_dir = tmp_path / "apply"

    settings = Settings(N_FOLDS=2, MODEL_SEED=5, WINSOR_P=0.1, ALPHA=0.2)

    train_models(dataset_dir, artifact_dir, settings=settings)

    feature_sources = sorted(artifact_dir.glob("feature_importance_*.csv"))
    assert feature_sources
    shap_sources: list[Path] = []
    for feature_source in feature_sources:
        shap_source = artifact_dir / feature_source.name.replace(
            "feature_importance_", "shap_summary_"
        )
        if not shap_source.exists():
            shap_source.write_text(
                "feature,mean_abs_shap,expected_value\nfeature_0,1.0,0.5\n",
                encoding="utf-8",
            )
        shap_sources.append(shap_source)
    discover_rules(artifact_dir, discovery_dir, settings=settings)

    hybrid_rows = [
        {
            "book": "PUTS_0DTE_11",
            "open_date": "2024-01-01",
            "pnl": 10.0,
            "prediction": 0.6,
            "keep": True,
        },
        {
            "book": "PUTS_0DTE_11",
            "open_date": "2024-01-02",
            "pnl": -4.0,
            "prediction": 0.4,
            "keep": True,
        },
    ]
    apply_dir.mkdir(parents=True, exist_ok=True)
    hybrid_path = apply_dir / "hybrid_input.csv"
    with hybrid_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["book", "open_date", "pnl", "prediction", "keep"]
        )
        writer.writeheader()
        writer.writerows(hybrid_rows)

    hybrid_result = select_hybrid_policy(
        apply_dir,
        {"tails": {}, "gammas": {}, "gamma_rules": {}},
        tmp_path / "hybrid" / "policy.json",
        settings=settings,
    )

    assert hybrid_result.accepted

    kit_dir = tmp_path / "kit"
    export_result = build_live_kit(
        artifact_dir,
        discovery_dir,
        hybrid_result.output_path,
        kit_dir,
        settings=settings,
        version="ci-test",
    )

    assert export_result.manifest_path.exists()
    assert export_result.digests_path.exists()

    manifest = json.loads(export_result.manifest_path.read_text())
    assert manifest["version"] == "ci-test"
    paths = [entry["path"] for entry in manifest["files"]]
    assert "features.json" in paths
    assert hybrid_result.output_path.name in paths
    assert all(
        path.startswith("models/") for path in paths if path.endswith("model.pkl")
    )

    digest_payload = json.loads(export_result.digests_path.read_text())
    assert digest_payload["files"] == export_result.digests

    hybrid_copy = kit_dir / hybrid_result.output_path.name
    assert hybrid_copy.exists()
    assert export_result.digests[hybrid_result.output_path.name] == _sha256(hybrid_copy)

    features_config = json.loads((kit_dir / "features.json").read_text())
    assert features_config == BOOK_FEATS_CONFIG

    live_kit_path = kit_dir / "live_kit.json"
    assert live_kit_path.exists()
    live_payload = json.loads(live_kit_path.read_text())
    assert set(live_payload.keys()) == {"calls", "hybrid_policy", "meta", "puts"}
    assert live_payload["hybrid_policy"] == json.loads(
        hybrid_copy.read_text(encoding="utf-8")
    )
    assert "features_by_book" in live_payload["meta"]
    assert live_payload["meta"]["features_by_book"]["PUTS_0DTE_11"]

    rules_payload = json.loads(
        (discovery_dir / "discovered_rules.json").read_text(encoding="utf-8")
    )
    puts_rules = rules_payload["PUTS_0DTE_11"]
    calls_rules = rules_payload["CALLS_0DTE_11"]

    assert math.isclose(
        live_payload["puts"]["PUTS_0DTE_11"]["keep"],
        puts_rules["keep_target"],
        rel_tol=1e-9,
    )
    assert (
        live_payload["calls"]["CALLS_0DTE_11"]["long_deciles"]
        == calls_rules["long_deciles"]
    )
    assert (
        live_payload["calls"]["CALLS_0DTE_11"]["short_deciles"]
        == calls_rules["short_deciles"]
    )
    assert live_payload["calls"]["CALLS_0DTE_11"]["edges"]

    for book in ["PUTS_0DTE_11", "CALLS_0DTE_11"]:
        model_path = kit_dir / "models" / f"{book}_model.pkl"
        assert model_path.exists()
        relative = f"models/{book}_model.pkl"
        assert export_result.digests[relative] == _sha256(model_path)
