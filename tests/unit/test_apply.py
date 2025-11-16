from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pytest

from xdte.config import Settings
from xdte.model.apply import (
    BOOKS,
    KILL_SWITCH_COLUMN,
    ApplyResult,
    _load_fold_artifacts,
    _load_trading_days,
    _train_edges,
    apply_models,
)


def _write_prediction_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_daily_context(
    path: Path, dates: Iterable[str], *, kill_flags: Iterable[bool] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    date_sequence = list(dates)
    flags = list(kill_flags or [])
    if len(flags) < len(date_sequence):
        flags.extend([False] * (len(date_sequence) - len(flags)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["open_date", KILL_SWITCH_COLUMN])
        writer.writeheader()
        for index, date in enumerate(date_sequence):
            flag = bool(flags[index]) if index < len(flags) else False
            writer.writerow({"open_date": date, KILL_SWITCH_COLUMN: flag})


def _prepare_fold_artifacts(base_dir: Path, book: str) -> None:
    train_rows = [
        {"book": book, "open_date": "2024-01-01", "pnl": 10.0, "score": 0.6},
        {"book": book, "open_date": "2024-01-02", "pnl": -5.0, "score": 0.3},
    ]
    test_rows = [
        {"book": book, "open_date": "2024-01-03", "pnl": 8.0, "score": 0.7},
        {"book": book, "open_date": "2024-01-04", "pnl": -2.0, "score": 0.2},
    ]
    fold_dir = base_dir / "discovery" / book / "fold_00"
    _write_prediction_rows(fold_dir / "train_predictions.csv", train_rows)
    _write_prediction_rows(fold_dir / "test_predictions.csv", test_rows)


def test_train_edges_uses_fallback_when_scores_sparse() -> None:
    scores = np.array([0.0, 0.0, 1.0])
    edges = _train_edges(scores, n=4)
    assert len(edges) >= 3
    assert pytest.approx(edges[0]) == 0.0
    assert pytest.approx(edges[-1]) == 1.0


def test_load_fold_artifacts_reads_prediction_series(tmp_path: Path) -> None:
    _prepare_fold_artifacts(tmp_path, "PUTS_0DTE_11")
    artifacts = _load_fold_artifacts(tmp_path)
    assert "PUTS_0DTE_11" in artifacts
    payload = artifacts["PUTS_0DTE_11"][0]
    train_scores = payload["train_scores"]
    test_dates = payload["test_dates"]
    assert isinstance(train_scores, pd.Series)
    assert isinstance(test_dates, pd.Series)
    assert list(train_scores.index.astype(str)) == ["2024-01-01", "2024-01-02"]
    assert list(test_dates.astype(str)) == ["2024-01-03", "2024-01-04"]


def test_load_trading_days_falls_back_to_predictions(tmp_path: Path) -> None:
    _prepare_fold_artifacts(tmp_path, "PUTS_0DTE_11")
    artifacts = _load_fold_artifacts(tmp_path)
    days = _load_trading_days(tmp_path, artifacts)
    expected = [
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-02"),
        np.datetime64("2024-01-03"),
        np.datetime64("2024-01-04"),
    ]
    assert list(days.astype("datetime64[D]")) == expected


def test_apply_models_initialises_daily_frame_and_defaults(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts"
    discovery_dir = tmp_path / "discovery"
    output_dir = tmp_path / "apply"
    _prepare_fold_artifacts(train_dir, "PUTS_0DTE_11")
    _write_daily_context(train_dir / "daily_context.csv", ["2024-01-01", "2024-01-02"])

    discovery_dir.mkdir(parents=True, exist_ok=True)
    rules_path = discovery_dir / "discovered_rules.json"
    rules = {"CALLS_1DTE_1515": {"type": "other"}}
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    result = apply_models(train_dir, discovery_dir, output_dir, settings=Settings())

    assert isinstance(result, ApplyResult)
    assert result.daily_pnl is not None
    assert list(result.daily_pnl.index.astype(str)) == ["2024-01-01", "2024-01-02"]
    assert "CALLS_1DTE_1515" in result.daily_pnl.columns
    assert (result.daily_pnl["CALLS_1DTE_1515"] == 0.0).all()


def test_apply_models_writes_hybrid_input_with_kill_switch(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts"
    discovery_dir = tmp_path / "discovery"
    output_dir = tmp_path / "apply"
    _prepare_fold_artifacts(train_dir, "PUTS_0DTE_11")
    _write_daily_context(
        train_dir / "daily_context.csv",
        ["2024-01-01", "2024-01-02"],
        kill_flags=[True, False],
    )

    discovery_dir.mkdir(parents=True, exist_ok=True)
    rules_path = discovery_dir / "discovered_rules.json"
    rules = {"PUTS_0DTE_11": {"type": "filter", "keep_target": 1.0}}
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    result = apply_models(train_dir, discovery_dir, output_dir, settings=Settings())

    assert result.daily_pnl is not None
    assert KILL_SWITCH_COLUMN in result.daily_pnl.columns

    hybrid_path = output_dir / "hybrid_input.csv"
    assert hybrid_path.exists()

    frame = pd.read_csv(hybrid_path)
    for book in BOOKS:
        assert book in frame.columns
    assert "Gated_Portfolio_Base" in frame.columns
    assert KILL_SWITCH_COLUMN in frame.columns
    assert frame[KILL_SWITCH_COLUMN].tolist() == [True, False]

    predictions_path = output_dir / "hybrid_predictions.csv"
    assert predictions_path.exists()
    predictions_frame = pd.read_csv(predictions_path)
    for column in ["book", "open_date", "pnl", "prediction", "fold", "keep"]:
        assert column in predictions_frame.columns
