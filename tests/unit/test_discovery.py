from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

import xdte.model.discovery as discovery_module
from xdte.config import Settings
from xdte.gamma import compute_percentiles
from xdte.model.discovery import (
    DiscoveryResult,
    _cvar,
    _edges_from_scores,
    _pf,
    _safe_bin,
    _winsor,
    discover_rules,
)


def _write_prediction_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book",
        "open_date",
        "pnl",
        "score",
        "L1_vvix_above_ema30",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "book": "BOOK_A",
            "open_date": "2024-01-01",
            "pnl": -100,
            "score": 0.1,
            "L1_vvix_above_ema30": False,
        },
        {
            "book": "BOOK_A",
            "open_date": "2024-01-02",
            "pnl": -10,
            "score": 0.5,
            "L1_vvix_above_ema30": False,
        },
        {
            "book": "BOOK_A",
            "open_date": "2024-01-03",
            "pnl": 10,
            "score": 0.9,
            "L1_vvix_above_ema30": True,
        },
        {
            "book": "BOOK_A",
            "open_date": "2024-01-04",
            "pnl": 100,
            "score": 0.7,
            "L1_vvix_above_ema30": True,
        },
    ]
    _write_prediction_rows(path, rows)


def _expected_put_keep_target(
    rows: Sequence[Mapping[str, object]], winsor_p: float
) -> float:
    frame = pd.DataFrame(rows)
    pnls = frame["pnl"].astype(float).to_numpy()
    scores = frame["score"].astype(float).to_numpy()
    base = discovery_module._winsor(pnls, p=winsor_p)
    base_cvar = discovery_module._cvar(base)
    target = base_cvar + abs(base_cvar) * discovery_module._PUT_CVAR_IMPROVE_FRAC
    grid_rows = []
    for keep in discovery_module._PUT_KEEP_GRID:
        threshold = float(np.nanquantile(scores, 1.0 - keep))
        mask = scores >= threshold
        kept = pnls[mask]
        winsorised = discovery_module._winsor(kept, p=winsor_p)
        grid_rows.append(
            {
                "keep": keep,
                "CVaR95": discovery_module._cvar(winsorised),
                "EDP": float(winsorised.mean()) if len(winsorised) else float("nan"),
                "PF": discovery_module._pf(winsorised),
            }
        )
    table = pd.DataFrame(grid_rows)
    feasible = table[table["CVaR95"] >= target]
    ordered = (feasible if not feasible.empty else table).sort_values(
        ["CVaR95", "EDP", "PF"], ascending=[False, False, False]
    )
    return float(ordered.iloc[0]["keep"])


def _expected_call_deciles(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    winsor_p: float,
    min_decile_days: int,
) -> tuple[list[int], list[int]]:
    train_scores = [float(row["score"]) for row in train_rows]
    edges = discovery_module._edges_from_scores(train_scores)
    assert edges is not None
    test_scores = [float(row["score"]) for row in test_rows]
    test_pnls = [float(row["pnl"]) for row in test_rows]
    deciles = discovery_module._safe_bin(test_scores, edges)
    frame = (
        pd.DataFrame({"dec": deciles, "pnl": test_pnls})
        .dropna(subset=["dec"])
        .astype({"dec": int})
    )
    if frame.empty:
        return [], []
    frame["pnl_w"] = discovery_module._winsor(frame["pnl"], p=winsor_p)
    grouped = (
        frame.groupby("dec")["pnl_w"]
        .agg(N="count", Short_EDP="mean", Long_EDP=lambda s: (-s).mean())
        .reset_index()
    )
    grouped = grouped[grouped["N"] >= min_decile_days]
    if grouped.empty:
        return [], []
    longs = (
        grouped.sort_values("Long_EDP", ascending=False)
        .head(discovery_module._CALLS_TOPK_LONG)["dec"]
        .astype(int)
        .tolist()
    )
    shorts = (
        grouped.sort_values("Short_EDP", ascending=False)
        .head(discovery_module._CALLS_TOPK_SHORT)["dec"]
        .astype(int)
        .tolist()
    )
    return longs, shorts


def test_discover_rules_winsorises_and_records_metadata(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts"
    predictions_dir = train_dir / "discovery" / "BOOK_A" / "fold_00"
    predictions_path = predictions_dir / "train_predictions.csv"
    _write_predictions(predictions_path)

    output_dir = tmp_path / "discovery"
    settings = Settings(WINSOR_P=0.2)

    result = discover_rules(train_dir, output_dir, settings=settings)

    assert isinstance(result, DiscoveryResult)
    assert result.folds
    fold = result.folds[0]
    assert fold.book == "BOOK_A"
    assert fold.fold == 0

    winsorised_path = fold.output_path
    with winsorised_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert [float(row["pnl_winsor"]) for row in rows] == pytest.approx(
        [-46.0, -10.0, 10.0, 46.0]
    )

    metadata_path = winsorised_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["book"] == "BOOK_A"
    assert metadata["fold"] == 0
    assert metadata["winsor_p"] == pytest.approx(0.2)
    assert metadata["keep_threshold"] == pytest.approx(0.78)
    assert metadata["winsor_limits"] == pytest.approx([-46.0, 46.0])
    assert set(metadata["deciles"].keys()) == {f"d{step}" for step in range(1, 11)}
    assert metadata["source"] == "train_predictions.csv"

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_percentiles = [
        [float(percentile), float(score)]
        for percentile, score in compute_percentiles([0.1, 0.5, 0.9, 0.7])
    ]
    assert manifest == {
        "books": {
            "BOOK_A": {
                "winsor_p": 0.2,
                "folds": [
                    {
                        "fold": 0,
                        "keep_threshold": metadata["keep_threshold"],
                        "winsor_limits": metadata["winsor_limits"],
                    }
                ],
                "prediction_percentiles": expected_percentiles,
            }
        }
    }


def test_discover_rules_writes_confidence_intervals(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts"

    puts_rows = [
        {
            "book": "PUTS_0DTE_11",
            "open_date": f"2024-01-{index:02d}",
            "pnl": pnl,
            "score": prediction,
            "L1_vvix_above_ema30": bool(index % 2),
        }
        for index, (pnl, prediction) in enumerate(
            (
                (-2.5, 0.05),
                (-1.0, 0.15),
                (-0.5, 0.4),
                (0.5, 0.6),
                (1.0, 0.7),
                (1.5, 0.8),
            ),
            start=1,
        )
    ]
    calls_rows = [
        {
            "book": "CALLS_0DTE_11",
            "open_date": f"2024-01-{index:02d}",
            "pnl": pnl,
            "score": prediction,
            "L1_vvix_above_ema30": bool(index % 2),
        }
        for index, (pnl, prediction) in enumerate(
            (
                (-3.0, 0.1),
                (-2.0, 0.2),
                (-1.0, 0.3),
                (-0.5, 0.4),
                (0.2, 0.5),
                (0.7, 0.6),
                (1.1, 0.7),
                (1.4, 0.8),
                (1.6, 0.9),
                (2.0, 1.0),
            ),
            start=1,
        )
    ]

    for fold in range(2):
        puts_dir = train_dir / "discovery" / "PUTS_0DTE_11" / f"fold_{fold:02d}"
        calls_dir = train_dir / "discovery" / "CALLS_0DTE_11" / f"fold_{fold:02d}"
        _write_prediction_rows(puts_dir / "train_predictions.csv", puts_rows)
        _write_prediction_rows(calls_dir / "train_predictions.csv", calls_rows)

    output_dir = tmp_path / "discovery"
    settings = Settings(WINSOR_P=0.1, MODEL_SEED=123)

    discover_rules(train_dir, output_dir, settings=settings)

    ci_path = output_dir / "rules_ci.json"
    assert ci_path.exists()
    first_payload = json.loads(ci_path.read_text(encoding="utf-8"))
    assert set(first_payload) == {"CALLS_0DTE_11", "PUTS_0DTE_11"}

    puts_payload = first_payload["PUTS_0DTE_11"]
    assert puts_payload["keep"] == pytest.approx(0.79)
    assert puts_payload["ci"] == pytest.approx([0.58, 0.8])

    calls_payload = first_payload["CALLS_0DTE_11"]
    assert calls_payload == {"long": {"10": 1.0}, "short": {"1": 1.0}}

    discover_rules(train_dir, output_dir, settings=settings)
    second_payload = json.loads(ci_path.read_text(encoding="utf-8"))
    assert second_payload == first_payload


def test_discover_rules_emits_discovered_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(discovery_module, "_MIN_DECILE_DAYS", 1)
    train_dir = tmp_path / "artifacts"

    puts_train_rows = [
        {
            "book": "PUTS_BOOK",
            "open_date": f"2024-01-{index:02d}",
            "pnl": float(index - 3),
            "score": float(index) / 10.0,
            "L1_vvix_above_ema30": False,
        }
        for index in range(1, 8)
    ]
    puts_test_rows = [
        {
            "book": "PUTS_BOOK",
            "open_date": f"2024-02-{index:02d}",
            "pnl": float(value),
            "score": float(score),
            "L1_vvix_above_ema30": False,
        }
        for index, (value, score) in enumerate(
            zip(
                [-8, -6, -5, -4, -2, -1, 1, 2, 3, 4, 5, 6],
                [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            ),
            start=1,
        )
    ]

    calls_train_rows = [
        {
            "book": "CALLS_BOOK",
            "open_date": f"2024-03-{index:02d}",
            "pnl": float(index - 6),
            "score": float(index) / 10.0,
            "L1_vvix_above_ema30": bool(index % 2),
        }
        for index in range(1, 15)
    ]
    calls_test_rows = [
        {
            "book": "CALLS_BOOK",
            "open_date": f"2024-04-{index:02d}",
            "pnl": float(value),
            "score": float(score),
            "L1_vvix_above_ema30": bool(index % 2),
        }
        for index, (value, score) in enumerate(
            zip(
                [-3, -2, -1, -0.5, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98],
            ),
            start=1,
        )
    ]

    puts_dir = train_dir / "discovery" / "PUTS_BOOK" / "fold_01"
    calls_dir = train_dir / "discovery" / "CALLS_BOOK" / "fold_01"
    _write_prediction_rows(puts_dir / "train_predictions.csv", puts_train_rows)
    _write_prediction_rows(puts_dir / "test_predictions.csv", puts_test_rows)
    _write_prediction_rows(calls_dir / "train_predictions.csv", calls_train_rows)
    _write_prediction_rows(calls_dir / "test_predictions.csv", calls_test_rows)

    output_dir = tmp_path / "discovery"
    settings = Settings(WINSOR_P=0.05)

    discover_rules(train_dir, output_dir, settings=settings)

    rules_path = output_dir / "discovered_rules.json"
    assert rules_path.exists()
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    assert set(payload) == {"CALLS_BOOK", "PUTS_BOOK"}

    expected_keep = _expected_put_keep_target(puts_test_rows, settings.WINSOR_P)
    assert payload["PUTS_BOOK"]["keep_target"] == pytest.approx(expected_keep)

    expected_longs, expected_shorts = _expected_call_deciles(
        calls_train_rows,
        calls_test_rows,
        settings.WINSOR_P,
        discovery_module._MIN_DECILE_DAYS,
    )
    assert payload["CALLS_BOOK"]["long_deciles"] == expected_longs
    assert payload["CALLS_BOOK"]["short_deciles"] == expected_shorts


def test_winsor_matches_nanpercentile_clipping() -> None:
    values = [10.0, -10.0, 0.0, np.nan, 5.0]
    result = _winsor(values, p=0.2)
    series = pd.Series(values)
    lo, hi = np.nanpercentile(series, [20, 80])
    expected = series.clip(lo, hi)
    pd.testing.assert_series_equal(result, expected)


def test_pf_handles_gains_losses_and_nan() -> None:
    values = [2.0, -1.0, np.nan, -4.0]
    assert _pf(values) == pytest.approx(2.0 / 5.0)
    assert math.isinf(_pf([1.0, 2.0]))
    assert math.isnan(_pf([np.nan, np.nan]))


def test_cvar_averages_worst_tail() -> None:
    values = [-5.0, -1.0, 3.0, 4.0]
    assert _cvar(values, alpha=0.5) == pytest.approx(-3.0)


def test_edges_from_scores_returns_edges_when_enough_data() -> None:
    scores = list(range(10))
    edges = _edges_from_scores(scores, n=4)
    assert edges is not None
    assert edges[0] == pytest.approx(min(scores))
    assert edges[-1] == pytest.approx(max(scores))


def test_edges_from_scores_handles_sparse_or_flat_scores() -> None:
    sparse = [1.0, np.nan, 2.0, np.nan]
    assert _edges_from_scores(sparse) is None

    flat = [5.0] * 10
    assert _edges_from_scores(flat) is None


def test_safe_bin_returns_nan_series_when_edges_missing() -> None:
    scores = [0.1, 0.5, 0.9]
    result = _safe_bin(scores, None)
    assert result.isna().all()


def test_safe_bin_matches_pandas_cut() -> None:
    scores = pd.Series([0.1, 0.5, 0.9])
    edges = np.array([0.1, 0.4, 0.7, 1.0])
    result = _safe_bin(scores, edges)
    expected = pd.Series(pd.cut(scores, edges, labels=False, include_lowest=True))
    pd.testing.assert_series_equal(result, expected)
