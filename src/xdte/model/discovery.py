"""Rule discovery helpers built on top of the training artefacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from random import Random
from typing import (
    Collection,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xdte.config import Settings, get_settings
from xdte.gamma import compute_percentiles


@dataclass(frozen=True)
class FoldDiscovery:
    """Winsorisation and threshold metadata for a single fold."""

    book: str
    fold: int
    keep_threshold: float
    deciles: Mapping[str, float]
    winsor_limits: tuple[float, float]
    source_path: Path
    output_path: Path


@dataclass(frozen=True)
class DiscoveryResult:
    """Container returned by :func:`discover_rules`."""

    folds: List[FoldDiscovery]
    output_dir: Path


__all__ = ["FoldDiscovery", "DiscoveryResult", "discover_rules"]


_ROUND_PLACES = 6
_BOOTSTRAP_ITERATIONS = 200
_CALLS_TOP_K = 1
_PUT_KEEP_GRID = [float(value) for value in np.round(np.linspace(0.60, 0.90, 7), 2)]
_PUT_CVAR_IMPROVE_FRAC = 0.10
_MIN_DECILE_DAYS = 20
_CALLS_TOPK_LONG = 2
_CALLS_TOPK_SHORT = 1


def _round_float(value: float) -> float:
    return float(round(value, _ROUND_PLACES))


def _winsor(series: Sequence[float] | pd.Series, p: float | None = None) -> pd.Series:
    """Winsorise a sequence using the percentile cutoffs from the notebook spec."""

    resolved = pd.Series(series)
    if len(resolved) == 0 or resolved.isna().all():
        return resolved
    percentile = get_settings().WINSOR_P if p is None else p
    resolved_values = resolved.to_numpy(dtype=float, copy=False)
    percentiles = cast(
        NDArray[np.float64],
        np.nanpercentile(
            resolved_values,
            np.array([100 * percentile, 100 * (1 - percentile)], dtype=float),
        ),
    )
    lo = float(percentiles[0])
    hi = float(percentiles[1])
    return resolved.clip(lo, hi)


def _pf(values: Sequence[float]) -> float:
    """Compute the profit factor from a sequence of PnL values."""

    series = pd.Series(values).dropna()
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    if losses > 0:
        return float(gains / losses)
    if gains > 0:
        return float("inf")
    return float(np.nan)


def _cvar(values: Sequence[float], alpha: float = 0.05) -> float:
    """Compute the CVaR using the notebook's implementation."""

    vector = pd.Series(values).dropna().values
    if vector.size == 0:
        return float(np.nan)
    k = max(1, int(np.floor(alpha * len(vector))))
    idx = np.argsort(vector)
    return float(np.mean(vector[idx[:k]]))


def _edges_from_scores(arr: Sequence[float], n: int = 10) -> NDArray[np.float64] | None:
    """Derive histogram edges mirroring the notebook logic."""

    series = pd.Series(arr).astype(float)
    if series.notna().sum() < 5:
        return None
    dropna = series.dropna().values
    edges = np.unique(np.nanpercentile(dropna, np.linspace(0, 100, n + 1)))
    if len(edges) < 3:
        edges = np.unique(np.nanpercentile(dropna, np.linspace(0, 100, 6)))
    return edges if len(edges) >= 3 else None


def _safe_bin(scores: Sequence[float], edges: NDArray[np.float64] | None) -> pd.Series:
    """Assign decile labels using the notebook's safety checks."""

    if edges is None:
        return pd.Series(np.nan, index=pd.RangeIndex(len(scores)))
    return pd.Series(pd.cut(scores, edges, labels=False, include_lowest=True))


def _coerce_numeric(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _serialise_value(value: object) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: List[str] = []
    seen = {"book", "open_date", "pnl"}
    for key in ("book", "open_date", "pnl"):
        if any(key in row for row in rows):
            columns.append(key)
    extra_keys: List[str] = []
    for row in rows:
        for key in row:
            if key in seen or key in columns:
                continue
            extra_keys.append(key)
    columns.extend(sorted(dict.fromkeys(extra_keys)))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised = {
                column: _serialise_value(row.get(column, "")) for column in columns
            }
            writer.writerow(serialised)


def _load_feature_columns(
    train_dir: Path, book: str, fold_dir_name: str, predictions_path: Path
) -> List[str]:
    """Load the feature column list recorded during training for the fold."""

    candidate_dirs = [predictions_path.parent]
    train_fold_dir = train_dir / "train" / book / fold_dir_name
    if train_fold_dir not in candidate_dirs:
        candidate_dirs.append(train_fold_dir)

    for directory in candidate_dirs:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        feature_columns_raw = payload.get("feature_columns")
        if isinstance(feature_columns_raw, Sequence):
            return [str(column) for column in feature_columns_raw]
    return []


def _iter_prediction_directories(
    root: Path, *, allowed_books: set[str] | None
) -> Iterator[tuple[str, Path]]:
    if not root.exists():
        return
    for book_dir in sorted(root.iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name
        if allowed_books is not None and book not in allowed_books:
            continue
        for fold_dir in sorted(book_dir.iterdir()):
            if not fold_dir.is_dir():
                continue
            predictions = fold_dir / "train_predictions.csv"
            if predictions.exists():
                yield book, predictions


def _iter_train_predictions(
    train_dir: Path, *, books: Collection[str] | None = None
) -> Iterator[tuple[str, Path]]:
    allowed = {str(book) for book in books} if books is not None else None
    discovery_root = train_dir / "discovery"
    yielded = False
    for entry in _iter_prediction_directories(discovery_root, allowed_books=allowed):
        yielded = True
        yield entry
    if yielded:
        return
    train_root = train_dir / "train"
    yield from _iter_prediction_directories(train_root, allowed_books=allowed)


def _load_prediction_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if key == "open_date" and value:
                    parsed[key] = datetime.fromisoformat(value)
                elif key in {"pnl", "prediction", "score"}:
                    parsed[key] = _coerce_numeric(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def _extract_prediction_value(row: Mapping[str, object]) -> float:
    score_value = _coerce_numeric(row.get("score"))
    if not math.isnan(score_value):
        return score_value
    return _coerce_numeric(row.get("prediction"))


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float((1 - weight) * ordered[lower] + weight * ordered[upper])


def _bootstrap_keep_threshold_ci(
    predictions: Sequence[float],
    winsor_p: float,
    alpha: float,
    rng: Random,
    *,
    iterations: int = _BOOTSTRAP_ITERATIONS,
) -> tuple[float, tuple[float, float]] | None:
    population = [value for value in predictions if not math.isnan(value)]
    size = len(population)
    if size == 0:
        return None

    keep_threshold = _quantile(list(population), 1.0 - winsor_p)

    if size == 1:
        rounded = _round_float(keep_threshold)
        return rounded, (rounded, rounded)

    samples: List[float] = []
    for _ in range(iterations):
        sample = [population[rng.randrange(size)] for _ in range(size)]
        samples.append(_quantile(sample, 1.0 - winsor_p))

    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    lower = _quantile(samples, lower_q)
    upper = _quantile(samples, upper_q)
    return _round_float(keep_threshold), (
        _round_float(lower),
        _round_float(upper),
    )


def _bootstrap_call_decile_stability(
    rows: Sequence[tuple[float, float]],
    winsor_p: float,
    rng: Random,
    *,
    iterations: int = _BOOTSTRAP_ITERATIONS,
    top_k: int = _CALLS_TOP_K,
) -> Dict[str, Dict[str, float]]:
    data = [row for row in rows if not any(math.isnan(value) for value in row)]
    size = len(data)
    if size == 0 or top_k <= 0:
        return {}

    long_counts: DefaultDict[int, int] = defaultdict(int)
    short_counts: DefaultDict[int, int] = defaultdict(int)
    effective = 0

    for _ in range(iterations):
        sample = [data[rng.randrange(size)] for _ in range(size)]
        pnls = [row[0] for row in sample]
        if not pnls:
            continue
        lower = _quantile(pnls, winsor_p)
        upper = _quantile(pnls, 1.0 - winsor_p)

        scored = sorted(sample, key=lambda item: item[1])
        n_scored = len(scored)
        if n_scored == 0:
            continue

        decile_values: DefaultDict[int, List[float]] = defaultdict(list)
        for index, (pnl, _) in enumerate(scored):
            decile = min(10, int(math.floor(index * 10 / n_scored)) + 1)
            winsorised = min(max(pnl, lower), upper)
            decile_values[decile].append(winsorised)

        if not decile_values:
            continue

        means: List[tuple[int, float]] = [
            (decile, sum(values) / len(values))
            for decile, values in decile_values.items()
        ]
        if not means:
            continue

        effective += 1
        means.sort(key=lambda item: (item[1], item[0]))
        for decile, _ in means[:top_k]:
            short_counts[decile] += 1
        for decile, _ in reversed(means[-top_k:]):
            long_counts[decile] += 1

    if effective == 0:
        return {}

    long_payload = {
        str(decile): _round_float(long_counts[decile] / effective)
        for decile in sorted(long_counts)
    }
    short_payload = {
        str(decile): _round_float(short_counts[decile] / effective)
        for decile in sorted(short_counts)
    }

    payload: Dict[str, Dict[str, float]] = {}
    if long_payload:
        payload["long"] = long_payload
    if short_payload:
        payload["short"] = short_payload
    return payload


def _compute_confidence_intervals(
    predictions: Mapping[str, Sequence[float]],
    call_rows: Mapping[str, Sequence[tuple[float, float]]],
    *,
    winsor_p: float,
    alpha: float,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    books = sorted(set(predictions) | set(call_rows))
    results: Dict[str, Dict[str, object]] = {}
    for index, book in enumerate(books):
        payload: Dict[str, object] = {}
        keep_rng = Random(seed + index * 2)
        call_rng = Random(seed + index * 2 + 1)

        if book.startswith("PUTS"):
            keep_data = _bootstrap_keep_threshold_ci(
                predictions.get(book, ()), winsor_p, alpha, keep_rng
            )
            if keep_data is not None:
                keep_value, (lower, upper) = keep_data
                payload["keep"] = keep_value
                payload["ci"] = [lower, upper]

        if book.startswith("CALLS"):
            stability = _bootstrap_call_decile_stability(
                call_rows.get(book, ()), winsor_p, call_rng
            )
            if stability:
                payload.update(stability)

        if payload:
            results[book] = payload
    return results


def _discover_put_rules(
    books: Collection[str],
    oos_rows: Mapping[str, Sequence[tuple[float, float]]],
    *,
    winsor_p: float,
) -> Dict[str, Dict[str, float]]:
    """Run the PUT_KEEP grid search using pooled OOS predictions."""

    rules: Dict[str, Dict[str, float]] = {}
    for book in sorted(books):
        entries = oos_rows.get(book, ())
        if not entries:
            rules[book] = {"keep_target": 1.0}
            continue

        pnl_values = [pnl for pnl, _ in entries]
        base_winsor = _winsor(pnl_values, p=winsor_p)
        base_cvar = _cvar(base_winsor)
        improve_target = base_cvar + abs(base_cvar) * _PUT_CVAR_IMPROVE_FRAC

        scores = np.array([score for _, score in entries], dtype=float)
        pnls = np.array(pnl_values, dtype=float)
        grid_rows: List[Dict[str, float]] = []
        for keep_target in _PUT_KEEP_GRID:
            if scores.size == 0 or pnls.size == 0:
                continue
            try:
                threshold = float(np.nanquantile(scores, 1.0 - keep_target))
            except ValueError:
                continue
            keep_mask = scores >= threshold
            kept = pnls[keep_mask]
            winsorised = _winsor(kept, p=winsor_p)
            edp = float(winsorised.mean()) if not winsorised.empty else float("nan")
            grid_rows.append(
                {
                    "keep": float(keep_target),
                    "EDP": edp,
                    "PF": _pf(winsorised),
                    "CVaR95": _cvar(winsorised),
                }
            )

        if not grid_rows:
            rules[book] = {"keep_target": 1.0}
            continue

        frame = pd.DataFrame(grid_rows)
        feasible = frame[frame["CVaR95"] >= improve_target]
        ordered = (feasible if not feasible.empty else frame).sort_values(
            ["CVaR95", "EDP", "PF"], ascending=[False, False, False]
        )
        choice = ordered.iloc[0]
        rules[book] = {"keep_target": float(choice["keep"])}

    return rules


def _discover_call_rules(
    books: Collection[str],
    decile_rows: Mapping[str, Sequence[tuple[int, float]]],
    *,
    winsor_p: float,
) -> Dict[str, Dict[str, List[int]]]:
    """Replicate the decile sweep from the notebook for CALLS books."""

    rules: Dict[str, Dict[str, List[int]]] = {}
    for book in sorted(books):
        entries = decile_rows.get(book, ())
        if not entries:
            rules[book] = {"long_deciles": [], "short_deciles": []}
            continue

        frame = pd.DataFrame(entries, columns=["decile", "pnl"])
        if frame.empty:
            rules[book] = {"long_deciles": [], "short_deciles": []}
            continue

        frame["pnl_w"] = _winsor(frame["pnl"], p=winsor_p)
        grouped = (
            frame.groupby("decile", dropna=False)["pnl_w"]
            .agg(N="count", Short_EDP="mean", Long_EDP=lambda s: (-s).mean())
            .reset_index()
        )
        grouped = grouped.dropna(subset=["decile"])
        if grouped.empty:
            rules[book] = {"long_deciles": [], "short_deciles": []}
            continue

        grouped["decile"] = grouped["decile"].astype(int)
        filtered = grouped[grouped["N"] >= _MIN_DECILE_DAYS]
        if filtered.empty:
            rules[book] = {"long_deciles": [], "short_deciles": []}
            continue

        long_deciles = (
            filtered.sort_values("Long_EDP", ascending=False)
            .head(_CALLS_TOPK_LONG)["decile"]
            .astype(int)
            .tolist()
        )
        short_deciles = (
            filtered.sort_values("Short_EDP", ascending=False)
            .head(_CALLS_TOPK_SHORT)["decile"]
            .astype(int)
            .tolist()
        )

        rules[book] = {
            "long_deciles": long_deciles,
            "short_deciles": short_deciles,
        }

    return rules


def discover_rules(
    train_dir: Path,
    output_dir: Path,
    *,
    settings: Settings | None = None,
    books: Collection[str] | None = None,
) -> DiscoveryResult:
    """Consume the fold training predictions and emit rule metadata."""

    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    active_settings = settings or get_settings()
    winsor_p = active_settings.WINSOR_P
    allowed_books = {str(book) for book in books} if books is not None else None

    folds: List[FoldDiscovery] = []
    summary: Dict[str, Dict[str, object]] = {}
    book_predictions: DefaultDict[str, List[float]] = defaultdict(list)
    book_call_rows: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    put_books: set[str] = set()
    call_books: set[str] = set()
    puts_oos_rows: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    calls_decile_rows: DefaultDict[str, List[Tuple[int, float]]] = defaultdict(list)

    for book, predictions_path in _iter_train_predictions(
        train_dir, books=allowed_books
    ):
        rows = _load_prediction_rows(predictions_path)
        if not rows:
            continue

        test_predictions_path = predictions_path.parent / "test_predictions.csv"
        test_rows = (
            _load_prediction_rows(test_predictions_path)
            if test_predictions_path.exists()
            else []
        )
        if book.startswith("PUTS"):
            put_books.add(book)
            for row in test_rows:
                pnl = _coerce_numeric(row.get("pnl"))
                score = _extract_prediction_value(row)
                if math.isnan(pnl) or math.isnan(score):
                    continue
                puts_oos_rows[book].append((pnl, score))
        if book.startswith("CALLS"):
            call_books.add(book)

        pnl_values = [
            value
            for value in (_coerce_numeric(row.get("pnl")) for row in rows)
            if not math.isnan(value)
        ]
        if pnl_values:
            lower = _quantile(pnl_values, winsor_p)
            upper = _quantile(pnl_values, 1.0 - winsor_p)
        else:
            lower = upper = 0.0

        winsorised_rows: List[Dict[str, object]] = []
        train_scores: List[float] = []
        for row in rows:
            pnl = _coerce_numeric(row.get("pnl"))
            if math.isnan(pnl):
                winsor_value = pnl
            else:
                winsor_value = min(max(pnl, lower), upper)
            new_row = dict(row)
            new_row["pnl_winsor"] = winsor_value
            winsorised_rows.append(new_row)

            prediction_value = _extract_prediction_value(row)
            if not math.isnan(prediction_value):
                train_scores.append(prediction_value)
                book_predictions[book].append(prediction_value)
                if not math.isnan(pnl):
                    book_call_rows[book].append((pnl, prediction_value))

        if book.startswith("CALLS") and train_scores and test_rows:
            edges = _edges_from_scores(train_scores)
            if edges is not None:
                test_scores = [_extract_prediction_value(row) for row in test_rows]
                test_pnls = [_coerce_numeric(row.get("pnl")) for row in test_rows]
                test_deciles = _safe_bin(test_scores, edges)
                for decile_value, pnl in zip(test_deciles, test_pnls):
                    if math.isnan(decile_value) or math.isnan(pnl):
                        continue
                    calls_decile_rows[book].append((int(decile_value), pnl))

        prediction_values = [
            value
            for value in (_extract_prediction_value(row) for row in rows)
            if not math.isnan(value)
        ]
        keep_threshold = (
            _quantile(prediction_values, 1.0 - winsor_p) if prediction_values else 0.0
        )

        deciles: Dict[str, float] = {}
        if prediction_values:
            for step in range(1, 11):
                quantile = step / 10.0
                deciles[f"d{step}"] = _quantile(prediction_values, quantile)

        fold = int(predictions_path.parent.name.split("_")[-1])
        fold_dir = resolved_output / book / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        output_path = fold_dir / "winsorised_train.csv"
        _write_rows(output_path, winsorised_rows)

        fold_dir_name = predictions_path.parent.name
        feature_columns = _load_feature_columns(
            train_dir, book, fold_dir_name, predictions_path
        )

        metadata_path = fold_dir / "metadata.json"
        metadata: Dict[str, object] = {
            "book": book,
            "fold": fold,
            "winsor_p": winsor_p,
            "keep_threshold": keep_threshold,
            "winsor_limits": [lower, upper],
            "deciles": deciles,
            "source": predictions_path.name,
        }
        if feature_columns:
            metadata["feature_columns"] = feature_columns
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        fold_discovery = FoldDiscovery(
            book=book,
            fold=fold,
            keep_threshold=keep_threshold,
            deciles=deciles,
            winsor_limits=(lower, upper),
            source_path=predictions_path,
            output_path=output_path,
        )
        folds.append(fold_discovery)

        book_summary = summary.setdefault(book, {})
        book_summary["winsor_p"] = winsor_p
        folds_list = cast(List[Dict[str, object]], book_summary.setdefault("folds", []))
        folds_list.append(
            {
                "fold": fold,
                "keep_threshold": keep_threshold,
                "winsor_limits": [lower, upper],
            }
        )
        if feature_columns and "feature_columns" not in book_summary:
            book_summary["feature_columns"] = feature_columns

    if summary:
        for book, predictions in book_predictions.items():
            if book not in summary:
                continue
            percentile_pairs = compute_percentiles(predictions)
            if percentile_pairs:
                summary[book]["prediction_percentiles"] = [
                    [float(percentile), float(score)]
                    for percentile, score in percentile_pairs
                ]
        with (resolved_output / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"books": summary}, handle, indent=2, sort_keys=True)

    ci_payload = _compute_confidence_intervals(
        book_predictions,
        book_call_rows,
        winsor_p=winsor_p,
        alpha=active_settings.ALPHA,
        seed=active_settings.MODEL_SEED,
    )
    if allowed_books is not None:
        ci_payload = {
            book: payload
            for book, payload in ci_payload.items()
            if book in allowed_books
        }
    with (resolved_output / "rules_ci.json").open("w", encoding="utf-8") as handle:
        json.dump(ci_payload, handle, indent=2, sort_keys=True)

    discovered_rules: Dict[str, Dict[str, object]] = {}
    for book, put_payload in _discover_put_rules(
        put_books, puts_oos_rows, winsor_p=winsor_p
    ).items():
        converted_put: Dict[str, object] = {
            key: cast(object, value) for key, value in put_payload.items()
        }
        discovered_rules[book] = converted_put
    for book, call_payload in _discover_call_rules(
        call_books, calls_decile_rows, winsor_p=winsor_p
    ).items():
        converted_call: Dict[str, object] = {
            key: cast(object, value) for key, value in call_payload.items()
        }
        discovered_rules[book] = converted_call
    with (resolved_output / "discovered_rules.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(discovered_rules, handle, indent=2, sort_keys=True)

    folds.sort(key=lambda item: (item.book, item.fold))
    return DiscoveryResult(folds=folds, output_dir=resolved_output)
