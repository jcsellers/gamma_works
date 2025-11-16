"""Model application helpers that operate on persisted artefacts."""

from __future__ import annotations

import csv
import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xdte.config import Settings, get_settings
from xdte.live.sessions import SUFFIX_TO_SESSION
from xdte.metrics import portfolio_metrics, portfolio_metrics_extended
from xdte.model._lgbm import LGBMRegressor
from xdte.model.discovery import _cvar


@dataclass(frozen=True)
class FoldApplyResult:
    """Prediction and evaluation summary for a single fold."""

    book: str
    fold: int
    predictions_path: Path
    metrics_path: Path
    keep_threshold: float | None


@dataclass(frozen=True)
class ApplyResult:
    """Aggregated result of :func:`apply_models`."""

    folds: List[FoldApplyResult]
    output_dir: Path
    daily_pnl: pd.DataFrame | None = None


@dataclass(frozen=True)
class FoldStabilityEntry:
    """Container for per-fold stability metrics prior to serialisation."""

    book: str
    fold: int
    train_edp: float
    test_edp: float
    train_pf: float
    test_pf: float
    test_cvar: float
    leak_score: float | None

    def as_row(
        self, *, stability_index: float | None, test_cvar_column: str
    ) -> Dict[str, object]:
        """Return the serialisable representation for ``fold_stability.csv``."""

        row: Dict[str, object] = {
            "book": self.book,
            "fold": self.fold,
            "train_EDP": _normalise_numeric(self.train_edp),
            "test_EDP": _normalise_numeric(self.test_edp),
            "train_PF": _normalise_numeric(self.train_pf),
            "test_PF": _normalise_numeric(self.test_pf),
            test_cvar_column: _normalise_numeric(self.test_cvar),
        }
        row["stability_index"] = _normalise_optional(stability_index)
        row["leak_score"] = _normalise_optional(self.leak_score)
        return row


__all__ = ["FoldApplyResult", "ApplyResult", "apply_models"]


_SESSION_SUFFIX_MAP: Mapping[str, str] = SUFFIX_TO_SESSION
BOOKS = ["PUTS_0DTE_11", "CALLS_0DTE_11", "PUTS_1DTE_1515", "CALLS_1DTE_1515"]
KILL_SWITCH_COLUMN = "L1_vvix_above_ema30"


def _train_edges(scores: Sequence[float], n: int = 10) -> NDArray[np.float64]:
    edges = np.asarray(
        np.unique(np.nanpercentile(scores, np.linspace(0, 100, n + 1))),
        dtype=float,
    )
    if len(edges) >= 3:
        return edges
    fallback = np.asarray(
        np.unique(np.nanpercentile(scores, np.linspace(0, 100, 6))),
        dtype=float,
    )
    return fallback


def _cvar_label(alpha: float) -> str:
    if math.isclose(alpha, 0.05, abs_tol=1e-9):
        return "CVaR95"
    percentile = int(round((1.0 - alpha) * 100.0))
    return f"CVaR{percentile}"


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


def _coerce_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _serialise_value(value: object) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _is_finite_number(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _normalise_numeric(value: float) -> float | None:
    if _is_finite_number(value):
        return float(value)
    return None


def _normalise_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return _normalise_numeric(float(value))


def _resolve_session(book: object, row: Mapping[str, object]) -> str:
    session_value = row.get("session")
    if isinstance(session_value, str) and session_value.strip():
        return session_value
    book_str = str(book)
    suffix = book_str.split("_")[-1]
    return _SESSION_SUFFIX_MAP.get(suffix, "UNKNOWN")


def _resolve_side(book: object, row: Mapping[str, object]) -> str:
    side_value = row.get("side")
    if isinstance(side_value, str) and side_value.strip():
        normalised = side_value.strip().lower()
        if normalised in {"long", "short"}:
            return normalised
    book_upper = str(book).upper()
    if "PUT" in book_upper:
        return "short"
    return "long"


def _prepare_deciles(payload: Mapping[str, object]) -> List[tuple[int, float]]:
    deciles_raw = payload.get("deciles")
    if not isinstance(deciles_raw, Mapping):
        return []
    deciles: List[tuple[int, float]] = []
    for key, value in deciles_raw.items():
        if not isinstance(key, str) or not key.startswith("d"):
            continue
        try:
            index = int(key[1:])
        except ValueError:
            continue
        numeric = _coerce_numeric(value)
        if math.isnan(numeric):
            continue
        deciles.append((index, numeric))
    deciles.sort(key=lambda item: item[0])
    return deciles


def _assign_decile(value: float, deciles: Sequence[tuple[int, float]]) -> int:
    if not deciles or math.isnan(value):
        return 10
    for index, threshold in deciles:
        if value <= threshold or math.isclose(
            value, threshold, rel_tol=1e-12, abs_tol=1e-12
        ):
            return index
    return 10


def _compute_group_metrics(
    values: Sequence[float], *, alpha: float
) -> Mapping[str, object]:
    pnl = [float(item) for item in values]
    metrics = (
        portfolio_metrics(pnl, alpha=alpha)
        if pnl
        else {"EDP": math.nan, "PF": math.nan, "CVaR": math.nan}
    )
    hit_rate = (sum(1 for value in pnl if value > 0.0) / len(pnl)) if pnl else math.nan
    gross_gain = sum(value for value in pnl if value > 0.0) if pnl else 0.0
    gross_loss = -sum(value for value in pnl if value < 0.0) if pnl else 0.0
    return {
        "N": len(pnl),
        "EDP": metrics.get("EDP", math.nan),
        "PF": metrics.get("PF", math.nan),
        "CVaR": metrics.get("CVaR", math.nan),
        "hit_rate": hit_rate,
        "gross_gain": float(gross_gain),
        "gross_loss": float(gross_loss),
    }


def _normalise_metrics(
    metrics: Mapping[str, object], *, cvar_column: str
) -> Dict[str, object]:
    count = _coerce_int(metrics.get("N"))
    gross_gain = _coerce_numeric(metrics.get("gross_gain"))
    gross_loss = _coerce_numeric(metrics.get("gross_loss"))
    payload: Dict[str, object] = {
        "N": count,
        "gross_gain": float(0.0 if math.isnan(gross_gain) else gross_gain),
        "gross_loss": float(0.0 if math.isnan(gross_loss) else gross_loss),
    }
    edp = _coerce_numeric(metrics.get("EDP"))
    pf = _coerce_numeric(metrics.get("PF"))
    cvar = _coerce_numeric(metrics.get("CVaR"))
    hit_rate = _coerce_numeric(metrics.get("hit_rate"))
    payload["EDP"] = _normalise_numeric(edp)
    payload["PF"] = _normalise_numeric(pf)
    payload[cvar_column] = _normalise_numeric(cvar)
    payload["hit_rate"] = _normalise_numeric(hit_rate)
    return payload


def _extract_kept_pnl(
    rows: Sequence[Mapping[str, object]], *, keep_threshold: float | None
) -> List[float]:
    kept: List[float] = []
    for row in rows:
        pnl_value = _coerce_numeric(row.get("pnl"))
        if math.isnan(pnl_value):
            continue
        if keep_threshold is not None:
            prediction_value = _extract_prediction_score(row)
            if math.isnan(prediction_value) or prediction_value < keep_threshold:
                continue
        kept.append(pnl_value)
    return kept


def _compute_leak_score(train_edp: float, test_edp: float) -> float | None:
    if not (_is_finite_number(train_edp) and _is_finite_number(test_edp)):
        return None
    if math.isclose(test_edp, 0.0, abs_tol=1e-12):
        return None
    return float(abs(train_edp - test_edp) / abs(test_edp))


def _compute_stability_index(values: Sequence[float]) -> float | None:
    finite = [value for value in values if _is_finite_number(value)]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    if math.isclose(mean, 0.0, abs_tol=1e-12):
        return None
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    stddev = math.sqrt(variance)
    if math.isclose(stddev, 0.0, abs_tol=1e-12):
        return 0.0
    return float(stddev / abs(mean))


def _write_fold_stability_report(
    rows: Sequence[Mapping[str, object]], *, path: Path, test_cvar_column: str
) -> None:
    columns = [
        "book",
        "fold",
        "train_EDP",
        "test_EDP",
        "train_PF",
        "test_PF",
        test_cvar_column,
        "stability_index",
        "leak_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised: Dict[str, object] = {}
            for column in columns:
                value = row.get(column)
                if isinstance(value, float) and not _is_finite_number(value):
                    serialised[column] = ""
                elif value is None:
                    serialised[column] = ""
                else:
                    serialised[column] = value
            writer.writerow(serialised)


def _write_calls_edge_csv(
    rows: Sequence[Mapping[str, object]], *, path: Path, cvar_column: str
) -> None:
    columns = [
        "book",
        "session",
        "decile",
        "side",
        "N",
        "EDP",
        "PF",
        cvar_column,
        "hit_rate",
        "gross_gain",
        "gross_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _generate_calls_edge_reports(
    records: Sequence[Mapping[str, object]], *, output_dir: Path, alpha: float
) -> None:
    cvar_column = _cvar_label(alpha)
    csv_path = output_dir / "calls_edge_report.csv"
    json_path = output_dir / "calls_edge_attribution.json"
    if not records:
        _write_calls_edge_csv([], path=csv_path, cvar_column=cvar_column)
        _write_json({}, json_path)
        return

    grouped: Dict[Tuple[str, str, int, str], List[float]] = defaultdict(list)
    side_totals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for record in records:
        pnl_value = _coerce_numeric(record.get("pnl"))
        if math.isnan(pnl_value):
            continue
        key: Tuple[str, str, int, str] = (
            str(record.get("book", "")),
            str(record.get("session", "UNKNOWN")),
            _coerce_int(record.get("decile", 10)),
            str(record.get("side", "long")),
        )
        grouped[key].append(pnl_value)
        side_key = (key[0], key[3])
        side_totals[side_key].append(pnl_value)

    rows: List[Dict[str, object]] = []
    book_deciles: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for key in sorted(grouped):
        book, session, decile, side = key
        metrics = _compute_group_metrics(grouped[key], alpha=alpha)
        normalised = _normalise_metrics(metrics, cvar_column=cvar_column)
        row = {
            "book": book,
            "session": session,
            "decile": decile,
            "side": side,
        }
        row.update(normalised)
        rows.append(row)

        book_deciles[book].append(
            {
                "session": session,
                "decile": decile,
                "side": side,
                **normalised,
            }
        )

    book_payloads: Dict[str, Dict[str, object]] = {}
    for book, decile_rows in book_deciles.items():
        decile_rows.sort(
            key=lambda item: (item.get("session", ""), item["decile"], item["side"])
        )
        totals: Dict[str, Dict[str, object]] = {}
        matching_sides = sorted(
            total_side for (total_book, total_side) in side_totals if total_book == book
        )
        for side in matching_sides:
            series = side_totals[(book, side)]
            metrics = _compute_group_metrics(series, alpha=alpha)
            totals[side] = _normalise_metrics(metrics, cvar_column=cvar_column)
        book_payloads[book] = {"by_decile": decile_rows, "totals": totals}

    _write_calls_edge_csv(rows, path=csv_path, cvar_column=cvar_column)
    _write_json(book_payloads, json_path)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    seen = {"book", "open_date", "pnl", "prediction", "fold", "keep"}
    ordered = [
        key
        for key in ("book", "open_date", "pnl", "prediction", "fold", "keep")
        if any(key in row for row in rows)
    ]
    columns.extend(ordered)
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


def _append_prediction_rows(
    container: List[Dict[str, object]],
    *,
    book: str,
    pnl_series: pd.Series,
    score_series: pd.Series,
    keep_series: pd.Series,
    fold: int,
) -> None:
    if pnl_series.empty:
        return
    aligned_scores = score_series.reindex(pnl_series.index)
    if aligned_scores.empty:
        aligned_scores = pd.Series(math.nan, index=pnl_series.index)
    aligned_keep = keep_series.reindex(pnl_series.index).fillna(False)
    for open_date, pnl_value, score_value, keep_value in zip(
        pnl_series.index,
        pnl_series.to_numpy(),
        aligned_scores.to_numpy(),
        aligned_keep.to_numpy(),
    ):
        if not isinstance(open_date, pd.Timestamp):
            continue
        if pd.isna(pnl_value):
            continue
        container.append(
            {
                "book": book,
                "open_date": open_date.to_pydatetime(),
                "pnl": float(pnl_value),
                "prediction": float(score_value),
                "fold": int(fold),
                "keep": bool(keep_value),
            }
        )


def _build_extended_row(
    *,
    book: str,
    mode: str,
    series: Sequence[float],
    alpha: float,
    cvar_column: str,
) -> Dict[str, object]:
    metrics = portfolio_metrics_extended(series, alpha=alpha)
    cvar_value = metrics.pop("CVaR", math.nan)
    row: Dict[str, object] = {
        "book": book,
        "mode": mode,
        "count": len(series),
    }
    for key, value in metrics.items():
        row[key] = _normalise_numeric(float(value))
    row[cvar_column] = _normalise_numeric(cvar_value)
    # Ensure canonical ordering keys exist even if metrics missing
    for key in (
        "EDP",
        "PF",
        "Sharpe",
        "Sortino",
        "Omega1.0",
        "MaxDrawdown",
        "WinRate",
        "Skewness",
        "Kurtosis",
        "TailHitRate",
        "MeanGain",
        "MeanLoss",
        "TopKLossConcentration",
    ):
        row.setdefault(key, None)
    return row


def _write_portfolio_metrics_extended_report(
    pnl_by_book: Mapping[str, Sequence[float]],
    kept_pnl_by_book: Mapping[str, Sequence[float]],
    *,
    path: Path,
    alpha: float,
) -> None:
    books = sorted(set(pnl_by_book) | set(kept_pnl_by_book))
    cvar_column = _cvar_label(alpha)
    rows: List[Dict[str, object]] = []
    for book in books:
        series = pnl_by_book.get(book, [])
        rows.append(
            _build_extended_row(
                book=book,
                mode="no_kill",
                series=series,
                alpha=alpha,
                cvar_column=cvar_column,
            )
        )
        kept_series = kept_pnl_by_book.get(book, [])
        rows.append(
            _build_extended_row(
                book=book,
                mode="hybrid",
                series=kept_series,
                alpha=alpha,
                cvar_column=cvar_column,
            )
        )

    all_no_kill = [value for values in pnl_by_book.values() for value in values]
    all_hybrid = [value for values in kept_pnl_by_book.values() for value in values]
    rows.append(
        _build_extended_row(
            book="ALL",
            mode="no_kill",
            series=all_no_kill,
            alpha=alpha,
            cvar_column=cvar_column,
        )
    )
    rows.append(
        _build_extended_row(
            book="ALL",
            mode="hybrid",
            series=all_hybrid,
            alpha=alpha,
            cvar_column=cvar_column,
        )
    )

    columns = [
        "book",
        "mode",
        "count",
        "EDP",
        "PF",
        cvar_column,
        "Sharpe",
        "Sortino",
        "Omega1.0",
        "MaxDrawdown",
        "WinRate",
        "Skewness",
        "Kurtosis",
        "TailHitRate",
        "MeanGain",
        "MeanLoss",
        "TopKLossConcentration",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised: Dict[str, object] = {}
            for column in columns:
                value = row.get(column)
                if column in {"book", "mode"}:
                    serialised[column] = str(value) if value is not None else ""
                elif column == "count":
                    serialised[column] = int(value) if isinstance(value, int) else 0
                else:
                    if isinstance(value, (int, float)):
                        serialised[column] = _normalise_numeric(float(value))
                    else:
                        serialised[column] = _normalise_numeric(_coerce_numeric(value))
            writer.writerow(serialised)


def _load_model(path: Path) -> LGBMRegressor:
    with path.open("rb") as handle:
        model = pickle.load(handle)  # nosec B301 - training outputs are locally trusted
    if not isinstance(model, LGBMRegressor):  # pragma: no cover - defensive guard
        msg = "Persisted model is not an LGBMRegressor"
        raise TypeError(msg)
    return model


def _load_json(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return cast(Mapping[str, object], data)
    return {}


def _iter_fold_directories(train_dir: Path) -> Iterator[tuple[str, Path]]:
    train_root = train_dir / "train"
    if not train_root.exists():
        return
    for book_dir in sorted(train_root.iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name
        for fold_dir in sorted(book_dir.iterdir()):
            if not fold_dir.is_dir():
                continue
            yield book, fold_dir


def _load_discovery_thresholds(
    discovery_dir: Path,
) -> Dict[str, Dict[int, Mapping[str, object]]]:
    thresholds: Dict[str, Dict[int, Mapping[str, object]]] = {}
    if not discovery_dir.exists():
        return thresholds
    for book_dir in sorted(discovery_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name
        fold_map: Dict[int, Mapping[str, object]] = {}
        for metadata_path in sorted(book_dir.glob("fold_*/metadata.json")):
            fold_str = metadata_path.parent.name.split("_")[-1]
            fold = int(fold_str)
            fold_map[fold] = _load_json(metadata_path)
        if fold_map:
            thresholds[book] = fold_map
    return thresholds


def _read_rows(path: Path) -> List[Mapping[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Mapping[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if key == "open_date" and value:
                    parsed[key] = datetime.fromisoformat(value)
                elif key == "pnl":
                    parsed[key] = _coerce_numeric(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def _extract_prediction_score(row: Mapping[str, object]) -> float:
    score_value = _coerce_numeric(row.get("score"))
    if not math.isnan(score_value):
        return score_value
    return _coerce_numeric(row.get("prediction"))


if TYPE_CHECKING:  # pragma: no cover - type checking aid
    from pandas import DataFrame


def _matrix_for_prediction(
    rows: List[Mapping[str, object]], feature_columns: List[str]
) -> "DataFrame":
    data = {
        column: [_coerce_numeric(row.get(column)) for row in rows]
        for column in feature_columns
    }
    return pd.DataFrame(data, columns=feature_columns)


def _write_json(payload: Mapping[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_discovered_rules(discovery_dir: Path) -> Mapping[str, Mapping[str, object]]:
    rules_path = discovery_dir / "discovered_rules.json"
    if not rules_path.exists():
        return {}
    try:
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, Mapping):
        return cast(Mapping[str, Mapping[str, object]], payload)
    return {}


def _fold_from_name(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def _load_prediction_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open_date", "pnl", "score"])
    frame = pd.read_csv(path)
    if "open_date" in frame.columns:
        frame["open_date"] = pd.to_datetime(frame["open_date"], errors="coerce")
    else:
        frame["open_date"] = pd.Series([pd.NaT] * len(frame), dtype="datetime64[ns]")
    if "score" not in frame.columns and "prediction" in frame.columns:
        frame["score"] = frame["prediction"]
    for column in ("pnl", "score"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _series_from_frame(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    dates = pd.to_datetime(frame.get("open_date"), errors="coerce")
    if isinstance(dates, pd.Series):
        mask = dates.notna()
        return pd.Series(values[mask].to_numpy(), index=dates[mask])
    return pd.Series(values.to_numpy())


def _load_fold_artifacts(train_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    discovery_root = train_dir / "discovery"
    fold_map: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    if not discovery_root.exists():
        return fold_map
    for book_dir in sorted(discovery_root.iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name
        for fold_dir in sorted(book_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            train_path = fold_dir / "train_predictions.csv"
            test_path = fold_dir / "test_predictions.csv"
            if not train_path.exists() and not test_path.exists():
                continue
            train_frame = _load_prediction_frame(train_path)
            test_frame = _load_prediction_frame(test_path)
            if "open_date" in test_frame.columns:
                test_dates = pd.Series(
                    pd.to_datetime(test_frame["open_date"], errors="coerce")
                ).dropna()
            else:
                test_dates = pd.Series(dtype="datetime64[ns]")
            payload: Dict[str, object] = {
                "fold": _fold_from_name(fold_dir.name),
                "train_scores": _series_from_frame(train_frame, "score"),
                "train_pnl": _series_from_frame(train_frame, "pnl"),
                "test_scores": _series_from_frame(test_frame, "score"),
                "test_pnl": _series_from_frame(test_frame, "pnl"),
                "test_dates": test_dates,
            }
            fold_map[book].append(payload)
    return fold_map


def _load_trading_days(
    train_dir: Path, fold_artifacts: Mapping[str, Sequence[Mapping[str, object]]]
) -> NDArray[np.datetime64]:
    candidates = [
        train_dir / "daily_context.csv",
        train_dir / "train" / "daily_context.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["open_date"])
        dates = pd.to_datetime(frame["open_date"], errors="coerce").dropna()
        if not dates.empty:
            unique_dates = np.sort(dates.unique())
            return np.asarray(unique_dates, dtype="datetime64[ns]")
    fallback: List[pd.Timestamp] = []
    for entries in fold_artifacts.values():
        for artifact in entries:
            for key in ("train_pnl", "test_pnl"):
                series = artifact.get(key)
                if isinstance(series, pd.Series):
                    fallback.extend(
                        [
                            cast(pd.Timestamp, idx)
                            for idx in series.index
                            if isinstance(idx, pd.Timestamp)
                        ]
                    )
            test_dates = artifact.get("test_dates")
            if isinstance(test_dates, pd.Series):
                fallback.extend(
                    [
                        cast(pd.Timestamp, value)
                        for value in test_dates
                        if isinstance(value, pd.Timestamp)
                    ]
                )
    if fallback:
        ordered = pd.to_datetime(fallback, errors="coerce").dropna().unique()
        sorted_ordered = np.sort(ordered)
        return np.asarray(sorted_ordered, dtype="datetime64[ns]")
    msg = "Unable to determine trading days; daily_context.csv is missing"
    raise FileNotFoundError(msg)


def apply_models(
    train_dir: Path,
    discovery_dir: Path,
    output_dir: Path,
    *,
    settings: Settings | None = None,
) -> ApplyResult:
    """Score the fold test rows and emit prediction artefacts."""

    resolved_apply = train_dir.resolve()
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    _ = settings or get_settings()
    rules = _load_discovered_rules(discovery_dir)
    fold_artifacts = _load_fold_artifacts(train_dir)
    days = _load_trading_days(train_dir, fold_artifacts)
    daily_index = pd.Index(days)
    out = pd.DataFrame(index=daily_index)
    prediction_rows: List[Dict[str, object]] = []

    for book in BOOKS:
        rule_payload = rules.get(book)
        if isinstance(rule_payload, Mapping):
            rule: Mapping[str, object] = rule_payload
        else:
            rule = cast(Mapping[str, object], {})
        rule_type = str(rule.get("type", ""))
        if not rule_type:
            if "keep_target" in rule:
                rule_type = "filter"
            elif any(key in rule for key in ("long_deciles", "short_deciles")):
                rule_type = "sweep"
        if rule_type == "filter":
            keep_value = rule.get("keep_target", 1.0)
            if isinstance(keep_value, (int, float)):
                keep = float(keep_value)
            else:
                keep = float(_coerce_numeric(keep_value))
            fold_entries = fold_artifacts.get(book, [])
            chunks: List[pd.Series] = []
            pooled_frames: List[pd.DataFrame] = []
            book_prediction_rows: List[Dict[str, object]] = []
            fallback_decision: tuple[bool, float, float] | None = None
            for artifact in fold_entries:
                train_scores_raw = artifact.get("train_scores")
                test_scores_raw = artifact.get("test_scores")
                train_pnl_raw = artifact.get("train_pnl")
                test_pnl_raw = artifact.get("test_pnl")
                test_dates_raw = artifact.get("test_dates")
                if not isinstance(train_scores_raw, pd.Series):
                    continue
                if not isinstance(test_scores_raw, pd.Series):
                    continue
                if not isinstance(train_pnl_raw, pd.Series):
                    continue
                if not isinstance(test_pnl_raw, pd.Series):
                    continue
                if not isinstance(test_dates_raw, pd.Series):
                    continue
                if test_pnl_raw.empty:
                    continue

                s_tr = pd.Series(train_scores_raw.to_numpy(), index=train_pnl_raw.index)
                s_te = pd.Series(test_scores_raw.to_numpy(), index=test_pnl_raw.index)
                p_tr = train_pnl_raw.copy()
                p_te = test_pnl_raw.copy()
                test_dates = pd.Series(test_dates_raw.to_numpy())

                fold_value = _coerce_int(artifact.get("fold"))
                if keep >= 0.999:
                    keep_mask = pd.Series(True, index=p_te.index)
                    g = p_te.copy()
                else:
                    if s_tr.empty:
                        continue
                    train_values = s_tr.to_numpy(dtype=float, copy=False)
                    tau_top = float(np.nanquantile(train_values, 1.0 - keep).item())
                    tau_bot = float(np.nanquantile(train_values, keep).item())
                    m_top_tr = s_tr >= tau_top
                    m_bot_tr = s_tr <= tau_bot
                    cvar_top = _cvar(p_tr.where(m_top_tr, 0.0))
                    cvar_bot = _cvar(p_tr.where(m_bot_tr, 0.0))
                    use_top = (
                        pd.notna(cvar_top)
                        and pd.notna(cvar_bot)
                        and (cvar_top >= cvar_bot)
                    )
                    if pd.isna(cvar_top) and pd.notna(cvar_bot):
                        use_top = False
                    if pd.notna(cvar_top) and pd.isna(cvar_bot):
                        use_top = True
                    if use_top:
                        keep_mask = s_te >= tau_top
                    else:
                        keep_mask = s_te <= tau_bot
                    g = p_te.where(keep_mask, 0.0)

                g.index = test_dates.to_numpy()
                chunks.append(g)

                keep_series = keep_mask.reindex(p_te.index).fillna(False)
                _append_prediction_rows(
                    book_prediction_rows,
                    book=book,
                    pnl_series=p_te,
                    score_series=s_te,
                    keep_series=keep_series,
                    fold=fold_value,
                )

                pooled_frames.append(
                    pd.DataFrame(
                        {
                            "open_date": test_dates.to_numpy(),
                            "PNL": p_te.to_numpy(),
                            "score": s_te.to_numpy(),
                        }
                    )
                )

            if chunks:
                ser = (
                    pd.concat(chunks)
                    .groupby(level=0)
                    .sum()
                    .reindex(daily_index)
                    .fillna(0.0)
                )
                if ser.mean() <= 0 and pooled_frames:
                    pooled = pd.concat(pooled_frames, ignore_index=True).dropna(
                        subset=["open_date"]
                    )
                    if not pooled.empty:
                        pooled["open_date"] = pd.to_datetime(
                            pooled["open_date"], errors="coerce"
                        )
                        pooled = pooled.dropna(subset=["open_date"])
                        if not pooled.empty:
                            scores = pooled["score"].to_numpy(dtype=float, copy=False)
                            if scores.size > 0:
                                tau_top = float(
                                    np.nanquantile(scores, 1.0 - keep).item()
                                )
                                tau_bot = float(np.nanquantile(scores, keep).item())
                                m_top = pooled["score"] >= tau_top
                                m_bot = pooled["score"] <= tau_bot
                                c_top = _cvar(pooled.loc[m_top, "PNL"])
                                c_bot = _cvar(pooled.loc[m_bot, "PNL"])
                                use_top = (
                                    pd.notna(c_top)
                                    and pd.notna(c_bot)
                                    and (c_top >= c_bot)
                                )
                                if pd.isna(c_top) and pd.notna(c_bot):
                                    use_top = False
                                if pd.notna(c_top) and pd.isna(c_bot):
                                    use_top = True
                                kept = (
                                    pooled.loc[m_top if use_top else m_bot]
                                    .groupby("open_date")["PNL"]
                                    .sum()
                                )
                                ser = kept.reindex(daily_index).fillna(0.0)
                                fallback_decision = (use_top, tau_top, tau_bot)
            else:
                ser = pd.Series(0.0, index=daily_index)

            out[book] = ser
            # Ensure 'keep' is set for all rows before appending.
            for row in book_prediction_rows:
                score_value = _coerce_numeric(row.get("prediction"))
                if math.isnan(score_value):
                    row["keep"] = False
                    continue
                if fallback_decision is not None:
                    use_top, tau_top, tau_bot = fallback_decision
                    row["keep"] = (
                        bool(score_value >= tau_top)
                        if use_top
                        else bool(score_value <= tau_bot)
                    )
                else:
                    # Set to a safe default if no fallback_decision; adjust as needed.
                    row["keep"] = False
            prediction_rows.extend(book_prediction_rows)
        elif rule_type == "sweep":

            def _normalise_deciles(values: object) -> set[int]:
                if isinstance(values, Sequence) and not isinstance(
                    values, (str, bytes)
                ):
                    result: set[int] = set()
                    for entry in values:
                        try:
                            result.add(int(entry))
                        except (TypeError, ValueError):
                            continue
                    return result
                if isinstance(values, (int, float)):
                    return {int(values)}
                return set()

            longs = _normalise_deciles(rule.get("long_deciles"))
            shorts = _normalise_deciles(rule.get("short_deciles"))
            sweep_chunks: List[pd.Series] = []

            for artifact in fold_artifacts.get(book, []):
                train_scores_raw = artifact.get("train_scores")
                test_scores_raw = artifact.get("test_scores")
                test_pnl_raw = artifact.get("test_pnl")
                test_dates_raw = artifact.get("test_dates")
                if not isinstance(train_scores_raw, pd.Series):
                    continue
                if not isinstance(test_scores_raw, pd.Series):
                    continue
                if not isinstance(test_pnl_raw, pd.Series):
                    continue
                if not isinstance(test_dates_raw, pd.Series):
                    continue
                if test_scores_raw.empty or test_pnl_raw.empty:
                    continue

                s_tr = train_scores_raw.copy()
                s_te = test_scores_raw.copy()
                p_te = test_pnl_raw.copy()
                test_dates = test_dates_raw.copy()
                fold_value = _coerce_int(artifact.get("fold"))

                edges = _train_edges(s_tr.values, n=10)
                dec = pd.Series(
                    pd.cut(
                        s_te.values,
                        edges,
                        labels=False,
                        include_lowest=True,
                    ),
                    index=s_te.index,
                )
                keep_mask = dec.isin(longs) | dec.isin(shorts)
                g = (-p_te).where(dec.isin(longs), 0.0) + (p_te).where(
                    dec.isin(shorts), 0.0
                )
                g.index = test_dates.to_numpy()
                sweep_chunks.append(g)

                _append_prediction_rows(
                    prediction_rows,
                    book=book,
                    pnl_series=p_te,
                    score_series=s_te,
                    keep_series=keep_mask,
                    fold=fold_value,
                )

            if sweep_chunks:
                out[book] = (
                    pd.concat(sweep_chunks)
                    .groupby(level=0)
                    .sum()
                    .reindex(daily_index)
                    .fillna(0.0)
                )
            else:
                out[book] = pd.Series(0.0, index=daily_index)
        else:
            out[book] = pd.Series(0.0, index=daily_index)

    out["Gated_Portfolio_Base"] = out[BOOKS].sum(axis=1)

    predictions_path = resolved_output / "hybrid_predictions.csv"
    _write_rows(predictions_path, prediction_rows)

    daily_context_path = resolved_apply / "daily_context.csv"
    if not daily_context_path.exists():
        msg = f"daily_context.csv is required to propagate kill switch flags (expected at {daily_context_path})"
        raise FileNotFoundError(msg)

    daily_context = pd.read_csv(daily_context_path)
    if "open_date" not in daily_context.columns:
        msg = "daily_context.csv must include an open_date column"
        raise ValueError(msg)
    if KILL_SWITCH_COLUMN not in daily_context.columns:
        daily_context[KILL_SWITCH_COLUMN] = False

    daily_context["open_date"] = pd.to_datetime(
        daily_context["open_date"], errors="coerce"
    )
    daily_context = daily_context.dropna(subset=["open_date"])
    daily_context = daily_context.set_index("open_date")
    daily_context = daily_context[[KILL_SWITCH_COLUMN]]

    final_df = out.merge(
        daily_context,
        left_index=True,
        right_index=True,
        how="left",
    )
    final_df[KILL_SWITCH_COLUMN] = (
        final_df[KILL_SWITCH_COLUMN].fillna(False).astype(bool)
    )

    hybrid_input_path = resolved_output / "hybrid_input.csv"
    final_df.to_csv(hybrid_input_path, index_label="open_date")

    return ApplyResult(folds=[], output_dir=resolved_output, daily_pnl=final_df)
