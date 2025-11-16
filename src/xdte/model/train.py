"""Model training utilities for the XDTE package."""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency at import time
    import shap as _shap_module
except Exception:  # pragma: no cover - best effort guard for missing SHAP
    _shap_module = None
shap = cast(Any, _shap_module)

from xdte.config import Settings, get_settings
from xdte.data.features import (
    BOOK_FEATS_CONFIG,
    generate_feature_panel,
    market_records_to_frame,
    trade_records_to_frame,
)
from xdte.data.loaders import PortfolioRecord, read_portfolio_file
from xdte.model._lgbm import LGBMRegressor
from xdte.model.wfo import WalkForwardSplitter


@dataclass(frozen=True)
class FoldArtifact:
    """Description of a persisted fold artefact."""

    book: str
    fold: int
    directory: Path
    discovery_dir: Path
    model_path: Path
    metadata_path: Path
    train_path: Path
    test_path: Path
    train_predictions_path: Path
    test_predictions_path: Path
    legacy_train_predictions_path: Path | None = None


@dataclass(frozen=True)
class TrainResult:
    """Aggregated result of a training run."""

    artifacts: List[FoldArtifact]
    feature_frame: pd.DataFrame


# ``BOOK_FEATS_CONFIG`` (defined in ``xdte.data.features``) intentionally omits
# columns (``gap``, ``movement``, ``opening_vix``, ``closing_vix``) that were
# temporarily kept in the package while the per-book feature mask was being
# ported from the notebook.  The columns still exist in the feature frame, so we
# preserve their historical ordering whenever the automatic fallback feature
# discovery path is used.
_LEGACY_FEATURE_PRIORITY = [
    "gap",
    "movement",
    "opening_vix",
    "closing_vix",
]

__all__ = [
    "FoldArtifact",
    "TrainResult",
    "load_backtest_dataset",
    "train_models",
]


_KILL_SWITCH_COLUMN = "L1_vvix_above_ema30"


logger = logging.getLogger(__name__)


def _iter_csv_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith("."):
            continue
        yield path


def _categorise_inputs(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    trades: list[Path] = []
    market: list[Path] = []
    for path in paths:
        if path.stem.startswith("daily_context"):
            continue
        if "market_stats" in path.name:
            market.append(path)
        else:
            trades.append(path)
    return trades, market


def _normalise_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported date value: {value!r}")


def _serialise_value(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return _normalise_datetime(value).date().isoformat()
    return value


def _coerce_numeric(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _normalise_kill_switch_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return False
        return bool(numeric)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes", "y"}:
            return True
        if lowered in {"false", "f", "0", "no", "n"}:
            return False
        try:
            numeric = float(value)
        except ValueError:
            return False
        if math.isnan(numeric):
            return False
        return bool(numeric)
    return False


def _extract_kill_switch_flag(row: Mapping[str, object]) -> bool:
    return _normalise_kill_switch_value(row.get(_KILL_SWITCH_COLUMN))


def _lookup_kill_switch_flag(
    lookup: Mapping[date, bool] | None, open_date_value: object
) -> bool:
    if lookup is None:
        return False
    try:
        normalised = _normalise_datetime(open_date_value)
    except (TypeError, ValueError):
        return False
    return bool(lookup.get(normalised.date(), False))


def _build_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[float],
    *,
    book: str,
    kill_switch_lookup: Mapping[date, bool] | None = None,
) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for row, prediction in zip(rows, predictions):
        if _KILL_SWITCH_COLUMN in row:
            kill_flag = _extract_kill_switch_flag(row)
        else:
            kill_flag = _lookup_kill_switch_flag(
                kill_switch_lookup, row.get("open_date")
            )
        payload.append(
            {
                "book": book,
                "open_date": row.get("open_date"),
                "pnl": _coerce_numeric(row.get("pnl")),
                "score": float(prediction),
                _KILL_SWITCH_COLUMN: kill_flag,
            }
        )
    return payload


def _build_kill_switch_lookup(frame: pd.DataFrame) -> dict[date, bool]:
    if _KILL_SWITCH_COLUMN not in frame.columns:
        return {}
    series = frame[["open_date", _KILL_SWITCH_COLUMN]].dropna(subset=["open_date"])
    series = series.drop_duplicates(subset=["open_date"], keep="last")
    lookup: dict[date, bool] = {}
    for open_date_value, kill_value in zip(
        series["open_date"], series[_KILL_SWITCH_COLUMN]
    ):
        try:
            normalised = _normalise_datetime(open_date_value)
        except (TypeError, ValueError):
            continue
        lookup[normalised.date()] = _normalise_kill_switch_value(kill_value)
    return lookup


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: List[str] = []
    seen = {"book", "open_date", "pnl"}
    for key in ("book", "open_date", "pnl"):
        if any(key in row for row in rows):
            columns.append(key)
    extra: List[str] = []
    for row in rows:
        for key in row:
            if key in seen or key in columns:
                continue
            extra.append(key)
    columns.extend(sorted(dict.fromkeys(extra)))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised = {
                column: _serialise_value(row.get(column, "")) for column in columns
            }
            writer.writerow(serialised)


def _load_daily_context(directory: Path) -> List[Dict[str, object]]:
    csv_path = directory / "daily_context.csv"
    json_path = directory / "daily_context.json"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: List[Dict[str, object]] = []
            for row in reader:
                if "open_date" not in row:
                    continue
                parsed: Dict[str, object] = {
                    "open_date": datetime.fromisoformat(row["open_date"]).date()
                }
                for key, value in row.items():
                    if key == "open_date":
                        continue
                    parsed[key] = _coerce_numeric(value)
                rows.append(parsed)
            return rows
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = []
        for entry in payload:
            if "open_date" not in entry:
                continue
            json_row: Dict[str, object] = {
                "open_date": datetime.fromisoformat(str(entry["open_date"])).date()
            }
            for key, value in entry.items():
                if key == "open_date":
                    continue
                json_row[key] = value
            rows.append(json_row)
        return rows
    return []


def load_backtest_dataset(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load trade, market, and daily inputs from ``directory``."""

    resolved = directory.resolve()
    trade_files, market_files = _categorise_inputs(_iter_csv_files(resolved))

    trades: List[PortfolioRecord] = []
    market: List[PortfolioRecord] = []

    for trade_file in trade_files:
        trades.extend(read_portfolio_file(trade_file))
    for market_file in market_files:
        market.extend(read_portfolio_file(market_file))

    if not market:
        if market_files:
            logger.info(
                "No usable market rows found in %s, falling back to trade records",
                resolved,
            )
        market = list(trades)

    daily_rows = _load_daily_context(resolved)

    trade_frame = trade_records_to_frame(trades)
    market_frame = market_records_to_frame(market)
    daily_frame = _daily_rows_to_frame(daily_rows)

    trade_dates = _extract_open_dates(trade_frame)
    if trade_dates:
        daily_dates = _extract_open_dates(daily_frame)
        if not daily_dates or trade_dates - daily_dates:
            min_trade = min(trade_dates)
            max_trade = max(trade_dates)
            raise ValueError(
                "daily_context.csv does not cover trade range "
                f"[{min_trade.isoformat()} – {max_trade.isoformat()}]. "
                f"Run xdte fetch-daily-context --data-dir {resolved}"
            )

    return trade_frame, market_frame, daily_frame


def _daily_rows_to_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open_date"])

    frame = pd.DataFrame(rows)
    if "open_date" in frame:
        frame["open_date"] = frame["open_date"].map(_normalise_datetime)
    return frame


def _extract_open_dates(frame: pd.DataFrame) -> set[date]:
    if "open_date" not in frame or frame.empty:
        return set()
    series = pd.to_datetime(frame["open_date"], errors="coerce")
    return {value.date() for value in series.dropna()}


def _prepare_features(frame: pd.DataFrame) -> List[str]:
    excluded = {"book", "open_date", "pnl"}
    columns = [str(column) for column in frame.columns if column not in excluded]
    return sorted(dict.fromkeys(columns))


_SESSION_SUFFIX_PATTERN = re.compile(r"(\d+)$")
_SESSION_AWARE_PREFIXES: tuple[str, ...] = (
    "VIX_Entry_",
    "Intraday_Move_OpenToEntry_",
    "t0_VIX_change_from_close_",
)


def _infer_session_suffix(book: str) -> str | None:
    match = _SESSION_SUFFIX_PATTERN.search(book)
    if match:
        return match.group(1)
    return None


def _expected_suffix(prefix: str, session_suffix: str) -> str | None:
    if prefix == "t0_VIX_change_from_close_":
        if session_suffix == "1515":
            return "15"
    return session_suffix


def _filter_feature_columns_for_book(book: str, columns: Sequence[str]) -> list[str]:
    session_suffix = _infer_session_suffix(book)
    if not session_suffix:
        return list(columns)

    filtered: list[str] = []
    for column in columns:
        keep = True
        for prefix in _SESSION_AWARE_PREFIXES:
            if column.startswith(prefix):
                expected = _expected_suffix(prefix, session_suffix)
                actual_suffix = column[len(prefix) :]
                if expected is not None and actual_suffix != expected:
                    keep = False
                break
        if keep:
            filtered.append(column)
    return filtered


def _build_training_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[List[Mapping[str, object]], List[List[float]], List[float]]:
    working = frame.copy()
    working["pnl"] = pd.to_numeric(working["pnl"], errors="coerce")
    working = working[working["pnl"].notna()]

    for column in feature_columns:
        if column not in working:
            working[column] = math.nan
    features_only = working.loc[:, feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    filtered_rows = working.to_dict("records")
    matrix = features_only.to_numpy(dtype=float).tolist()
    target = working["pnl"].astype(float).tolist()
    return filtered_rows, matrix, target


def _extract_shap_array(payload: object) -> NDArray[np.float64] | None:
    candidate = payload
    if isinstance(candidate, list):
        candidate = candidate[0]
    values = getattr(candidate, "values", candidate)
    if isinstance(values, list):
        values = values[0]
    try:
        array = np.asarray(values, dtype=float)
    except Exception:  # pragma: no cover - defensive guard
        return None
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[0]
    return None


def _normalise_expected_value(value: object) -> float | None:
    try:
        array = np.asarray(value, dtype=float)
    except Exception:  # pragma: no cover - defensive guard
        return None
    if array.size == 0:
        return None
    mean_value = float(array.mean())
    if math.isnan(mean_value):
        return None
    return mean_value


def _compute_global_shap_summary(
    model: LGBMRegressor,
    matrix: Sequence[Sequence[float]],
) -> Tuple[NDArray[np.float64], int, float | None] | None:
    if shap is None:
        return None
    if not matrix or len(matrix) < 2:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_payload: object = explainer(np.asarray(matrix, dtype=float))
    except Exception:  # pragma: no cover - defensive guard
        return None
    shap_array = _extract_shap_array(shap_payload)
    if shap_array is None or shap_array.size == 0:
        return None
    expected_value = _normalise_expected_value(
        getattr(explainer, "expected_value", None)
    )
    return np.abs(shap_array).sum(axis=0), int(shap_array.shape[0]), expected_value


def _extract_gain_importance(
    model: LGBMRegressor,
) -> NDArray[np.float64] | None:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            importance = np.asarray(
                booster.feature_importance(importance_type="gain"), dtype=float
            )
        except Exception:  # pragma: no cover - defensive guard
            importance = None
        else:
            if importance.size > 0:
                return importance.reshape(-1)
    raw_importance = getattr(model, "feature_importances_", None)
    if raw_importance is None:
        return None
    try:
        importance_array = np.asarray(raw_importance, dtype=float)
    except Exception:  # pragma: no cover - defensive guard
        return None
    if importance_array.size == 0:
        return None
    return importance_array.reshape(-1)


def train_models(
    data_dir: Path,
    artifact_dir: Path,
    *,
    settings: Settings | None = None,
    books: Sequence[str] | None = None,
    min_train_size: int = 5,
    min_test_size: int = 1,
) -> TrainResult:
    """Train LightGBM quantile models and persist fold artefacts."""

    resolved_artifact_dir = artifact_dir.resolve()
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_data_dir = data_dir.resolve()

    daily_context_source = resolved_data_dir / "daily_context.csv"
    daily_context_target = resolved_artifact_dir / "daily_context.csv"
    if daily_context_source.exists():
        shutil.copyfile(daily_context_source, daily_context_target)

    active_settings = settings or get_settings()
    trade_frame, market_frame, daily_frame = load_backtest_dataset(data_dir)

    feature_frame = generate_feature_panel(trade_frame, market_frame, daily_frame)
    feature_df = feature_frame.reset_index()
    if feature_df.empty:
        return TrainResult(artifacts=[], feature_frame=feature_frame)

    feature_df["book"] = feature_df["book"].astype(str)
    feature_df["open_date"] = pd.to_datetime(feature_df["open_date"])
    kill_switch_lookup = _build_kill_switch_lookup(feature_df)

    available_books = sorted(feature_df["book"].unique())
    if books is not None:
        selected_books = [book for book in available_books if book in books]
    else:
        selected_books = [book for book in available_books if book != "UNKNOWN"]

    splitter = WalkForwardSplitter(
        settings=active_settings,
        min_train_size=min_train_size,
        min_test_size=min_test_size,
    )

    artifacts: List[FoldArtifact] = []
    manifest: Dict[str, Dict[str, object]] = {}

    for book in selected_books:
        book_frame = feature_df[feature_df["book"] == book].copy()
        if book_frame.empty:
            continue
        book_frame.sort_values("open_date", inplace=True)
        dates = [value.to_pydatetime() for value in book_frame["open_date"]]
        splits = splitter.splits(dates)
        if not splits:
            unique_dates = {value.date() for value in book_frame["open_date"]}
            logger.warning(
                "Skipping book %s because no walk-forward splits are available (%d available dates)",
                book,
                len(unique_dates),
            )
            continue

        if book in BOOK_FEATS_CONFIG:
            configured = [
                column
                for column in BOOK_FEATS_CONFIG[book]
                if column in book_frame.columns
            ]
            feature_columns = list(dict.fromkeys(configured))
        else:
            fallback = _prepare_features(book_frame)
            legacy = [
                column for column in _LEGACY_FEATURE_PRIORITY if column in fallback
            ]
            ordered = legacy + fallback
            feature_columns = list(dict.fromkeys(ordered))
        feature_columns = _filter_feature_columns_for_book(book, feature_columns)
        if not feature_columns:
            logger.warning(
                "Skipping book %s because no usable features remain after filtering",
                book,
            )
            continue
        book_manifest: List[Dict[str, object]] = []
        shap_sum: NDArray[np.float64] | None = None
        shap_rows = 0
        shap_expected: List[float] = []
        gain_sum: NDArray[np.float64] | None = None

        book_frame["open_date_date"] = book_frame["open_date"].dt.date

        for fold_index, (train_dates, test_dates) in enumerate(splits, start=1):
            train_dates_set = {_normalise_datetime(item).date() for item in train_dates}
            test_dates_set = {_normalise_datetime(item).date() for item in test_dates}

            train_mask = book_frame["open_date_date"].isin(train_dates_set)
            test_mask = book_frame["open_date_date"].isin(test_dates_set)
            train_rows_df = book_frame[train_mask].copy()
            test_rows_df = book_frame[test_mask].copy()
            if train_rows_df.empty or test_rows_df.empty:
                continue

            filtered_rows, X_train, y_train = _build_training_matrix(
                train_rows_df.drop(columns=["open_date_date"]), feature_columns
            )
            if len(filtered_rows) < 2:
                continue

            model = LGBMRegressor(
                objective="quantile",
                alpha=active_settings.ALPHA,
                random_state=active_settings.MODEL_SEED,
                n_estimators=100,
                learning_rate=0.1,
                n_jobs=1,
                verbose=-1,
            )
            model.fit(X_train, y_train)

            fold_dir = resolved_artifact_dir / "train" / book / f"fold_{fold_index:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            discovery_fold_dir = (
                resolved_artifact_dir / "discovery" / book / f"fold_{fold_index:02d}"
            )
            discovery_fold_dir.mkdir(parents=True, exist_ok=True)

            model_path = fold_dir / "model.pkl"
            metadata_path = fold_dir / "metadata.json"
            train_path = fold_dir / "train.csv"
            test_path = fold_dir / "test.csv"
            legacy_predictions_path = fold_dir / "train_predictions.csv"
            train_predictions_path = discovery_fold_dir / "train_predictions.csv"
            test_predictions_path = discovery_fold_dir / "test_predictions.csv"

            with model_path.open("wb") as handle:
                pickle.dump(model, handle)

            train_serialised = [dict(row) for row in filtered_rows]
            test_serialised = test_rows_df.drop(columns=["open_date_date"]).to_dict(
                "records"
            )

            _write_rows(train_path, train_serialised)
            _write_rows(test_path, test_serialised)

            prediction_frame = pd.DataFrame(X_train, columns=feature_columns)
            predictions = list(model.predict(prediction_frame))
            train_prediction_rows = _build_prediction_rows(
                filtered_rows,
                predictions,
                book=book,
                kill_switch_lookup=kill_switch_lookup,
            )

            test_filtered_rows, X_test, _ = _build_training_matrix(
                test_rows_df.drop(columns=["open_date_date"]), feature_columns
            )
            if X_test:
                test_frame = pd.DataFrame(X_test, columns=feature_columns)
                test_predictions = list(model.predict(test_frame))
                test_prediction_rows = _build_prediction_rows(
                    test_filtered_rows,
                    test_predictions,
                    book=book,
                    kill_switch_lookup=kill_switch_lookup,
                )
            else:
                test_prediction_rows = []

            _write_rows(train_predictions_path, train_prediction_rows)
            _write_rows(test_predictions_path, test_prediction_rows)
            shutil.copyfile(train_predictions_path, legacy_predictions_path)

            metadata: Dict[str, object] = {
                "book": book,
                "fold": fold_index,
                "feature_columns": feature_columns,
                "target": "pnl",
                "train_dates": [value.isoformat() for value in train_dates],
                "test_dates": [value.isoformat() for value in test_dates],
                "train_size": int(len(filtered_rows)),
                "test_size": int(len(test_rows_df)),
            }
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)

            artifact = FoldArtifact(
                book=book,
                fold=fold_index,
                directory=fold_dir,
                discovery_dir=discovery_fold_dir,
                model_path=model_path,
                metadata_path=metadata_path,
                train_path=train_path,
                test_path=test_path,
                train_predictions_path=train_predictions_path,
                test_predictions_path=test_predictions_path,
                legacy_train_predictions_path=legacy_predictions_path,
            )
            artifacts.append(artifact)
            book_manifest.append(
                {
                    "fold": fold_index,
                    "path": artifact.directory.relative_to(
                        resolved_artifact_dir
                    ).as_posix(),
                    "train_size": metadata["train_size"],
                    "test_size": metadata["test_size"],
                }
            )

            shap_summary = _compute_global_shap_summary(model, X_train)
            if shap_summary is not None:
                fold_sum, fold_count, fold_expected_value = shap_summary
                if shap_sum is None:
                    shap_sum = np.zeros_like(fold_sum, dtype=float)
                if shap_sum.shape == fold_sum.shape:
                    shap_sum += fold_sum
                    shap_rows += fold_count
                    if fold_expected_value is not None and not math.isnan(
                        fold_expected_value
                    ):
                        shap_expected.append(float(fold_expected_value))

            gain_vector = _extract_gain_importance(model)
            if gain_vector is not None:
                if gain_sum is None:
                    gain_sum = np.zeros_like(gain_vector, dtype=float)
                if gain_sum.shape == gain_vector.shape:
                    gain_sum += gain_vector

        if book_manifest:
            book_entry: Dict[str, object] = {
                "folds": book_manifest,
                "feature_columns": feature_columns,
            }
            if shap_sum is not None and shap_rows > 0:
                mean_abs = shap_sum / float(shap_rows)
                importance: Dict[str, float] = {}
                shap_pairs: List[tuple[str, float]] = []
                for column, value in zip(feature_columns, mean_abs.tolist()):
                    numeric = float(value) if math.isfinite(value) else 0.0
                    importance[column] = numeric
                    shap_pairs.append((column, numeric))
                expected_values = [
                    value for value in shap_expected if math.isfinite(value)
                ]
                book_expected_value: float | None
                if expected_values:
                    book_expected_value = float(
                        sum(expected_values) / len(expected_values)
                    )
                else:
                    book_expected_value = None
                book_entry["shap"] = {
                    "expected_value": book_expected_value,
                    "importance": importance,
                }
                shap_summary_rows = sorted(
                    shap_pairs, key=lambda item: item[1], reverse=True
                )
            else:
                shap_summary_rows = []

            importance_source = ""
            ranked_importance: List[tuple[str, float]] = []
            if shap_sum is not None and shap_rows > 0:
                importance_source = "shap"
                ranked_importance = shap_summary_rows
            elif gain_sum is not None:
                gain_pairs: List[tuple[str, float]] = []
                for column, value in zip(feature_columns, gain_sum.tolist()):
                    numeric = float(value) if math.isfinite(value) else 0.0
                    gain_pairs.append((column, numeric))
                ranked_importance = sorted(
                    gain_pairs, key=lambda item: item[1], reverse=True
                )
                if ranked_importance:
                    importance_source = "gain"
            else:
                ranked_importance = []

            if ranked_importance and importance_source:
                feature_importance_path = (
                    resolved_artifact_dir / f"feature_importance_{book}.csv"
                )
                with feature_importance_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["rank", "feature", "importance", "method"],
                    )
                    writer.writeheader()
                    for index, (feature_name, feature_value) in enumerate(
                        ranked_importance, start=1
                    ):
                        writer.writerow(
                            {
                                "rank": index,
                                "feature": feature_name,
                                "importance": feature_value,
                                "method": importance_source,
                            }
                        )

            if shap_summary_rows and shap_sum is not None and shap_rows > 0:
                shap_summary_path = resolved_artifact_dir / f"shap_summary_{book}.csv"
                with shap_summary_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["feature", "mean_abs_shap", "expected_value"],
                    )
                    writer.writeheader()
                    expected_str: str | float
                    if book_entry.get("shap") and isinstance(
                        book_entry["shap"], Mapping
                    ):
                        expected_value = cast(
                            Mapping[str, object], book_entry["shap"]
                        ).get("expected_value")
                    else:
                        expected_value = None
                    for feature_name, feature_value in shap_summary_rows:
                        if isinstance(expected_value, (int, float)) and math.isfinite(
                            float(expected_value)
                        ):
                            expected_str = float(expected_value)
                        else:
                            expected_str = ""
                        writer.writerow(
                            {
                                "feature": feature_name,
                                "mean_abs_shap": feature_value,
                                "expected_value": expected_str,
                            }
                        )
            manifest[book] = book_entry

    if manifest:
        manifest_path = resolved_artifact_dir / "train" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({"books": manifest}, handle, indent=2, sort_keys=True)

    artifacts.sort(key=lambda item: (item.book, item.fold))
    return TrainResult(artifacts=artifacts, feature_frame=feature_frame)
