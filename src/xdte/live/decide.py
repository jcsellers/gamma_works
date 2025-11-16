"""Live decision entrypoints for XDTE."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import pickle
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    TypedDict,
    cast,
)

import numpy as np

from xdte.config import Settings, get_settings
from xdte.live.providers import YFinanceDailyContextProvider, YFinanceMarketDataProvider
from xdte.live.sessions import (
    SESSION_TO_SUFFIX,
    SUFFIX_ELEVEN_AM,
    SUFFIX_FIFTEEN_FIFTEEN,
)


class MarketDataProvider(Protocol):
    """Callable returning live market information for a session."""

    def __call__(self, session: str) -> Mapping[str, object]:
        """Return a mapping describing the session level market snapshot."""


class DailyContextProvider(Protocol):
    """Callable returning lagged daily context for the supplied open date."""

    def __call__(self, open_date: date) -> Mapping[str, object]:
        """Return a mapping of lagged daily indicators for ``open_date``."""


@dataclass(frozen=True)
class DecisionRecord:
    """Single decision emitted for live execution."""

    session: str
    book: str
    action: str
    value: object


@dataclass(frozen=True)
class SessionDecision:
    """Feature snapshot and guardrail feedback for a decision window."""

    session: str
    open_date: date
    features: Mapping[str, object]
    warnings: Sequence[str]


class WarningPayload(TypedDict):
    """Structured payload describing a warning emitted during a live run."""

    level: str
    message: str
    source: str
    timestamp: str


@dataclass(frozen=True)
class LiveDecisionResult:
    """Container describing the outcome of a live decision run."""

    kit_dir: Path
    manifest: Mapping[str, object]
    policy: Mapping[str, object]
    sessions: Sequence[SessionDecision]
    decisions: Sequence[DecisionRecord]
    warnings: Sequence[WarningPayload]
    generated_at: datetime
    decision_context_id: str | None = None
    frozen_snapshot: Mapping[str, object] | None = None


@dataclass(frozen=True)
class FoldPrediction:
    """Prediction detail for a single fold."""

    fold: int
    prediction: float
    threshold: float | None
    keep: bool


@dataclass(frozen=True)
class BookPrediction:
    """Aggregated prediction summary for a book."""

    prediction: float
    threshold: float | None
    keep: bool
    folds: Sequence[FoldPrediction]
    shap_expected_value: float | None = None
    shap_importance: Mapping[str, float] | None = None
    shap_contributions: Sequence[tuple[str, float]] = ()
    shap_top_contributors: Sequence[tuple[str, float]] = ()


__all__ = [
    "DecisionRecord",
    "SessionDecision",
    "LiveDecisionResult",
    "KitIntegrityError",
    "run_live_decisions",
]


_SESSION_SUFFIX = SESSION_TO_SUFFIX
_BOOK_SESSION = {
    "CALLS_0DTE_11": "11:00",
    "PUTS_0DTE_11": "11:00",
    "CALLS_1DTE_1515": "15:15",
    "PUTS_1DTE_1515": "15:15",
}
_DAILY_FEATURES = [
    "DoW",
    "L1_TS",
    "L1_VIX_Close",
    "L1_VIX_pct",
    "L1_vvix_close",
    "L1_vvix_pct",
    "L1_vvix_ema20",
    "L1_vvix_ema30",
    "L1_SPX_ATR_Pct",
    "L1_SPX_Drawdown_Pct",
    "L1_rv5",
    "L1_rv20",
    "L1_vvix_above_ema20",
    "L1_vvix_above_ema30",
]

_MISSING = object()

_FROZEN_SNAPSHOT_FILENAME = "_frozen_decision_snapshot.json"


def _json_ready(value: object) -> object:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item) for item in value]
    return value


def _json_ready_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_ready(value) for key, value in mapping.items()}


def _load_frozen_snapshot(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to load frozen snapshot %s: %s", path, exc)
        return None
    if not isinstance(payload, Mapping):
        logger.warning("Frozen snapshot at %s is not a mapping", path)
        return None
    return cast(Mapping[str, object], payload)


def _persist_frozen_snapshot(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_context_fingerprint(
    sessions: Sequence[SessionDecision],
    decisions: Sequence[DecisionRecord],
) -> Mapping[str, object]:
    sessions_payload = [
        {
            "session": session.session,
            "open_date": session.open_date.isoformat(),
            "features": _json_ready_mapping(session.features),
            "warnings": list(session.warnings),
        }
        for session in sessions
    ]
    decisions_payload = [
        {
            "session": decision.session,
            "book": decision.book,
            "action": decision.action,
            "value": _json_ready(decision.value),
        }
        for decision in decisions
    ]
    return {
        "sessions": sessions_payload,
        "decisions": decisions_payload,
    }


def _build_warning(
    message: str,
    *,
    source: str,
    level: str = "WARNING",
    timestamp: datetime | None = None,
) -> WarningPayload:
    current_time = timestamp or datetime.now(timezone.utc)
    return WarningPayload(
        level=level,
        message=message,
        source=source,
        timestamp=current_time.isoformat(),
    )


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):  # pragma: no cover - defensive guard
        msg = f"Expected mapping payload in {path}"
        raise TypeError(msg)
    return cast(Mapping[str, object], data)


def _coerce_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:  # pragma: no cover - defensive guard
            return None
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result):
            return None
        return result
    if isinstance(value, str) and value:
        try:
            result = float(value)
        except ValueError:
            return None
        if math.isnan(result):
            return None
        return result
    return None


def _build_market_features(
    session: str, payload: Mapping[str, object]
) -> tuple[date | None, Mapping[str, object], list[str]]:
    suffix = _SESSION_SUFFIX.get(session)
    if suffix is None:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported session: {session}")

    warnings: list[str] = []
    open_date = _coerce_date(payload.get("open_date"))
    vix_value = _coerce_float(
        payload.get("vix") if "vix" in payload else payload.get("VIX")
    )
    if vix_value is None:
        warnings.append(f"{session}: missing VIX input")

    # Use explicit key checking to avoid skipping 0/0.0 values
    entry_level_val = (
        payload["spx_entry"]
        if "spx_entry" in payload
        else (
            payload["spx_last"]
            if "spx_last" in payload
            else (
                payload["spx_close"]
                if "spx_close" in payload
                else payload["entry_level"] if "entry_level" in payload else None
            )
        )
    )
    entry_level = _coerce_float(entry_level_val)
    open_level_val = (
        payload["spx_open"]
        if "spx_open" in payload
        else (
            payload["open_level"]
            if "open_level" in payload
            else payload["spx_start"] if "spx_start" in payload else None
        )
    )
    open_level = _coerce_float(open_level_val)
    move_value = _coerce_float(payload.get("intraday_move"))
    if move_value is None and entry_level is not None and open_level is not None:
        move_value = entry_level - open_level
    if move_value is None:
        warnings.append(f"{session}: missing intraday move input")

    features: MutableMapping[str, object] = {}
    features[f"VIX_Entry_{suffix}"] = vix_value
    features[f"Intraday_Move_OpenToEntry_{suffix}"] = move_value
    gap_value = _coerce_float(payload.get("gap"))
    if gap_value is not None:
        features["gap"] = gap_value
    movement_value = _coerce_float(payload.get("movement"))
    if movement_value is not None:
        features["movement"] = movement_value
    else:
        features["movement"] = move_value
    closing_vix = _coerce_float(payload.get("closing_vix"))
    if closing_vix is not None:
        features["closing_vix"] = closing_vix
    opening_vix = _coerce_float(payload.get("opening_vix"))
    if opening_vix is not None:
        features["opening_vix"] = opening_vix
    elif vix_value is not None:
        features["opening_vix"] = vix_value
    return open_date, features, warnings


def _build_daily_features(
    open_date: date, payload: Mapping[str, object]
) -> tuple[Mapping[str, object], list[str]]:
    features: MutableMapping[str, object] = {"DoW": open_date.weekday()}
    warnings: list[str] = []

    for key in _DAILY_FEATURES:
        if key == "DoW":
            continue
        if key in payload:
            value = payload[key]
            if key.startswith("L1_vvix_above_"):
                features[key] = bool(value)
            else:
                features[key] = _coerce_float(value)
        else:
            features[key] = None
            warnings.append(f"daily: missing {key}")

    close_value = _coerce_float(payload.get("L1_vvix_close"))
    ema20 = _coerce_float(payload.get("L1_vvix_ema20"))
    ema30 = _coerce_float(payload.get("L1_vvix_ema30"))

    if "L1_vvix_above_ema20" not in payload:
        if close_value is not None and ema20 is not None:
            features["L1_vvix_above_ema20"] = close_value > ema20
        else:
            features["L1_vvix_above_ema20"] = False
    if "L1_vvix_above_ema30" not in payload:
        if close_value is not None and ema30 is not None:
            features["L1_vvix_above_ema30"] = close_value > ema30
        else:
            features["L1_vvix_above_ema30"] = False

    return features, warnings


def _decisions_from_policy(policy: Mapping[str, object]) -> list[DecisionRecord]:
    decisions: list[DecisionRecord] = []

    tails = policy.get("tails", {})
    if isinstance(tails, Mapping):
        for book, bounds in sorted(tails.items()):
            session = _BOOK_SESSION.get(str(book), "UNKNOWN")
            decisions.append(
                DecisionRecord(
                    session=session, book=str(book), action="SET_TAILS", value=bounds
                )
            )

    gammas = policy.get("gammas", {})
    if isinstance(gammas, Mapping):
        for book, gamma in sorted(gammas.items()):
            session = _BOOK_SESSION.get(str(book), "UNKNOWN")
            decisions.append(
                DecisionRecord(
                    session=session, book=str(book), action="ADJUST_GAMMA", value=gamma
                )
            )

    decisions.sort(key=lambda record: (record.session, record.book, record.action))
    return decisions


def _load_model(path: Path) -> Any:
    with path.open("rb") as handle:
        model = pickle.load(handle)  # nosec B301 - models are trusted local artifacts
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not expose a predict method")
    return model


def _feature_vector(
    columns: Sequence[str], features: Mapping[str, object]
) -> list[float]:
    vector: list[float] = []
    for column in columns:
        value = features.get(column, None)
        numeric = _coerce_float(value)
        vector.append(numeric if numeric is not None else math.nan)
    return vector


def _load_keep_threshold(discovery_dir: Path, book: str, fold: int) -> float | None:
    metadata_path = discovery_dir / book / f"fold_{fold:02d}" / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = _load_json_mapping(metadata_path)
    threshold = _coerce_float(metadata.get("keep_threshold"))
    return threshold


def _load_shap_summary(
    models_dir: Path, book: str
) -> tuple[Sequence[tuple[str, float]], float | None]:
    shap_path = models_dir / f"shap_summary_{book}.csv"
    if not shap_path.exists():
        return (), None
    contributions: list[tuple[str, float]] = []
    expected_value: float | None = None
    try:
        with shap_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if expected_value is None:
                    candidate = _coerce_float(row.get("expected_value"))
                    if candidate is not None:
                        expected_value = candidate
                feature = row.get("feature")
                if not isinstance(feature, str) or not feature:
                    continue
                raw_value: object = row.get("mean_abs_shap")
                value = _coerce_float(raw_value)
                if value is None:
                    continue
                contributions.append((feature, value))
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to load SHAP summary for %s: %s", book, exc)
        return (), None
    contributions.sort(key=lambda item: item[1], reverse=True)
    return tuple(contributions), expected_value


def _coerce_int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return 0
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _validate_feature_columns(
    book: str,
    feature_columns: Sequence[str],
    features: Mapping[str, object],
) -> tuple[bool, str | None]:
    missing: list[str] = []
    invalid: list[str] = []

    for column in feature_columns:
        value = features.get(column)
        if value is None:
            missing.append(column)
            continue
        numeric = _coerce_float(value)
        if numeric is None or math.isnan(numeric):
            invalid.append(column)

    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append("missing: " + ", ".join(sorted(missing)))
        if invalid:
            parts.append("invalid: " + ", ".join(sorted(invalid)))
        detail = "; ".join(parts)
        message = f"{book}: required feature validation failed ({detail})"
        return False, message
    return True, None


def _score_live_predictions(
    models_dir: Path,
    discovery_dir: Path,
    book_features: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, BookPrediction], list[WarningPayload], dict[str, str]]:
    summary: dict[str, BookPrediction] = {}
    warnings: list[WarningPayload] = []
    skipped: dict[str, str] = {}

    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        warnings.append(
            _build_warning(
                f"Model manifest not found: {manifest_path}",
                source="model_scoring",
            )
        )
        return summary, warnings, skipped

    manifest = _load_json_mapping(manifest_path)
    books_meta = manifest.get("books", {})
    if not isinstance(books_meta, Mapping):
        warnings.append(
            _build_warning(
                "Model manifest missing 'books' mapping",
                source="model_scoring",
            )
        )
        return summary, warnings, skipped

    for book, features in book_features.items():
        meta = books_meta.get(book)
        if not isinstance(meta, Mapping):
            warnings.append(
                _build_warning(
                    f"Model metadata missing for {book}",
                    source="model_scoring",
                )
            )
            continue
        columns_raw = meta.get("feature_columns")
        if not isinstance(columns_raw, Sequence):
            warnings.append(
                _build_warning(
                    f"No feature columns recorded for {book}",
                    source="model_scoring",
                )
            )
            continue
        feature_columns = [str(column) for column in columns_raw]
        duplicates = [
            column for column, count in Counter(feature_columns).items() if count > 1
        ]
        if duplicates:
            detail = (
                f"{book}: duplicate feature columns detected ("
                + ", ".join(sorted(set(duplicates)))
                + ")"
            )
            logger.critical(detail)
            warning_payload = _build_warning(
                detail,
                source="model_scoring",
                level="CRITICAL",
            )
            warnings.append(warning_payload)
            skipped[str(book)] = detail
            continue
        valid, failure_reason = _validate_feature_columns(
            str(book), feature_columns, features
        )
        if not valid and failure_reason:
            logger.critical(failure_reason)
            warning_payload = _build_warning(
                failure_reason,
                source="model_scoring",
                level="CRITICAL",
            )
            warnings.append(warning_payload)
            skipped[str(book)] = failure_reason
            continue
        fold_entries_raw = meta.get("folds", [])
        fold_entries = (
            fold_entries_raw if isinstance(fold_entries_raw, Sequence) else []
        )
        fold_results: list[FoldPrediction] = []
        shap_expected_value = None
        shap_importance: Mapping[str, float] | None = None
        shap_contributions: Sequence[tuple[str, float]] = ()
        shap_meta = meta.get("shap")
        if isinstance(shap_meta, Mapping):
            shap_expected_value = _coerce_float(shap_meta.get("expected_value"))
            importance_raw = shap_meta.get("importance", {})
            if isinstance(importance_raw, Mapping):
                importance_values: dict[str, float] = {}
                for feature, value in importance_raw.items():
                    if not isinstance(feature, str):
                        continue
                    numeric = _coerce_float(value)
                    if numeric is None:
                        continue
                    importance_values[feature] = numeric
                if importance_values:
                    shap_importance = importance_values

        shap_summary_rows, shap_summary_expected = _load_shap_summary(
            models_dir, str(book)
        )
        if shap_summary_rows:
            shap_contributions = shap_summary_rows
            if shap_importance is None:
                shap_importance = {
                    feature: value for feature, value in shap_summary_rows
                }
        if shap_expected_value is None and shap_summary_expected is not None:
            shap_expected_value = shap_summary_expected

        for entry in fold_entries:
            if not isinstance(entry, Mapping):
                continue
            fold_index = _coerce_int_value(entry.get("fold"))
            path_value = entry.get("path")
            if not isinstance(path_value, str):
                warnings.append(
                    _build_warning(
                        f"Fold path missing for {book} fold {fold_index:02d}",
                        source="model_scoring",
                    )
                )
                continue
            fold_dir = models_dir / path_value
            model_path = fold_dir / "model.pkl"
            if not model_path.exists():
                warnings.append(
                    _build_warning(
                        f"Model artifact missing for {book} fold {fold_index:02d}",
                        source="model_scoring",
                    )
                )
                continue
            try:
                model = _load_model(model_path)
                vector = [_feature_vector(feature_columns, features)]
                predictor = cast(
                    Callable[[Sequence[Sequence[float]]], Sequence[Any]],
                    getattr(model, "predict"),
                )
                raw_prediction = predictor(vector)
                if not raw_prediction:
                    raise ValueError("Model returned no predictions")
                prediction = float(raw_prediction[0])
            except Exception as exc:  # pragma: no cover - defensive guard
                warnings.append(
                    _build_warning(
                        f"Prediction failed for {book} fold {fold_index:02d}: {exc}",
                        source="model_scoring",
                    )
                )
                continue
            threshold = _load_keep_threshold(discovery_dir, book, fold_index)
            keep_flag = (
                threshold is not None
                and not math.isnan(threshold)
                and prediction >= threshold
            )
            fold_results.append(
                FoldPrediction(
                    fold=fold_index,
                    prediction=prediction,
                    threshold=threshold,
                    keep=keep_flag,
                )
            )

        if not fold_results:
            warnings.append(
                _build_warning(
                    f"No predictions generated for {book}",
                    source="model_scoring",
                )
            )
            continue

        mean_prediction = statistics.fmean(
            fold_summary.prediction for fold_summary in fold_results
        )
        thresholds: list[float] = []
        for fold_summary in fold_results:
            current_threshold = fold_summary.threshold
            if current_threshold is None:
                continue
            if math.isnan(current_threshold):
                continue
            thresholds.append(current_threshold)
        if thresholds:
            mean_threshold = statistics.fmean(thresholds)
            keep_flag = mean_prediction >= mean_threshold
            threshold_value: float | None = mean_threshold
        else:
            keep_flag = True  # Default to True if no valid thresholds exist
            threshold_value = None
        ordered_folds = sorted(fold_results, key=lambda fold_summary: fold_summary.fold)
        top_contributors: Sequence[tuple[str, float]] = tuple(shap_contributions[:3])
        summary[book] = BookPrediction(
            prediction=mean_prediction,
            threshold=threshold_value,
            keep=keep_flag,
            folds=tuple(ordered_folds),
            shap_expected_value=shap_expected_value,
            shap_importance=shap_importance,
            shap_contributions=shap_contributions,
            shap_top_contributors=top_contributors,
        )

    return summary, warnings, skipped


def _normalise_float(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return float(value)


def _decisions_from_predictions(
    predictions: Mapping[str, BookPrediction],
    skipped: Mapping[str, str] | None = None,
) -> list[DecisionRecord]:
    decisions: list[DecisionRecord] = []
    for book, summary in sorted(predictions.items()):
        session = _BOOK_SESSION.get(book, "UNKNOWN")
        folds = [
            {
                "fold": fold.fold,
                "prediction": _normalise_float(fold.prediction),
                "threshold": _normalise_float(fold.threshold),
                "keep": fold.keep,
            }
            for fold in summary.folds
        ]
        value: dict[str, object] = {
            "keep": summary.keep,
            "prediction": _normalise_float(summary.prediction),
            "threshold": _normalise_float(summary.threshold),
            "folds": folds,
        }
        shap_payload: dict[str, object] | None = None
        contributions_payload: list[dict[str, object]] = []
        if summary.shap_contributions:
            for feature, impact in summary.shap_contributions:
                normalised_impact = _normalise_float(impact)
                if normalised_impact is None:
                    continue
                contributions_payload.append(
                    {
                        "feature": feature,
                        "impact": normalised_impact,
                        "mean_abs_shap": normalised_impact,
                        "label": f"{feature}: {impact:+.3f}",
                    }
                )
            if contributions_payload:
                shap_payload = {"contributions": contributions_payload}
        top_entries: list[dict[str, object]] = []
        if summary.shap_top_contributors:
            for feature, impact in summary.shap_top_contributors:
                normalised_impact = _normalise_float(impact)
                if normalised_impact is None:
                    continue
                top_entries.append(
                    {
                        "feature": feature,
                        "impact": normalised_impact,
                        "label": f"{feature}: {impact:+.3f}",
                    }
                )
        elif contributions_payload:
            top_entries = contributions_payload[:3]
        if top_entries:
            if shap_payload is None:
                shap_payload = {}
            shap_payload["top_contributors"] = top_entries
        if summary.shap_expected_value is not None:
            normalised_expected = _normalise_float(summary.shap_expected_value)
            if normalised_expected is not None:
                if shap_payload is None:
                    shap_payload = {}
                shap_payload["expected_value"] = normalised_expected
        if summary.shap_importance:
            ordered = sorted(
                summary.shap_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top_global: list[dict[str, object]] = []
            for feature, importance in ordered[:5]:
                normalised_importance = _normalise_float(importance)
                if normalised_importance is None:
                    continue
                top_global.append(
                    {"feature": feature, "importance": normalised_importance}
                )
            if top_global:
                if shap_payload is None:
                    shap_payload = {}
                shap_payload["global_importance"] = top_global
        if shap_payload:
            value["shap"] = shap_payload
        decisions.append(
            DecisionRecord(
                session=session,
                book=book,
                action="TRADE_SIGNAL",
                value=value,
            )
        )
    if skipped:
        for book, reason in sorted(skipped.items()):
            session = _BOOK_SESSION.get(book, "UNKNOWN")
            decisions.append(
                DecisionRecord(
                    session=session,
                    book=book,
                    action="SKIP",
                    value={"reason": reason},
                )
            )
    return decisions


def run_live_decisions(
    kit_dir: Path,
    *,
    settings: Settings | None = None,
    market_provider: MarketDataProvider | None = None,
    daily_provider: DailyContextProvider | None = None,
    cache_dir: Path | str | None = None,
    cache_freshness: timedelta | None = timedelta(minutes=15),
    freeze_now: bool = True,
) -> LiveDecisionResult:
    """Load a Live Kit bundle and emit operational decisions."""

    resolved = kit_dir.resolve()
    manifest_path = resolved / "manifest.json"
    policy_path = resolved / "hybrid" / "policy.json"

    if not manifest_path.exists():  # pragma: no cover - defensive guard
        msg = f"Live Kit manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    active_settings = settings or get_settings()
    _ = active_settings  # honour the interface even if unused for now

    snapshot_path = resolved / _FROZEN_SNAPSHOT_FILENAME
    frozen_snapshot_raw = _load_frozen_snapshot(snapshot_path) if freeze_now else None

    snapshot_market_payloads: dict[str, dict[str, object]] = {}
    snapshot_daily_payload: dict[str, object] | None = None
    stored_generated_at: datetime | None = None
    stored_context_id: str | None = None
    stored_extra_daily: list[str] | None = None
    stored_warning_objects: list[WarningPayload] | None = None
    stored_open_date: date | None = None

    if frozen_snapshot_raw is not None:
        market_payload = frozen_snapshot_raw.get("market")
        if isinstance(market_payload, Mapping):
            snapshot_market_payloads = {
                str(session): dict(cast(Mapping[str, object], payload))
                for session, payload in market_payload.items()
                if isinstance(payload, Mapping)
            }
        stored_daily_payload = frozen_snapshot_raw.get("daily")
        if isinstance(stored_daily_payload, Mapping):
            snapshot_daily_payload = dict(stored_daily_payload)
        generated_at_value = frozen_snapshot_raw.get("generated_at")
        if isinstance(generated_at_value, str):
            try:
                stored_generated_at = datetime.fromisoformat(generated_at_value)
            except ValueError:  # pragma: no cover - defensive guard
                stored_generated_at = None
        context_value = frozen_snapshot_raw.get("decision_context_id")
        if isinstance(context_value, str):
            stored_context_id = context_value
        extra_daily = frozen_snapshot_raw.get("extra_daily_warnings")
        if isinstance(extra_daily, Sequence) and not isinstance(
            extra_daily, (str, bytes)
        ):
            stored_extra_daily = [str(entry) for entry in extra_daily]
        warnings_payload = frozen_snapshot_raw.get("warnings")
        if isinstance(warnings_payload, Sequence) and not isinstance(
            warnings_payload, (str, bytes)
        ):
            stored_warning_objects = []
            for entry in warnings_payload:
                if isinstance(entry, Mapping):
                    stored_warning_objects.append(cast(WarningPayload, dict(entry)))
        open_date_value = frozen_snapshot_raw.get("open_date")
        if isinstance(open_date_value, str) and open_date_value:
            try:
                stored_open_date = datetime.fromisoformat(open_date_value).date()
            except ValueError:  # pragma: no cover - defensive guard
                stored_open_date = None

    warnings: list[WarningPayload] = []
    fetched_fresh_data = False

    default_daily: YFinanceDailyContextProvider | None = None
    default_market: YFinanceMarketDataProvider | None = None

    if market_provider is None:
        default_daily = YFinanceDailyContextProvider(
            cache_dir=cache_dir,
            cache_freshness=cache_freshness,
        )
        default_market = YFinanceMarketDataProvider(
            daily_context=default_daily,
            cache_dir=cache_dir,
            cache_freshness=cache_freshness,
        )
        market_provider = default_market
        # Also use the default daily provider when creating all defaults
        if daily_provider is None:
            daily_provider = default_daily

    providers_to_check: list[object] = []
    if default_market is not None:
        providers_to_check.append(default_market)
    elif market_provider is not None:
        providers_to_check.append(market_provider)
    if default_daily is not None:
        providers_to_check.append(default_daily)
    elif daily_provider is not None:
        providers_to_check.append(daily_provider)

    for provider in providers_to_check:
        provider_warnings = getattr(provider, "warnings", None)
        if provider_warnings:
            for message in cast(Iterable[str], provider_warnings):
                warnings.append(
                    _build_warning(
                        str(message),
                        source=type(provider).__name__,
                    )
                )

    manifest = _load_json_mapping(manifest_path)
    policy: Mapping[str, object]
    if policy_path.exists():
        policy = _load_json_mapping(policy_path)
    else:  # pragma: no cover - defensive guard
        policy = cast(Mapping[str, object], {})

    session_snapshots: list[SessionDecision] = []
    open_date: date | None = None

    session_features: dict[str, Mapping[str, object]] = {}
    session_warnings: dict[str, list[str]] = {}

    for session in _SESSION_SUFFIX:
        if frozen_snapshot_raw is not None and session in snapshot_market_payloads:
            payload = snapshot_market_payloads.get(session, {})
            payload_mapping = dict(payload)
        else:
            if market_provider is None:  # pragma: no cover - defensive guard
                raise RuntimeError("Market provider unavailable for live run")
            fetched_payload = market_provider(session)
            payload_mapping = dict(fetched_payload)
            fetched_fresh_data = True
            if freeze_now:
                snapshot_market_payloads[session] = _json_ready_mapping(payload_mapping)
        session_open_date, features, session_messages = _build_market_features(
            session, payload_mapping
        )
        if session_open_date is not None:
            if open_date is None:
                open_date = session_open_date
            elif open_date != session_open_date:
                session_messages.append(
                    "session open_date mismatch with other sessions"
                )
        session_features[session] = features
        session_warnings[session] = session_messages

    if open_date is None:
        open_date = stored_open_date or date.today()

    extra_daily_warnings: list[str] = []
    if stored_extra_daily is not None:
        extra_daily_warnings = list(stored_extra_daily)

    if frozen_snapshot_raw is not None and snapshot_daily_payload is not None:
        daily_payload_mapping = dict(snapshot_daily_payload)
    elif daily_provider is not None:
        try:
            fetched_daily = daily_provider(open_date)
        except Exception as exc:
            failure_warning = (
                f"daily provider failed for {open_date.isoformat()}: {exc}"
            )
            warning_payload = _build_warning(
                failure_warning,
                source=type(daily_provider).__name__,
            )
            warnings.append(warning_payload)
            extra_daily_warnings.append(failure_warning)
            daily_payload_mapping = {}
            fetched_fresh_data = True
        else:
            daily_payload_mapping = dict(fetched_daily)
            fetched_fresh_data = True
            if freeze_now:
                snapshot_daily_payload = _json_ready_mapping(daily_payload_mapping)
    else:
        daily_payload_mapping = {}

    daily_payload: Mapping[str, object] = daily_payload_mapping
    daily_features, daily_warnings = _build_daily_features(open_date, daily_payload)
    if extra_daily_warnings:
        daily_warnings = [*extra_daily_warnings, *daily_warnings]

    session_combined_features: dict[str, Mapping[str, object]] = {}
    for session in _SESSION_SUFFIX:
        combined: dict[str, object] = dict(daily_features)
        combined.update(session_features[session])
        combined["open_date"] = open_date
        suffix = _SESSION_SUFFIX[session]
        if suffix == SUFFIX_ELEVEN_AM:
            entry = _coerce_float(combined.get("VIX_Entry_11"))
            l1_close = _coerce_float(combined.get("L1_VIX_Close"))
            if entry is not None and l1_close is not None:
                combined["t0_VIX_change_from_close_11"] = entry - l1_close
        if suffix == SUFFIX_FIFTEEN_FIFTEEN:
            entry = _coerce_float(combined.get("VIX_Entry_1515"))
            l1_close = _coerce_float(combined.get("L1_VIX_Close"))
            if entry is not None and l1_close is not None:
                combined["t0_VIX_change_from_close_15"] = entry - l1_close
        session_warning_messages = list(session_warnings[session])
        session_warning_messages.extend(daily_warnings)
        session_snapshots.append(
            SessionDecision(
                session=session,
                open_date=open_date,
                features=combined,
                warnings=session_warning_messages,
            )
        )
        session_combined_features[session] = combined

    book_feature_rows: dict[str, Mapping[str, object]] = {}
    for book, session in _BOOK_SESSION.items():
        session_feature_payload = session_combined_features.get(session)
        if session_feature_payload is not None:
            combined_payload: dict[str, object] = dict(session_feature_payload)
            for other_session, other_payload in session_combined_features.items():
                if other_session == session:
                    continue
                for key, value in other_payload.items():
                    if key not in combined_payload:
                        combined_payload[key] = value
            book_feature_rows[book] = combined_payload

    _verify_export_digests(resolved, manifest)

    models_dir = resolved / "models"
    discovery_dir = resolved / "discovery"
    prediction_summary, prediction_warnings, skipped_books = _score_live_predictions(
        models_dir, discovery_dir, book_feature_rows
    )
    warnings.extend(prediction_warnings)

    decisions = _decisions_from_policy(policy)
    decisions.extend(_decisions_from_predictions(prediction_summary, skipped_books))
    decisions.sort(key=lambda record: (record.session, record.book, record.action))

    fingerprint = _build_context_fingerprint(session_snapshots, decisions)
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()
    resolved_context_id = f"sha256:{digest}"
    context_changed = (
        stored_context_id is None or stored_context_id != resolved_context_id
    )
    decision_context_id = (
        resolved_context_id if context_changed else cast(str, stored_context_id)
    )

    if stored_warning_objects is not None:
        # Persisted warnings should stay visible on subsequent runs, but they
        # must not eclipse any fresh warnings raised while rebuilding the
        # snapshot.  Merge the stored payloads with the newly generated ones
        # when new data was fetched, otherwise fall back to the stored list so
        # offline replays remain stable.
        if not warnings:
            warnings = list(stored_warning_objects)
        elif fetched_fresh_data:
            seen_keys: set[tuple[str, str, str]] = {
                (
                    warning["level"],
                    warning["message"],
                    warning["source"],
                )
                for warning in warnings
            }
            for warning in stored_warning_objects:
                warning_signature = (
                    warning["level"],
                    warning["message"],
                    warning["source"],
                )
                if warning_signature not in seen_keys:
                    warnings.append(warning)
                    seen_keys.add(warning_signature)
        else:
            warnings = list(stored_warning_objects)

    if stored_generated_at is not None and not context_changed:
        generated_at = stored_generated_at
    else:
        generated_at = datetime.now(timezone.utc)

    snapshot_payload: Mapping[str, object] | None = None
    if freeze_now:
        serialised_market = {
            session: snapshot_market_payloads.get(session, {})
            for session in sorted(_SESSION_SUFFIX)
        }
        serialised_daily = snapshot_daily_payload or {}
        snapshot_payload = {
            "market": serialised_market,
            "daily": serialised_daily,
            "open_date": open_date.isoformat() if open_date else None,
            "generated_at": generated_at.isoformat(),
            "decision_context_id": decision_context_id,
            "warnings": [_json_ready(warning) for warning in warnings],
            "extra_daily_warnings": list(extra_daily_warnings),
        }
        # Persist the snapshot if it is new or has changed
        should_persist = False
        if frozen_snapshot_raw is None:
            should_persist = True
        else:
            try:
                # Compare the new payload to the existing one
                existing_payload = frozen_snapshot_raw
                # Use json.dumps with sort_keys for deterministic comparison
                new_json = json.dumps(snapshot_payload, sort_keys=True, default=str)
                existing_json = json.dumps(
                    existing_payload, sort_keys=True, default=str
                )
                if new_json != existing_json:
                    should_persist = True
            except Exception:
                # If comparison fails, be conservative and persist
                should_persist = True
        if should_persist:
            _persist_frozen_snapshot(snapshot_path, snapshot_payload)

    return LiveDecisionResult(
        kit_dir=resolved,
        manifest=manifest,
        policy=policy,
        sessions=tuple(session_snapshots),
        decisions=tuple(decisions),
        warnings=tuple(warnings),
        generated_at=generated_at,
        decision_context_id=decision_context_id,
        frozen_snapshot=snapshot_payload,
    )


class KitIntegrityError(RuntimeError):
    """Raised when a live kit fails integrity validation."""


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_export_digests(kit_dir: Path, manifest: Mapping[str, object]) -> None:
    files = manifest.get("files")
    if not files:
        return
    if not isinstance(files, Sequence):
        msg = "Live kit manifest 'files' entry must be a list"
        raise KitIntegrityError(msg)

    missing: list[str] = []
    mismatched: list[str] = []

    for entry in files:
        if not isinstance(entry, Mapping):
            continue
        path_value = entry.get("path")
        digest_value = entry.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise KitIntegrityError("Manifest entry missing file path")
        if not isinstance(digest_value, str) or not digest_value:
            raise KitIntegrityError(
                f"Manifest entry for {path_value} is missing a sha256 digest"
            )
        file_path = kit_dir / path_value
        if not file_path.exists():
            missing.append(path_value)
            continue
        actual = _compute_sha256(file_path)
        if actual != digest_value:
            mismatched.append(path_value)

    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append("missing: " + ", ".join(sorted(missing)))
        if mismatched:
            parts.append("mismatched: " + ", ".join(sorted(mismatched)))
        detail = "; ".join(parts)
        raise KitIntegrityError(f"Kit digest validation failed; {detail}")


logger = logging.getLogger(__name__)
