"""Live kit assembly helpers for XDTE."""

from __future__ import annotations

import json
import math
import pickle
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Collection, Dict, List, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xdte.config import Settings, get_settings
from xdte.data.features import BOOK_FEATS_CONFIG
from xdte.model._lgbm import LGBMRegressor
from xdte.model.apply import BOOKS


@dataclass(frozen=True)
class ExportedFile:
    """Description of a file included in the exported kit."""

    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class ExportResult:
    """Metadata returned by :func:`build_live_kit`."""

    kit_dir: Path
    manifest_path: Path
    digests_path: Path
    files: Sequence[ExportedFile]
    digests: Mapping[str, str]


__all__ = ["ExportedFile", "ExportResult", "build_live_kit"]


def _compute_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    include: Callable[[Path], bool] | None = None,
) -> List[Path]:
    copied: List[Path] = []
    if not source.exists():
        return copied
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if include is not None and not include(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _normalise_book_filter(books: Collection[str] | None) -> set[str] | None:
    if books is None:
        return None
    return {str(book) for book in books if str(book).strip()}


def _book_directory_filter(
    source: Path, allowed_books: set[str] | None
) -> Callable[[Path], bool] | None:
    if allowed_books is None:
        return None
    existing_books = (
        {entry.name for entry in source.iterdir() if entry.is_dir()}
        if source.exists()
        else set()
    )

    def include(relative: Path) -> bool:
        parts = relative.parts
        if not parts:
            return True
        first = parts[0]
        if first in allowed_books:
            return True
        if first in existing_books:
            return False
        return True

    return include


def _extract_book_from_filename(name: str, prefix: str) -> str | None:
    if not name.startswith(prefix) or not name.endswith(".csv"):
        return None
    book = name[len(prefix) : -4]
    return book or None


def _filter_book_manifest(path: Path, allowed_books: set[str] | None) -> None:
    if allowed_books is None or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive guard
        return
    books_section = payload.get("books")
    if not isinstance(books_section, Mapping):
        return
    filtered = {
        book: data for book, data in books_section.items() if book in allowed_books
    }
    payload["books"] = filtered
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _filter_rules_ci(path: Path, allowed_books: set[str] | None) -> None:
    if allowed_books is None or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive guard
        return
    if not isinstance(payload, Mapping):
        return
    filtered = {book: data for book, data in payload.items() if book in allowed_books}
    path.write_text(
        json.dumps(filtered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_shap_payload(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    shap_payload: Dict[str, Dict[str, object]] = {}
    if not manifest_path.exists():
        return shap_payload
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive guard
        return shap_payload
    books_meta = manifest.get("books", {})
    if not isinstance(books_meta, Mapping):
        return shap_payload
    for book, metadata in books_meta.items():
        if not isinstance(metadata, Mapping):
            continue
        shap_meta = metadata.get("shap")
        if not isinstance(shap_meta, Mapping):
            continue
        importance_raw = shap_meta.get("importance", {})
        if not isinstance(importance_raw, Mapping):
            continue
        importance: Dict[str, float] = {}
        for feature, value in importance_raw.items():
            if not isinstance(feature, str):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric):
                continue
            importance[feature] = numeric
        expected_value_raw = shap_meta.get("expected_value")
        expected_value: float | None = None
        if expected_value_raw is not None:
            try:
                numeric = float(expected_value_raw)
            except (TypeError, ValueError):
                numeric = math.nan
            if not math.isnan(numeric):
                expected_value = numeric
        shap_payload[book] = {
            "expected_value": expected_value,
            "importance": importance,
        }
    return shap_payload


def _serialise_settings(settings: Settings) -> Dict[str, object]:
    return {
        "alpha": float(settings.ALPHA),
        "n_folds": int(settings.N_FOLDS),
        "winsor_p": float(settings.WINSOR_P),
        "kill_switch_tails": {
            book: [float(lower), float(upper)]
            for book, (lower, upper) in settings.KILL_SWITCH_TAILS.items()
        },
        "kill_switch_gammas": {
            book: float(value) for book, value in settings.KILL_SWITCH_GAMMAS.items()
        },
    }


def _prepare_feature_matrix(
    frame: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    matrix = frame.loc[:, list(features)].copy()
    for column in matrix.columns:
        series = matrix[column]
        if not (
            pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        ):
            matrix[column] = pd.to_numeric(series, errors="coerce")
    return matrix


def _load_book_panel(train_dir: Path) -> pd.DataFrame:
    train_root = train_dir / "train"
    if not train_root.exists():
        msg = f"Training directory does not exist: {train_root}"
        raise FileNotFoundError(msg)

    frames: List[pd.DataFrame] = []
    for book_dir in sorted(train_root.iterdir()):
        if not book_dir.is_dir():
            continue
        for fold_dir in sorted(book_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            for name in ("train.csv", "test.csv"):
                csv_path = fold_dir / name
                if csv_path.is_file():
                    frames.append(pd.read_csv(csv_path))

    if not frames:
        msg = f"No training CSV files were found under {train_root}"
        raise FileNotFoundError(msg)

    panel = pd.concat(frames, ignore_index=True)
    if "PNL" not in panel.columns:
        if "pnl" in panel.columns:
            panel["PNL"] = panel["pnl"]
        else:
            msg = "Training panel is missing the 'PNL' column"
            raise KeyError(msg)
    panel["PNL"] = pd.to_numeric(panel["PNL"], errors="coerce")
    if "pnl" in panel.columns:
        panel = panel.drop(columns=["pnl"])

    panel["open_date"] = pd.to_datetime(panel["open_date"], errors="coerce")
    panel = panel.dropna(subset=["book", "open_date"])
    panel["book"] = panel["book"].astype(str)
    panel = panel.drop_duplicates().sort_values(["book", "open_date"])
    panel = panel.reset_index(drop=True)
    return panel


def _fit_full_model(
    book: str,
    book_panel: pd.DataFrame,
    features: Sequence[str],
    *,
    alpha: float,
    seed: int,
) -> LGBMRegressor:
    if not features:
        msg = f"No configured features available for book {book}"
        raise ValueError(msg)
    df = (
        book_panel[book_panel["book"] == book]
        .copy()
        .sort_values("open_date")
        .reset_index(drop=True)
    )
    if df.empty:
        msg = f"No training rows available for book {book}"
        raise ValueError(msg)
    X = _prepare_feature_matrix(df, features)
    y = df["PNL"].fillna(0.0)
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _train_decile_edges(scores: Sequence[float], n: int = 10) -> NDArray[np.float64]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    edges = np.asarray(
        np.unique(np.nanpercentile(arr, np.linspace(0, 100, n + 1))), dtype=float
    )
    if len(edges) >= 3:
        return edges
    fallback = np.asarray(
        np.unique(np.nanpercentile(arr, np.linspace(0, 100, 6))), dtype=float
    )
    return fallback


def _cvar(values: Sequence[float], alpha: float = 0.05) -> float:
    vector = pd.Series(values).dropna().to_numpy(dtype=float, copy=False)
    if vector.size == 0:
        return float("nan")
    k = max(1, int(np.floor(alpha * len(vector))))
    idx = np.argsort(vector)
    return float(np.mean(vector[idx[:k]]))


def build_live_kit(
    train_dir: Path,
    discovery_dir: Path,
    hybrid_policy_path: Path,
    output_dir: Path,
    *,
    settings: Settings | None = None,
    version: str = "v1",
    books: Collection[str] | None = None,
) -> ExportResult:
    """Assemble an inference-ready live kit bundle."""

    if not hybrid_policy_path.exists():
        raise FileNotFoundError(
            f"Hybrid policy file not found: {hybrid_policy_path}"  # pragma: no cover
        )

    active_settings = settings or get_settings()
    allowed_books = _normalise_book_filter(books)

    book_panel = _load_book_panel(train_dir)
    if allowed_books is not None:
        book_panel = book_panel[book_panel["book"].isin(allowed_books)].copy()
    if book_panel.empty:
        raise ValueError("Book panel is empty after applying the requested filters")

    rules_path = discovery_dir / "discovered_rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(
            f"Discovered rules file not found: {rules_path}"  # pragma: no cover
        )
    discovered_rules = json.loads(rules_path.read_text(encoding="utf-8"))
    if allowed_books is not None:
        discovered_rules = {
            book: payload
            for book, payload in discovered_rules.items()
            if book in allowed_books
        }

    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    models_dir = resolved_output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    hybrid_target = resolved_output / hybrid_policy_path.name
    shutil.copy2(hybrid_policy_path, hybrid_target)

    features_path = resolved_output / "features.json"
    features_path.write_text(
        json.dumps(BOOK_FEATS_CONFIG, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    kit: Dict[str, object] = {
        "meta": {
            "train_end": str(book_panel["open_date"].max().date()),
            "features_by_book": {},
            "version": version,
        },
        "puts": {},
        "calls": {},
        "hybrid_policy": json.loads(hybrid_policy_path.read_text(encoding="utf-8")),
    }
    meta_section = cast(Dict[str, object], kit["meta"])
    features_by_book = cast(Dict[str, List[str]], meta_section["features_by_book"])
    kit_puts = cast(Dict[str, Dict[str, object]], kit["puts"])
    kit_calls = cast(Dict[str, Dict[str, object]], kit["calls"])

    available_books = {
        str(book) for book in book_panel["book"].astype(str).unique().tolist()
    }
    books_to_process = [
        book
        for book in BOOKS
        if book in available_books and (allowed_books is None or book in allowed_books)
    ]

    copied_paths: List[Path] = [hybrid_target, features_path]

    for book in books_to_process:
        feats = [
            column
            for column in BOOK_FEATS_CONFIG.get(book, [])
            if column in book_panel.columns
        ]
        features_by_book[book] = feats

        model = _fit_full_model(
            book,
            book_panel,
            feats,
            alpha=active_settings.ALPHA,
            seed=active_settings.MODEL_SEED,
        )
        model_path = models_dir / f"{book}_model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        copied_paths.append(model_path)

        book_rows = (
            book_panel[book_panel["book"] == book]
            .copy()
            .sort_values("open_date")
            .reset_index(drop=True)
        )
        X = _prepare_feature_matrix(book_rows, feats)
        train_scores = pd.Series(
            model.predict(X), index=book_rows.index, dtype="float64"
        )
        train_pnl = book_rows["PNL"].fillna(0.0)

        if book.startswith("PUTS"):
            rule_payload = discovered_rules.get(book)
            if not isinstance(rule_payload, Mapping):
                raise ValueError(f"Missing PUT rules for {book}")
            keep = float(rule_payload.get("keep_target", 1.0))
            scores = train_scores.to_numpy(dtype=float, copy=False)
            if scores.size == 0:
                tau_top = float("nan")
                tau_bot = float("nan")
            else:
                tau_top = float(np.nanquantile(scores, 1.0 - keep))
                tau_bot = float(np.nanquantile(scores, keep))
            mask_top = train_scores >= tau_top
            mask_bot = train_scores <= tau_bot
            c_top = _cvar(train_pnl.where(mask_top, 0.0))
            c_bot = _cvar(train_pnl.where(mask_bot, 0.0))
            use_top = (
                pd.notna(c_top) and pd.notna(c_bot) and (float(c_top) >= float(c_bot))
            ) or (pd.isna(c_bot) and pd.notna(c_top))
            kit_puts[book] = {
                "keep": keep,
                "tau_top": float(tau_top),
                "tau_bot": float(tau_bot),
                "use_top": bool(use_top),
            }
        else:
            rule_payload = discovered_rules.get(book)
            if not isinstance(rule_payload, Mapping):
                raise ValueError(f"Missing CALL rules for {book}")
            edges = _train_decile_edges(
                train_scores.to_numpy(dtype=float, copy=False), n=10
            )
            kit_calls[book] = {
                "edges": edges.astype(float).tolist(),
                "long_deciles": [
                    int(value) for value in rule_payload.get("long_deciles", [])
                ],
                "short_deciles": [
                    int(value) for value in rule_payload.get("short_deciles", [])
                ],
            }

    live_kit_payload_path = resolved_output / "live_kit.json"
    with live_kit_payload_path.open("w", encoding="utf-8") as handle:
        json.dump(kit, handle, indent=2, sort_keys=True)
    copied_paths.append(live_kit_payload_path)

    exported_files: List[ExportedFile] = []
    for path in sorted(copied_paths):
        relative = path.relative_to(resolved_output).as_posix()
        digest = _compute_sha256(path)
        exported_files.append(
            ExportedFile(path=path, relative_path=relative, sha256=digest)
        )

    manifest_path = resolved_output / "manifest.json"
    manifest_payload = {
        "version": version,
        "settings": _serialise_settings(active_settings),
        "files": [
            {
                "path": item.relative_path,
                "sha256": item.sha256,
            }
            for item in exported_files
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)

    manifest_digest = _compute_sha256(manifest_path)
    digests: Dict[str, str] = {
        item.relative_path: item.sha256 for item in exported_files
    }
    digests[manifest_path.relative_to(resolved_output).as_posix()] = manifest_digest

    digests_path = resolved_output / "digests.json"
    with digests_path.open("w", encoding="utf-8") as handle:
        json.dump({"files": digests}, handle, indent=2, sort_keys=True)

    return ExportResult(
        kit_dir=resolved_output,
        manifest_path=manifest_path,
        digests_path=digests_path,
        files=exported_files,
        digests=digests,
    )
