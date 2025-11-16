"""Persistence helpers used by the CLI entry points."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .formatting import _format_csv_value, _format_top_features, _timestamp_for_filename
from .live.actions import FinalAction


def _build_csv_rows(
    actions: Sequence[FinalAction],
    *,
    decision_context_id: str,
    generated_at: str,
) -> tuple[list[str], list[dict[str, str]]]:
    base_fields = [
        "decision_context_id",
        "generated_at",
        "session",
        "book",
        "rule_action",
        "final_action",
        "size_gamma",
        "score",
        "margin_to_threshold",
        "kill_flag",
        "policy_type",
        "decile",
        "reason_text",
        "confidence",
        "top_features",
    ]
    threshold_fields: set[str] = set()
    rows: list[dict[str, str]] = []

    for action in actions:
        rationale = action.rationale
        confidence_value: object = ""
        top_features_value = ""
        if isinstance(rationale, Mapping):
            confidence_value = rationale.get("confidence", "")
            top_features_value = _format_top_features(rationale.get("top_features"))

        row: dict[str, str] = {
            "decision_context_id": decision_context_id,
            "generated_at": generated_at,
            "session": action.session,
            "book": action.book,
            "rule_action": action.rule_action,
            "final_action": action.final_action,
            "size_gamma": _format_csv_value(action.size_gamma),
            "score": _format_csv_value(action.score),
            "margin_to_threshold": _format_csv_value(action.margin_to_threshold),
            "kill_flag": _format_csv_value(action.kill_flag),
            "policy_type": action.policy_type,
            "decile": _format_csv_value(action.decile),
            "reason_text": action.reason_text,
            "confidence": _format_csv_value(confidence_value),
            "top_features": top_features_value,
        }

        for key in sorted(action.thresholds):
            column = f"threshold_{key}"
            threshold_fields.add(column)
            row[column] = _format_csv_value(action.thresholds[key])

        rows.append(row)

    fieldnames = base_fields + sorted(threshold_fields)
    for row in rows:
        for field in fieldnames:
            row.setdefault(field, "")
    return fieldnames, rows


def _get_daily_runs_dir() -> Path:
    """Return the directory that stores persisted decision runs."""

    return Path(os.environ.get("XDTE_DAILY_RUNS_DIR", "_daily_runs"))


def _persist_decision_run(
    payload: Mapping[str, object],
    *,
    actions: Sequence[FinalAction],
    generated_at: datetime,
) -> None:
    """Persist a decision run as JSON and CSV artefacts on disk."""

    timestamp = _timestamp_for_filename(generated_at)
    log_dir = _get_daily_runs_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        json_path = log_dir / f"decisions_{timestamp}.json"
        json_payload = json.dumps(payload, indent=2, sort_keys=True, default=str)
        json_path.write_text(json_payload, encoding="utf-8")

        csv_path = log_dir / f"decisions_{timestamp}.csv"
        fieldnames, rows = _build_csv_rows(
            actions,
            decision_context_id=str(payload.get("decision_context_id", "")),
            generated_at=str(payload.get("generated_at", "")),
        )
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.warning(
            "Failed to persist decision run to '%s': %s", log_dir, exc, exc_info=True
        )


__all__ = ["_get_daily_runs_dir", "_persist_decision_run"]
