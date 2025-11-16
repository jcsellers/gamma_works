"""Formatting helpers used across the XDTE CLI."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import Mapping, Sequence


def _format_feature_value(value: object) -> str:
    """Render arbitrary values into human-readable strings."""

    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _json_safe(value: object) -> object:
    """Return a JSON-serialisable representation of ``value``."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return value


def _format_top_features(top_features: object) -> str:
    """Summarise the most impactful features for a decision."""

    if not isinstance(top_features, Sequence) or isinstance(top_features, (str, bytes)):
        return ""

    rendered: list[str] = []
    for entry in top_features:
        if (
            isinstance(entry, Sequence)
            and not isinstance(entry, (str, bytes))
            and len(entry) >= 2
        ):
            feature, impact = entry[0], entry[1]
            if isinstance(feature, str) and isinstance(impact, (int, float)):
                numeric = float(impact)
                if math.isfinite(numeric):
                    rendered.append(f"{feature}:{numeric:+0.2f}")
        elif isinstance(entry, Mapping):
            feature = entry.get("feature")
            impact = entry.get("impact")
            if isinstance(feature, str) and isinstance(impact, (int, float)):
                numeric = float(impact)
                if math.isfinite(numeric):
                    rendered.append(f"{feature}:{numeric:+0.2f}")
        if len(rendered) >= 3:
            break
    return ", ".join(rendered)


def _render_confidence(confidence: object) -> str:
    """Visualise confidence scores as a five-star rating."""

    if isinstance(confidence, bool):
        return "n/a"
    if isinstance(confidence, (int, float)):
        numeric = float(confidence)
        if math.isfinite(numeric):
            clamped = min(max(numeric, 0.0), 1.0)
            filled = int(round(clamped * 5))
            empty = 5 - filled
            return f"{'★' * filled}{'☆' * empty}"
    return "n/a"


def _format_warning_entry(warning: Mapping[str, object]) -> str:
    """Render structured warning payloads into log strings."""

    level = warning.get("level", "WARNING")
    source = warning.get("source", "unknown")
    message = warning.get("message", "")
    timestamp = warning.get("timestamp")
    formatted = f"[{level}] {source}: {message}"
    if isinstance(timestamp, str) and timestamp:
        return f"{timestamp} {formatted}"
    return formatted


def _format_csv_value(value: object) -> str:
    """Normalise values for CSV export."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return _format_feature_value(value)


def _timestamp_for_filename(generated_at: datetime) -> str:
    """Return a filesystem-friendly timestamp string."""

    aware = (
        generated_at.replace(tzinfo=timezone.utc)
        if generated_at.tzinfo is None
        else generated_at.astimezone(timezone.utc)
    )
    base = aware.strftime("%Y%m%dT%H%M%S")
    if aware.microsecond:
        return f"{base}_{aware.microsecond:06d}Z"
    return f"{base}Z"


__all__ = [
    "_format_csv_value",
    "_format_feature_value",
    "_format_top_features",
    "_format_warning_entry",
    "_json_safe",
    "_render_confidence",
    "_timestamp_for_filename",
]
