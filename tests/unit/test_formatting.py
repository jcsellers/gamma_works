from __future__ import annotations

from datetime import date, datetime, timezone

from xdte.formatting import (
    _format_csv_value,
    _format_feature_value,
    _format_top_features,
    _format_warning_entry,
    _json_safe,
    _render_confidence,
    _timestamp_for_filename,
)


def test_format_feature_value_handles_special_cases() -> None:
    assert _format_feature_value(float("nan")) == "nan"
    assert _format_feature_value(date(2024, 7, 1)) == "2024-07-01"
    assert _format_feature_value({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_json_safe_normalises_containers() -> None:
    payload = {
        "when": datetime(2024, 7, 1, 15, 30, tzinfo=timezone.utc),
        "values": [1, float("inf"), {"foo": float("nan")}, date(2024, 7, 2)],
    }
    safe = _json_safe(payload)
    assert safe["when"] == "2024-07-01T15:30:00+00:00"
    assert safe["values"][0] == 1
    assert safe["values"][1] is None
    assert safe["values"][2]["foo"] is None
    assert safe["values"][3] == "2024-07-02"


def test_format_top_features_limits_results() -> None:
    top_features = [
        ("feature_a", 0.6),
        ("feature_b", -0.2),
        ("feature_c", 0.1),
        ("feature_d", 0.5),
    ]
    assert (
        _format_top_features(top_features)
        == "feature_a:+0.60, feature_b:-0.20, feature_c:+0.10"
    )


def test_render_confidence_maps_scores_to_stars() -> None:
    assert _render_confidence(True) == "n/a"
    assert _render_confidence(0.6) == "★★★☆☆"
    assert _render_confidence(0.95) == "★★★★★"
    assert _render_confidence(-1) == "☆☆☆☆☆"


def test_format_warning_entry_includes_timestamp() -> None:
    warning = {
        "level": "WARNING",
        "source": "test",
        "message": "boom",
        "timestamp": "2024-07-01T15:30:00Z",
    }
    assert _format_warning_entry(warning) == "2024-07-01T15:30:00Z [WARNING] test: boom"


def test_format_csv_value_delegates_to_feature_value() -> None:
    assert _format_csv_value(True) == "true"
    assert _format_csv_value(None) == ""
    assert _format_csv_value({"foo": "bar"}) == '{"foo": "bar"}'


def test_timestamp_for_filename_preserves_microseconds() -> None:
    naive = datetime(2024, 7, 1, 15, 30, 45, 123456)
    aware = datetime(2024, 7, 1, 15, 30, 45, tzinfo=timezone.utc)
    assert _timestamp_for_filename(naive) == "20240701T153045_123456Z"
    assert _timestamp_for_filename(aware) == "20240701T153045Z"
