from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import pickle
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from xdte.live import decide as decide_module
from xdte.live.decide import LiveDecisionResult, run_live_decisions
from xdte.live.providers import build_stub_daily_provider, build_stub_market_provider


class DummyModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, matrix: list[list[float]]) -> list[float]:
        return [self.value for _ in matrix]


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _make_kit(
    tmp_path: Path,
    policy: Mapping[str, object],
    *,
    predictions: Mapping[str, float],
    thresholds: Mapping[str, float],
    feature_columns: Sequence[str] | None = None,
    include_shap_summary: bool = False,
) -> Path:
    kit_dir = tmp_path / "kit"
    (kit_dir / "hybrid").mkdir(parents=True, exist_ok=True)
    (kit_dir / "hybrid" / "policy.json").write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8",
    )

    models_dir = kit_dir / "models"
    discovery_dir = kit_dir / "discovery"
    books_payload: dict[str, object] = {}

    resolved_columns = list(
        feature_columns
        or [
            "Intraday_Move_OpenToEntry_11",
            "Intraday_Move_OpenToEntry_1515",
            "VIX_Entry_11",
            "VIX_Entry_1515",
            "gap",
            "movement",
            "opening_vix",
            "closing_vix",
        ]
    )

    for book, prediction in predictions.items():
        fold_dir = models_dir / "train" / book / "fold_01"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "model.pkl").open("wb") as handle:
            pickle.dump(DummyModel(prediction), handle)
        books_payload[book] = {
            "feature_columns": resolved_columns,
            "folds": [
                {
                    "fold": 1,
                    "path": f"train/{book}/fold_01",
                }
            ],
        }
        discovery_meta_dir = discovery_dir / book / "fold_01"
        discovery_meta_dir.mkdir(parents=True, exist_ok=True)
        keep_threshold = thresholds.get(book, 0.0)
        (discovery_meta_dir / "metadata.json").write_text(
            json.dumps({"keep_threshold": keep_threshold}, indent=2),
            encoding="utf-8",
        )
        if include_shap_summary:
            shap_path = models_dir / f"shap_summary_{book}.csv"
            shap_path.parent.mkdir(parents=True, exist_ok=True)
            with shap_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["feature", "mean_abs_shap", "expected_value"],
                )
                writer.writeheader()
                expected_value = 0.1
                for index, feature_name in enumerate(resolved_columns[:3], start=1):
                    weight = round(0.3 - 0.05 * (index - 1), 3)
                    writer.writerow(
                        {
                            "feature": feature_name,
                            "mean_abs_shap": weight,
                            "expected_value": expected_value,
                        }
                    )

    (models_dir / "manifest.json").write_text(
        json.dumps({"books": books_payload}, indent=2),
        encoding="utf-8",
    )

    manifest_entries: list[dict[str, object]] = []
    for path in sorted(kit_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        relative = path.relative_to(kit_dir).as_posix()
        manifest_entries.append({"path": relative, "sha256": _compute_sha256(path)})

    (kit_dir / "manifest.json").write_text(
        json.dumps({"version": "ci", "files": manifest_entries}, indent=2),
        encoding="utf-8",
    )

    return kit_dir


def _build_daily_stub(open_date: date) -> Mapping[str, object]:
    return {
        "L1_TS": 0.9,
        "L1_VIX_Close": 16.5,
        "L1_VIX_pct": 0.55,
        "L1_vvix_close": 93.0,
        "L1_vvix_pct": 0.6,
        "L1_vvix_ema20": 88.0,
        "L1_vvix_ema30": 87.5,
        "L1_SPX_ATR_Pct": 0.012,
        "L1_SPX_Drawdown_Pct": -0.015,
        "L1_rv5": 0.18,
        "L1_rv20": 0.19,
        "L1_vvix_above_ema20": True,
        "L1_vvix_above_ema30": True,
        "gap": 4.5,
        "movement": 3.25,
        "closing_vix": 16.5,
        "opening_vix": 16.2,
    }


def _build_market_stub(open_date: date) -> Mapping[str, Mapping[str, object]]:
    return {
        "11:00": {
            "open_date": open_date,
            "spx_open": 4800.0,
            "spx_last": 4805.0,
            "intraday_move": 5.0,
            "vix": 15.2,
            "gap": 4.5,
            "movement": 5.0,
            "closing_vix": 16.5,
            "opening_vix": 16.2,
        },
        "15:15": {
            "open_date": open_date,
            "spx_open": 4795.0,
            "spx_last": 4788.0,
            "intraday_move": -7.0,
            "vix": 15.6,
            "gap": 4.5,
            "movement": -7.0,
            "closing_vix": 16.5,
            "opening_vix": 16.2,
        },
    }


def test_run_live_decisions_scores_predictions(tmp_path: Path) -> None:
    policy = {
        "tails": {
            "PUTS_0DTE_11": [0.1, 0.9],
            "CALLS_1DTE_1515": [0.2, 0.8],
        },
        "gammas": {
            "PUTS_0DTE_11": 0.95,
            "CALLS_0DTE_11": 1.05,
        },
    }
    predictions = {
        "CALLS_0DTE_11": 0.65,
        "PUTS_0DTE_11": 0.72,
        "CALLS_1DTE_1515": 0.48,
        "PUTS_1DTE_1515": 0.52,
    }
    thresholds = {
        "CALLS_0DTE_11": 0.6,
        "PUTS_0DTE_11": 0.7,
        "CALLS_1DTE_1515": 0.5,
        "PUTS_1DTE_1515": 0.6,
    }
    kit_dir = _make_kit(
        tmp_path, policy, predictions=predictions, thresholds=thresholds
    )

    open_date = date(2024, 7, 1)
    market_payloads = _build_market_stub(open_date)
    market_provider = build_stub_market_provider(
        open_date=open_date,
        eleven_payload={
            key: value
            for key, value in market_payloads["11:00"].items()
            if key != "open_date"
        },
        fifteen_payload={
            key: value
            for key, value in market_payloads["15:15"].items()
            if key != "open_date"
        },
    )
    daily_provider = build_stub_daily_provider(
        open_date=open_date,
        payload=_build_daily_stub(open_date),
    )

    result = run_live_decisions(
        kit_dir,
        market_provider=market_provider,
        daily_provider=daily_provider,
    )

    assert isinstance(result, LiveDecisionResult)
    assert not result.warnings

    session_map = {session.session: session for session in result.sessions}
    assert session_map["11:00"].features["gap"] == pytest.approx(4.5)
    assert session_map["15:15"].features["movement"] == pytest.approx(-7.0)

    trade_signals = {
        (record.session, record.book): record.value
        for record in result.decisions
        if record.action == "TRADE_SIGNAL"
    }
    assert trade_signals[("11:00", "CALLS_0DTE_11")]["keep"] is True
    assert trade_signals[("15:15", "CALLS_1DTE_1515")]["keep"] is False
    assert (
        pytest.approx(trade_signals[("11:00", "PUTS_0DTE_11")]["prediction"], rel=1e-6)
        == 0.72
    )
    assert trade_signals[("15:15", "PUTS_1DTE_1515")]["threshold"] == pytest.approx(0.6)


def test_run_live_decisions_uses_shap_summary(tmp_path: Path) -> None:
    policy = {"tails": {}, "gammas": {}}
    predictions = {
        "CALLS_0DTE_11": 0.65,
        "PUTS_0DTE_11": 0.72,
    }
    thresholds = {book: 0.6 for book in predictions}
    kit_dir = _make_kit(
        tmp_path,
        policy,
        predictions=predictions,
        thresholds=thresholds,
        include_shap_summary=True,
    )

    open_date = date(2024, 7, 1)
    market_payloads = _build_market_stub(open_date)
    market_provider = build_stub_market_provider(
        open_date=open_date,
        eleven_payload={
            key: value
            for key, value in market_payloads["11:00"].items()
            if key != "open_date"
        },
        fifteen_payload={
            key: value
            for key, value in market_payloads["15:15"].items()
            if key != "open_date"
        },
    )
    daily_provider = build_stub_daily_provider(
        open_date=open_date,
        payload=_build_daily_stub(open_date),
    )

    result = run_live_decisions(
        kit_dir,
        market_provider=market_provider,
        daily_provider=daily_provider,
    )

    trade_signals = {
        (record.session, record.book): record.value
        for record in result.decisions
        if record.action == "TRADE_SIGNAL"
    }
    signal_payload = trade_signals[("11:00", "CALLS_0DTE_11")]
    shap_payload = signal_payload.get("shap")
    assert isinstance(shap_payload, Mapping)
    contributions = shap_payload.get("contributions")
    assert isinstance(contributions, list)
    assert contributions
    assert contributions[0]["feature"] == "Intraday_Move_OpenToEntry_11"
    assert contributions[0]["impact"] == pytest.approx(0.3)
    assert shap_payload.get("expected_value") == pytest.approx(0.1)
    top_contributors = shap_payload.get("top_contributors")
    assert isinstance(top_contributors, list)
    assert top_contributors[0]["feature"] == contributions[0]["feature"]


def test_run_live_decisions_uses_default_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = {"tails": {}, "gammas": {}, "gamma_rules": {}}
    predictions = {
        book: 0.6
        for book in [
            "CALLS_0DTE_11",
            "PUTS_0DTE_11",
            "CALLS_1DTE_1515",
            "PUTS_1DTE_1515",
        ]
    }
    thresholds = {book: 0.5 for book in predictions}
    kit_dir = _make_kit(
        tmp_path, policy, predictions=predictions, thresholds=thresholds
    )

    class DummyDaily:
        def __init__(self) -> None:
            self.calls: list[date] = []
            self.warnings = ["daily-warning"]

        def __call__(self, open_date: date) -> Mapping[str, object]:
            self.calls.append(open_date)
            return _build_daily_stub(open_date)

    class DummyMarket:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.calls: list[str] = []
            self.warnings = ["market-warning"]

        def __call__(self, session: str) -> Mapping[str, object]:
            self.calls.append(session)
            return _build_market_stub(date(2024, 7, 1))[session]

    daily_stub = DummyDaily()
    market_stub = DummyMarket()

    monkeypatch.setattr(
        "xdte.live.decide.YFinanceDailyContextProvider",
        lambda **_kwargs: daily_stub,
    )
    monkeypatch.setattr(
        "xdte.live.decide.YFinanceMarketDataProvider",
        lambda **_kwargs: market_stub,
    )

    result = run_live_decisions(kit_dir)

    assert isinstance(result, LiveDecisionResult)
    assert set(market_stub.calls) == {"11:00", "15:15"}
    assert daily_stub.calls
    assert any(warning["message"] == "daily-warning" for warning in result.warnings)
    assert any(warning["message"] == "market-warning" for warning in result.warnings)


def test_run_live_decisions_warns_on_daily_provider_failure(tmp_path: Path) -> None:
    policy = {"tails": {}, "gammas": {}, "gamma_rules": {}}
    predictions = {
        "CALLS_0DTE_11": 0.5,
        "PUTS_0DTE_11": 0.5,
    }
    thresholds = {book: 0.4 for book in predictions}
    kit_dir = _make_kit(
        tmp_path,
        policy,
        predictions=predictions,
        thresholds=thresholds,
    )

    open_date = date(2024, 7, 1)
    market_payloads = _build_market_stub(open_date)
    market_provider = build_stub_market_provider(
        open_date=open_date,
        eleven_payload={
            key: value
            for key, value in market_payloads["11:00"].items()
            if key != "open_date"
        },
        fifteen_payload={
            key: value
            for key, value in market_payloads["15:15"].items()
            if key != "open_date"
        },
    )

    class FailingDailyProvider:
        def __init__(self) -> None:
            self.calls: list[date] = []

        def __call__(self, target_date: date) -> Mapping[str, object]:
            self.calls.append(target_date)
            raise RuntimeError("boom")

    daily_provider = FailingDailyProvider()

    result = run_live_decisions(
        kit_dir,
        market_provider=market_provider,
        daily_provider=daily_provider,
    )

    assert daily_provider.calls == [open_date]
    assert any(
        "daily provider failed" in warning["message"] for warning in result.warnings
    )
    for session in result.sessions:
        assert any("daily provider failed" in warning for warning in session.warnings)
    assert result.decisions


def test_run_live_decisions_skips_books_with_invalid_features(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy = {"tails": {}, "gammas": {}, "gamma_rules": {}}
    predictions = {
        "CALLS_0DTE_11": 0.65,
        "PUTS_0DTE_11": 0.72,
        "CALLS_1DTE_1515": 0.48,
        "PUTS_1DTE_1515": 0.52,
    }
    thresholds = {book: 0.5 for book in predictions}
    kit_dir = _make_kit(
        tmp_path,
        policy,
        predictions=predictions,
        thresholds=thresholds,
    )

    open_date = date(2024, 7, 1)
    market_payloads = _build_market_stub(open_date)
    market_payloads["11:00"].pop("vix")

    market_provider = build_stub_market_provider(
        open_date=open_date,
        eleven_payload={
            key: value
            for key, value in market_payloads["11:00"].items()
            if key != "open_date"
        },
        fifteen_payload={
            key: value
            for key, value in market_payloads["15:15"].items()
            if key != "open_date"
        },
    )
    daily_provider = build_stub_daily_provider(
        open_date=open_date,
        payload=_build_daily_stub(open_date),
    )

    with caplog.at_level(logging.CRITICAL):
        result = run_live_decisions(
            kit_dir,
            market_provider=market_provider,
            daily_provider=daily_provider,
        )

    skip_actions = {
        (record.session, record.book): record.value
        for record in result.decisions
        if record.action == "SKIP"
    }
    assert ("11:00", "CALLS_0DTE_11") in skip_actions
    assert ("11:00", "PUTS_0DTE_11") in skip_actions
    reason = skip_actions[("11:00", "CALLS_0DTE_11")]["reason"]
    assert "required feature validation failed" in reason

    critical_warnings = [
        warning for warning in result.warnings if warning["level"] == "CRITICAL"
    ]
    assert critical_warnings
    assert any("VIX_Entry_11" in warning["message"] for warning in critical_warnings)
    assert any(
        "required feature validation failed" in record.message
        for record in caplog.records
    )


def test_feature_vector_missing_columns_yield_nan() -> None:
    columns = ["L1_VIX_pct", "L1_VIX_Close"]
    features = {"L1_VIX_Close": 16.5}

    vector = decide_module._feature_vector(columns, features)

    assert math.isnan(vector[0])
    assert vector[1] == 16.5


def test_run_live_decisions_freeze_reuses_cached_snapshot(tmp_path: Path) -> None:
    policy = {"tails": {}, "gammas": {}}
    predictions = {
        "CALLS_0DTE_11": 0.6,
        "PUTS_0DTE_11": 0.55,
        "CALLS_1DTE_1515": 0.52,
        "PUTS_1DTE_1515": 0.51,
    }
    thresholds = {book: 0.5 for book in predictions}
    kit_dir = _make_kit(
        tmp_path, policy, predictions=predictions, thresholds=thresholds
    )

    open_date = date(2024, 7, 1)
    market_payload = _build_market_stub(open_date)
    daily_payload = _build_daily_stub(open_date)

    def market_provider(session: str) -> Mapping[str, object]:
        return market_payload[session]

    def daily_provider(_open_date: date) -> Mapping[str, object]:
        return daily_payload

    result_first = run_live_decisions(
        kit_dir,
        market_provider=market_provider,
        daily_provider=daily_provider,
        freeze_now=True,
    )

    snapshot_path = kit_dir / "_frozen_decision_snapshot.json"
    assert snapshot_path.exists()
    assert result_first.decision_context_id
    assert result_first.frozen_snapshot is not None

    def failing_market_provider(_session: str) -> Mapping[str, object]:
        raise AssertionError("market provider should not be called when frozen")

    def failing_daily_provider(_open_date: date) -> Mapping[str, object]:
        raise AssertionError("daily provider should not be called when frozen")

    result_second = run_live_decisions(
        kit_dir,
        market_provider=failing_market_provider,
        daily_provider=failing_daily_provider,
        freeze_now=True,
    )

    assert result_second.decision_context_id == result_first.decision_context_id
    assert result_second.generated_at == result_first.generated_at
    assert result_second.frozen_snapshot == result_first.frozen_snapshot
    assert [session.features for session in result_second.sessions] == [
        session.features for session in result_first.sessions
    ]


def test_run_live_decisions_refreshes_context_id_when_policy_changes(
    tmp_path: Path,
) -> None:
    policy = {"tails": {}, "gammas": {"CALLS_0DTE_11": 1.05}}
    predictions = {
        "CALLS_0DTE_11": 0.6,
        "PUTS_0DTE_11": 0.55,
        "CALLS_1DTE_1515": 0.52,
        "PUTS_1DTE_1515": 0.51,
    }
    thresholds = {book: 0.5 for book in predictions}
    kit_dir = _make_kit(
        tmp_path, policy, predictions=predictions, thresholds=thresholds
    )

    open_date = date(2024, 7, 1)
    market_payload = _build_market_stub(open_date)
    daily_payload = _build_daily_stub(open_date)

    def market_provider(session: str) -> Mapping[str, object]:
        return market_payload[session]

    def daily_provider(_open_date: date) -> Mapping[str, object]:
        return daily_payload

    result_first = run_live_decisions(
        kit_dir,
        market_provider=market_provider,
        daily_provider=daily_provider,
        freeze_now=True,
    )

    snapshot_path = kit_dir / "_frozen_decision_snapshot.json"
    assert snapshot_path.exists()
    first_context_id = result_first.decision_context_id
    assert first_context_id

    policy_path = kit_dir / "hybrid" / "policy.json"
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_payload.setdefault("gammas", {})["CALLS_0DTE_11"] = 1.15
    policy_path.write_text(
        json.dumps(policy_payload, indent=2),
        encoding="utf-8",
    )

    manifest_path = kit_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest_payload.get("files", []):
        if (
            isinstance(entry, dict)
            and entry.get("path") == "hybrid/policy.json"
            and isinstance(entry.get("sha256"), str)
        ):
            entry["sha256"] = _compute_sha256(policy_path)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )

    def failing_market_provider(_session: str) -> Mapping[str, object]:
        raise AssertionError("market provider should not be called when frozen")

    def failing_daily_provider(_open_date: date) -> Mapping[str, object]:
        raise AssertionError("daily provider should not be called when frozen")

    result_second = run_live_decisions(
        kit_dir,
        market_provider=failing_market_provider,
        daily_provider=failing_daily_provider,
        freeze_now=True,
    )

    assert result_second.decision_context_id != first_context_id

    stored_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert (
        stored_snapshot.get("decision_context_id") == result_second.decision_context_id
    )


def test_run_live_decisions_rejects_duplicate_feature_columns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.clear()
    policy: dict[str, object] = {"tails": {}, "gammas": {}, "gamma_rules": {}}
    predictions = {"CALLS_0DTE_11": 0.65}
    thresholds = {"CALLS_0DTE_11": 0.5}
    feature_columns = [
        "L1_VIX_pct",
        "L1_VIX_pct",
        "Intraday_Move_OpenToEntry_11",
        "VIX_Entry_11",
        "gap",
        "movement",
    ]
    kit_dir = _make_kit(
        tmp_path,
        policy,
        predictions=predictions,
        thresholds=thresholds,
        feature_columns=feature_columns,
    )

    open_date = date(2024, 7, 1)
    market_payloads = _build_market_stub(open_date)
    market_provider = build_stub_market_provider(
        open_date=open_date,
        eleven_payload={
            key: value
            for key, value in market_payloads["11:00"].items()
            if key != "open_date"
        },
        fifteen_payload={
            key: value
            for key, value in market_payloads["15:15"].items()
            if key != "open_date"
        },
    )
    daily_provider = build_stub_daily_provider(
        open_date=open_date,
        payload=_build_daily_stub(open_date),
    )

    with caplog.at_level(logging.CRITICAL):
        result = run_live_decisions(
            kit_dir,
            market_provider=market_provider,
            daily_provider=daily_provider,
        )

    duplicate_warnings = [
        warning
        for warning in result.warnings
        if "duplicate feature columns" in warning["message"]
    ]
    assert duplicate_warnings
    assert all(warning["level"] == "CRITICAL" for warning in duplicate_warnings)
    assert any(
        record.levelno == logging.CRITICAL
        and "duplicate feature columns" in record.message
        for record in caplog.records
    )
    assert all(
        not (record.book == "CALLS_0DTE_11" and record.action == "TRADE_SIGNAL")
        for record in result.decisions
    )
    skip_reasons = [
        record.value["reason"]
        for record in result.decisions
        if record.book == "CALLS_0DTE_11" and record.action == "SKIP"
    ]
    assert skip_reasons and "duplicate feature columns" in skip_reasons[0]
