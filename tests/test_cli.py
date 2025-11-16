import csv
import hashlib
import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence

import pytest
from click.testing import CliRunner

from tests.unit.test_live_decide import _build_daily_stub, _build_market_stub, _make_kit
from xdte.cli import app
from xdte.config import Settings
from xdte.live.actions import FinalAction
from xdte.live.decide import DecisionRecord, LiveDecisionResult, SessionDecision
from xdte.model.export import ExportedFile, ExportResult
from xdte.model.hybrid import HybridPolicy, HybridSelectionResult

runner = CliRunner()


def _build_corruptible_kit(tmp_path: Path) -> Path:
    kit_dir = tmp_path / "kit"
    (kit_dir / "hybrid").mkdir(parents=True, exist_ok=True)
    policy_path = kit_dir / "hybrid" / "policy.json"
    policy_path.write_text(
        json.dumps({"tails": {}, "gammas": {}, "gamma_rules": {}}, indent=2),
        encoding="utf-8",
    )
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    manifest_payload = {
        "version": "ci",
        "files": [
            {"path": "hybrid/policy.json", "sha256": digest},
        ],
    }
    (kit_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    return kit_dir


def test_help_lists_expected_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "tune", "export-kit", "decide", "validate-backtest"):
        assert command in result.stdout


def test_decide_reports_dual_sessions_and_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_at = datetime(2024, 7, 5, 15, 30, tzinfo=timezone.utc)

    final_actions = [
        FinalAction(
            session="11:00",
            book="CALLS_0DTE_11",
            rule_action="GO_LONG",
            final_action="GO_LONG",
            size_gamma=1.1,
            score=0.42,
            margin_to_threshold=0.18,
            kill_flag=False,
            policy_type="gamma_rule",
            thresholds={"keep": 0.52},
            decile=3,
            reason_text="Follow gamma ramp",
            rationale={
                "policy_type": "gamma_rule",
                "kill_flag": False,
                "margin_to_threshold": 0.18,
                "reason_text": "Follow gamma ramp",
                "top_features": ["feature_a:+0.12"],
            },
        ),
        FinalAction(
            session="15:15",
            book="PUTS_0DTE_1515",
            rule_action="GO_SHORT",
            final_action="SKIP",
            size_gamma=0.4,
            score=-0.27,
            margin_to_threshold=-0.05,
            kill_flag=True,
            policy_type="kill_day",
            thresholds={},
            decile=None,
            reason_text="Kill day",
            rationale={
                "policy_type": "kill_day",
                "kill_flag": True,
                "margin_to_threshold": -0.05,
                "reason_text": "Kill day",
                "confidence": 0.81,
            },
        ),
    ]

    sessions = (
        SessionDecision(
            session="11:00",
            open_date=date(2024, 7, 5),
            features={"metric": 1.0},
            warnings=("low volume",),
        ),
        SessionDecision(
            session="15:15",
            open_date=date(2024, 7, 5),
            features={"metric": 2.0},
            warnings=(),
        ),
    )

    decisions = (
        DecisionRecord(
            session="11:00",
            book="CALLS_0DTE_11",
            action="TRADE_SIGNAL",
            value={"keep": True},
        ),
        DecisionRecord(
            session="15:15",
            book="PUTS_0DTE_1515",
            action="TRADE_SIGNAL",
            value={"keep": False},
        ),
    )

    decision_result = LiveDecisionResult(
        kit_dir=Path("kit"),
        manifest={"version": "ci"},
        policy={"variant": "test"},
        sessions=sessions,
        decisions=decisions,
        warnings=(),
        generated_at=generated_at,
    )

    monkeypatch.setattr(
        "xdte.cli.run_live_decisions", lambda *_args, **_kwargs: decision_result
    )
    monkeypatch.setattr(
        "xdte.cli.compile_operational_directives",
        lambda *_args, **_kwargs: final_actions,
    )

    with runner.isolated_filesystem():
        kit_dir = Path("kit")
        kit_dir.mkdir()
        monkeypatch.setenv("XDTE_DAILY_RUNS_DIR", str(Path("operator_logs")))

        result = runner.invoke(
            app,
            [
                "decide",
                "--kit-dir",
                str(kit_dir),
                "--json",
                "--offline-market",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)

        assert {session["session"] for session in payload["sessions"]} == {
            "11:00",
            "15:15",
        }
        assert len(payload["actions"]) == 2
        why_payloads = [entry["why"] for entry in payload["actions"]]
        for why in why_payloads:
            assert why["policy_type"] in {"gamma_rule", "kill_day"}
            assert "reason_text" in why
            assert "margin_to_threshold" in why
            assert "kill_flag" in why

        result_text = runner.invoke(
            app,
            [
                "decide",
                "--kit-dir",
                str(kit_dir),
                "--offline-market",
            ],
        )

        assert result_text.exit_code == 0
        output = result_text.stdout
        assert "Session 11:00" in output
        assert "Session 15:15" in output
        header_line = next(line for line in output.splitlines() if "WHY" in line)
        assert "WHY" in header_line
        assert "Follow gamma ramp" in output
        assert "Kill day" in output


def test_train_invokes_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    train_call: dict[str, object] = {}
    discover_call: dict[str, object] = {}
    apply_call: dict[str, object] = {}

    def fake_train(
        data_dir: Path,
        artifact_dir: Path,
        *,
        books: Optional[Sequence[str]] = None,
        settings: Settings,
    ) -> SimpleNamespace:
        train_call.update(
            {
                "data_dir": data_dir,
                "artifact_dir": artifact_dir,
                "settings": settings,
                "books": books,
            }
        )
        return SimpleNamespace(artifacts=[object()])

    def fake_discover(
        artifact_dir: Path,
        discovery_dir: Path,
        *,
        settings: Settings,
    ) -> SimpleNamespace:
        discover_call.update(
            {
                "artifact_dir": artifact_dir,
                "discovery_dir": discovery_dir,
                "settings": settings,
            }
        )
        return SimpleNamespace(folds=[object()])

    def fake_apply(
        artifact_dir: Path,
        discovery_dir: Path,
        apply_dir: Path,
        *,
        settings: Settings,
    ) -> SimpleNamespace:
        apply_call.update(
            {
                "artifact_dir": artifact_dir,
                "discovery_dir": discovery_dir,
                "apply_dir": apply_dir,
                "settings": settings,
            }
        )
        return SimpleNamespace(folds=[object()])

    monkeypatch.setattr("xdte.cli.train_models", fake_train)
    monkeypatch.setattr("xdte.cli.discover_rules", fake_discover)
    monkeypatch.setattr("xdte.cli.apply_models", fake_apply)

    with caplog.at_level(logging.INFO):
        result = runner.invoke(
            app,
            [
                "--verbose",
                "train",
                "--backtest-dir",
                str(backtest_dir),
                "--out-dir",
                str(out_dir),
                "--books",
                "ALPHA",
                "--books",
                "BETA",
                "--alpha",
                "0.2",
            ],
        )

    assert result.exit_code == 0

    expected_artifacts = (out_dir / "artifacts").resolve()
    expected_discovery = (out_dir / "discovery").resolve()
    expected_apply = (out_dir / "apply").resolve()

    assert train_call["data_dir"] == backtest_dir.resolve()
    assert train_call["artifact_dir"] == expected_artifacts
    assert isinstance(train_call["settings"], Settings)
    assert train_call["settings"].ALPHA == 0.2
    assert train_call["books"] == ("ALPHA", "BETA")

    assert discover_call["artifact_dir"] == expected_artifacts
    assert discover_call["discovery_dir"] == expected_discovery

    assert apply_call["artifact_dir"] == expected_artifacts
    assert apply_call["discovery_dir"] == expected_discovery
    assert apply_call["apply_dir"] == expected_apply
    assert "Training complete" in caplog.text


def test_train_reports_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("xdte.cli.train_models", explode)

    result = runner.invoke(
        app,
        [
            "train",
            "--backtest-dir",
            str(backtest_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code != 0
    assert "Training failed" in result.output


def test_tune_evaluates_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    apply_dir = artifacts_dir / "apply"
    apply_dir.mkdir(parents=True)

    candidates_path = tmp_path / "candidates.json"
    payload = {
        "candidates": [{"tails": {}, "gammas": {"BOOK": 0.9}, "gamma_rules": {}}]
    }
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")

    selection_result = HybridSelectionResult(
        baseline_metrics={},
        evaluations=[],
        selected_policy=HybridPolicy(tails={}, gammas={}),
        accepted=True,
        output_path=artifacts_dir / "hybrid" / "policy.json",
    )

    captured: dict[str, object] = {}

    def fake_select(
        apply_path: Path,
        candidates: object,
        output_path: Path,
        *,
        settings: Settings,
    ) -> HybridSelectionResult:
        captured["apply"] = apply_path
        captured["candidates"] = candidates
        captured["output"] = output_path
        return selection_result

    monkeypatch.setattr("xdte.cli.select_hybrid_policy", fake_select)

    with caplog.at_level(logging.INFO):
        result = runner.invoke(
            app,
            [
                "--verbose",
                "tune",
                "--artifacts-dir",
                str(artifacts_dir),
                "--candidates",
                str(candidates_path),
            ],
        )

    assert result.exit_code == 0
    assert captured["apply"] == apply_dir.resolve()
    assert isinstance(captured["candidates"], list)
    assert "Hybrid tuning" in caplog.text


def test_tune_reports_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(["bad"]), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "tune",
            "--artifacts-dir",
            str(artifacts_dir),
            "--candidates",
            str(candidates_path),
        ],
    )

    assert result.exit_code != 0
    assert "Candidates file" in result.output


def test_export_builds_live_kit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    hybrid_dir = artifacts_dir / "hybrid"
    hybrid_dir.mkdir(parents=True)
    (hybrid_dir / "policy.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "kit"

    export_result = ExportResult(
        kit_dir=out_dir,
        manifest_path=out_dir / "manifest.json",
        digests_path=out_dir / "digests.json",
        files=[ExportedFile(path=out_dir, relative_path=".", sha256="deadbeef")],
        digests={"manifest.json": "deadbeef"},
    )

    captured: dict[str, object] = {}

    def fake_build(
        train_dir: Path,
        discovery_dir: Path,
        hybrid_policy_path: Path,
        output_dir: Path,
        *,
        settings: Settings,
        version: str,
    ) -> ExportResult:
        captured["train"] = train_dir
        captured["discovery"] = discovery_dir
        captured["hybrid"] = hybrid_policy_path
        captured["output"] = output_dir
        captured["version"] = version
        return export_result

    monkeypatch.setattr("xdte.cli.build_live_kit", fake_build)

    with caplog.at_level(logging.INFO):
        result = runner.invoke(
            app,
            [
                "--verbose",
                "export-kit",
                "--artifacts-dir",
                str(artifacts_dir),
                "--out",
                str(out_dir),
                "--version",
                "ci",
            ],
        )

    assert result.exit_code == 0
    assert captured["train"] == (artifacts_dir / "artifacts").resolve()
    assert captured["discovery"] == (artifacts_dir / "discovery").resolve()
    assert captured["hybrid"] == (artifacts_dir / "hybrid" / "policy.json").resolve()
    assert captured["output"] == out_dir.resolve()
    assert captured["version"] == "ci"
    assert "Live Kit exported" in caplog.text


def test_decide_outputs_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()

    sessions = [
        SessionDecision(
            session="11:00",
            open_date=date(2024, 7, 1),
            features={
                "open_date": date(2024, 7, 1),
                "VIX_Entry_11": 17.5,
                "Intraday_Move_OpenToEntry_11": 4.0,
            },
            warnings=(),
        )
    ]
    decision_result = LiveDecisionResult(
        kit_dir=kit_dir,
        manifest={"version": "ci"},
        policy={"settings": {"no_trade_margin": 0.05, "keep_quorum": 0.5}},
        sessions=sessions,
        decisions=[
            DecisionRecord(
                session="11:00",
                book="BOOK",
                action="TRADE_SIGNAL",
                value={
                    "keep": True,
                    "prediction": 0.74,
                    "threshold": 0.62,
                    "folds": [
                        {"keep": True},
                        {"keep": True},
                        {"keep": False},
                    ],
                    "shap": {
                        "top_contributors": [
                            {"feature": "gap", "impact": 0.12},
                            {"feature": "movement", "impact": -0.08},
                        ]
                    },
                },
            ),
            DecisionRecord(
                session="11:00", book="BOOK", action="ADJUST_GAMMA", value=0.9
            ),
        ],
        warnings=(),
        generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        "xdte.cli.run_live_decisions", lambda *_args, **_kwargs: decision_result
    )

    result = runner.invoke(
        app,
        ["decide", "--kit-dir", str(kit_dir)],
    )

    assert result.exit_code == 0
    assert "BOOK" in result.stdout
    assert "Session 11:00" in result.stdout
    assert "KEEP_SHORT" in result.stdout
    assert "+0.120" in result.stdout
    assert "gap:+0.12" in result.stdout
    assert "Conf" in result.stdout
    assert "★★★★☆" in result.stdout


def test_decide_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()

    sessions = [
        SessionDecision(
            session="15:15",
            open_date=date(2024, 7, 1),
            features={"open_date": date(2024, 7, 1)},
            warnings=("missing feature",),
        )
    ]
    warning_payload = {
        "level": "WARNING",
        "message": "connectivity",
        "source": "unit",
        "timestamp": "2024-07-01T12:00:00+00:00",
    }
    decision_result = LiveDecisionResult(
        kit_dir=kit_dir,
        manifest={"version": "ci"},
        policy={},
        sessions=sessions,
        decisions=[],
        warnings=(warning_payload,),
        generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        "xdte.cli.run_live_decisions", lambda *_args, **_kwargs: decision_result
    )

    result = runner.invoke(
        app,
        ["decide", "--kit-dir", str(kit_dir), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kit_dir"] == str(kit_dir.resolve())
    assert payload["sessions"][0]["session"] == "15:15"
    assert payload["warnings"] == [warning_payload]
    assert payload["generated_at"] == "2024-07-01T00:00:00+00:00"
    assert payload["actions"] == []


def test_decide_json_output_includes_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()

    sessions = [
        SessionDecision(
            session="11:00",
            open_date=date(2024, 7, 1),
            features={},
            warnings=(),
        )
    ]
    decision_result = LiveDecisionResult(
        kit_dir=kit_dir,
        manifest={"version": "ci"},
        policy={},
        sessions=sessions,
        decisions=[
            DecisionRecord(
                session="11:00",
                book="BOOK",
                action="TRADE_SIGNAL",
                value={
                    "keep": True,
                    "prediction": 0.72,
                    "threshold": 0.65,
                    "folds": [
                        {"keep": True},
                        {"keep": False},
                        {"keep": True},
                    ],
                },
            )
        ],
        warnings=(),
        generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        "xdte.cli.run_live_decisions", lambda *_args, **_kwargs: decision_result
    )

    result = runner.invoke(
        app,
        ["decide", "--kit-dir", str(kit_dir), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["actions"][0]["why"]["confidence"] == pytest.approx(
        (2 / 3 + (1 - math.exp(-0.07 / 0.05))) / 2
    )


def test_decide_offline_persists_daily_run_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_at = datetime(2024, 7, 4, 15, 30, tzinfo=timezone.utc)

    final_actions = [
        FinalAction(
            session="11:00",
            book="BOOK",
            rule_action="KEEP_SHORT",
            final_action="KEEP",
            size_gamma=0.75,
            score=0.12,
            margin_to_threshold=0.05,
            kill_flag=False,
            policy_type="hybrid",
            thresholds={"keep": 0.6},
            decile=3,
            reason_text="Maintain position",
            rationale={
                "confidence": 0.82,
                "top_features": [["feature_a", 0.1], ["feature_b", -0.05]],
            },
        )
    ]

    with runner.isolated_filesystem():
        kit_dir = Path("kit")
        kit_dir.mkdir()

        decision_result = LiveDecisionResult(
            kit_dir=kit_dir.resolve(),
            manifest={"version": "ci"},
            policy={"variant": "test"},
            sessions=(
                SessionDecision(
                    session="11:00",
                    open_date=date(2024, 7, 4),
                    features={"metric": 1.0},
                    warnings=(),
                ),
            ),
            decisions=(
                DecisionRecord(
                    session="11:00",
                    book="BOOK",
                    action="TRADE_SIGNAL",
                    value={"keep": True},
                ),
            ),
            warnings=(),
            generated_at=generated_at,
        )

        monkeypatch.setattr(
            "xdte.cli.run_live_decisions", lambda *_args, **_kwargs: decision_result
        )
        monkeypatch.setattr(
            "xdte.cli.compile_operational_directives",
            lambda *_args, **_kwargs: final_actions,
        )

        result = runner.invoke(
            app,
            [
                "decide",
                "--kit-dir",
                str(kit_dir),
                "--json",
                "--offline-market",
            ],
        )

        assert result.exit_code == 0

        payload = json.loads(result.stdout)
        timestamp = generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = Path("_daily_runs")
        json_path = log_dir / f"decisions_{timestamp}.json"
        csv_path = log_dir / f"decisions_{timestamp}.csv"

        assert json_path.exists()
        assert csv_path.exists()
        assert json.loads(json_path.read_text(encoding="utf-8")) == payload

        with csv_path.open(encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert len(rows) == 1
        row = rows[0]
        assert row["decision_context_id"] == payload["decision_context_id"]
        assert row["generated_at"] == payload["generated_at"]
        assert row["session"] == "11:00"
        assert row["book"] == "BOOK"
        assert row["rule_action"] == "KEEP_SHORT"
        assert row["final_action"] == "KEEP"
        assert row["size_gamma"] == "0.75"
        assert row["score"] == "0.12"
        assert row["margin_to_threshold"] == "0.05"
        assert row["kill_flag"] == "false"
        assert row["policy_type"] == "hybrid"
        assert row["decile"] == "3"
        assert row["reason_text"] == "Maintain position"
        assert row["confidence"] == "0.82"
        assert row["top_features"] == "feature_a:+0.10, feature_b:-0.05"
        assert row["threshold_keep"] == "0.6"


def test_decide_freeze_reuses_snapshot_offline_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {"tails": {}, "gammas": {}}
    predictions = {
        "CALLS_0DTE_11": 0.55,
        "PUTS_0DTE_11": 0.65,
        "CALLS_1DTE_1515": 0.48,
        "PUTS_1DTE_1515": 0.52,
    }
    thresholds = {
        "CALLS_0DTE_11": 0.5,
        "PUTS_0DTE_11": 0.6,
        "CALLS_1DTE_1515": 0.45,
        "PUTS_1DTE_1515": 0.5,
    }

    market_runs = {"count": 0}
    daily_runs = {"count": 0}

    def fake_offline_market(open_date: date, *, settings: Settings):
        base_payload = _build_market_stub(open_date)
        market_runs["count"] += 1

        if market_runs["count"] > 1:

            def provider(_session: str) -> Mapping[str, object]:
                raise AssertionError(
                    "market provider should not be accessed after freeze"
                )

            return provider

        def provider(session: str) -> Mapping[str, object]:
            return base_payload[session]

        return provider

    def fake_offline_daily(open_date: date, *, settings: Settings):
        payload = _build_daily_stub(open_date)
        daily_runs["count"] += 1

        if daily_runs["count"] > 1:

            def provider(_open_date: date) -> Mapping[str, object]:
                raise AssertionError(
                    "daily provider should not be accessed after freeze"
                )

            return provider

        def provider(_open_date: date) -> Mapping[str, object]:
            return payload

        return provider

    monkeypatch.setattr("xdte.cli._build_offline_market", fake_offline_market)
    monkeypatch.setattr("xdte.cli._build_offline_daily", fake_offline_daily)

    with runner.isolated_filesystem():
        kit_dir = _make_kit(
            Path.cwd(),
            policy,
            predictions=predictions,
            thresholds=thresholds,
        )

        result_first = runner.invoke(
            app,
            ["decide", "--kit-dir", str(kit_dir), "--offline-market", "--json"],
        )
        assert result_first.exit_code == 0
        payload_first = json.loads(result_first.stdout)

        result_second = runner.invoke(
            app,
            ["decide", "--kit-dir", str(kit_dir), "--offline-market", "--json"],
        )
        assert result_second.exit_code == 0
        payload_second = json.loads(result_second.stdout)

        assert payload_first == payload_second
        assert (
            payload_first["decision_context_id"]
            == payload_second["decision_context_id"]
        )
        snapshot_path = kit_dir / "_frozen_decision_snapshot.json"
        assert snapshot_path.exists()
        assert "frozen_snapshot" in payload_first
        assert (
            payload_first["frozen_snapshot"]["decision_context_id"]
            == payload_first["decision_context_id"]
        )


def test_decide_freeze_reuses_snapshot_offline_daily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {"tails": {}, "gammas": {}}
    predictions = {
        "CALLS_0DTE_11": 0.58,
        "PUTS_0DTE_11": 0.61,
        "CALLS_1DTE_1515": 0.53,
        "PUTS_1DTE_1515": 0.47,
    }
    thresholds = {
        "CALLS_0DTE_11": 0.52,
        "PUTS_0DTE_11": 0.6,
        "CALLS_1DTE_1515": 0.5,
        "PUTS_1DTE_1515": 0.5,
    }

    daily_runs = {"count": 0}

    def fake_offline_daily(open_date: date, *, settings: Settings):
        payload = _build_daily_stub(open_date)
        daily_runs["count"] += 1

        if daily_runs["count"] > 1:

            def provider(_open_date: date) -> Mapping[str, object]:
                raise AssertionError(
                    "daily provider should not be accessed after freeze"
                )

            return provider

        def provider(_open_date: date) -> Mapping[str, object]:
            return payload

        return provider

    monkeypatch.setattr("xdte.cli._build_offline_daily", fake_offline_daily)

    first_market = {
        "11:00": {"intraday_move": 5.0, "open_date": "2024-07-01"},
        "15:15": {"intraday_move": -4.0, "open_date": "2024-07-01"},
    }
    second_market = {
        "11:00": {"intraday_move": 12.0, "open_date": "2024-07-01"},
        "15:15": {"intraday_move": -9.5, "open_date": "2024-07-01"},
    }

    def _serialise_market(payload: Mapping[str, Mapping[str, object]]) -> str:
        return json.dumps(
            {session: dict(values) for session, values in payload.items()}
        )

    with runner.isolated_filesystem():
        kit_dir = _make_kit(
            Path.cwd(),
            policy,
            predictions=predictions,
            thresholds=thresholds,
        )
        market_path = Path("market.json")
        market_path.write_text(_serialise_market(first_market), encoding="utf-8")

        result_first = runner.invoke(
            app,
            [
                "decide",
                "--kit-dir",
                str(kit_dir),
                "--offline-daily",
                "--market-json",
                str(market_path),
                "--json",
            ],
        )
        assert result_first.exit_code == 0
        payload_first = json.loads(result_first.stdout)

        market_path.write_text(_serialise_market(second_market), encoding="utf-8")

        result_second = runner.invoke(
            app,
            [
                "decide",
                "--kit-dir",
                str(kit_dir),
                "--offline-daily",
                "--market-json",
                str(market_path),
                "--json",
            ],
        )
        assert result_second.exit_code == 0
        payload_second = json.loads(result_second.stdout)

        assert payload_first == payload_second
        assert (
            payload_first["decision_context_id"]
            == payload_second["decision_context_id"]
        )
        assert "frozen_snapshot" in payload_first


def test_decide_errors_on_digest_mismatch(tmp_path: Path) -> None:
    kit_dir = _build_corruptible_kit(tmp_path)
    policy_path = kit_dir / "hybrid" / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["decide", "--kit-dir", str(kit_dir), "--offline-market"],
    )

    assert result.exit_code != 0
    assert "Kit digest validation failed" in result.output


def test_cli_pipeline_creates_consumable_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    def fake_train_models(
        data_dir: Path,
        artifact_dir: Path,
        *,
        books: Optional[Sequence[str]] = None,
        settings: Settings,
    ) -> SimpleNamespace:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "fold-0.model").write_text("stub", encoding="utf-8")
        calls["train"] = {
            "data_dir": data_dir,
            "artifact_dir": artifact_dir,
            "books": books,
            "settings": settings,
        }
        return SimpleNamespace(artifacts=[object()])

    def fake_discover(
        artifact_dir: Path,
        discovery_dir: Path,
        *,
        settings: Settings,
    ) -> SimpleNamespace:
        discovery_dir.mkdir(parents=True, exist_ok=True)
        (discovery_dir / "rules.json").write_text("{}", encoding="utf-8")
        calls["discover"] = {
            "artifact_dir": artifact_dir,
            "discovery_dir": discovery_dir,
        }
        return SimpleNamespace(folds=[object()])

    def fake_apply(
        artifact_dir: Path,
        discovery_dir: Path,
        apply_dir: Path,
        *,
        settings: Settings,
    ) -> SimpleNamespace:
        apply_dir.mkdir(parents=True, exist_ok=True)
        hybrid_rows = apply_dir / "hybrid_input.csv"
        hybrid_rows.write_text(
            "book,open_date,pnl,prediction,keep\nBOOK,2024-07-01,1.0,0.5,1\n",
            encoding="utf-8",
        )
        calls["apply"] = {
            "artifact_dir": artifact_dir,
            "discovery_dir": discovery_dir,
            "apply_dir": apply_dir,
            "hybrid": hybrid_rows,
        }
        return SimpleNamespace(rows=1, output_path=hybrid_rows)

    monkeypatch.setattr("xdte.cli.train_models", fake_train_models)
    monkeypatch.setattr("xdte.cli.discover_rules", fake_discover)
    monkeypatch.setattr("xdte.cli.apply_models", fake_apply)

    with runner.isolated_filesystem():
        backtest_dir = Path("backtest")
        backtest_dir.mkdir()
        (backtest_dir / "dummy.csv").write_text("book\nBOOK", encoding="utf-8")

        artifacts_root = Path("artifacts")
        artifacts_root.mkdir()

        candidates_path = Path("candidates.json")
        candidates_payload = {
            "candidates": [{"tails": {}, "gammas": {"BOOK": 0.9}, "gamma_rules": {}}]
        }
        candidates_path.write_text(json.dumps(candidates_payload), encoding="utf-8")

        def fake_select(
            apply_path: Path,
            candidates: object,
            output_path: Path,
            *,
            settings: Settings,
        ) -> HybridSelectionResult:
            output_path.write_text(
                json.dumps({"tails": {}, "gammas": {"BOOK": 0.9}, "gamma_rules": {}}),
                encoding="utf-8",
            )
            calls["tune"] = {
                "apply": apply_path,
                "candidates": candidates,
                "output": output_path,
            }
            return HybridSelectionResult(
                baseline_metrics={},
                evaluations=[],
                selected_policy=HybridPolicy(tails={}, gammas={"BOOK": 0.9}),
                accepted=True,
                output_path=output_path,
            )

        def fake_build(
            train_dir: Path,
            discovery_dir: Path,
            hybrid_policy_path: Path,
            output_dir: Path,
            *,
            settings: Settings,
            version: str,
        ) -> ExportResult:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"version": version}), encoding="utf-8")
            digests_path = output_dir / "digests.json"
            digests_path.write_text(json.dumps({"files": {}}), encoding="utf-8")
            calls["export"] = {
                "train": train_dir,
                "discovery": discovery_dir,
                "hybrid": hybrid_policy_path,
                "output": output_dir,
                "version": version,
            }
            exported_file = ExportedFile(
                path=manifest_path,
                relative_path="manifest.json",
                sha256="deadbeef",
            )
            return ExportResult(
                kit_dir=output_dir,
                manifest_path=manifest_path,
                digests_path=digests_path,
                files=[exported_file],
                digests={"manifest.json": "deadbeef"},
            )

        def fake_run_live_decisions(
            kit_dir: Path,
            *,
            settings: Settings,
            market_provider: Optional[object] = None,
            daily_provider: Optional[object] = None,
            **_kwargs: object,
        ) -> LiveDecisionResult:
            calls["decide"] = {"kit_dir": kit_dir}
            sessions = [
                SessionDecision(
                    session="11:00",
                    open_date=date(2024, 7, 1),
                    features={"open_date": date(2024, 7, 1)},
                    warnings=("lagging data",),
                )
            ]
            warning_payload = {
                "level": "WARNING",
                "message": "check hybrid",
                "source": "FakeDecision",
                "timestamp": "2024-07-01T13:00:00+00:00",
            }
            return LiveDecisionResult(
                kit_dir=kit_dir,
                manifest={"version": "ci"},
                policy={"gammas": {"BOOK": 0.9}},
                sessions=sessions,
                decisions=[
                    DecisionRecord(
                        session="11:00",
                        book="BOOK",
                        action="ADJUST_GAMMA",
                        value=0.9,
                    )
                ],
                warnings=(warning_payload,),
                generated_at=datetime(2024, 7, 1, 13, tzinfo=timezone.utc),
            )

        monkeypatch.setattr("xdte.cli.select_hybrid_policy", fake_select)
        monkeypatch.setattr("xdte.cli.build_live_kit", fake_build)
        monkeypatch.setattr("xdte.cli.run_live_decisions", fake_run_live_decisions)

        train_result = runner.invoke(
            app,
            [
                "train",
                "--backtest-dir",
                str(backtest_dir),
                "--out-dir",
                str(artifacts_root),
            ],
        )
        assert train_result.exit_code == 0

        artifacts_dir = (artifacts_root / "artifacts").resolve()
        discovery_dir = (artifacts_root / "discovery").resolve()
        apply_dir = (artifacts_root / "apply").resolve()

        assert artifacts_dir.exists()
        assert discovery_dir.exists()
        assert apply_dir.exists()
        assert (artifacts_dir / "fold-0.model").exists()
        assert (discovery_dir / "rules.json").exists()
        assert (apply_dir / "hybrid_input.csv").exists()

        tune_result = runner.invoke(
            app,
            [
                "tune",
                "--artifacts-dir",
                str(artifacts_root),
                "--candidates",
                str(candidates_path),
            ],
        )
        assert tune_result.exit_code == 0

        policy_path = artifacts_dir.parent / "hybrid" / "policy.json"
        assert policy_path.exists()

        export_dir = Path("kit")
        export_result = runner.invoke(
            app,
            [
                "export-kit",
                "--artifacts-dir",
                str(artifacts_root),
                "--out",
                str(export_dir),
                "--version",
                "ci",
            ],
        )
        assert export_result.exit_code == 0

        manifest_path = export_dir / "manifest.json"
        digests_path = export_dir / "digests.json"
        assert manifest_path.exists()
        assert digests_path.exists()

        decide_result = runner.invoke(app, ["decide", "--kit-dir", str(export_dir)])
        assert decide_result.exit_code == 0
        assert "[WARNING] FakeDecision: check hybrid" in decide_result.output
        assert "Decisions generated" in decide_result.stdout

        assert calls["train"]["data_dir"] == backtest_dir.resolve()
        assert calls["tune"]["apply"] == apply_dir
        assert isinstance(calls["tune"]["candidates"], list)
        assert calls["tune"]["output"] == policy_path
        assert calls["export"]["hybrid"] == policy_path.resolve()
        assert calls["decide"]["kit_dir"] == export_dir.resolve()


def test_cli_sequential_pipeline_with_offline_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    def fake_train_models(
        data_dir: Path,
        artifact_dir: Path,
        *,
        books: Optional[Sequence[str]] = None,
        settings: Settings,
    ) -> SimpleNamespace:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "fold-0.model").write_text("stub", encoding="utf-8")
        calls["train"] = {
            "data_dir": data_dir,
            "artifact_dir": artifact_dir,
            "books": books,
            "settings": settings,
        }
        return SimpleNamespace(artifacts=[object(), object()])

    def fake_discover(
        artifact_dir: Path,
        discovery_dir: Path,
        *,
        settings: Settings,
    ) -> SimpleNamespace:
        discovery_dir.mkdir(parents=True, exist_ok=True)
        (discovery_dir / "rules.json").write_text("{}", encoding="utf-8")
        calls["discover"] = {
            "artifact_dir": artifact_dir,
            "discovery_dir": discovery_dir,
        }
        return SimpleNamespace(folds=[object()])

    def fake_apply(
        artifact_dir: Path,
        discovery_dir: Path,
        apply_dir: Path,
        *,
        settings: Settings,
    ) -> SimpleNamespace:
        apply_dir.mkdir(parents=True, exist_ok=True)
        hybrid_rows = apply_dir / "hybrid_input.csv"
        hybrid_rows.write_text(
            "book,open_date,pnl,prediction,keep\nBOOK,2024-07-01,1.0,0.5,1\n",
            encoding="utf-8",
        )
        calls["apply"] = {
            "artifact_dir": artifact_dir,
            "discovery_dir": discovery_dir,
            "apply_dir": apply_dir,
            "hybrid": hybrid_rows,
        }
        return SimpleNamespace(rows=1, output_path=hybrid_rows)

    def fake_select(
        apply_path: Path,
        candidates: object,
        output_path: Path,
        *,
        settings: Settings,
    ) -> HybridSelectionResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"tails": {}, "gammas": {"BOOK": 0.9}, "gamma_rules": {}}),
            encoding="utf-8",
        )
        calls["tune"] = {
            "apply": apply_path,
            "candidates": candidates,
            "output": output_path,
        }
        return HybridSelectionResult(
            baseline_metrics={},
            evaluations=[],
            selected_policy=HybridPolicy(tails={}, gammas={"BOOK": 0.9}),
            accepted=True,
            output_path=output_path,
        )

    def fake_build(
        train_dir: Path,
        discovery_dir: Path,
        hybrid_policy_path: Path,
        output_dir: Path,
        *,
        settings: Settings,
        version: str,
    ) -> ExportResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"version": version}), encoding="utf-8")
        digests_path = output_dir / "digests.json"
        digests_path.write_text(json.dumps({"files": {}}), encoding="utf-8")
        calls["export"] = {
            "train": train_dir,
            "discovery": discovery_dir,
            "hybrid": hybrid_policy_path,
            "output": output_dir,
            "version": version,
        }
        exported_file = ExportedFile(
            path=manifest_path,
            relative_path="manifest.json",
            sha256="deadbeef",
        )
        return ExportResult(
            kit_dir=output_dir,
            manifest_path=manifest_path,
            digests_path=digests_path,
            files=[exported_file],
            digests={"manifest.json": "deadbeef"},
        )

    daily_builder_calls: dict[str, object] = {}
    market_builder_calls: dict[str, object] = {}

    def fake_daily_builder(open_date: date) -> Callable[[date], dict[str, object]]:
        daily_builder_calls["open_date"] = open_date

        def provider(_open_date: date) -> dict[str, object]:
            return {"stub": "daily"}

        daily_builder_calls["provider"] = provider
        return provider

    def fake_market_builder(open_date: date) -> Callable[[str], dict[str, object]]:
        market_builder_calls["open_date"] = open_date

        def provider(_session: str) -> dict[str, object]:
            return {"stub": "market"}

        market_builder_calls["provider"] = provider
        return provider

    def fake_run_live_decisions(
        kit_dir: Path,
        *,
        settings: Settings,
        market_provider: Optional[Callable[[str], dict[str, object]]],
        daily_provider: Optional[Callable[[date], dict[str, object]]],
        **_kwargs: object,
    ) -> LiveDecisionResult:
        calls["decide"] = {
            "kit_dir": kit_dir,
            "settings": settings,
            "market_provider": market_provider,
            "daily_provider": daily_provider,
        }
        sessions = [
            SessionDecision(
                session="11:00",
                open_date=date(2024, 7, 1),
                features={"open_date": date(2024, 7, 1)},
                warnings=("lagging data",),
            )
        ]
        warning_payload = {
            "level": "WARNING",
            "message": "check hybrid",
            "source": "FakeDecision",
            "timestamp": "2024-07-01T13:30:00+00:00",
        }
        return LiveDecisionResult(
            kit_dir=kit_dir,
            manifest={"version": "ci"},
            policy={"gammas": {"BOOK": 0.9}},
            sessions=sessions,
            decisions=[
                DecisionRecord(
                    session="11:00",
                    book="BOOK",
                    action="ADJUST_GAMMA",
                    value=0.9,
                )
            ],
            warnings=(warning_payload,),
            generated_at=datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
        )

    monkeypatch.setattr("xdte.cli.train_models", fake_train_models)
    monkeypatch.setattr("xdte.cli.discover_rules", fake_discover)
    monkeypatch.setattr("xdte.cli.apply_models", fake_apply)
    monkeypatch.setattr("xdte.cli.select_hybrid_policy", fake_select)
    monkeypatch.setattr("xdte.cli.build_live_kit", fake_build)
    monkeypatch.setattr("xdte.cli._build_offline_daily", fake_daily_builder)
    monkeypatch.setattr("xdte.cli._build_offline_market", fake_market_builder)
    monkeypatch.setattr("xdte.cli.run_live_decisions", fake_run_live_decisions)

    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    (backtest_dir / "dummy.csv").write_text("book\nBOOK", encoding="utf-8")

    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()

    candidates_path = tmp_path / "candidates.json"
    candidates_payload = {
        "candidates": [{"tails": {}, "gammas": {"BOOK": 0.9}, "gamma_rules": {}}]
    }
    candidates_path.write_text(json.dumps(candidates_payload), encoding="utf-8")

    export_dir = tmp_path / "kit"

    with caplog.at_level(logging.INFO):
        train_result = runner.invoke(
            app,
            [
                "--verbose",
                "train",
                "--backtest-dir",
                str(backtest_dir),
                "--out-dir",
                str(artifacts_root),
            ],
        )
    assert train_result.exit_code == 0
    assert "Training complete" in caplog.text
    caplog.clear()

    artifacts_dir = (artifacts_root / "artifacts").resolve()
    discovery_dir = (artifacts_root / "discovery").resolve()
    apply_dir = (artifacts_root / "apply").resolve()

    assert artifacts_dir.exists()
    assert discovery_dir.exists()
    assert apply_dir.exists()
    assert (artifacts_dir / "fold-0.model").exists()
    assert (discovery_dir / "rules.json").exists()
    assert (apply_dir / "hybrid_input.csv").exists()

    with caplog.at_level(logging.INFO):
        tune_result = runner.invoke(
            app,
            [
                "--verbose",
                "tune",
                "--artifacts-dir",
                str(artifacts_root),
                "--candidates",
                str(candidates_path),
            ],
        )
    assert tune_result.exit_code == 0
    assert "Hybrid tuning Accepted" in caplog.text
    caplog.clear()

    policy_path = artifacts_dir.parent / "hybrid" / "policy.json"
    assert policy_path.exists()

    with caplog.at_level(logging.INFO):
        export_result = runner.invoke(
            app,
            [
                "--verbose",
                "export-kit",
                "--artifacts-dir",
                str(artifacts_root),
                "--out",
                str(export_dir),
                "--version",
                "ci",
            ],
        )
    assert export_result.exit_code == 0
    assert "Live Kit exported" in caplog.text
    caplog.clear()

    manifest_path = export_dir / "manifest.json"
    digests_path = export_dir / "digests.json"
    assert manifest_path.exists()
    assert digests_path.exists()

    decide_result = runner.invoke(
        app,
        [
            "decide",
            "--kit-dir",
            str(export_dir),
            "--offline-market",
        ],
    )
    assert decide_result.exit_code == 0
    assert "[WARNING] FakeDecision: check hybrid" in decide_result.output
    assert "Decisions generated from kit" in decide_result.stdout

    assert calls["train"]["data_dir"] == backtest_dir.resolve()
    assert calls["tune"]["apply"] == apply_dir
    assert isinstance(calls["tune"]["candidates"], list)
    assert calls["export"]["hybrid"] == policy_path.resolve()
    assert calls["decide"]["kit_dir"] == export_dir.resolve()
    assert calls["decide"]["market_provider"] is market_builder_calls["provider"]
    assert calls["decide"]["daily_provider"] is daily_builder_calls["provider"]
    assert daily_builder_calls["open_date"] == market_builder_calls["open_date"]


def test_validate_backtest_command_validates_manifest(tmp_path: Path) -> None:
    manifest_path = (
        Path(__file__).resolve().parent / "fixtures" / "backtest" / "manifest.json"
    )
    data_dir = tmp_path / "backtest"
    data_dir.mkdir()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dir = Path("data/backtest_data/original")
    for name in manifest["sources"].keys():
        (data_dir / name).write_bytes((source_dir / name).read_bytes())

    result = runner.invoke(
        app,
        [
            "validate-backtest",
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0
    assert "validated successfully" in result.stdout

    tampered = data_dir / next(iter(manifest["sources"]))
    tampered.write_text("tampered", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "validate-backtest",
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code != 0
    assert "failed validation" in result.output
