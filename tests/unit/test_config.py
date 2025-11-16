"""Tests for the :mod:`xdte.config` module."""

from __future__ import annotations

import argparse
import json
from typing import Iterator

import pytest

from xdte.config import Settings, get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Ensure the module cache is reset between tests."""

    reset_settings_cache()
    yield
    reset_settings_cache()


def test_defaults_match_notebook() -> None:
    """The defaults must reflect the historical research notebook."""

    settings = Settings()

    assert settings.N_FOLDS == 8
    assert settings.ALPHA == pytest.approx(0.05)
    assert settings.WINSOR_P == pytest.approx(0.01)
    tails = settings.KILL_SWITCH_TAILS
    puts_low, puts_high = tails["PUTS_0DTE_11"]
    calls_low, calls_high = tails["CALLS_0DTE_11"]
    assert puts_low == pytest.approx(0.045)
    assert puts_high == pytest.approx(0.961)
    assert calls_low == pytest.approx(0.0345)
    assert calls_high == pytest.approx(0.952)

    gammas = settings.KILL_SWITCH_GAMMAS
    assert gammas["PUTS_1DTE_1515"] == pytest.approx(1.0)
    assert gammas["CALLS_1DTE_1515"] == pytest.approx(1.0)
    assert settings.MODEL_SEED == 42
    assert settings.TUNER_SEED == 42


def test_environment_overrides() -> None:
    """Environment variables override defaults and cast values."""

    environ = {
        "XDTE_N_FOLDS": "12",
        "XDTE_ALPHA": "0.1",
        "XDTE_WINSOR_P": "0.02",
        "XDTE_KILL_SWITCH_TAILS": json.dumps({"PUTS_0DTE_11": [0.1, 0.9]}),
        "XDTE_KILL_SWITCH_GAMMAS": json.dumps({"CALLS_1DTE_1515": 0.75}),
        "XDTE_MODEL_SEED": "7",
        "XDTE_TUNER_SEED": "11",
    }

    settings = Settings.load(environ=environ)

    assert settings.N_FOLDS == 12
    assert settings.ALPHA == pytest.approx(0.1)
    assert settings.WINSOR_P == pytest.approx(0.02)
    env_tails = settings.KILL_SWITCH_TAILS["PUTS_0DTE_11"]
    assert env_tails[0] == pytest.approx(0.1)
    assert env_tails[1] == pytest.approx(0.9)
    assert settings.KILL_SWITCH_GAMMAS["CALLS_1DTE_1515"] == pytest.approx(0.75)
    assert settings.MODEL_SEED == 7
    assert settings.TUNER_SEED == 11


def test_cli_overrides_take_precedence() -> None:
    """Command-line arguments take precedence over environment variables."""

    environ = {"XDTE_N_FOLDS": "6"}
    args = ["--n-folds", "3", "--model-seed", "99"]

    settings = Settings.load(environ=environ, cli_args=args)

    assert settings.N_FOLDS == 3
    assert settings.MODEL_SEED == 99


def test_cli_parser_round_trip() -> None:
    """`Settings.add_cli_arguments` integrates with :mod:`argparse`."""

    parser = argparse.ArgumentParser(add_help=False)
    Settings.add_cli_arguments(parser)
    tails = json.dumps({"PUTS_0DTE_11": [0.2, 0.8]})
    gammas = json.dumps({"PUTS_1DTE_1515": 0.5})

    namespace = parser.parse_args(
        [
            "--n-folds",
            "4",
            "--alpha",
            "0.2",
            "--winsor-p",
            "0.03",
            "--kill-switch-tails",
            tails,
            "--kill-switch-gammas",
            gammas,
            "--model-seed",
            "13",
            "--tuner-seed",
            "17",
        ]
    )

    assert namespace.N_FOLDS == 4
    settings = Settings.load(cli_args=[], parser=parser)
    assert settings.N_FOLDS == 8

    parsed = Settings.load(
        cli_args=[
            "--n-folds",
            "4",
            "--alpha",
            "0.2",
            "--winsor-p",
            "0.03",
            "--kill-switch-tails",
            tails,
            "--kill-switch-gammas",
            gammas,
            "--model-seed",
            "13",
            "--tuner-seed",
            "17",
        ],
        parser=parser,
    )

    assert parsed.N_FOLDS == 4
    assert parsed.ALPHA == pytest.approx(0.2)
    assert parsed.WINSOR_P == pytest.approx(0.03)
    parsed_tails = parsed.KILL_SWITCH_TAILS["PUTS_0DTE_11"]
    assert parsed_tails[0] == pytest.approx(0.2)
    assert parsed_tails[1] == pytest.approx(0.8)
    assert parsed.KILL_SWITCH_GAMMAS["PUTS_1DTE_1515"] == pytest.approx(0.5)
    assert parsed.MODEL_SEED == 13
    assert parsed.TUNER_SEED == 17


def test_get_settings_caches_and_refreshes() -> None:
    """The module-level cache should be reusable and resettable."""

    first = get_settings()
    second = get_settings()

    assert first is second

    refreshed = get_settings(refresh=True, cli_args=["--model-seed", "101"])

    assert refreshed is not first
    assert refreshed.MODEL_SEED == 101
