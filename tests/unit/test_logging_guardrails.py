"""Logging guardrail tests for the CLI."""

from __future__ import annotations

import logging

import pytest

from xdte.cli import app


def test_cli_configures_structured_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the CLI initialises logging with a structured format."""

    captured_config: dict[str, object] = {}

    def fake_basic_config(*args: object, **kwargs: object) -> None:
        captured_config.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    assert app.callback is not None
    app.callback(verbose=0)

    assert captured_config.get("level") == logging.WARNING

    log_format = captured_config.get("format")
    assert isinstance(log_format, str)
    assert "%(levelname)s" in log_format
    assert "%(message)s" in log_format


def test_cli_verbose_mode_promotes_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure verbosity flags promote logging levels without prints."""

    levels: list[int] = []

    def fake_basic_config(*args: object, **kwargs: object) -> None:
        if isinstance(kwargs.get("level"), int):
            levels.append(kwargs["level"])

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    assert app.callback is not None
    for verbosity in range(3):
        app.callback(verbose=verbosity)

    assert levels == [logging.WARNING, logging.INFO, logging.DEBUG]
