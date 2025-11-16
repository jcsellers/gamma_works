"""Configuration primitives for the XDTE package."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from importlib import resources
from typing import Any, Callable, Mapping, Optional, Sequence

from .live.sessions import REQUIRED_SESSIONS

ENV_PREFIX = "XDTE_"


def _load_stub_json(name: str) -> object:
    """Load a stub JSON payload from the installed package resources."""

    path = resources.files("xdte").joinpath(name)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_offline_daily_payload() -> dict[str, object]:
    """Return the default offline payload used for daily context stubs."""

    payload = _load_stub_json("offline_daily_stub.json")
    if not isinstance(payload, Mapping):
        msg = "Offline daily stub must decode to a mapping"
        raise ValueError(msg)
    return dict(payload)


def _default_offline_market_payload() -> dict[str, Mapping[str, object]]:
    """Return the default offline payload used for market data stubs."""

    payload = _load_stub_json("offline_market_stub.json")
    if not isinstance(payload, Mapping):
        msg = "Offline market stub must decode to a mapping"
        raise ValueError(msg)
    parsed: dict[str, Mapping[str, object]] = {}
    for session, session_payload in payload.items():
        if not isinstance(session, str):  # pragma: no cover - defensive guard
            msg = "Offline market stub keys must be strings"
            raise ValueError(msg)
        if not isinstance(session_payload, Mapping):
            msg = "Offline market stub payloads must be mappings"
            raise ValueError(msg)
        parsed[session] = dict(session_payload)
    return parsed


def _parse_mapping(value: str) -> dict[str, object]:
    """Parse a JSON mapping payload from a string."""

    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):
        msg = "Expected a JSON object payload"
        raise ValueError(msg)
    return dict(loaded)


def _parse_nested_mapping(value: str) -> dict[str, Mapping[str, object]]:
    """Parse a JSON mapping of mapping payload from a string."""

    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):
        msg = "Expected a JSON object payload"
        raise ValueError(msg)
    parsed: dict[str, Mapping[str, object]] = {}
    for key, payload in loaded.items():
        if not isinstance(key, str):
            msg = "Mapping keys must be strings"
            raise ValueError(msg)
        if not isinstance(payload, Mapping):
            msg = "Nested payloads must be JSON objects"
            raise ValueError(msg)
        parsed[key] = dict(payload)
    return parsed


def _default_kill_switch_tails() -> dict[str, tuple[float, float]]:
    """Return the legacy default kill-switch tails mapping."""

    return {
        "PUTS_0DTE_11": (0.045, 0.961),
        "CALLS_0DTE_11": (0.0345, 0.952),
    }


def _default_kill_switch_gammas() -> dict[str, float]:
    """Return the legacy default kill-switch gamma scaling mapping."""

    return {
        "PUTS_1DTE_1515": 1.0,
        "CALLS_1DTE_1515": 1.0,
    }


def _parse_tails(value: str) -> dict[str, tuple[float, float]]:
    """Parse a JSON string into a kill-switch tails mapping."""

    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):  # pragma: no cover - defensive guard
        msg = "Kill-switch tails must decode to a mapping"
        raise ValueError(msg)
    parsed: dict[str, tuple[float, float]] = {}
    for key, pair in loaded.items():
        if not isinstance(key, str):  # pragma: no cover - defensive guard
            msg = "Kill-switch tails keys must be strings"
            raise ValueError(msg)
        if not isinstance(pair, Sequence) or len(pair) != 2:
            msg = "Kill-switch tails values must be 2-length sequences"
            raise ValueError(msg)
        low, high = float(pair[0]), float(pair[1])
        parsed[key] = (low, high)
    return parsed


def _parse_gammas(value: str) -> dict[str, float]:
    """Parse a JSON string into a kill-switch gamma mapping."""

    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):  # pragma: no cover - defensive guard
        msg = "Kill-switch gammas must decode to a mapping"
        raise ValueError(msg)
    parsed: dict[str, float] = {}
    for key, gamma in loaded.items():
        if not isinstance(key, str):  # pragma: no cover - defensive guard
            msg = "Kill-switch gamma keys must be strings"
            raise ValueError(msg)
        parsed[key] = float(gamma)
    return parsed


_FIELD_CASTERS: Mapping[str, Callable[[str], Any]] = {
    "N_FOLDS": int,
    "ALPHA": float,
    "WINSOR_P": float,
    "KILL_SWITCH_TAILS": _parse_tails,
    "KILL_SWITCH_GAMMAS": _parse_gammas,
    "MODEL_SEED": int,
    "TUNER_SEED": int,
    "CONFIDENCE_MARGIN_SCALE": float,
    "OFFLINE_DAILY_PAYLOAD": _parse_mapping,
    "OFFLINE_MARKET_PAYLOAD": _parse_nested_mapping,
}

_SESSION_KEY_HELP = "', '".join(REQUIRED_SESSIONS)


@dataclass(frozen=True)
class Settings:
    """Package configuration with legacy defaults and override hooks."""

    N_FOLDS: int = 8
    ALPHA: float = 0.05
    WINSOR_P: float = 0.01
    KILL_SWITCH_TAILS: Mapping[str, tuple[float, float]] = field(
        default_factory=_default_kill_switch_tails
    )
    KILL_SWITCH_GAMMAS: Mapping[str, float] = field(
        default_factory=_default_kill_switch_gammas
    )
    MODEL_SEED: int = 42
    TUNER_SEED: int = 42
    CONFIDENCE_MARGIN_SCALE: float = 0.05
    OFFLINE_DAILY_PAYLOAD: Mapping[str, object] = field(
        default_factory=_default_offline_daily_payload
    )
    OFFLINE_MARKET_PAYLOAD: Mapping[str, Mapping[str, object]] = field(
        default_factory=_default_offline_market_payload
    )

    @classmethod
    def _read_env(cls, environ: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
        env: Mapping[str, str] = os.environ if environ is None else environ
        overrides: dict[str, Any] = {}
        for field_info in dataclass_fields(cls):
            env_key = f"{ENV_PREFIX}{field_info.name}"
            value = env.get(env_key)
            if value is None:
                continue
            caster = _FIELD_CASTERS[field_info.name]
            overrides[field_info.name] = caster(value)
        return overrides

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register CLI arguments corresponding to settings overrides."""

        defaults = cls()
        parser.add_argument(
            "--n-folds",
            type=int,
            dest="N_FOLDS",
            default=None,
            help=f"Number of walk-forward folds (default: {defaults.N_FOLDS})",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            dest="ALPHA",
            default=None,
            help=f"Risk level for CVaR calculations (default: {defaults.ALPHA})",
        )
        parser.add_argument(
            "--winsor-p",
            type=float,
            dest="WINSOR_P",
            default=None,
            help=f"Winsorisation percentile used in discovery (default: {defaults.WINSOR_P})",
        )
        parser.add_argument(
            "--kill-switch-tails",
            type=_parse_tails,
            dest="KILL_SWITCH_TAILS",
            default=None,
            help="JSON mapping of kill-switch tail quantiles",
        )
        parser.add_argument(
            "--kill-switch-gammas",
            type=_parse_gammas,
            dest="KILL_SWITCH_GAMMAS",
            default=None,
            help="JSON mapping of kill-switch gamma scalars",
        )
        parser.add_argument(
            "--model-seed",
            type=int,
            dest="MODEL_SEED",
            default=None,
            help=f"Random seed for model training (default: {defaults.MODEL_SEED})",
        )
        parser.add_argument(
            "--tuner-seed",
            type=int,
            dest="TUNER_SEED",
            default=None,
            help=f"Random seed for tuning (default: {defaults.TUNER_SEED})",
        )
        parser.add_argument(
            "--confidence-margin-scale",
            type=float,
            dest="CONFIDENCE_MARGIN_SCALE",
            default=None,
            help=(
                "Scale factor applied to the absolute decision margin when "
                "computing confidence (default: "
                f"{defaults.CONFIDENCE_MARGIN_SCALE})"
            ),
        )
        parser.add_argument(
            "--offline-daily-payload",
            type=_parse_mapping,
            dest="OFFLINE_DAILY_PAYLOAD",
            default=None,
            help=(
                "JSON mapping providing feature values for the offline "
                "daily context stub"
            ),
        )
        parser.add_argument(
            "--offline-market-payload",
            type=_parse_nested_mapping,
            dest="OFFLINE_MARKET_PAYLOAD",
            default=None,
            help=(
                "JSON mapping keyed by session name providing market data "
                "payloads for the offline stub "
                f"(keys: '{_SESSION_KEY_HELP}')."
            ),
        )

    @classmethod
    def _read_cli(
        cls,
        args: Optional[Sequence[str]] = None,
        parser: Optional[argparse.ArgumentParser] = None,
    ) -> dict[str, Any]:
        if parser is None:
            cli_parser = argparse.ArgumentParser(add_help=False)
            cls.add_cli_arguments(cli_parser)
        else:
            cli_parser = parser
        namespace, _ = cli_parser.parse_known_args(args)
        overrides: dict[str, Any] = {}
        for field_info in dataclass_fields(cls):
            value = getattr(namespace, field_info.name, None)
            if value is not None:
                overrides[field_info.name] = value
        return overrides

    @classmethod
    def load(
        cls,
        *,
        environ: Optional[Mapping[str, str]] = None,
        cli_args: Optional[Sequence[str]] = None,
        parser: Optional[argparse.ArgumentParser] = None,
    ) -> "Settings":
        """Create a settings object from defaults with env/CLI overrides applied."""

        overrides: dict[str, Any] = {}
        overrides.update(cls._read_env(environ=environ))
        overrides.update(cls._read_cli(args=cli_args, parser=parser))
        return cls(**overrides)


_SETTINGS_CACHE: Optional[Settings] = None


def get_settings(
    *,
    refresh: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    cli_args: Optional[Sequence[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> Settings:
    """Return a cached :class:`Settings` instance, refreshing when requested."""

    global _SETTINGS_CACHE
    if refresh or _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = Settings.load(
            environ=environ,
            cli_args=cli_args,
            parser=parser,
        )
    return _SETTINGS_CACHE


def reset_settings_cache() -> None:
    """Reset the cached settings to force re-computation on next access."""

    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
