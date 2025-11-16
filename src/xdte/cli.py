"""Command line entry points for the XDTE package."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, cast

import click
import pandas as pd

from .config import Settings, get_settings
from .data.features import compute_daily_context_features, write_daily_context_csv
from .data.loaders import read_portfolio_file
from .formatting import (
    _format_feature_value,
    _format_top_features,
    _format_warning_entry,
    _json_safe,
    _render_confidence,
)
from .live.actions import FinalAction, compile_operational_directives
from .live.decide import (
    DailyContextProvider,
    LiveDecisionResult,
    MarketDataProvider,
    run_live_decisions,
)
from .live.providers import (
    YFinanceDailyContextProvider,
    build_stub_daily_provider,
    build_stub_market_provider,
)
from .live.sessions import REQUIRED_SESSIONS, SESSION_ELEVEN_AM, SESSION_FIFTEEN_FIFTEEN
from .model.apply import apply_models
from .model.discovery import discover_rules
from .model.export import ExportResult, build_live_kit
from .model.hybrid import HybridSelectionResult, select_hybrid_policy
from .model.train import TrainResult, train_models
from .persistence import _persist_decision_run

_SESSION_KEY_HELP = "', '".join(REQUIRED_SESSIONS)


def _discover_trade_files(directory: Path) -> list[Path]:
    """Return trade CSV files, ignoring market stats and helper payloads."""

    files: list[Path] = []
    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith("."):
            continue
        if path.stem.startswith("daily_context"):
            continue
        if "market_stats" in path.name:
            continue
        files.append(path)
    return files


def _infer_trade_date_range(directory: Path) -> tuple[date, date]:
    """Compute the inclusive trade date span for the bundle."""

    trade_files = _discover_trade_files(directory)
    open_dates: list[date] = []
    for trade_file in trade_files:
        for record in read_portfolio_file(trade_file):
            open_dates.append(record.open_timestamp.date())
    if not open_dates:
        raise click.ClickException(
            f"No trade rows found in {directory}. Ensure the directory contains portfolio CSVs."
        )
    return min(open_dates), max(open_dates)


def _build_daily_rows_from_frame(frame: Any) -> pd.DataFrame:
    """Normalise the provider frame into the schema expected by feature helpers."""

    if frame is None:
        return pd.DataFrame(columns=["open_date"])
    try:
        working = frame.copy()
    except AttributeError:
        return pd.DataFrame(columns=["open_date"])

    if isinstance(working.columns, pd.MultiIndex):
        working.columns = [
            "_".join(str(part) for part in column if part)
            for column in working.columns.to_list()
        ]

    index = working.index
    if not isinstance(index, pd.DatetimeIndex):
        raise click.ClickException("Daily provider did not return a DatetimeIndex")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = index.normalize()
    working = working.set_index(index)

    column_map = {
        "Close_^GSPC": "SPX_Close",
        "Open_^GSPC": "SPX_Open",
        "High_^GSPC": "SPX_High",
        "Low_^GSPC": "SPX_Low",
        "Close_^VIX": "VIX_Close",
        "Close_^VIX3M": "VIX3M_Close",
        "Close_^VVIX": "VVIX_Close",
    }

    rows = pd.DataFrame(index=index)
    rows.index.name = "open_date"
    for source, dest in column_map.items():
        if source in working:
            rows[dest] = pd.Series(working[source].to_numpy(), index=index)

    return rows.reset_index()


def create_parser() -> argparse.ArgumentParser:
    """Create an argument parser that exposes XDTE configuration knobs.

    Returns
    -------
    argparse.ArgumentParser
        Parser pre-populated with the :class:`~xdte.config.Settings`
        overrides so command-line utilities and tests can share a consistent
        interface.
    """

    parser = argparse.ArgumentParser(prog="xdte")
    Settings.add_cli_arguments(parser)
    return parser


def parse_settings(args: Optional[Sequence[str]] = None) -> Settings:
    """Parse CLI arguments and return a settings instance.

    Parameters
    ----------
    args:
        Optional iterable of argument strings. When ``None`` the active
        ``sys.argv`` is parsed, mimicking normal command-line execution.

    Returns
    -------
    Settings
        Materialised configuration reflecting defaults, environment overrides
        and command-line flags.

    Raises
    ------
    SystemExit
        Propagated if parsing fails. This mirrors ``argparse``'s behaviour and
        ensures CLI consumers receive helpful usage messages.
    """

    parser = create_parser()
    return Settings.load(cli_args=args, parser=parser)


def load_settings(args: Optional[Sequence[str]] = None) -> Settings:
    """Return cached settings while honouring CLI arguments for overrides.

    Parameters
    ----------
    args:
        Optional iterable of arguments. Passing a sequence bypasses the cached
        settings instance to honour ad-hoc overrides (primarily for tests).

    Returns
    -------
    Settings
        Cached :class:`~xdte.config.Settings` instance, refreshed when explicit
        arguments are supplied.
    """

    refresh = bool(args)
    return get_settings(refresh=refresh, cli_args=args, parser=create_parser())


def _settings_cli_args(
    *,
    n_folds: Optional[int],
    alpha: Optional[float],
    winsor_p: Optional[float],
    kill_switch_tails: Optional[str],
    kill_switch_gammas: Optional[str],
    model_seed: Optional[int],
    tuner_seed: Optional[int],
    confidence_margin_scale: Optional[float],
    offline_daily_payload: Optional[str],
    offline_market_payload: Optional[str],
) -> list[str]:
    """Return CLI-style arguments for settings overrides."""

    args: list[str] = []
    if n_folds is not None:
        args.extend(["--n-folds", str(n_folds)])
    if alpha is not None:
        args.extend(["--alpha", str(alpha)])
    if winsor_p is not None:
        args.extend(["--winsor-p", str(winsor_p)])
    if kill_switch_tails is not None:
        args.extend(["--kill-switch-tails", kill_switch_tails])
    if kill_switch_gammas is not None:
        args.extend(["--kill-switch-gammas", kill_switch_gammas])
    if model_seed is not None:
        args.extend(["--model-seed", str(model_seed)])
    if tuner_seed is not None:
        args.extend(["--tuner-seed", str(tuner_seed)])
    if confidence_margin_scale is not None:
        args.extend(["--confidence-margin-scale", str(confidence_margin_scale)])
    if offline_daily_payload is not None:
        args.extend(["--offline-daily-payload", offline_daily_payload])
    if offline_market_payload is not None:
        args.extend(["--offline-market-payload", offline_market_payload])
    return args


def _load_settings_from_options(
    *,
    n_folds: Optional[int],
    alpha: Optional[float],
    winsor_p: Optional[float],
    kill_switch_tails: Optional[str],
    kill_switch_gammas: Optional[str],
    model_seed: Optional[int],
    tuner_seed: Optional[int],
    confidence_margin_scale: Optional[float],
    offline_daily_payload: Optional[str],
    offline_market_payload: Optional[str],
) -> Settings:
    """Materialise settings using the provided option overrides."""

    args = _settings_cli_args(
        n_folds=n_folds,
        alpha=alpha,
        winsor_p=winsor_p,
        kill_switch_tails=kill_switch_tails,
        kill_switch_gammas=kill_switch_gammas,
        model_seed=model_seed,
        tuner_seed=tuner_seed,
        confidence_margin_scale=confidence_margin_scale,
        offline_daily_payload=offline_daily_payload,
        offline_market_payload=offline_market_payload,
    )
    return parse_settings(args if args else None)


def _build_offline_daily(
    open_date: date, *, settings: Settings
) -> DailyContextProvider:
    payload = dict(settings.OFFLINE_DAILY_PAYLOAD)
    return cast(
        DailyContextProvider,
        build_stub_daily_provider(open_date=open_date, payload=payload),
    )


def _build_offline_market(open_date: date, *, settings: Settings) -> MarketDataProvider:
    market_payloads = dict(settings.OFFLINE_MARKET_PAYLOAD)
    missing = [
        session for session in REQUIRED_SESSIONS if session not in market_payloads
    ]
    if missing:
        missing_list = ", ".join(sorted(missing))
        msg = f"Offline market payload must define sessions for {missing_list}"
        raise ValueError(msg)
    session_payloads = {
        session: dict(market_payloads[session]) for session in REQUIRED_SESSIONS
    }
    return cast(
        MarketDataProvider,
        build_stub_market_provider(
            open_date=open_date,
            eleven_payload=session_payloads[SESSION_ELEVEN_AM],
            fifteen_payload=session_payloads[SESSION_FIFTEEN_FIFTEEN],
        ),
    )


def _offline_daily_provider(
    open_date: date, settings: Settings
) -> DailyContextProvider:
    builder: Callable[..., DailyContextProvider] = _build_offline_daily
    try:
        try:
            return builder(open_date, settings=settings)
        except TypeError as exc:
            if "settings" not in str(exc):
                raise
            return builder(open_date)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _offline_market_provider(open_date: date, settings: Settings) -> MarketDataProvider:
    builder: Callable[..., MarketDataProvider] = _build_offline_market
    try:
        try:
            return builder(open_date, settings=settings)
        except TypeError as exc:
            if "settings" not in str(exc):
                raise
            return builder(open_date)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _load_candidates_from_file(
    path: Path,
) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        candidates = payload.get("candidates", payload)
        if isinstance(candidates, Mapping):
            return cast(Mapping[str, object], candidates)
        if isinstance(candidates, Sequence):
            candidate_maps: list[Mapping[str, object]] = []
            for item in candidates:
                if not isinstance(item, Mapping):
                    break
                candidate_maps.append(cast(Mapping[str, object], item))
            else:
                return candidate_maps
    elif isinstance(payload, Sequence):
        payload_maps: list[Mapping[str, object]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                break
            payload_maps.append(cast(Mapping[str, object], item))
        else:
            return payload_maps
    msg = "Candidates file must contain a mapping or sequence payload"
    raise ValueError(msg)


def _format_actions_table(
    result: LiveDecisionResult, actions: Sequence[FinalAction]
) -> str:
    if not actions:
        return "No actions generated."

    lines: list[str] = []
    for session in result.sessions:
        lines.append(
            "Session {session} (open date {date})".format(
                session=session.session, date=session.open_date.isoformat()
            )
        )
        feature_subset = {
            key: value
            for key, value in session.features.items()
            if key.startswith("VIX_Entry")
            or key.startswith("Intraday_Move_OpenToEntry")
            or key in {"DoW", "L1_vvix_above_ema20", "L1_vvix_above_ema30"}
        }
        if feature_subset:
            for key in sorted(feature_subset):
                lines.append(f"  {key}: {_format_feature_value(feature_subset[key])}")
        if session.warnings:
            lines.append("  Warnings:")
            for warning in session.warnings:
                lines.append(f"    - {warning}")
        lines.append("")

    header = (
        f"{'Session':<8}"
        f"{'Book':<20}"
        f"{'Rule':<10}"
        f"{'Final':<10}"
        f"{'Gamma':<8}"
        f"{'Score':<10}"
        f"{'Margin':<10}"
        f"{'Kill':<6}"
        f"{'Policy':<8}"
        f"{'Conf':<8}"
        "WHY"
    )
    separator = "-" * len(header)
    lines.append(header)
    lines.append(separator)
    for action in actions:
        margin = action.margin_to_threshold
        score = action.score
        confidence = action.rationale.get("confidence")
        gamma_str = f"{action.size_gamma:.2f}"
        margin_str = "n/a"
        if isinstance(margin, (int, float)) and math.isfinite(float(margin)):
            margin_str = f"{float(margin):+0.3f}"
        score_str = "n/a"
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            score_str = f"{float(score):+0.3f}"
        kill_str = "Y" if action.kill_flag else "N"
        reason = action.reason_text or "n/a"
        features_summary = _format_top_features(action.rationale.get("top_features"))
        if features_summary:
            reason = f"{reason} | {features_summary}"
        confidence_str = _render_confidence(confidence)

        lines.append(
            "{session:<8}{book:<20}{rule:<10}{final:<10}{gamma:<8}{score:<10}{margin:<10}{kill:<6}{policy:<8}{conf:<8}{reason}".format(
                session=action.session,
                book=action.book,
                rule=action.rule_action,
                final=action.final_action,
                gamma=gamma_str,
                score=score_str,
                margin=margin_str,
                kill=kill_str,
                policy=action.policy_type,
                conf=confidence_str,
                reason=reason,
            )
        )
    return "\n".join(lines)


def _validate_directory(path: Path, *, param_name: str) -> Path:
    """Ensure ``path`` is an existing directory, raising a CLI-friendly error."""

    if not path.exists():  # pragma: no cover - Click ensures this, but defensive
        raise click.BadParameter(f"{param_name} does not exist: {path}")
    if not path.is_dir():
        raise click.BadParameter(f"{param_name} must be a directory: {path}")
    return path


def _validate_output_parent(path: Path, *, param_name: str) -> Path:
    """Ensure the parent directory for a file output exists."""

    parent = path.parent
    if not parent.exists():
        raise click.BadParameter(
            f"Parent directory for {param_name} does not exist: {parent}"
        )
    return path


def _settings_options(func: Callable[..., None]) -> Callable[..., None]:
    """Apply common settings options to a Click command."""

    option_decorators = [
        click.option(
            "--n-folds",
            type=int,
            default=None,
            help="Number of walk-forward folds used for model training.",
            show_default=False,
        ),
        click.option(
            "--alpha",
            type=float,
            default=None,
            help="Risk level for Conditional Value-at-Risk calculations.",
            show_default=False,
        ),
        click.option(
            "--winsor-p",
            type=float,
            default=None,
            help="Winsorisation percentile for discovery stage metrics.",
            show_default=False,
        ),
        click.option(
            "--kill-switch-tails",
            type=str,
            default=None,
            help="JSON mapping defining tail quantiles for hybrid kill-switch controls.",
            show_default=False,
        ),
        click.option(
            "--kill-switch-gammas",
            type=str,
            default=None,
            help="JSON mapping defining gamma scalars for hybrid controls.",
            show_default=False,
        ),
        click.option(
            "--model-seed",
            type=int,
            default=None,
            help="Random seed for model training.",
            show_default=False,
        ),
        click.option(
            "--tuner-seed",
            type=int,
            default=None,
            help="Random seed for tuning components.",
            show_default=False,
        ),
        click.option(
            "--confidence-margin-scale",
            type=float,
            default=None,
            help="Scale factor applied to decision margin when computing confidence.",
            show_default=False,
        ),
        click.option(
            "--offline-daily-payload",
            type=str,
            default=None,
            help="JSON mapping used for the offline daily context stub.",
            show_default=False,
        ),
        click.option(
            "--offline-market-payload",
            type=str,
            default=None,
            help="JSON mapping used for the offline market data stub.",
            show_default=False,
        ),
    ]
    for decorator in reversed(option_decorators):
        func = decorator(func)
    return func


@click.group(help="XDTE machine learning selector pipeline entry points.")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase logging verbosity. Repeat for more detailed logs.",
)
def app(verbose: int) -> None:
    """Entry point for the XDTE CLI group.

    Parameters
    ----------
    verbose:
        Verbosity counter supplied by Click. ``0`` keeps logging at ``WARNING``
        level, ``1`` promotes to ``INFO`` and ``2`` or more enables debugging
        output.
    """

    level = logging.WARNING
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


@app.command("fetch-daily-context")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing the frozen backtest CSV bundle.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
    envvar="XDTE_CACHE_DIR",
    show_default=False,
    help="Optional directory for caching yfinance downloads.",
)
@click.option(
    "--cache-freshness-minutes",
    type=float,
    default=None,
    show_default=False,
    help="Override the cache freshness window (minutes).",
)
def fetch_daily_context(
    data_dir: Path,
    cache_dir: Path | None,
    cache_freshness_minutes: float | None,
) -> None:
    """Fetch lagged daily context rows for the backtest bundle."""

    resolved_dir = _validate_directory(data_dir, param_name="--data-dir").resolve()
    min_trade, max_trade = _infer_trade_date_range(resolved_dir)
    fetch_start = min_trade - timedelta(days=90)

    cache_freshness: timedelta | None = None
    if cache_freshness_minutes is not None:
        cache_freshness = timedelta(minutes=cache_freshness_minutes)

    try:
        provider = YFinanceDailyContextProvider(
            cache_dir=cache_dir,
            cache_freshness=cache_freshness,
        )
    except Exception as exc:  # pragma: no cover - provider import/runtime guard
        raise click.ClickException(
            f"Failed to initialise daily provider: {exc}"
        ) from exc

    provider_frame = provider.to_frame()
    daily_rows = _build_daily_rows_from_frame(provider_frame)
    if daily_rows.empty:
        raise click.ClickException("Daily provider did not return any usable rows")

    daily_rows["open_date"] = pd.to_datetime(daily_rows["open_date"])
    filtered_rows = daily_rows[
        (daily_rows["open_date"] >= pd.Timestamp(fetch_start))
        & (daily_rows["open_date"] <= pd.Timestamp(max_trade))
    ]
    if filtered_rows.empty:
        raise click.ClickException(
            "Daily provider did not return rows covering the requested trade range"
        )

    feature_frame = compute_daily_context_features(filtered_rows)
    trimmed = feature_frame.loc[
        (feature_frame.index.date >= min_trade)
        & (feature_frame.index.date <= max_trade)
    ]
    if trimmed.empty:
        raise click.ClickException(
            "Failed to compute engineered daily context features for the trade range"
        )

    output_frame = trimmed.reset_index()
    write_daily_context_csv(
        resolved_dir,
        output_frame,
        warnings=getattr(provider, "warnings", None),
    )

    click.echo(
        "daily_context.csv updated for trade range "
        f"[{min_trade.isoformat()} – {max_trade.isoformat()}]"
    )


@app.command()
@click.option(
    "--backtest-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing frozen backtest datasets.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory where training artifacts should be written.",
)
@click.option(
    "--books",
    multiple=True,
    metavar="BOOK",
    help=(
        "Limit training to the specified books. "
        "Provide the option multiple times to select more than one book."
    ),
)
@_settings_options
def train(
    backtest_dir: Path,
    out_dir: Path,
    books: tuple[str, ...],
    n_folds: Optional[int],
    alpha: Optional[float],
    winsor_p: Optional[float],
    kill_switch_tails: Optional[str],
    kill_switch_gammas: Optional[str],
    model_seed: Optional[int],
    tuner_seed: Optional[int],
    confidence_margin_scale: Optional[float],
    offline_daily_payload: Optional[str],
    offline_market_payload: Optional[str],
) -> None:
    """Run the training pipeline on historical backtest data."""

    resolved_backtest = _validate_directory(
        backtest_dir, param_name="--backtest-dir"
    ).resolve()
    resolved_out = _validate_directory(out_dir, param_name="--out-dir").resolve()

    settings = _load_settings_from_options(
        n_folds=n_folds,
        alpha=alpha,
        winsor_p=winsor_p,
        kill_switch_tails=kill_switch_tails,
        kill_switch_gammas=kill_switch_gammas,
        model_seed=model_seed,
        tuner_seed=tuner_seed,
        confidence_margin_scale=confidence_margin_scale,
        offline_daily_payload=offline_daily_payload,
        offline_market_payload=offline_market_payload,
    )
    artifacts_dir = resolved_out / "artifacts"
    discovery_dir = resolved_out / "discovery"
    apply_dir = resolved_out / "apply"

    try:
        train_result: TrainResult = train_models(
            resolved_backtest,
            artifacts_dir,
            settings=settings,
            books=books or None,
        )
        discover_rules(
            artifacts_dir,
            discovery_dir,
            settings=settings,
        )
        apply_models(
            artifacts_dir,
            discovery_dir,
            apply_dir,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - safety guard
        raise click.ClickException(f"Training failed: {exc}") from exc

    logging.info(
        "Training complete.\n"
        f"- Artifacts: {artifacts_dir}\n"
        f"- Discovery: {discovery_dir}\n"
        f"- Apply: {apply_dir}\n"
        f"Trained {len(train_result.artifacts)} fold models."
    )


@app.command()
@click.option(
    "--artifacts-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing training artifacts for tuning.",
)
@click.option(
    "--candidates",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="JSON file describing candidate hybrid policies.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Target file path for the selected hybrid policy.",
)
@_settings_options
def tune(
    artifacts_dir: Path,
    candidates: Path,
    out: Optional[Path],
    n_folds: Optional[int],
    alpha: Optional[float],
    winsor_p: Optional[float],
    kill_switch_tails: Optional[str],
    kill_switch_gammas: Optional[str],
    model_seed: Optional[int],
    tuner_seed: Optional[int],
    confidence_margin_scale: Optional[float],
    offline_daily_payload: Optional[str],
    offline_market_payload: Optional[str],
) -> None:
    """Tune the hybrid policy using existing training artifacts."""

    resolved_artifacts = _validate_directory(
        artifacts_dir, param_name="--artifacts-dir"
    ).resolve()
    resolved_candidates = candidates.resolve()

    settings = _load_settings_from_options(
        n_folds=n_folds,
        alpha=alpha,
        winsor_p=winsor_p,
        kill_switch_tails=kill_switch_tails,
        kill_switch_gammas=kill_switch_gammas,
        model_seed=model_seed,
        tuner_seed=tuner_seed,
        confidence_margin_scale=confidence_margin_scale,
        offline_daily_payload=offline_daily_payload,
        offline_market_payload=offline_market_payload,
    )
    try:
        candidate_payload = _load_candidates_from_file(resolved_candidates)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    policy_path = out or (resolved_artifacts / "hybrid" / "policy.json")
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    apply_dir = resolved_artifacts / "apply"
    try:
        selection: HybridSelectionResult = select_hybrid_policy(
            apply_dir,
            candidate_payload,
            policy_path,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - safety guard
        raise click.ClickException(f"Tuning failed: {exc}") from exc

    status = "accepted" if selection.accepted else "rejected"
    logging.info(
        "Hybrid tuning {status}. Selected policy written to {path}".format(
            status=status.capitalize(), path=selection.output_path
        )
    )


@app.command("export-kit")
@click.option(
    "--artifacts-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing tuned artifacts for export.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
    help="Directory where the Live Kit should be written.",
)
@click.option(
    "--version",
    type=str,
    default="v1",
    show_default=True,
    help="Version string embedded in the Live Kit manifest.",
)
@_settings_options
def export_kit(
    artifacts_dir: Path,
    out: Path,
    version: str,
    n_folds: Optional[int],
    alpha: Optional[float],
    winsor_p: Optional[float],
    kill_switch_tails: Optional[str],
    kill_switch_gammas: Optional[str],
    model_seed: Optional[int],
    tuner_seed: Optional[int],
    confidence_margin_scale: Optional[float],
    offline_daily_payload: Optional[str],
    offline_market_payload: Optional[str],
) -> None:
    """Export a versioned Live Kit from tuned artifacts."""

    resolved_artifacts = _validate_directory(
        artifacts_dir, param_name="--artifacts-dir"
    ).resolve()
    resolved_out = out.resolve()

    settings = _load_settings_from_options(
        n_folds=n_folds,
        alpha=alpha,
        winsor_p=winsor_p,
        kill_switch_tails=kill_switch_tails,
        kill_switch_gammas=kill_switch_gammas,
        model_seed=model_seed,
        tuner_seed=tuner_seed,
        confidence_margin_scale=confidence_margin_scale,
        offline_daily_payload=offline_daily_payload,
        offline_market_payload=offline_market_payload,
    )
    train_dir = resolved_artifacts / "artifacts"
    discovery_dir = resolved_artifacts / "discovery"
    hybrid_policy_path = resolved_artifacts / "hybrid" / "policy.json"

    try:
        export_result: ExportResult = build_live_kit(
            train_dir,
            discovery_dir,
            hybrid_policy_path,
            resolved_out,
            settings=settings,
            version=version,
        )
    except Exception as exc:  # pragma: no cover - safety guard
        raise click.ClickException(f"Export failed: {exc}") from exc

    logging.info(
        "Live Kit exported to {kit}. Manifest: {manifest}".format(
            kit=export_result.kit_dir, manifest=export_result.manifest_path
        )
    )


@app.command()
@click.option(
    "--kit-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing the exported Live Kit.",
)
@click.option(
    "--market-json",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=None,
    show_default=False,
    help=f"JSON payload with per-session market inputs (keys: '{_SESSION_KEY_HELP}').",
)
@click.option(
    "--daily-json",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=None,
    show_default=False,
    help="Optional JSON payload providing lagged daily indicators.",
)
@click.option(
    "--json/--no-json",
    "json_output",
    default=False,
    help="Emit decisions as JSON instead of table output.",
    show_default=True,
)
@click.option(
    "--offline-market",
    is_flag=True,
    default=False,
    help="Use built-in offline market data stub instead of yfinance.",
    show_default=False,
)
@click.option(
    "--offline-daily",
    is_flag=True,
    default=False,
    help="Use built-in offline daily context stub instead of yfinance.",
    show_default=False,
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
    envvar="XDTE_CACHE_DIR",
    show_default=False,
    help="Optional directory for caching yfinance downloads during live runs.",
)
@click.option(
    "--freeze-now/--no-freeze-now",
    default=True,
    show_default=True,
    help="Freeze live provider payloads on first run and reuse them for repeats.",
)
@_settings_options
def decide(
    kit_dir: Path,
    market_json: Path | None,
    daily_json: Path | None,
    json_output: bool,
    offline_market: bool,
    offline_daily: bool,
    cache_dir: Path | None,
    freeze_now: bool,
    n_folds: Optional[int],
    alpha: Optional[float],
    winsor_p: Optional[float],
    kill_switch_tails: Optional[str],
    kill_switch_gammas: Optional[str],
    model_seed: Optional[int],
    tuner_seed: Optional[int],
    confidence_margin_scale: Optional[float],
    offline_daily_payload: Optional[str],
    offline_market_payload: Optional[str],
) -> None:
    """Produce live decisions using a previously exported kit."""

    resolved_kit = _validate_directory(kit_dir, param_name="--kit-dir").resolve()

    settings = _load_settings_from_options(
        n_folds=n_folds,
        alpha=alpha,
        winsor_p=winsor_p,
        kill_switch_tails=kill_switch_tails,
        kill_switch_gammas=kill_switch_gammas,
        model_seed=model_seed,
        tuner_seed=tuner_seed,
        confidence_margin_scale=confidence_margin_scale,
        offline_daily_payload=offline_daily_payload,
        offline_market_payload=offline_market_payload,
    )
    if offline_market and market_json is not None:
        raise click.UsageError("Cannot combine --offline-market with --market-json")
    if offline_daily and daily_json is not None:
        raise click.UsageError("Cannot combine --offline-daily with --daily-json")

    market_provider = None
    if market_json is not None:
        market_payload = json.loads(market_json.read_text(encoding="utf-8"))
        if not isinstance(market_payload, Mapping):
            raise click.ClickException("Market JSON must be a mapping keyed by session")

        def _market_provider(session: str) -> Mapping[str, object]:
            data = market_payload.get(session)
            if not isinstance(data, Mapping):
                raise RuntimeError(f"Market JSON missing mapping for session {session}")
            return cast(Mapping[str, object], data)

        market_provider = _market_provider

    daily_provider = None
    if daily_json is not None:
        daily_payload = json.loads(daily_json.read_text(encoding="utf-8"))
        if not isinstance(daily_payload, Mapping):
            raise click.ClickException("Daily JSON must contain a mapping payload")

        def _daily_provider(open_date: date) -> Mapping[str, object]:
            key = open_date.isoformat()
            data = daily_payload.get(key)
            if isinstance(data, Mapping):
                return cast(Mapping[str, object], data)
            return cast(Mapping[str, object], daily_payload)

        daily_provider = _daily_provider

    stub_open_date = date.today()
    if offline_daily:
        if daily_provider is not None:
            raise click.UsageError(
                "Cannot use --offline-daily when a daily provider is already set (e.g., from --daily-json)"
            )
        daily_provider = _offline_daily_provider(stub_open_date, settings)
    if offline_market:
        if daily_provider is not None:
            raise click.UsageError(
                "Cannot use --offline-market when a daily provider is already set (e.g., from --daily-json or --offline-daily)"
            )
        daily_provider = _offline_daily_provider(stub_open_date, settings)
        if market_provider is not None:
            raise click.UsageError(
                "Cannot use --offline-market when a market provider is already set (e.g., from --market-json)"
            )
        market_provider = _offline_market_provider(stub_open_date, settings)

    try:
        decision_result = run_live_decisions(
            resolved_kit,
            settings=settings,
            market_provider=market_provider,
            daily_provider=daily_provider,
            cache_dir=cache_dir,
            freeze_now=freeze_now,
        )
    except Exception as exc:  # pragma: no cover - safety guard
        raise click.ClickException(f"Decision run failed: {exc}") from exc

    final_actions = compile_operational_directives(decision_result, settings=settings)

    sessions_payload = [
        {
            "session": session.session,
            "open_date": session.open_date.isoformat(),
            "features": {
                key: _json_safe(value) for key, value in session.features.items()
            },
            "warnings": list(session.warnings),
        }
        for session in decision_result.sessions
    ]
    decisions_payload = [
        {
            "session": record.session,
            "book": record.book,
            "action": record.action,
            "value": _json_safe(record.value),
        }
        for record in decision_result.decisions
    ]
    actions_payload: list[dict[str, object]] = []
    for action in final_actions:
        thresholds_payload = {
            key: _json_safe(value) for key, value in action.thresholds.items()
        }
        rationale_payload: MutableMapping[str, object] = dict(action.rationale)
        rationale_payload.setdefault("kill_flag", action.kill_flag)
        rationale_payload.setdefault("policy_type", action.policy_type)
        rationale_payload.setdefault("score", action.score)
        rationale_payload.setdefault("margin_to_threshold", action.margin_to_threshold)
        rationale_payload.setdefault("reason_text", action.reason_text)
        if thresholds_payload:
            rationale_payload.setdefault("thresholds", thresholds_payload)
        if action.decile is not None:
            rationale_payload.setdefault("decile", action.decile)
        why_payload = _json_safe(rationale_payload)
        action_entry = {
            "session": action.session,
            "book": action.book,
            "rule_action": action.rule_action,
            "final_action": action.final_action,
            "size_gamma": action.size_gamma,
            "score": action.score,
            "margin_to_threshold": action.margin_to_threshold,
            "kill_flag": action.kill_flag,
            "policy_type": action.policy_type,
            "thresholds": thresholds_payload,
            "decile": action.decile,
            "reason_text": action.reason_text,
            "why": why_payload,
            "rationale": why_payload,
        }
        actions_payload.append(action_entry)

    context_fingerprint = {
        "sessions": sessions_payload,
        "decisions": decisions_payload,
        "actions": actions_payload,
    }
    decision_context_id = decision_result.decision_context_id
    if not decision_context_id:
        digest = hashlib.sha256(
            json.dumps(context_fingerprint, sort_keys=True).encode("utf-8")
        ).hexdigest()
        decision_context_id = f"sha256:{digest}"

    payload: dict[str, object] = {
        "kit_dir": str(decision_result.kit_dir),
        "manifest": decision_result.manifest,
        "policy": decision_result.policy,
        "generated_at": decision_result.generated_at.isoformat(),
        "decision_context_id": decision_context_id,
        "sessions": sessions_payload,
        "decisions": decisions_payload,
        "warnings": [dict(warning) for warning in decision_result.warnings],
        "actions": actions_payload,
    }

    if decision_result.frozen_snapshot is not None:
        payload["frozen_snapshot"] = _json_safe(decision_result.frozen_snapshot)

    _persist_decision_run(
        payload,
        actions=final_actions,
        generated_at=decision_result.generated_at,
    )

    if json_output:
        click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        if decision_result.warnings:
            for warning in decision_result.warnings:
                click.echo(
                    f"Warning: {_format_warning_entry(warning)}",
                    err=True,
                )
        table = _format_actions_table(decision_result, final_actions)
        click.echo(table)
        click.echo(f"Decisions generated from kit at {decision_result.kit_dir}")
        click.echo(
            f"Live decision run generated at {decision_result.generated_at.isoformat()}"
        )


@app.command("validate-backtest")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=Path("data/backtest_data/original"),
    show_default=True,
    help="Directory containing the frozen backtest CSV bundle.",
)
@click.option(
    "--manifest",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=Path("tests/fixtures/backtest/manifest.json"),
    show_default=True,
    help="Manifest describing expected SHA-256 digests for the bundle.",
)
def validate_backtest(data_dir: Path, manifest: Path) -> None:
    """Validate that the backtest bundle matches the recorded manifest."""

    from .data.validation import (
        BacktestDataValidationError,
        validate_backtest_directory,
    )

    try:
        validate_backtest_directory(data_dir, manifest)
    except BacktestDataValidationError as exc:  # pragma: no cover - Click handles exit
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Backtest data validated successfully against {manifest}")


def main() -> None:
    """Invoke :func:`app` when the package is executed as a script."""

    app()


__all__ = [
    "app",
    "create_parser",
    "parse_settings",
    "load_settings",
    "main",
]
