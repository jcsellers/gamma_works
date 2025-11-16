"""Data providers for live decisioning."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, cast
from zoneinfo import ZoneInfo

from xdte.live.calendar import resolve_open_date
from xdte.live.sessions import SESSION_TIMES


def _optional_import(name: str) -> Any:
    """Return an optional dependency if available."""

    try:  # pragma: no cover - import guarded for optional dependencies
        return importlib.import_module(name)
    except ModuleNotFoundError:  # pragma: no cover - handled lazily
        return None


yf = _optional_import("yfinance")
np = _optional_import("numpy")
pd = _optional_import("pandas")


__all__ = [
    "YFinanceDailyContextProvider",
    "YFinanceMarketDataProvider",
    "build_stub_daily_provider",
    "build_stub_market_provider",
]


_EASTERN_TZ = "America/New_York"
_EASTERN_ZONE = ZoneInfo(_EASTERN_TZ)


def _default_clock() -> datetime:
    return datetime.now(tz=_EASTERN_ZONE)


def _require_pandas() -> Any:
    if pd is None or np is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "pandas and numpy must be installed to use yfinance-backed providers"
        )
    return pd


def _require_yfinance() -> Any:
    if yf is None:  # pragma: no cover - runtime guard
        raise RuntimeError("yfinance must be installed to use the live data providers")
    return yf


def _flatten_columns(frame: Any) -> Any:
    pandas_mod = _require_pandas()
    frame = cast(Any, frame)
    multiindex_cls = getattr(pandas_mod, "MultiIndex", None)
    if multiindex_cls is not None and isinstance(frame.columns, multiindex_cls):
        new_columns = [
            "_".join(str(part) for part in column if part)
            for column in frame.columns.to_list()
        ]
        frame = frame.copy()
        frame.columns = new_columns
    return frame


def _ensure_timezone(index: Any) -> Any:
    _require_pandas()
    index = cast(Any, index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(_EASTERN_TZ)


def _cache_paths(cache_dir: str | Path | None, stem: str) -> tuple[Path, Path] | None:
    if cache_dir is None:
        return None
    directory = Path(cache_dir)
    return directory / f"{stem}.parquet", directory / f"{stem}.json"


def _read_cached_frame(
    cache_dir: str | Path | None,
    stem: str,
    warnings: list[str],
) -> tuple[Any, Mapping[str, object]] | None:
    paths = _cache_paths(cache_dir, stem)
    if paths is None:
        return None
    frame_path, metadata_path = paths
    if not frame_path.exists() or not metadata_path.exists():
        return None
    pandas_mod = _require_pandas()
    try:
        frame = pandas_mod.read_parquet(frame_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.append(f"Failed to load {stem} cache: {exc}")
        return None
    try:
        metadata_raw = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_raw) if metadata_raw else {}
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.append(f"Failed to read {stem} cache metadata: {exc}")
        return frame, {}
    if not isinstance(metadata, Mapping):  # pragma: no cover - defensive guard
        warnings.append(f"Unexpected metadata payload in {metadata_path}")
        return frame, {}
    return frame, cast(Mapping[str, object], metadata)


def _write_cached_frame(
    cache_dir: str | Path | None,
    stem: str,
    frame: Any,
    warnings: list[str],
) -> None:
    paths = _cache_paths(cache_dir, stem)
    if paths is None:
        return
    frame_path, metadata_path = paths
    pandas_mod = _require_pandas()
    frame_to_save = cast(Any, frame)
    try:
        directory = frame_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        if getattr(frame_to_save, "empty", True):
            frame_to_write = pandas_mod.DataFrame()
        else:
            frame_to_write = frame_to_save.sort_index()
        frame_to_write.to_parquet(frame_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.append(f"Failed to persist {stem} cache: {exc}")
        return
    latest_index: datetime | None = None
    if not getattr(frame_to_save, "empty", True):
        index = getattr(frame_to_save, "index", None)
        if index is not None and len(index) > 0:
            try:
                pandas_mod = _require_pandas()
                timestamp = pandas_mod.Timestamp(index[-1])
                latest_index = timestamp.to_pydatetime()
            except Exception:  # pragma: no cover - defensive guard
                latest_index = None
    metadata = {
        "refreshed_at": datetime.now(tz=_EASTERN_ZONE).isoformat(),
        "last_index": latest_index.isoformat() if latest_index is not None else None,
    }
    try:
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.append(f"Failed to persist {stem} cache metadata: {exc}")


def _coerce_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    try:
        pandas_mod = _require_pandas()
    except RuntimeError:  # pragma: no cover - pandas guard
        return None
    try:
        timestamp = pandas_mod.Timestamp(value)
    except Exception:  # pragma: no cover - defensive guard
        return None
    result = timestamp.to_pydatetime()
    if isinstance(result, datetime):
        return result
    if isinstance(result, date):
        return datetime(result.year, result.month, result.day)
    return None


def _as_eastern(dt_value: datetime | None) -> datetime | None:
    if dt_value is None:
        return None
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=_EASTERN_ZONE)
    return dt_value.astimezone(_EASTERN_ZONE)


def _merge_frames(existing: Any | None, new_frame: Any | None) -> Any:
    pandas_mod = _require_pandas()
    frames: list[Any] = []
    if existing is not None and not getattr(existing, "empty", True):
        frames.append(existing)
    if new_frame is not None and not getattr(new_frame, "empty", True):
        frames.append(new_frame)
    if not frames:
        return pandas_mod.DataFrame()
    combined = pandas_mod.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


@dataclass
class YFinanceDailyContextProvider:
    """Fetch lagged daily context using yfinance."""

    period: str = "10y"
    tickers: Mapping[str, str] | None = None
    frame: Any | None = None
    cache_dir: str | Path | None = None
    cache_freshness: timedelta | None = None

    def __post_init__(self) -> None:
        self._warnings: list[str] = []
        if self.tickers is None:
            tickers: Mapping[str, str] = {
                "spx": "^GSPC",
                "vix": "^VIX",
                "vvix": "^VVIX",
                "vix3m": "^VIX3M",
            }
        else:
            tickers = self.tickers
        self._tickers: Mapping[str, str] = dict(tickers)
        self._data = (
            self.frame.copy()
            if self.frame is not None
            else self._initialise_daily_frame()
        )
        self._features = self._compute_features(self._data)

    @property
    def warnings(self) -> Iterable[str]:
        return tuple(self._warnings)

    def _initialise_daily_frame(self) -> Any:
        cached = _read_cached_frame(self.cache_dir, "yfinance_daily", self._warnings)
        if cached is None:
            frame = self._download_daily_frame()
            _write_cached_frame(self.cache_dir, "yfinance_daily", frame, self._warnings)
            return frame
        frame, metadata = cached
        metadata_mapping = dict(metadata)
        if getattr(frame, "empty", True):
            return self._refresh_daily_frame(None, metadata_mapping, incremental=False)
        lagging = self._daily_data_lagging(frame, metadata_mapping)
        stale = self._cache_is_stale(metadata_mapping)
        if lagging:
            return self._refresh_daily_frame(frame, metadata_mapping, incremental=True)
        if stale:
            return self._refresh_daily_frame(frame, metadata_mapping, incremental=False)
        return frame

    def _cache_is_stale(self, metadata: Mapping[str, object]) -> bool:
        if self.cache_freshness is None:
            return False
        refreshed_at = _as_eastern(_coerce_datetime(metadata.get("refreshed_at")))
        if refreshed_at is None:
            return True
        now = datetime.now(tz=_EASTERN_ZONE)
        return refreshed_at + self.cache_freshness < now

    def _daily_data_lagging(self, frame: Any, metadata: Mapping[str, object]) -> bool:
        latest = self._latest_index(frame)
        if latest is None:
            latest = _as_eastern(_coerce_datetime(metadata.get("last_index")))
        if latest is None:
            return True
        now = datetime.now(tz=_EASTERN_ZONE)
        return now.date() > latest.date()

    def _latest_index(self, frame: Any) -> datetime | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        index = getattr(frame, "index", None)
        if index is None or len(index) == 0:
            return None
        raw_value = index[-1]
        coerced = _coerce_datetime(raw_value)
        if coerced is None:
            try:
                pandas_mod = _require_pandas()
                timestamp = pandas_mod.Timestamp(raw_value)
            except Exception:
                return None
            coerced = _coerce_datetime(timestamp)
            if coerced is None:
                return None
        if coerced.tzinfo is None:
            return coerced.replace(tzinfo=_EASTERN_ZONE)
        return coerced.astimezone(_EASTERN_ZONE)

    def _refresh_daily_frame(
        self,
        existing: Any | None,
        metadata: Mapping[str, object],
        *,
        incremental: bool,
    ) -> Any:
        start: object | None = None
        if incremental:
            latest = self._latest_index(existing)
            if latest is None:
                latest = _as_eastern(_coerce_datetime(metadata.get("last_index")))
            if latest is not None:
                start_date = (latest + timedelta(days=1)).date()
                start = start_date.isoformat()
        new_frame = self._download_daily_frame(start=start)
        merged = _merge_frames(existing if incremental else None, new_frame)
        _write_cached_frame(self.cache_dir, "yfinance_daily", merged, self._warnings)
        return merged

    def _download_daily_frame(self, start: object | None = None) -> Any:
        pandas_mod = _require_pandas()
        yfinance_mod = _require_yfinance()
        download = cast(Callable[..., Any], getattr(yfinance_mod, "download"))
        download_kwargs: dict[str, object] = {
            "tickers": list(self._tickers.values()),
            "interval": "1d",
            "group_by": "ticker",
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }
        if start is None:
            download_kwargs["period"] = self.period
        else:
            download_kwargs["start"] = start
        try:
            frame = download(**download_kwargs)
        except Exception as exc:  # pragma: no cover - network failure guard
            self._warnings.append(f"yfinance daily download failed: {exc}")
            return pandas_mod.DataFrame()
        dataframe_cls = getattr(pandas_mod, "DataFrame", None)
        if (
            dataframe_cls is not None
            and isinstance(frame, dataframe_cls)
            and not frame.empty
        ):
            return frame
        self._warnings.append("yfinance returned no daily data")
        return pandas_mod.DataFrame()

    def _compute_features(self, frame: Any) -> Mapping[date, Mapping[str, object]]:
        pandas_mod = _require_pandas()
        frame = cast(Any, frame)
        if getattr(frame, "empty", True):
            return {}

        flattened = _flatten_columns(frame)
        flattened = flattened.sort_index()
        index = flattened.index
        datetime_index_cls = getattr(pandas_mod, "DatetimeIndex", None)
        if datetime_index_cls is None or not isinstance(index, datetime_index_cls):
            raise TypeError("Expected DatetimeIndex from yfinance daily data")

        dates = index.normalize()
        spx_close = flattened.get("Close_^GSPC")
        spx_open = flattened.get("Open_^GSPC")
        spx_high = flattened.get("High_^GSPC")
        spx_low = flattened.get("Low_^GSPC")
        vix_close = flattened.get("Close_^VIX")
        vix_open = flattened.get("Open_^VIX")
        vix3m_close = flattened.get("Close_^VIX3M")
        vvix_close = flattened.get("Close_^VVIX")

        daily = pandas_mod.DataFrame(index=dates)
        if spx_close is not None:
            daily["SPX_Close"] = spx_close.to_numpy()
        if spx_open is not None:
            daily["SPX_Open"] = spx_open.to_numpy()
        if spx_high is not None and spx_low is not None:
            daily["SPX_ATR_Pct"] = (
                (spx_high - spx_low).ewm(span=14, adjust=False).mean() / spx_close
                if spx_close is not None
                else math.nan
            )
        if spx_close is not None:
            rolling_max = spx_close.rolling(252, min_periods=20).max()
            daily["SPX_Drawdown_Pct"] = (spx_close - rolling_max) / rolling_max
        if vix_close is not None and vix3m_close is not None:
            # Division may produce NaN or Inf when vix3m_close is zero; these are treated as missing values in downstream feature engineering.
            with np.errstate(divide="ignore", invalid="ignore"):
                daily["TS_eff"] = vix_close / vix3m_close
        if vix_close is not None:
            daily["VIX_Close"] = vix_close
            daily["vix_pct"] = vix_close.rank(pct=True)
            daily["VIX_pct"] = daily["vix_pct"]
        if vvix_close is not None:
            daily["VVIX_Close"] = vvix_close
            daily["vvix_pct"] = vvix_close.rank(pct=True)
            daily["vvix_ema20"] = vvix_close.ewm(span=20, adjust=False).mean()
            daily["vvix_ema30"] = vvix_close.ewm(span=30, adjust=False).mean()
        if spx_close is not None:
            returns = spx_close.pct_change().fillna(0.0)
            daily["rv5"] = returns.rolling(5).std() * math.sqrt(252)
            daily["rv20"] = returns.rolling(20).std() * math.sqrt(252)
        if vix_open is not None:
            daily["VIX_Open"] = vix_open
        if spx_close is not None and spx_open is not None:
            daily["gap"] = spx_open - spx_close.shift(1)
            daily["movement"] = spx_close - spx_open

        daily["DoW"] = daily.index.dayofweek

        lag_map = {
            "TS_eff": "L1_TS",
            "VIX_Close": "L1_VIX_Close",
            "VIX_pct": "L1_VIX_pct",
            "VVIX_Close": "L1_vvix_close",
            "vvix_pct": "L1_vvix_pct",
            "vvix_ema20": "L1_vvix_ema20",
            "vvix_ema30": "L1_vvix_ema30",
            "SPX_ATR_Pct": "L1_SPX_ATR_Pct",
            "SPX_Drawdown_Pct": "L1_SPX_Drawdown_Pct",
            "rv5": "L1_rv5",
            "rv20": "L1_rv20",
        }

        features: dict[date, MutableMapping[str, object]] = {}
        for current_date, row in daily.iterrows():
            dow_value = row.get("DoW", math.nan)
            payload: MutableMapping[str, object] = {
                "DoW": (
                    int(dow_value)
                    if not pd.isna(dow_value)
                    else int(current_date.weekday())
                ),
            }
            for src, dest in lag_map.items():
                if src not in daily:
                    payload[dest] = None
                    continue
                value = daily[src].shift(1).get(current_date, math.nan)
                payload[dest] = float(value) if not math.isnan(value) else None
            close_value = payload.get("L1_vvix_close")
            ema20 = payload.get("L1_vvix_ema20")
            ema30 = payload.get("L1_vvix_ema30")
            if isinstance(close_value, (int, float)) and isinstance(
                ema20, (int, float)
            ):
                payload["L1_vvix_above_ema20"] = bool(close_value > ema20)
            else:
                payload["L1_vvix_above_ema20"] = False
            if isinstance(close_value, (int, float)) and isinstance(
                ema30, (int, float)
            ):
                payload["L1_vvix_above_ema30"] = bool(close_value > ema30)
            else:
                payload["L1_vvix_above_ema30"] = False

            # Non-lagged values used by the models
            payload["gap"] = (
                float(row.get("gap", math.nan))
                if not math.isnan(row.get("gap", math.nan))
                else None
            )
            payload["movement"] = (
                float(row.get("movement", math.nan))
                if not math.isnan(row.get("movement", math.nan))
                else None
            )
            payload["closing_vix"] = (
                float(row.get("VIX_Close", math.nan))
                if not math.isnan(row.get("VIX_Close", math.nan))
                else None
            )
            payload["opening_vix"] = (
                float(row.get("VIX_Open", math.nan))
                if not math.isnan(row.get("VIX_Open", math.nan))
                else None
            )
            features[current_date.date()] = payload

        return features

    def __call__(self, open_date: date) -> Mapping[str, object]:
        return self._features.get(open_date, {})

    def to_frame(self) -> Any:
        """Return a copy of the raw yfinance frame backing the provider."""

        if self._data is None:
            pandas_mod = _require_pandas()
            return pandas_mod.DataFrame()
        return self._data.copy()


@dataclass
class YFinanceMarketDataProvider:
    """Produce per-session market snapshots backed by yfinance."""

    period: str = "5d"
    interval: str = "5m"
    index_ticker: str = "^GSPC"
    vix_ticker: str = "^VIX"
    frame: Any | None = None
    daily_context: YFinanceDailyContextProvider | None = None
    clock: Callable[[], datetime] = _default_clock
    cache_dir: str | Path | None = None
    cache_freshness: timedelta | None = None

    def __post_init__(self) -> None:
        self._warnings: list[str] = []
        self._raw_frame = (
            self.frame.copy()
            if self.frame is not None
            else self._initialise_intraday_frame()
        )
        self._session_payloads = self._prepare_session_payloads(self._raw_frame)

    @property
    def warnings(self) -> Iterable[str]:
        return tuple(self._warnings)

    def _current_time(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=_EASTERN_ZONE)
        return now.astimezone(_EASTERN_ZONE)

    def _initialise_intraday_frame(self) -> Any:
        cached = _read_cached_frame(self.cache_dir, "yfinance_intraday", self._warnings)
        if cached is None:
            frame = self._download_intraday_frame()
            _write_cached_frame(
                self.cache_dir, "yfinance_intraday", frame, self._warnings
            )
            return frame
        frame, metadata = cached
        metadata_mapping = dict(metadata)
        if getattr(frame, "empty", True):
            return self._refresh_intraday_frame(
                None, metadata_mapping, incremental=False
            )
        lagging = self._intraday_data_lagging(frame, metadata_mapping)
        stale = self._cache_is_stale(metadata_mapping)
        if lagging:
            return self._refresh_intraday_frame(
                frame, metadata_mapping, incremental=True
            )
        if stale:
            return self._refresh_intraday_frame(
                frame, metadata_mapping, incremental=False
            )
        return frame

    def _cache_is_stale(self, metadata: Mapping[str, object]) -> bool:
        if self.cache_freshness is None:
            return False
        refreshed_at = _as_eastern(_coerce_datetime(metadata.get("refreshed_at")))
        if refreshed_at is None:
            return True
        now = datetime.now(tz=_EASTERN_ZONE)
        return refreshed_at + self.cache_freshness < now

    def _intraday_data_lagging(
        self, frame: Any, metadata: Mapping[str, object]
    ) -> bool:
        latest = self._latest_index(frame)
        if latest is None:
            latest = _as_eastern(_coerce_datetime(metadata.get("last_index")))
        if latest is None:
            return True
        interval_delta = self._interval_delta()
        now = self._current_time()
        if interval_delta is None:
            return now.date() > latest.date()
        return latest + interval_delta < now

    def _latest_index(self, frame: Any) -> datetime | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        index = getattr(frame, "index", None)
        if index is None or len(index) == 0:
            return None
        raw_value = index[-1]
        coerced = _coerce_datetime(raw_value)
        if coerced is None:
            try:
                pandas_mod = _require_pandas()
                timestamp = pandas_mod.Timestamp(raw_value)
            except Exception:
                return None
            coerced = _coerce_datetime(timestamp)
            if coerced is None:
                return None
        if coerced.tzinfo is None:
            return coerced.replace(tzinfo=_EASTERN_ZONE)
        return coerced.astimezone(_EASTERN_ZONE)

    def _interval_delta(self) -> timedelta | None:
        try:
            pandas_mod = _require_pandas()
        except RuntimeError:  # pragma: no cover - pandas guard
            return None
        try:
            delta = pandas_mod.to_timedelta(self.interval)
        except Exception:
            return None
        try:
            result = delta.to_pytimedelta()
        except AttributeError:  # pragma: no cover - defensive guard
            return None
        return cast(timedelta, result)

    def _refresh_intraday_frame(
        self,
        existing: Any | None,
        metadata: Mapping[str, object],
        *,
        incremental: bool,
    ) -> Any:
        start: object | None = None
        if incremental:
            latest = self._latest_index(existing)
            if latest is None:
                latest = _as_eastern(_coerce_datetime(metadata.get("last_index")))
            interval_delta = self._interval_delta()
            if latest is not None:
                if interval_delta is not None:
                    start_dt = latest + interval_delta
                else:
                    start_dt = latest
                start = start_dt.isoformat()
        new_frame = self._download_intraday_frame(start=start)
        merged = _merge_frames(existing if incremental else None, new_frame)
        _write_cached_frame(self.cache_dir, "yfinance_intraday", merged, self._warnings)
        return merged

    def _download_intraday_frame(self, start: object | None = None) -> Any:
        pandas_mod = _require_pandas()
        yfinance_mod = _require_yfinance()
        download = cast(Callable[..., Any], getattr(yfinance_mod, "download"))
        download_kwargs: dict[str, object] = {
            "tickers": [self.index_ticker, self.vix_ticker],
            "interval": self.interval,
            "group_by": "ticker",
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }
        if start is None:
            download_kwargs["period"] = self.period
        else:
            download_kwargs["start"] = start
        try:
            frame = download(**download_kwargs)
        except Exception as exc:  # pragma: no cover - network failure guard
            self._warnings.append(f"yfinance intraday download failed: {exc}")
            return pandas_mod.DataFrame()
        dataframe_cls = getattr(pandas_mod, "DataFrame", None)
        if (
            dataframe_cls is not None
            and isinstance(frame, dataframe_cls)
            and not frame.empty
        ):
            return frame
        self._warnings.append("yfinance returned no intraday data")
        return pandas_mod.DataFrame()

    def _prepare_session_payloads(
        self, frame: Any
    ) -> Mapping[str, Mapping[str, object]]:
        pandas_mod = _require_pandas()
        frame = cast(Any, frame)
        if getattr(frame, "empty", True):
            return {}

        flattened = _flatten_columns(frame)
        flattened = flattened.sort_index()
        index = flattened.index
        datetime_index_cls = getattr(pandas_mod, "DatetimeIndex", None)
        if datetime_index_cls is None or not isinstance(index, datetime_index_cls):
            raise TypeError("Expected DatetimeIndex from yfinance intraday data")
        eastern_index = _ensure_timezone(index)
        flattened = flattened.copy()
        flattened.index = eastern_index

        spx_close = flattened.get(f"Close_{self.index_ticker}")
        spx_open = flattened.get(f"Open_{self.index_ticker}")
        vix_close = flattened.get(f"Close_{self.vix_ticker}")
        vix_open = flattened.get(f"Open_{self.vix_ticker}")
        if spx_close is None:
            self._warnings.append(
                "Missing SPX close series from yfinance intraday data"
            )
            return {}
        if vix_close is None:
            self._warnings.append(
                "Missing VIX close series from yfinance intraday data"
            )
            return {}

        latest_timestamp = spx_close.index[-1]
        latest_date = latest_timestamp.date()
        payloads: dict[str, MutableMapping[str, object]] = {}
        current_time = self._current_time()
        calendar_warning_emitted = False
        missing_intraday_warning_emitted = False
        for session, session_time in SESSION_TIMES.items():
            session_reference = current_time.replace(
                hour=session_time.hour,
                minute=session_time.minute,
                second=session_time.second,
                microsecond=0,
            )
            open_date = resolve_open_date(session_reference)
            if not calendar_warning_emitted and open_date != session_reference.date():
                self._warnings.append(
                    "Trading calendar resolved open_date "
                    f"{open_date.isoformat()} for wall-clock date "
                    f"{session_reference.date().isoformat()}"
                )
                calendar_warning_emitted = True

            day_mask = spx_close.index.date == open_date
            day_slice = spx_close[day_mask]
            if day_slice.empty:
                if not missing_intraday_warning_emitted:
                    self._warnings.append(
                        "Missing intraday data for resolved open_date "
                        f"{open_date.isoformat()}; using latest available data "
                        f"from {latest_date.isoformat()}"
                    )
                    missing_intraday_warning_emitted = True
                day_slice = spx_close
            session_dt = datetime.combine(
                open_date, session_time, tzinfo=day_slice.index.tz
            )
            upto_session = day_slice.loc[day_slice.index <= session_dt]
            if upto_session.empty:
                upto_session = day_slice
            spx_last = float(upto_session.iloc[-1])
            if spx_open is not None:
                day_open_slice = spx_open[day_mask]
                spx_open_value = (
                    float(day_open_slice.iloc[0])
                    if not day_open_slice.empty
                    else float(spx_open.iloc[-1])
                )
            else:
                spx_open_value = float(day_slice.iloc[0])
            intraday_move = spx_last - spx_open_value

            vix_day_slice = vix_close[vix_close.index.date == open_date]
            if vix_day_slice.empty:
                vix_day_slice = vix_close
            upto_vix = vix_day_slice.loc[vix_day_slice.index <= session_dt]
            if upto_vix.empty:
                upto_vix = vix_day_slice
            vix_last = float(upto_vix.iloc[-1])
            if vix_open is not None:
                vix_open_slice = vix_open[vix_open.index.date == open_date]
                opening_vix = (
                    float(vix_open_slice.iloc[0])
                    if not vix_open_slice.empty
                    else float(vix_open.iloc[-1])
                )
            else:
                opening_vix = vix_last

            payload: MutableMapping[str, object] = {
                "open_date": open_date,
                "spx_open": spx_open_value,
                "spx_last": spx_last,
                "intraday_move": intraday_move,
                "vix": vix_last,
                "opening_vix": opening_vix,
            }

            if self.daily_context is not None:
                daily_payload = self.daily_context(open_date)
                for key in ("gap", "movement", "closing_vix"):
                    value = daily_payload.get(key)
                    if value is not None:
                        payload[key] = value
            payload.setdefault("gap", None)
            movement_value = payload.get("movement", intraday_move)
            if isinstance(movement_value, (int, float)):
                payload["movement"] = float(movement_value)
            else:
                payload["movement"] = intraday_move
            payload.setdefault("closing_vix", payload["vix"])
            payloads[session] = payload

        return payloads

    def __call__(self, session: str) -> Mapping[str, object]:
        payload = self._session_payloads.get(session)
        if payload is None:
            raise RuntimeError(f"No market snapshot available for session {session}")
        return payload


def build_stub_market_provider(
    *,
    open_date: date,
    eleven_payload: Mapping[str, object],
    fifteen_payload: Mapping[str, object],
) -> YFinanceMarketDataProvider:
    class _StubProvider(YFinanceMarketDataProvider):
        def __init__(self) -> None:
            self._warnings: list[str] = []
            self._session_payloads = {
                "11:00": {"open_date": open_date, **eleven_payload},
                "15:15": {"open_date": open_date, **fifteen_payload},
            }

        @property
        def warnings(self) -> Iterable[str]:
            return ()

        def __call__(self, session: str) -> Mapping[str, object]:
            if session not in self._session_payloads:
                raise RuntimeError(f"Unknown session: {session}")
            return self._session_payloads[session]

    return _StubProvider()


def build_stub_daily_provider(
    *,
    open_date: date,
    payload: Mapping[str, object],
) -> YFinanceDailyContextProvider:
    class _StubDailyProvider(YFinanceDailyContextProvider):
        def __init__(self) -> None:
            self._features = {open_date: dict(payload)}
            self._warnings: list[str] = []

        @property
        def warnings(self) -> Iterable[str]:
            return ()

        def __call__(self, target_date: date) -> Mapping[str, object]:
            return self._features.get(target_date, {})

    return _StubDailyProvider()
