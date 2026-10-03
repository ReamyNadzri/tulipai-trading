"""Candle data loading/saving.

Canonical format everywhere in TulipAI: a DataFrame indexed by the bar OPEN time as a
tz-aware UTC DatetimeIndex, with float columns open/high/low/close/volume and an
optional ``spread`` column in price units. Prices are BID prices (what MT5 charts show).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import tf_minutes

OHLC = ["open", "high", "low", "close"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().strip("<>").lower() for c in out.columns]
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"tick_volume": "volume"})
    if "tickvol" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"tickvol": "volume"})
    missing = [c for c in OHLC if c not in out.columns]
    if missing:
        raise ValueError(f"Candle data is missing columns {missing}; got {list(out.columns)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Candle data must be indexed by a DatetimeIndex")
    idx = out.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    out.index.name = "time"
    keep = OHLC + ["volume"] + (["spread"] if "spread" in out.columns else [])
    out = out[keep].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=OHLC)
    # Repair rows where high/low do not envelope open/close (bad ticks in some exports).
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


def _detect_sep(first_line: str) -> str:
    for sep in ("\t", ";", ","):
        if sep in first_line:
            return sep
    return ","


def load_csv(path: str | Path, utc_offset_hours: float = 0.0) -> pd.DataFrame:
    """Load candles from CSV.

    Understands TulipAI's own format, MetaTrader 5 "Export bars" files
    (<DATE> <TIME> <OPEN> ...), TradingView exports (unix ``time``) and generic files
    with a datetime/timestamp column. ``utc_offset_hours`` converts broker server time
    (e.g. MT5 exports, usually UTC+2/+3) to UTC.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as fh:
        first = fh.readline()
    raw = pd.read_csv(path, sep=_detect_sep(first))
    raw.columns = [str(c).strip().strip("<>").lower() for c in raw.columns]

    if "date" in raw.columns and "time" in raw.columns:
        stamp = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str), format="mixed")
    else:
        col = next((c for c in ("time", "datetime", "timestamp", "date", "gmt time", "local time") if c in raw.columns), None)
        if col is None:
            raise ValueError(f"No time column found in {path}; columns: {list(raw.columns)}")
        series = raw[col]
        if pd.api.types.is_numeric_dtype(series):
            unit = "ms" if series.abs().max() > 1e11 else "s"
            stamp = pd.to_datetime(series, unit=unit, utc=True)
        else:
            stamp = pd.to_datetime(series, utc=True, format="mixed", dayfirst=False)
    idx = pd.DatetimeIndex(stamp)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    if utc_offset_hours:
        idx = idx - pd.Timedelta(hours=float(utc_offset_hours))
    raw.index = idx
    drop = [c for c in ("date", "time", "datetime", "timestamp", "gmt time", "local time") if c in raw.columns]
    raw = raw.drop(columns=drop)
    if "spread" in raw.columns and ("tickvol" in raw.columns or raw["spread"].abs().max() > 50):
        # MT5 "Export bars" files give spread in points; without the symbol's point size it
        # cannot be converted safely, so fall back to the configured spread.
        raw = raw.drop(columns=["spread"])
    return normalize(raw)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "time"
    out.to_csv(path)
    return path


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample candles to a higher timeframe (bars labelled by their open time)."""
    rule = f"{tf_minutes(tf)}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "spread" in df.columns:
        agg["spread"] = "mean"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


def utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def slice_time(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= utc_ts(start)]
    if end is not None:
        df = df[df.index < utc_ts(end)]
    return df
