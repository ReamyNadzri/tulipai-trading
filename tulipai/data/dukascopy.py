"""Free historical XAUUSD spot candles from Dukascopy's public datafeed.

Each trading day is one LZMA-compressed ``.bi5`` file of 1-minute BID candles. Each record
is 24 bytes big-endian: seconds-from-midnight (int32), open, close, low, high (int32,
price x ``price_scale``) and volume (float32). Files are cached on disk, so re-running a
download only fetches missing days.

Use this on your own PC when you want years of history without MT5. MT5's own history
(``tulip fetch --source mt5``) is better for final tests because it matches your broker.
"""

from __future__ import annotations

import lzma
import struct
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .io import normalize, resample

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">iiiiif")


def parse_bi5_candles(raw: bytes, day: date, price_scale: float = 1000.0) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    data = lzma.decompress(raw)
    n = len(data) // RECORD.size
    arr = np.frombuffer(data[: n * RECORD.size], dtype=np.dtype(
        [("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]
    ))
    o, c, lo, hi = (arr[k].astype(float) / price_scale for k in ("o", "c", "l", "h"))
    # Guard against a field-order surprise: the high must envelope open and close.
    if np.mean(hi >= np.maximum(o, c)) < 0.99 and np.mean(c >= np.maximum(o, hi)) > 0.99:
        hi, c = c, hi
    start = pd.Timestamp(day, tz="UTC")
    idx = start + pd.to_timedelta(arr["t"].astype(np.int64), unit="s")
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": arr["v"].astype(float)}, index=idx)
    return df[df["volume"] > 0]


def _fetch(url: str, session: requests.Session, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return b""
            resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    return b""


def download(
    start: str,
    end: str,
    timeframe: str = "M15",
    instrument: str = "XAUUSD",
    cache_dir: str | Path = "data/cache/dukascopy",
    price_scale: float = 1000.0,
    progress: bool = True,
) -> pd.DataFrame:
    cache = Path(cache_dir) / instrument
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    d0, d1 = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    frames = []
    day = d0
    total = (d1 - d0).days or 1
    while day < d1:
        if day.weekday() != 5:  # no Saturday data
            fname = cache / f"{day.isoformat()}.bi5"
            if fname.exists():
                raw = fname.read_bytes()
            else:
                url = f"{BASE_URL}/{instrument}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
                raw = _fetch(url, session)
                fname.write_bytes(raw)
            frames.append(parse_bi5_candles(raw, day, price_scale))
        day += timedelta(days=1)
        if progress and day.day == 1:
            print(f"  dukascopy: {day.isoformat()} ({(day - d0).days / total:.0%})", flush=True)
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError("Dukascopy returned no data for that range")
    m1 = normalize(pd.concat(frames))
    median = float(m1["close"].median())
    if not 100 < median < 50000:
        print(f"WARNING: median price {median:.2f} looks wrong; try a different price_scale")
    return m1 if timeframe.upper() == "M1" else resample(m1, timeframe)
