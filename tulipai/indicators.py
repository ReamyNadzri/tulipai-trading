"""Technical indicators (pure pandas/numpy, all strictly causal).

``Features`` wraps a candle frame and memoises indicator series so strategies and the
walk-forward optimiser can request the same indicator many times for free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import tf_minutes


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = rma(delta.clip(lower=0.0), n)
    loss = rma((-delta).clip(lower=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(loss != 0.0, 100.0)
    return out.where(gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr_s = rma(true_range(high, low, close), n)
    plus_di = 100.0 * rma(plus_dm, n) / tr_s
    minus_di = 100.0 * rma(minus_dm, n) / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return rma(dx.fillna(0.0), n).where(plus_di.notna()), plus_di, minus_di


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def donchian(high: pd.Series, low: pd.Series, n: int):
    return high.rolling(n, min_periods=n).max(), low.rolling(n, min_periods=n).min()


class Features:
    """Memoised indicator access on one candle frame."""

    def __init__(self, df: pd.DataFrame, timeframe: str = "M15"):
        self.df = df
        self.timeframe = timeframe
        self.bar = pd.Timedelta(minutes=tf_minutes(timeframe))
        self._cache: dict = {}

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.df.index

    @property
    def open(self) -> pd.Series:
        return self.df["open"]

    @property
    def high(self) -> pd.Series:
        return self.df["high"]

    @property
    def low(self) -> pd.Series:
        return self.df["low"]

    @property
    def close(self) -> pd.Series:
        return self.df["close"]

    def ema(self, n: int) -> pd.Series:
        return self._memo(("ema", n), lambda: ema(self.close, n))

    def sma(self, n: int) -> pd.Series:
        return self._memo(("sma", n), lambda: sma(self.close, n))

    def rsi(self, n: int = 14) -> pd.Series:
        return self._memo(("rsi", n), lambda: rsi(self.close, n))

    def atr(self, n: int = 14) -> pd.Series:
        return self._memo(("atr", n), lambda: atr(self.high, self.low, self.close, n))

    def adx(self, n: int = 14):
        return self._memo(("adx", n), lambda: adx(self.high, self.low, self.close, n))

    def bollinger(self, n: int = 20, k: float = 2.0):
        return self._memo(("bb", n, k), lambda: bollinger(self.close, n, k))

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        return self._memo(("macd", fast, slow, signal), lambda: macd(self.close, fast, slow, signal))

    def hour(self) -> np.ndarray:
        return self._memo("hour", lambda: np.asarray(self.index.hour))

    def day(self) -> pd.DatetimeIndex:
        return self._memo("day", lambda: self.index.normalize())

    def local_hour(self, tz: str) -> np.ndarray:
        """Hour of each bar's open in a market time zone (DST-aware, e.g. Europe/London)."""
        return self._memo(("local_hour", tz), lambda: np.asarray(self.index.tz_convert(tz).hour))

    def local_day(self, tz: str) -> pd.Index:
        """Calendar date of each bar in a market time zone, for per-session grouping."""
        return self._memo(("local_day", tz), lambda: pd.Index(self.index.tz_convert(tz).date))

    def htf_ema_slope(self, tf: str = "H4", n: int = 50) -> pd.Series:
        """+1/-1/0 trend of an EMA on a higher timeframe, using ONLY completed HTF bars.

        Each base bar sees the HTF EMA as of the last HTF bar that had fully closed by the
        time the base bar closed - the same information a live trader has.
        """

        def build():
            rule = f"{tf_minutes(tf)}min"
            htf_close = self.close.resample(rule, label="left", closed="left").last().dropna()
            e = ema(htf_close, n)
            trend = np.sign(htf_close - e) * (e.diff() * (htf_close - e) > 0)
            trend.index = trend.index + pd.Timedelta(minutes=tf_minutes(tf))  # available at HTF close
            base_close_times = self.index + self.bar
            aligned = trend.reindex(trend.index.union(base_close_times)).ffill().reindex(base_close_times)
            return pd.Series(aligned.to_numpy(), index=self.index).fillna(0.0)

        return self._memo(("htf", tf, n), build)
