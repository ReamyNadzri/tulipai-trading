"""Synthetic gold-like candles for tests, demos and engine sanity checks.

These are NOT real prices. They exist so the whole pipeline can be exercised without a
data feed, and so the backtester can be checked for look-ahead bias: on a driftless
random walk (``random_walk=True``) no strategy should make money after costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import tf_minutes

# Relative volatility by UTC hour: quiet Asia, London open pickup, NY overlap peak.
_HOURLY_VOL = np.array(
    [0.55, 0.5, 0.5, 0.5, 0.5, 0.55, 0.6, 0.85, 1.1, 1.1, 1.0, 0.95,
     1.2, 1.5, 1.6, 1.45, 1.25, 1.0, 0.85, 0.75, 0.7, 0.5, 0.45, 0.5]
)


def synthetic_gold(
    start: str = "2024-01-01",
    days: int = 365,
    timeframe: str = "M15",
    seed: int = 7,
    start_price: float = 2050.0,
    annual_vol: float = 0.16,
    random_walk: bool = False,
    spread: float | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = tf_minutes(timeframe)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=int(days * 1440 / minutes), freq=f"{minutes}min")
    # Market open Sun 22:00 -> Fri 21:00 UTC, with a 21:00-22:00 daily break.
    wd, hr = idx.weekday, idx.hour
    open_mask = ~((wd == 5) | ((wd == 6) & (hr < 22)) | ((wd == 4) & (hr >= 21)) | (hr == 21))
    idx = idx[open_mask]
    n = len(idx)

    bars_per_year = 252 * 1380 / minutes
    sigma = annual_vol / np.sqrt(bars_per_year)
    vol_mult = _HOURLY_VOL[idx.hour] / _HOURLY_VOL.mean()
    if random_walk:
        drift = np.zeros(n)
        shocks = rng.standard_normal(n)
    else:
        # Persistent regimes (trend up / trend down / range) plus vol clustering and fat tails.
        regime = np.zeros(n, dtype=int)
        state = 0
        for i in range(1, n):
            if rng.random() < 0.004:
                state = rng.choice([-1, 0, 1])
            regime[i] = state
        drift = regime * sigma * 0.08
        shocks = rng.standard_t(df=4, size=n) / np.sqrt(2.0)
        garch = np.ones(n)
        for i in range(1, n):
            garch[i] = 0.94 * garch[i - 1] + 0.06 * min(shocks[i - 1] ** 2, 9.0)
        vol_mult = vol_mult * np.sqrt(garch)

    rets = drift + sigma * vol_mult * shocks
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    bar_sigma = sigma * vol_mult * close
    wick_up = np.abs(rng.standard_normal(n)) * bar_sigma * 0.6
    wick_dn = np.abs(rng.standard_normal(n)) * bar_sigma * 0.6
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    volume = np.round(500 * vol_mult * (1 + rng.random(n)))

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df.index.name = "time"
    if spread is not None:
        df["spread"] = float(spread)
    return df.round({"open": 3, "high": 3, "low": 3, "close": 3})
