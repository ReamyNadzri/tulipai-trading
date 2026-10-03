"""Asian-range breakout at the London open.

Gold ranges during the Asian session and frequently picks a direction when London
liquidity arrives. We mark the Asian high/low, then take the FIRST close beyond it (plus a
buffer) during the London morning, in the direction allowed by the H4 trend.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import Features
from .base import Strategy


class SessionBreakout(Strategy):
    name = "session_breakout"
    default_params = {
        "range_start_hour": 0,
        "range_end_hour": 7,
        "trade_end_hour": 13,
        "buffer_atr": 0.1,
        "min_width_atr": 2.0,
        "max_width_atr": 12.0,
        "sl_range_frac": 0.5,
        "min_sl_atr": 1.0,
        "max_sl_atr": 3.5,
        "rr": 1.5,
        "htf_filter": True,
        "atr_n": 14,
    }
    param_grid = {
        "range_end_hour": [6, 7, 8],
        "buffer_atr": [0.0, 0.1, 0.25],
        "rr": [1.2, 1.5, 2.0],
    }

    @property
    def warmup(self) -> int:
        return 120

    def generate(self, f: Features) -> pd.DataFrame:
        p = self.params
        h, l, c = f.high, f.low, f.close
        a = f.atr(p["atr_n"])
        hour = f.hour()
        day = f.day()

        in_range = (hour >= p["range_start_hour"]) & (hour < p["range_end_hour"])
        rng_hi = h.where(in_range).groupby(day).transform("max")
        rng_lo = l.where(in_range).groupby(day).transform("min")
        width = rng_hi - rng_lo
        window = pd.Series((hour >= p["range_end_hour"]) & (hour < p["trade_end_hour"]), index=f.index)
        valid = window & (width >= p["min_width_atr"] * a) & (width <= p["max_width_atr"] * a)
        buf = p["buffer_atr"] * a

        long_raw = valid & (c > rng_hi + buf)
        short_raw = valid & (c < rng_lo - buf)
        if p["htf_filter"]:
            trend = f.htf_ema_slope("H4", 50)
            long_raw &= trend >= 0
            short_raw &= trend <= 0
        # Only the first breakout of the day in each direction.
        long_sig = long_raw & (long_raw.astype(int).groupby(day).cumsum() == 1)
        short_sig = short_raw & (short_raw.astype(int).groupby(day).cumsum() == 1)

        sl = (p["sl_range_frac"] * width).clip(lower=p["min_sl_atr"] * a, upper=p["max_sl_atr"] * a)
        tp = p["rr"] * sl
        beyond = (c - rng_hi).where(long_sig, rng_lo - c)
        strength = 0.5 + (beyond / a).clip(0, 2) / 4
        return self._frame(f, long_sig, short_sig, sl, tp, strength)
