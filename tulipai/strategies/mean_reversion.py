"""Range-regime mean reversion.

When ADX says there is no trend, fade stretches outside the Bollinger band once a candle
closes back inside it, targeting the middle band.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import Features
from .base import Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"
    default_params = {
        "bb_n": 20,
        "bb_k": 2.2,
        "adx_max": 20.0,
        "rsi_low": 32.0,
        "rsi_high": 68.0,
        "sl_atr": 1.2,
        "min_rr": 1.0,
        "max_tp_atr": 3.0,
        "atr_n": 14,
    }
    param_grid = {
        "bb_k": [2.0, 2.2, 2.5],
        "adx_max": [18.0, 22.0],
        "sl_atr": [1.0, 1.2, 1.5],
    }

    @property
    def warmup(self) -> int:
        return 100

    def generate(self, f: Features) -> pd.DataFrame:
        p = self.params
        o, h, l, c = f.open, f.high, f.low, f.close
        mid, upper, lower = f.bollinger(p["bb_n"], p["bb_k"])
        a = f.atr(p["atr_n"])
        adx, _, _ = f.adx(14)
        r = f.rsi(14)
        ranging = adx < p["adx_max"]

        long_sig = ranging & (l < lower) & (c > lower) & (c > o) & (r.shift(1) < p["rsi_low"])
        short_sig = ranging & (h > upper) & (c < upper) & (c < o) & (r.shift(1) > p["rsi_high"])
        sl = p["sl_atr"] * a
        tp_long = (mid - c).clip(upper=p["max_tp_atr"] * a)
        tp_short = (c - mid).clip(upper=p["max_tp_atr"] * a)
        tp = tp_long.where(long_sig, tp_short)
        enough_room = tp >= p["min_rr"] * sl
        strength = ((p["adx_max"] - adx) / p["adx_max"]).clip(0, 1)
        return self._frame(f, long_sig & enough_room, short_sig & enough_room, sl, tp, strength)
