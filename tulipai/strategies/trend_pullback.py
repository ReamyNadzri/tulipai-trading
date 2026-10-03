"""Trend-following pullback entries.

Only trade in the direction of an established trend (EMA50 vs EMA200, rising/falling
EMA200, ADX above a floor, H4 trend agreeing), and only after price pulls back into the
fast EMA and a candle closes back in the trend direction.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import Features
from .base import Strategy


class TrendPullback(Strategy):
    name = "trend_pullback"
    default_params = {
        "ema_fast": 20,
        "ema_mid": 50,
        "ema_slow": 200,
        "slope_bars": 10,
        "adx_min": 20.0,
        "touch_atr": 0.25,
        "rsi_low": 40.0,
        "rsi_high": 70.0,
        "sl_atr": 1.5,
        "rr": 2.0,
        "htf_filter": True,
        "atr_n": 14,
    }
    param_grid = {
        "adx_min": [18.0, 22.0, 26.0],
        "sl_atr": [1.2, 1.5, 2.0],
        "rr": [1.5, 2.0, 2.5],
    }

    @property
    def warmup(self) -> int:
        return int(self.params["ema_slow"]) + 50

    def generate(self, f: Features) -> pd.DataFrame:
        p = self.params
        o, h, l, c = f.open, f.high, f.low, f.close
        ef, em, es = f.ema(p["ema_fast"]), f.ema(p["ema_mid"]), f.ema(p["ema_slow"])
        a = f.atr(p["atr_n"])
        adx, _, _ = f.adx(14)
        r = f.rsi(14)
        slope = es - es.shift(p["slope_bars"])

        up = (em > es) & (c > es) & (slope > 0) & (adx >= p["adx_min"])
        down = (em < es) & (c < es) & (slope < 0) & (adx >= p["adx_min"])
        if p["htf_filter"]:
            trend = f.htf_ema_slope("H4", 50)
            up &= trend > 0
            down &= trend < 0

        long_sig = up & (l <= ef + p["touch_atr"] * a) & (c > ef) & (c > o) & r.between(p["rsi_low"], p["rsi_high"])
        short_sig = (
            down & (h >= ef - p["touch_atr"] * a) & (c < ef) & (c < o)
            & r.between(100 - p["rsi_high"], 100 - p["rsi_low"])
        )
        sl = p["sl_atr"] * a
        tp = p["rr"] * sl
        strength = (adx / 40.0).clip(0, 1)
        return self._frame(f, long_sig, short_sig, sl, tp, strength)
