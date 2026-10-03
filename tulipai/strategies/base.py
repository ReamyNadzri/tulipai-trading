"""Strategy interface.

A strategy turns candles into a signal frame aligned to the candle index. A signal on
bar ``i`` is computed from data up to and including bar ``i``'s CLOSE and is executed at
the next bar's open - identical in backtest, replay and live trading.

Signal frame columns:
    signal    +1 buy, -1 sell, 0 nothing
    sl_dist   stop-loss distance in price units (> 0 when signal != 0)
    tp_dist   take-profit distance in price units
    strength  0..1 rough conviction, used for ensemble priority and the AI context
    strategy  name of the strategy that produced the signal
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ..indicators import Features

SIGNAL_COLUMNS = ["signal", "sl_dist", "tp_dist", "strength", "strategy"]


class Strategy(ABC):
    name: str = "base"
    default_params: dict[str, Any] = {}
    # Small grids for walk-forward optimisation. Keep them small: every extra knob is
    # another way to overfit.
    param_grid: dict[str, list] = {}

    def __init__(self, **params: Any):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}; valid: {sorted(self.default_params)}")
        self.params = {**self.default_params, **params}

    @property
    def warmup(self) -> int:
        return 250

    @abstractmethod
    def generate(self, f: Features) -> pd.DataFrame: ...

    def _frame(self, f: Features, long: pd.Series, short: pd.Series, sl: pd.Series, tp: pd.Series,
               strength: pd.Series | float = 0.5) -> pd.DataFrame:
        long = long.fillna(False).astype(bool)
        short = short.fillna(False).astype(bool)
        sig = np.where(long & ~short, 1, np.where(short & ~long, -1, 0))
        valid = (sl > 0) & (tp > 0) & np.isfinite(sl) & np.isfinite(tp)
        sig = np.where(valid, sig, 0)
        out = pd.DataFrame(index=f.index)
        out["signal"] = sig.astype(int)
        out["sl_dist"] = np.where(sig != 0, sl, 0.0)
        out["tp_dist"] = np.where(sig != 0, tp, 0.0)
        st = strength if isinstance(strength, pd.Series) else pd.Series(strength, index=f.index)
        out["strength"] = np.where(sig != 0, st.clip(0, 1).fillna(0.5), 0.0)
        out["strategy"] = np.where(sig != 0, self.name, "")
        return out

    def describe(self, verbose: bool = False) -> str:
        if not verbose:
            changed = {k: v for k, v in self.params.items() if self.default_params.get(k) != v}
            return f"{self.name}({', '.join(f'{k}={v}' for k, v in changed.items())})" if changed else self.name
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
