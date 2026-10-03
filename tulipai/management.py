"""Open-position management rules, evaluated once per closed bar.

Shared by backtest and live so both manage trades identically: weekend flattening, a time
stop, moving the stop to break-even (+ a small lock) and an ATR trailing stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ManagementConfig


@dataclass
class ManageAction:
    kind: str = "none"  # none | modify | close
    new_sl: float | None = None
    reason: str = ""


def manage_position(
    side: int,
    entry: float,
    sl: float,
    initial_sl_dist: float,
    bars_held: int,
    price: float,
    atr: float,
    bar_close_time: pd.Timestamp,
    cfg: ManagementConfig,
) -> ManageAction:
    """``price`` is the exit-side price at the bar close (bid for longs, ask for shorts)."""
    if cfg.close_before_weekend and bar_close_time.weekday() == 4 and bar_close_time.hour >= cfg.weekend_close_hour_utc:
        return ManageAction("close", reason="weekend")
    if cfg.max_bars_in_trade and bars_held >= cfg.max_bars_in_trade:
        return ManageAction("close", reason="time")
    if initial_sl_dist <= 0:
        return ManageAction()

    r_now = side * (price - entry) / initial_sl_dist
    best = sl
    if cfg.breakeven_at_r and r_now >= cfg.breakeven_at_r:
        be = entry + side * cfg.breakeven_lock_r * initial_sl_dist
        if side * (be - best) > 0:
            best = be
    if cfg.trail_start_r and r_now >= cfg.trail_start_r and atr > 0:
        trail = price - side * cfg.trail_atr_mult * atr
        if side * (trail - best) > 0:
            best = trail
    if side * (best - sl) > 1e-9:
        return ManageAction("modify", new_sl=best, reason="breakeven/trail")
    return ManageAction()
