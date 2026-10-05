"""Fill and exit model shared by the backtester and the replay broker.

Prices in candle data are BID. Like MT5: longs open at ASK and close at BID; shorts open
at BID and close at ASK (= bid + spread). A long's stop/target trigger on the bid, a
short's on the ask. Stops that gap are filled at the open; if a bar touches both stop and
target we assume the stop came first (pessimistic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def spread_array(df, floor: float) -> np.ndarray:
    """Per-bar spread in price units, never below ``floor``.

    MT5 stores a single spread value per bar (typically the tightest of the bar), which
    understates what market orders really pay, so the configured spread acts as a floor.
    """
    if "spread" in df.columns:
        return np.maximum(df["spread"].to_numpy(dtype=float), floor)
    return np.full(len(df), float(floor))


_ROLL: dict = {"years": None, "t": np.empty(0, dtype=np.int64), "cw": np.zeros(1)}


def _rollover_table(y0: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """UTC instants (ns) of every weekday 17:00 New York rollover in years y0..y1 (cached and
    widened on demand), plus cumulative swap weights (Wednesday = 3)."""
    have = _ROLL["years"]
    if have is None or y0 < have[0] or y1 > have[1]:
        lo, hi = (y0, y1) if have is None else (min(y0, have[0]), max(y1, have[1]))
        days = pd.date_range(f"{lo - 1}-12-25", f"{hi + 1}-01-07", freq="D")
        days = days[days.weekday < 5]
        rolls = (days + pd.Timedelta(hours=17)).tz_localize(NY)
        w = np.where(days.weekday == 2, 3.0, 1.0)
        _ROLL.update(years=(lo, hi), t=rolls.tz_convert("UTC").as_unit("ns").asi8, cw=np.concatenate([[0.0], np.cumsum(w)]))
    return _ROLL["t"], _ROLL["cw"]


def rollover_nights(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
    """How many daily rollovers (17:00 New York) a position was held across, weighted the
    way brokers charge overnight swap: the Wednesday rollover counts 3x to cover the weekend."""
    if exit_time <= entry_time:
        return 0.0
    t, cw = _rollover_table(entry_time.year, exit_time.year)
    i = int(np.searchsorted(t, entry_time.value, side="right"))  # rollovers strictly after the entry
    j = int(np.searchsorted(t, exit_time.value, side="left"))  # ... and strictly before the exit
    return float(cw[j] - cw[i]) if j > i else 0.0


def swap_money(side: int, lots: float, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
               swap_long: float, swap_short: float) -> float:
    """Overnight financing for a closed position, account currency (negative = cost)."""
    nights = rollover_nights(entry_time, exit_time)
    return (swap_long if side > 0 else swap_short) * lots * nights if nights else 0.0


def entry_fill(side: int, bid_open: float, spread: float, slippage: float) -> float:
    return bid_open + spread + slippage if side > 0 else bid_open - slippage


def exit_fill_market(side: int, bid: float, spread: float, slippage: float) -> float:
    return bid - slippage if side > 0 else bid + spread + slippage


def initial_levels(side: int, fill: float, sl_dist: float, tp_dist: float) -> tuple[float, float]:
    return fill - side * sl_dist, fill + side * tp_dist


def _stop_reason(side: int, entry: float, sl: float) -> str:
    return "trail" if side * (sl - entry) >= 0 else "sl"


def check_exit(side: int, entry: float, sl: float, tp: float, o: float, h: float, l: float,
               spread: float, slippage: float):
    """Return (exit_price, reason) if the bar hits the stop or target, else None."""
    if side > 0:
        if o <= sl:
            return o - slippage, _stop_reason(side, entry, sl)
        if tp > 0 and o >= tp:
            return o, "tp"
        if l <= sl:
            return sl - slippage, _stop_reason(side, entry, sl)
        if tp > 0 and h >= tp:
            return tp, "tp"
        return None
    ao, ah, al = o + spread, h + spread, l + spread
    if ao >= sl:
        return ao + slippage, _stop_reason(side, entry, sl)
    if tp > 0 and ao <= tp:
        return ao, "tp"
    if ah >= sl:
        return sl + slippage, _stop_reason(side, entry, sl)
    if tp > 0 and al <= tp:
        return tp, "tp"
    return None


def money(side: int, entry: float, exit_price: float, lots: float, contract_size: float,
          value_multiplier: float, commission_per_lot: float = 0.0) -> float:
    return side * (exit_price - entry) * lots * contract_size * value_multiplier - commission_per_lot * lots


def simulate_trade(o, h, l, c, spread, entry_idx: int, side: int, sl_dist: float, tp_dist: float,
                   slippage: float, max_bars: int):
    """Stand-alone trade (no management) from the open of ``entry_idx``.

    Returns (exit_idx, entry_price, exit_price, reason, r_multiple), or None when the data
    ends before the trade resolves. Used to label signals for ML and to score trades the AI
    or ML filter vetoed ("shadow" trades).
    """
    n = len(o)
    if entry_idx >= n:
        return None
    quote = o[entry_idx] + spread[entry_idx] if side > 0 else o[entry_idx]
    sl, tp = initial_levels(side, quote, sl_dist, tp_dist)
    fill = entry_fill(side, o[entry_idx], spread[entry_idx], slippage)
    last = min(n - 1, entry_idx + max_bars - 1) if max_bars else n - 1
    for j in range(entry_idx, last + 1):
        hit = check_exit(side, fill, sl, tp, o[j], h[j], l[j], spread[j], slippage)
        if hit:
            px, reason = hit
            return j, fill, px, reason, side * (px - fill) / sl_dist
    if max_bars and last == entry_idx + max_bars - 1:
        px = exit_fill_market(side, c[last], spread[last], slippage)
        return last, fill, px, "time", side * (px - fill) / sl_dist
    return None
