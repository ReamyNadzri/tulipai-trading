"""Fill and exit model shared by the backtester and the replay broker.

Prices in candle data are BID. Like MT5: longs open at ASK and close at BID; shorts open
at BID and close at ASK (= bid + spread). A long's stop/target trigger on the bid, a
short's on the ask. Stops that gap are filled at the open; if a bar touches both stop and
target we assume the stop came first (pessimistic).
"""

from __future__ import annotations


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
