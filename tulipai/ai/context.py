"""Builds the compact market briefing Claude reads before every decision."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.io import resample
from ..indicators import Features, adx, atr, ema, rsi


def session_name(t: pd.Timestamp) -> str:
    h = t.hour
    if 12 <= h < 16:
        return "London/New York overlap"
    if 7 <= h < 12:
        return "London"
    if 16 <= h < 21:
        return "New York"
    return "Asia (thin liquidity)"


def _tf_block(df: pd.DataFrame, label: str) -> str:
    if len(df) < 60:
        return f"{label}: not enough bars"
    c = df["close"]
    slow_n = 200 if len(df) >= 220 else 100
    e50, e200 = ema(c, 50), ema(c, slow_n)
    a = atr(df["high"], df["low"], c, 14)
    ad, pdi, mdi = adx(df["high"], df["low"], c, 14)
    r = rsi(c, 14)
    last = c.iloc[-1]
    trend = "UP" if e50.iloc[-1] > e200.iloc[-1] and last > e50.iloc[-1] else (
        "DOWN" if e50.iloc[-1] < e200.iloc[-1] and last < e50.iloc[-1] else "MIXED")
    chg = (last / c.iloc[-21] - 1) * 100 if len(c) > 21 else np.nan
    return (f"{label}: trend {trend} | close {last:.2f} | EMA50 {e50.iloc[-1]:.2f} | EMA{slow_n} {e200.iloc[-1]:.2f} | "
            f"ATR {a.iloc[-1]:.2f} | ADX {ad.iloc[-1]:.0f} (+DI {pdi.iloc[-1]:.0f} / -DI {mdi.iloc[-1]:.0f}) | "
            f"RSI {r.iloc[-1]:.0f} | 20-bar change {chg:+.2f}%")


def build_context(
    *,
    now: pd.Timestamp,
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    feats: Features,
    mode: str,
    signal: Optional[dict] = None,
    ml_prob: Optional[float] = None,
    spread: float = 0.0,
    account: Optional[dict] = None,
    positions: Optional[list[dict]] = None,
    risk_state: Optional[dict] = None,
    events: Optional[pd.DataFrame] = None,
    headlines: Optional[list[dict]] = None,
    recent_trades: Optional[pd.DataFrame] = None,
    other_markets: Optional[dict[str, pd.DataFrame]] = None,
) -> str:
    c = candles["close"]
    a = float(feats.atr(14).iloc[-1])
    last = float(c.iloc[-1])
    lines = [f"# Snapshot {now:%Y-%m-%d %H:%M} UTC ({now:%A}) - session: {session_name(now)}"]
    lines.append(f"Mode: {mode.upper()} | symbol {symbol} {timeframe} | last close {last:.2f} | "
                 f"spread {spread:.2f} | ATR14 {a:.2f} ({a / last * 100:.2f}% of price)")

    lines.append("\n## Multi-timeframe")
    lines.append(_tf_block(candles, timeframe))
    for tf in ("H1", "H4"):
        lines.append(_tf_block(resample(candles, tf), tf))

    day = resample(candles, "D1").tail(8)
    if len(day):
        lines.append("\n## Daily candles (UTC days, last 8)")
        for t, r in day.iterrows():
            lines.append(f"{t:%m-%d %a} O {r.open:.2f} H {r.high:.2f} L {r.low:.2f} C {r.close:.2f} "
                         f"({(r.close / r.open - 1) * 100:+.2f}%)")

    today = candles[candles.index >= now.normalize()]
    asia = today[today.index.hour < 7]
    per_day = int(pd.Timedelta(days=1) / feats.bar)
    lvl = [f"20-day high {candles['high'].tail(20 * per_day).max():.2f} / low {candles['low'].tail(20 * per_day).min():.2f}"]
    if len(today):
        lvl.append(f"today high {today['high'].max():.2f} / low {today['low'].min():.2f}")
    if len(asia):
        lvl.append(f"Asian range {asia['low'].min():.2f}-{asia['high'].max():.2f}")
    if len(day) >= 2:
        y = day.iloc[-2]
        lvl.append(f"yesterday H {y.high:.2f} L {y.low:.2f} C {y.close:.2f}")
    lines.append("\n## Key levels\n" + " | ".join(lvl))

    tail = candles.tail(16)
    lines.append(f"\n## Last {len(tail)} {timeframe} bars (time O H L C)")
    for t, r in tail.iterrows():
        lines.append(f"{t:%H:%M} {r.open:.2f} {r.high:.2f} {r.low:.2f} {r.close:.2f}")

    if other_markets:
        lines.append("\n## Related markets")
        for name, df in other_markets.items():
            if df is None or len(df) < 25:
                continue
            cc = df["close"]
            lines.append(f"{name}: last {cc.iloc[-1]:.3f} | 1-day {(cc.iloc[-1] / cc.iloc[-25] - 1) * 100:+.2f}% "
                         f"| 5-day {(cc.iloc[-1] / cc.iloc[max(0, len(cc) - 121)] - 1) * 100:+.2f}%")

    if signal:
        rr = signal["tp_dist"] / signal["sl_dist"] if signal["sl_dist"] else 0
        lines.append("\n## Quant strategy proposal")
        lines.append(f"{signal['strategy']}: {'BUY' if signal['signal'] > 0 else 'SELL'} | SL {signal['sl_dist']:.2f} "
                     f"({signal['sl_dist'] / a:.2f} ATR) | TP {signal['tp_dist']:.2f} ({signal['tp_dist'] / a:.2f} ATR) "
                     f"| R:R {rr:.2f} | strength {signal.get('strength', 0):.2f}")
        if ml_prob is not None and np.isfinite(ml_prob):
            lines.append(f"ML meta-model win probability for this setup: {ml_prob:.2f}")
    else:
        lines.append("\n## Quant strategy proposal\nNone this bar.")

    lines.append("\n## Economic calendar (USD, next 24h)")
    if events is not None and len(events):
        for r in events.itertuples():
            lines.append(f"{r.time:%a %H:%M}Z [{r.impact}] {r.title} (forecast {r.forecast or '-'}, previous {r.previous or '-'})")
    else:
        lines.append("No events loaded.")

    lines.append("\n## Headlines (most recent first)")
    if headlines:
        for h in headlines:
            stamp = f"{h['time']:%m-%d %H:%M}Z " if h.get("time") is not None else ""
            lines.append(f"- {stamp}{h['title']} ({h.get('source', '')})")
    else:
        lines.append("No headlines available.")

    lines.append("\n## Account")
    if account:
        lines.append(f"balance {account['balance']:.2f} {account['currency']} | equity {account['equity']:.2f} | "
                     f"demo {account['is_demo']}")
    if risk_state:
        lines.append(f"trades today {risk_state.get('trades_today', 0)} | consecutive losses "
                     f"{risk_state.get('consecutive_losses', 0)} | day start equity {risk_state.get('day_start_equity', 0):.2f}")
    if positions:
        for p in positions:
            lines.append(f"open: {'BUY' if p['side'] > 0 else 'SELL'} {p['volume']} @ {p['entry']:.2f} SL {p['sl']:.2f} "
                         f"TP {p['tp']:.2f} P/L {p.get('profit', 0):.2f}")
    else:
        lines.append("no open positions")

    if recent_trades is not None and len(recent_trades):
        lines.append("\n## Recent closed trades (newest first)")
        for r in recent_trades.itertuples():
            lines.append(f"{str(r.close_time)[:16]} {'BUY' if r.side > 0 else 'SELL'} {r.strategy} "
                         f"R {r.r_multiple:+.2f} ({r.exit_reason})")

    task = ("Decide whether to APPROVE the proposed trade (same direction) or VETO it (HOLD)."
            if mode == "filter" and signal else
            "Decide BUY, SELL or HOLD for the next bar.")
    lines.append(f"\n## Task\n{task}")
    return "\n".join(lines)
