"""Bar-by-bar backtester.

Event order for every bar ``i`` (identical to what the live engine does in real time):
    1. at the open: execute closes/entries decided at the previous bar's close
    2. during the bar: stops and targets (bid/ask aware, pessimistic when ambiguous)
    3. at the close: mark to market, update risk state, manage open trades
    4. at the close: read bar ``i``'s signal -> risk checks -> size -> enter at bar i+1 open
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..execution import check_exit, entry_fill, exit_fill_market, initial_levels, money, spread_array
from ..indicators import Features
from ..management import manage_position
from ..risk import RiskManager, in_blackout
from .metrics import compute_metrics

# ml_gate(i, side, sl_dist, tp_dist, strategy) -> probability the trade wins
MLGate = Callable[[int, int, float, float, str], float]


@dataclass
class _Pos:
    id: int
    side: int
    entry_idx: int
    signal_idx: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    lots: float
    sl_dist: float
    tp_dist: float
    strategy: str
    ml_prob: float = float("nan")
    close_pending: Optional[str] = None
    sl_initial: float = 0.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    initial_balance: float
    blocked: dict = field(default_factory=dict)
    label: str = ""

    @property
    def final_balance(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.initial_balance

    def metrics(self) -> dict:
        return compute_metrics(self.trades, self.equity, self.initial_balance)


TRADE_COLUMNS = [
    "id", "side", "strategy", "signal_time", "entry_time", "exit_time", "entry", "exit", "sl_initial",
    "tp", "lots", "sl_dist", "tp_dist", "pnl", "r_multiple", "exit_reason", "bars_held", "signal_idx",
    "ml_prob",
]


class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        start: int = 0,
        end: Optional[int] = None,
        initial_balance: Optional[float] = None,
        features: Optional[Features] = None,
        ml_gate: Optional[MLGate] = None,
        ml_threshold: float = 0.5,
        news_events: Optional[np.ndarray] = None,
        label: str = "",
    ) -> BacktestResult:
        cfg = self.cfg
        bt, sym, mg = cfg.backtest, cfg.symbol, cfg.management
        n = len(df)
        end = n if end is None else min(end, n)
        balance = float(bt.initial_balance if initial_balance is None else initial_balance)

        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        spread = spread_array(df, bt.spread)
        feats = features or Features(df, sym.timeframe)
        atr = feats.atr(14).to_numpy(dtype=float)
        sig = signals["signal"].to_numpy()
        sl_arr = signals["sl_dist"].to_numpy(dtype=float)
        tp_arr = signals["tp_dist"].to_numpy(dtype=float)
        strat = signals["strategy"].to_numpy()
        bar = pd.Timedelta(minutes=cfg.tf_minutes)
        close_times = df.index + bar

        risk = RiskManager(cfg.risk, cfg.tf_minutes)
        slip, cs, vm, comm = bt.slippage, sym.contract_size, sym.value_multiplier, bt.commission_per_lot
        positions: list[_Pos] = []
        pending: Optional[dict] = None
        trades: list[dict] = []
        blocked: Counter = Counter()
        equity = np.full(end - start, np.nan)
        next_id = 1

        def close(p: _Pos, i: int, price: float, reason: str, exit_time: pd.Timestamp) -> None:
            nonlocal balance
            pnl = money(p.side, p.entry, price, p.lots, cs, vm, comm)
            balance += pnl
            risk_money = p.sl_dist * p.lots * cs * vm
            trades.append({
                "id": p.id, "side": p.side, "strategy": p.strategy, "signal_time": df.index[p.signal_idx],
                "entry_time": p.entry_time, "exit_time": exit_time,
                "entry": p.entry, "exit": price, "sl_initial": p.sl_initial, "tp": p.tp,
                "lots": p.lots, "sl_dist": p.sl_dist, "tp_dist": p.tp_dist, "pnl": pnl,
                "r_multiple": pnl / risk_money if risk_money > 0 else 0.0, "exit_reason": reason,
                "bars_held": int(round((exit_time - p.entry_time) / bar)), "signal_idx": p.signal_idx, "ml_prob": p.ml_prob,
            })
            risk.on_trade_closed(pnl, exit_time)

        for i in range(start, end):
            t_close = close_times[i]
            # 1) open of bar i: deferred closes, then the pending entry
            if positions and any(p.close_pending for p in positions):
                keep = []
                for p in positions:
                    if p.close_pending:
                        close(p, i, exit_fill_market(p.side, o[i], spread[i], slip), p.close_pending, df.index[i])
                    else:
                        keep.append(p)
                positions = keep
            if pending is not None:
                side = pending["side"]
                # Like an MT5 market order: SL/TP are absolute levels from the quoted price,
                # the fill itself may slip.
                quote = o[i] + spread[i] if side > 0 else o[i]
                sl, tp = initial_levels(side, quote, pending["sl_dist"], pending["tp_dist"])
                fill = entry_fill(side, o[i], spread[i], slip)
                positions.append(_Pos(next_id, side, i, pending["signal_idx"], df.index[i], fill, sl, tp,
                                      pending["lots"], pending["sl_dist"], pending["tp_dist"],
                                      pending["strategy"], pending["ml_prob"], sl_initial=sl))
                next_id += 1
                pending = None

            # 2) intrabar stops/targets
            if positions:
                keep = []
                for p in positions:
                    hit = check_exit(p.side, p.entry, p.sl, p.tp, o[i], h[i], l[i], spread[i], slip)
                    if hit:
                        close(p, i, hit[0], hit[1], t_close)
                    else:
                        keep.append(p)
                positions = keep

            # 3) close of bar i: mark to market, risk state, management
            floating = 0.0
            for p in positions:
                px = c[i] if p.side > 0 else c[i] + spread[i]
                floating += money(p.side, p.entry, px, p.lots, cs, vm, 0.0)
            eq = balance + floating
            risk.update(t_close, eq)
            equity[i - start] = eq
            for p in positions:
                px = c[i] if p.side > 0 else c[i] + spread[i]
                held = int(round((t_close - p.entry_time) / bar))  # wall-clock bars, same as live
                act = manage_position(p.side, p.entry, p.sl, p.sl_dist, held, px, atr[i], t_close, mg)
                if act.kind == "close":
                    p.close_pending = act.reason
                elif act.kind == "modify":
                    p.sl = act.new_sl

            # 4) entry decision on bar i's signal
            s = int(sig[i])
            if s == 0 or i + 1 >= end:
                continue
            n_open = sum(1 for p in positions if not p.close_pending)
            news = "high-impact event" if news_events is not None and in_blackout(
                news_events, t_close, cfg.risk.news_blackout_before_min, cfg.risk.news_blackout_after_min) else None
            ok, why = risk.can_open(t_close, eq, n_open, spread[i], atr[i], news)
            if not ok:
                blocked[why.split(":")[0].split(" (")[0]] += 1
                continue
            ok, why, sl_d, tp_d = risk.validate_levels(sl_arr[i], tp_arr[i], atr[i])
            if not ok:
                blocked["levels"] += 1
                continue
            prob = float("nan")
            if ml_gate is not None:
                prob = float(ml_gate(i, s, sl_d, tp_d, str(strat[i])))
                if prob < ml_threshold:
                    blocked["ml filter"] += 1
                    continue
            loss_per_lot = sl_d * cs * vm + comm
            lots, why = risk.position_size(eq, loss_per_lot, sym.volume_min, sym.volume_step, sym.volume_max)
            if lots <= 0:
                blocked["size"] += 1
                continue
            pending = {"side": s, "sl_dist": sl_d, "tp_dist": tp_d, "lots": lots, "strategy": str(strat[i]),
                       "signal_idx": i, "ml_prob": prob}
            risk.on_trade_opened(t_close)

        # Flatten at the end of the test so every trade is counted.
        last = end - 1
        for p in positions:
            close(p, last, exit_fill_market(p.side, c[last], spread[last], slip), "end", close_times[last])
        if len(equity):
            equity[-1] = balance

        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        eq_series = pd.Series(equity, index=close_times[start:end], name="equity")
        return BacktestResult(trades_df, eq_series, float(bt.initial_balance if initial_balance is None else initial_balance),
                              dict(blocked), label)
