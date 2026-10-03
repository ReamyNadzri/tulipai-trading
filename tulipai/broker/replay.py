"""Replay broker: drives the LIVE engine over historical candles with simulated fills.

Used to (a) test the live engine end-to-end without MT5 and (b) prove that live logic and
the backtester agree. Fills/exits use the same model as the backtester (execution.py).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import Config
from ..execution import check_exit, entry_fill, exit_fill_market, money, spread_array
from .base import AccountInfo, Broker, BrokerPosition, ClosedTrade, OrderResult, SymbolSpec, Tick


class ReplayBroker(Broker):
    def __init__(self, df: pd.DataFrame, cfg: Config, start_index: int, initial_balance: float | None = None):
        self.df = df
        self.cfg = cfg
        self.symbol = "XAUUSD.replay"
        self.i = max(1, start_index)
        self.o, self.h, self.l, self.c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        bt = cfg.backtest
        self.sp = spread_array(df, bt.spread)
        self.slip, self.comm = bt.slippage, bt.commission_per_lot
        self.cs, self.vm = cfg.symbol.contract_size, cfg.symbol.value_multiplier
        self.balance = float(bt.initial_balance if initial_balance is None else initial_balance)
        self.bar = pd.Timedelta(minutes=cfg.tf_minutes)
        self._pos: dict[int, BrokerPosition] = {}
        self._closed: dict[int, ClosedTrade] = {}
        self._next = 1000

    # ------------------------------------------------------------------ clock
    def has_more(self) -> bool:
        return self.i < len(self.df)

    def advance(self) -> bool:
        """Play out the current bar (stops/targets), then move to the next bar."""
        i = self.i
        for t, p in list(self._pos.items()):
            hit = check_exit(p.side, p.entry, p.sl, p.tp, self.o[i], self.h[i], self.l[i], self.sp[i], self.slip)
            if hit:
                self._finish(t, hit[0], self.df.index[i] + self.bar, hit[1])
        self.i += 1
        return self.has_more()

    def _finish(self, ticket: int, price: float, when: pd.Timestamp, reason: str) -> None:
        p = self._pos.pop(ticket)
        pnl = money(p.side, p.entry, price, p.volume, self.cs, self.vm, self.comm)
        self.balance += pnl
        self._closed[ticket] = ClosedTrade(ticket, price, when, pnl, reason)

    # ------------------------------------------------------------------ Broker API
    def connect(self) -> AccountInfo:
        return self.account()

    def account(self) -> AccountInfo:
        bid = self.o[self.i] if self.has_more() else self.c[-1]
        sp = self.sp[min(self.i, len(self.sp) - 1)]
        floating = sum(money(p.side, p.entry, bid if p.side > 0 else bid + sp, p.volume, self.cs, self.vm)
                       for p in self._pos.values())
        return AccountInfo(0, self.balance, self.balance + floating, self.balance, "USC", 1000, True, "replay")

    def spec(self) -> SymbolSpec:
        s = self.cfg.symbol
        return SymbolSpec(self.symbol, 8, 1e-8, s.contract_size, s.volume_min, s.volume_step, s.volume_max)

    def now(self) -> pd.Timestamp:
        return self.df.index[self.i]

    def tick(self) -> Tick:
        bid = self.o[self.i]
        return Tick(self.now(), bid, bid + self.sp[self.i])

    def candles(self, count: int) -> pd.DataFrame:
        return self.df.iloc[max(0, self.i - count): self.i]

    def positions(self) -> list[BrokerPosition]:
        return list(self._pos.values())

    def loss_per_lot(self, side: int, entry: float, sl: float) -> float:
        return abs(entry - sl) * self.cs * self.vm + self.comm

    def open_market(self, side: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult:
        fill = entry_fill(side, self.o[self.i], self.sp[self.i], self.slip)
        ticket = self._next
        self._next += 1
        self._pos[ticket] = BrokerPosition(ticket, side, volume, fill, sl, tp, self.now(), comment=comment,
                                           magic=self.cfg.account.magic)
        return OrderResult(True, ticket, fill, volume, 10009, "replay fill")

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> OrderResult:
        p = self._pos.get(ticket)
        if p is None:
            return OrderResult(False, ticket, message="no such position")
        p.sl, p.tp = sl, tp
        return OrderResult(True, ticket, retcode=10009)

    def close_position(self, ticket: int) -> OrderResult:
        p = self._pos.get(ticket)
        if p is None:
            return OrderResult(False, ticket, message="no such position")
        price = exit_fill_market(p.side, self.o[self.i], self.sp[self.i], self.slip)
        self._finish(ticket, price, self.now(), "manual")
        return OrderResult(True, ticket, price, p.volume, 10009)

    def flatten_at_close(self) -> None:
        """End of replay: close everything at the last bar's close (backtester convention)."""
        last = len(self.df) - 1
        for t, p in list(self._pos.items()):
            self._finish(t, exit_fill_market(p.side, self.c[last], self.sp[last], self.slip),
                         self.df.index[last] + self.bar, "end")

    def closed_trade(self, ticket: int) -> Optional[ClosedTrade]:
        return self._closed.get(ticket)
