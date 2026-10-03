"""Paper broker: real-time prices from a data broker (MT5), fills simulated in memory.

Lets you watch the whole system trade live market data without sending a single order,
even to a demo account.
"""

from __future__ import annotations

import threading
from typing import Optional

import pandas as pd

from ..config import Config
from ..execution import swap_money
from .base import AccountInfo, Broker, BrokerPosition, ClosedTrade, OrderResult, SymbolSpec, Tick


class PaperBroker(Broker):
    def __init__(self, data: Broker, cfg: Config, initial_balance: float | None = None):
        self.data = data
        self.cfg = cfg
        self.balance = float(initial_balance or cfg.backtest.initial_balance)
        self.currency = "USC"
        self._pos: dict[int, BrokerPosition] = {}
        self._closed: dict[int, ClosedTrade] = {}
        self._next = 1
        self._lock = threading.RLock()  # the panel thread and the engine thread share this broker

    @property
    def symbol(self) -> str:  # type: ignore[override]
        return self.data.symbol

    def connect(self) -> AccountInfo:
        real = self.data.connect()
        self.currency = real.currency
        return self.account()

    def shutdown(self) -> None:
        self.data.shutdown()

    def _pnl(self, side: int, entry: float, exit_price: float, volume: float) -> float:
        comm = self.cfg.backtest.commission_per_lot
        per_lot = self.data.loss_per_lot(side, entry, exit_price) - comm
        sign = 1.0 if side * (exit_price - entry) >= 0 else -1.0
        return sign * per_lot * volume - comm * volume

    def _poll(self) -> None:
        with self._lock:
            if self._pos:
                self._poll_locked()

    def _poll_locked(self) -> None:
        t = self.data.tick()
        for ticket, p in list(self._pos.items()):
            px = t.bid if p.side > 0 else t.ask
            hit_sl = (px <= p.sl) if p.side > 0 else (px >= p.sl)
            hit_tp = p.tp > 0 and ((px >= p.tp) if p.side > 0 else (px <= p.tp))
            if hit_sl or hit_tp:
                self._finish(ticket, px, t.time, "sl" if hit_sl else "tp")

    def _finish(self, ticket: int, price: float, when: pd.Timestamp, reason: str) -> None:
        p = self._pos.pop(ticket, None)
        if p is None:
            return
        pnl = self._pnl(p.side, p.entry, price, p.volume)
        bt = self.cfg.backtest
        pnl += swap_money(p.side, p.volume, p.open_time, when, bt.swap_long, bt.swap_short)
        self.balance += pnl
        self._closed[ticket] = ClosedTrade(ticket, price, when, pnl, reason)

    def account(self) -> AccountInfo:
        self._poll()
        floating = 0.0
        if self._pos:
            t = self.data.tick()
            floating = sum(self._pnl(p.side, p.entry, t.bid if p.side > 0 else t.ask, p.volume) for p in self._pos.values())
        return AccountInfo(0, self.balance, self.balance + floating, self.balance, self.currency, 0, True, "paper")

    def spec(self) -> SymbolSpec:
        return self.data.spec()

    def tick(self) -> Tick:
        return self.data.tick()

    def now(self) -> pd.Timestamp:
        return self.data.now()

    def candles(self, count: int) -> pd.DataFrame:
        return self.data.candles(count)

    def candles_for(self, symbol: str, timeframe: str, count: int):
        return self.data.candles_for(symbol, timeframe, count)

    def positions(self) -> list[BrokerPosition]:
        self._poll()
        return list(self._pos.values())

    def loss_per_lot(self, side: int, entry: float, sl: float) -> float:
        return self.data.loss_per_lot(side, entry, sl)

    def open_market(self, side: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult:
        t = self.data.tick()
        fill = t.ask if side > 0 else t.bid
        with self._lock:
            ticket = self._next
            self._next += 1
            self._pos[ticket] = BrokerPosition(ticket, side, volume, fill, sl, tp, t.time, comment=comment,
                                               magic=self.cfg.account.magic)
        return OrderResult(True, ticket, fill, volume, 10009, "paper fill")

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> OrderResult:
        p = self._pos.get(ticket)
        if p is None:
            return OrderResult(False, ticket, message="no such position")
        p.sl, p.tp = sl, tp
        return OrderResult(True, ticket, retcode=10009)

    def close_position(self, ticket: int) -> OrderResult:
        with self._lock:
            p = self._pos.get(ticket)
            if p is None:
                return OrderResult(False, ticket, message="no such position")
            t = self.data.tick()
            px = t.bid if p.side > 0 else t.ask
            self._finish(ticket, px, t.time, "manual")
            return OrderResult(True, ticket, px, p.volume, 10009)

    def closed_trade(self, ticket: int) -> Optional[ClosedTrade]:
        return self._closed.get(ticket)
