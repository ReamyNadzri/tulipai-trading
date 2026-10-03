"""Broker interface used by the live engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SymbolSpec:
    name: str
    digits: int
    point: float
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    stops_level: float = 0.0  # minimum SL/TP distance from price, in price units
    currency_profit: str = "USD"


@dataclass
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin_free: float
    currency: str
    leverage: int
    is_demo: bool
    server: str = ""
    company: str = ""
    name: str = ""


@dataclass
class Tick:
    time: pd.Timestamp
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class BrokerPosition:
    ticket: int
    side: int
    volume: float
    entry: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    profit: float = 0.0
    comment: str = ""
    magic: int = 0


@dataclass
class OrderResult:
    ok: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    retcode: int = 0
    message: str = ""


@dataclass
class ClosedTrade:
    ticket: int
    exit_price: float
    exit_time: pd.Timestamp
    pnl: float  # profit + commission + swap, account currency
    reason: str = ""
    extra: dict = field(default_factory=dict)


class Broker(ABC):
    """All times are UTC. Candles are BID prices, indexed by bar open time, closed bars only."""

    symbol: str = ""

    @abstractmethod
    def connect(self) -> AccountInfo: ...

    def shutdown(self) -> None:  # pragma: no cover - optional
        pass

    @abstractmethod
    def account(self) -> AccountInfo: ...

    @abstractmethod
    def spec(self) -> SymbolSpec: ...

    @abstractmethod
    def tick(self) -> Tick: ...

    @abstractmethod
    def now(self) -> pd.Timestamp: ...

    @abstractmethod
    def candles(self, count: int) -> pd.DataFrame: ...

    def candles_for(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Candles of another symbol (DXY, US10Y, ...) for AI context. Optional."""
        return None

    @abstractmethod
    def positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def loss_per_lot(self, side: int, entry: float, sl: float) -> float:
        """Account-currency loss of 1.0 lot if price moves from ``entry`` to ``sl``."""

    @abstractmethod
    def open_market(self, side: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult: ...

    @abstractmethod
    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> OrderResult: ...

    @abstractmethod
    def close_position(self, ticket: int) -> OrderResult: ...

    @abstractmethod
    def closed_trade(self, ticket: int) -> Optional[ClosedTrade]: ...
