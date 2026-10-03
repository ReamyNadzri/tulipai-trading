"""A small in-memory stand-in for the ``MetaTrader5`` package.

It mimics the parts of the API TulipAI uses (server-time timestamps, structured rate
arrays, filling-mode bitmasks, retcodes, positions and deals) over a candle DataFrame,
so the MT5 adapter, the live engine and the control panel can be tested on Linux.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408
ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2
ORDER_TIME_GTC = 0
ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
SYMBOL_TRADE_MODE_DISABLED, SYMBOL_TRADE_MODE_FULL = 0, 4
DEAL_ENTRY_IN, DEAL_ENTRY_OUT = 0, 1
DEAL_REASON_CLIENT, DEAL_REASON_EXPERT, DEAL_REASON_SL, DEAL_REASON_TP = 0, 3, 4, 5
TRADE_RETCODE_DONE = 10009


class FakeMT5:
    def __init__(self, df: pd.DataFrame, start_index: int = 400, offset_hours: int = 2, currency: str = "USC",
                 demo: bool = True, filling_mask: int = SYMBOL_FILLING_IOC, symbol: str = "XAUUSDc",
                 balance: float = 10000.0, spread: float = 0.30, fail_login: bool = False):
        # expose module-level constants as attributes, like the real module
        for k, v in globals().items():
            if k.isupper():
                setattr(self, k, v)
        self.df = df
        self.i = start_index
        self.offset = offset_hours * 3600
        self.currency = currency
        self.demo = demo
        self.filling_mask = filling_mask
        self.symbol = symbol
        self.balance = balance
        self.spread = spread
        self.fail_login = fail_login
        self.initialized_with: dict = {}
        self.requests: list[dict] = []
        self.positions: dict[int, NS] = {}
        self.deals: list[NS] = []
        self.next_ticket = 5000
        self.algo_trading = True

    # ------------------------------------------------------------- session
    def initialize(self, **kwargs):
        self.initialized_with = kwargs
        return not self.fail_login

    def shutdown(self):
        return True

    def last_error(self):
        return (1, "Success") if not self.fail_login else (-6, "Terminal: Authorization failed")

    def terminal_info(self):
        return NS(trade_allowed=self.algo_trading, connected=True)

    def account_info(self):
        eq = self.balance + sum(self._floating(p) for p in self.positions.values())
        return NS(login=int(self.initialized_with.get("login", 12345678)), balance=self.balance, equity=eq,
                  margin_free=eq, currency=self.currency, leverage=1000,
                  trade_mode=ACCOUNT_TRADE_MODE_DEMO if self.demo else ACCOUNT_TRADE_MODE_REAL,
                  server=self.initialized_with.get("server", "Fake-MT5Trial"), company="Fake Broker", name="Test")

    # ------------------------------------------------------------- symbols
    def _sym(self, name):
        return NS(name=name, digits=3, point=0.001, trade_contract_size=100.0, volume_min=0.01, volume_step=0.01,
                  volume_max=200.0, trade_stops_level=0, filling_mode=self.filling_mask, trade_tick_size=0.001,
                  trade_tick_value=0.1, currency_profit="USD", trade_mode=SYMBOL_TRADE_MODE_FULL, visible=True)

    def symbols_get(self, group=None):
        names = ["EURUSDc", "XAUUSD", self.symbol, "XAGUSDc", "US500c"]
        return [self._sym(n) for n in dict.fromkeys(names)]

    def symbol_info(self, name):
        return self._sym(name) if name in (self.symbol, "XAUUSD", "XAGUSDc") else None

    def symbol_select(self, name, enable=True):
        return name in (self.symbol, "XAUUSD", "XAGUSDc")

    def _bid(self):
        return float(self.df["open"].iloc[self.i])

    def symbol_info_tick(self, name):
        t = int(self.df.index[self.i].timestamp()) + self.offset  # server time of the current bar
        return NS(time=t, bid=self._bid(), ask=self._bid() + self.spread)

    # ------------------------------------------------------------- data
    def copy_rates_from_pos(self, symbol, tf, start_pos, count):
        if symbol not in (self.symbol, "XAGUSDc", "XAUUSD"):
            return None
        end = self.i - (start_pos - 1)  # start_pos=1 -> closed bars only
        sub = self.df.iloc[max(0, end - count): end]
        arr = np.zeros(len(sub), dtype=[("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
                                         ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"),
                                         ("real_volume", "<u8")])
        arr["time"] = sub.index.as_unit("s").asi8 + self.offset
        for k in ("open", "high", "low", "close"):
            arr[k] = sub[k].to_numpy()
        arr["tick_volume"] = sub["volume"].to_numpy().astype(int)
        arr["spread"] = int(round(self.spread / 0.001))
        return arr

    def copy_rates_range(self, symbol, tf, date_from, date_to):
        t0 = pd.Timestamp(date_from).tz_localize("UTC") if pd.Timestamp(date_from).tzinfo is None else pd.Timestamp(date_from)
        t1 = pd.Timestamp(date_to).tz_localize("UTC") if pd.Timestamp(date_to).tzinfo is None else pd.Timestamp(date_to)
        srv = self.df.index + pd.Timedelta(seconds=self.offset)
        mask = (srv >= t0) & (srv < t1)
        idx = np.flatnonzero(mask)
        if not len(idx):
            return None
        save = self.i
        self.i = idx[-1] + 1
        out = self.copy_rates_from_pos(symbol, tf, 1, len(idx))
        self.i = save
        return out

    # ------------------------------------------------------------- trading
    def order_calc_profit(self, otype, symbol, volume, price_open, price_close):
        sign = 1 if otype == ORDER_TYPE_BUY else -1
        return sign * (price_close - price_open) * volume * 100.0

    def _floating(self, p):
        bid = self._bid()
        px = bid if p.type == POSITION_TYPE_BUY else bid + self.spread
        return (1 if p.type == POSITION_TYPE_BUY else -1) * (px - p.price_open) * p.volume * 100.0

    def order_send(self, req):
        self.requests.append(dict(req))
        if not self.algo_trading:
            return NS(retcode=10027, order=0, deal=0, price=0.0, volume=0.0, comment="AutoTrading disabled by client")
        if req["action"] == TRADE_ACTION_SLTP:
            p = self.positions.get(req["position"])
            if p is None:
                return NS(retcode=10013, order=0, deal=0, price=0.0, volume=0.0, comment="invalid request")
            p.sl, p.tp = req["sl"], req["tp"]
            return NS(retcode=TRADE_RETCODE_DONE, order=0, deal=0, price=0.0, volume=0.0, comment="done")
        allowed = {ORDER_FILLING_RETURN}
        if self.filling_mask & SYMBOL_FILLING_FOK:
            allowed.add(ORDER_FILLING_FOK)
        if self.filling_mask & SYMBOL_FILLING_IOC:
            allowed.add(ORDER_FILLING_IOC)
        if req.get("type_filling") not in allowed or req.get("type_filling") == ORDER_FILLING_RETURN and self.filling_mask:
            return NS(retcode=10030, order=0, deal=0, price=0.0, volume=0.0, comment="Unsupported filling mode")
        bid = self._bid()
        now = int(self.df.index[self.i].timestamp()) + self.offset
        if "position" in req:  # closing deal
            p = self.positions.pop(req["position"], None)
            if p is None:
                return NS(retcode=10013, order=0, deal=0, price=0.0, volume=0.0, comment="no position")
            px = bid if p.type == POSITION_TYPE_BUY else bid + self.spread
            self._out_deal(p, px, now, DEAL_REASON_EXPERT)
            return NS(retcode=TRADE_RETCODE_DONE, order=self._tick(), deal=self._tick(), price=px, volume=p.volume, comment="done")
        px = bid + self.spread if req["type"] == ORDER_TYPE_BUY else bid
        t = self._tick()
        self.positions[t] = NS(ticket=t, type=POSITION_TYPE_BUY if req["type"] == ORDER_TYPE_BUY else POSITION_TYPE_SELL,
                               volume=req["volume"], price_open=px, sl=req["sl"], tp=req["tp"], time=now, profit=0.0,
                               comment=req.get("comment", ""), magic=req.get("magic", 0), symbol=req["symbol"])
        self.deals.append(NS(ticket=self._tick(), position_id=t, entry=DEAL_ENTRY_IN, price=px, time=now, profit=0.0,
                             commission=0.0, swap=0.0, fee=0.0, reason=DEAL_REASON_EXPERT))
        return NS(retcode=TRADE_RETCODE_DONE, order=t, deal=t, price=px, volume=req["volume"], comment="done")

    def _tick(self):
        self.next_ticket += 1
        return self.next_ticket

    def _out_deal(self, p, px, when, reason):
        pnl = (1 if p.type == POSITION_TYPE_BUY else -1) * (px - p.price_open) * p.volume * 100.0
        self.balance += pnl
        self.deals.append(NS(ticket=self._tick(), position_id=p.ticket, entry=DEAL_ENTRY_OUT, price=px, time=when,
                             profit=pnl, commission=0.0, swap=0.0, fee=0.0, reason=reason))

    def positions_get(self, symbol=None, ticket=None):
        ps = list(self.positions.values())
        if ticket is not None:
            ps = [p for p in ps if p.ticket == ticket]
        if symbol is not None:
            ps = [p for p in ps if p.symbol == symbol]
        for p in ps:
            p.profit = self._floating(p)
        return tuple(ps)

    def history_deals_get(self, *args, position=None, **kw):
        if position is not None:
            return tuple(d for d in self.deals if d.position_id == position)
        return tuple(self.deals)

    # ------------------------------------------------------------- clock
    def advance(self):
        """Play out the current bar's high/low against stops, then move to the next bar."""
        row = self.df.iloc[self.i]
        when = int((self.df.index[self.i] + pd.Timedelta(minutes=15)).timestamp()) + self.offset
        for t, p in list(self.positions.items()):
            if p.type == POSITION_TYPE_BUY:
                if p.sl and row.low <= p.sl:
                    self.positions.pop(t); self._out_deal(p, p.sl, when, DEAL_REASON_SL)
                elif p.tp and row.high >= p.tp:
                    self.positions.pop(t); self._out_deal(p, p.tp, when, DEAL_REASON_TP)
            else:
                if p.sl and row.high + self.spread >= p.sl:
                    self.positions.pop(t); self._out_deal(p, p.sl, when, DEAL_REASON_SL)
                elif p.tp and row.low + self.spread <= p.tp:
                    self.positions.pop(t); self._out_deal(p, p.tp, when, DEAL_REASON_TP)
        self.i += 1
        return self.i < len(self.df)
