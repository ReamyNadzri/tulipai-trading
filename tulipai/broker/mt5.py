"""MetaTrader 5 adapter (Windows + the official ``MetaTrader5`` Python package).

``mt5.initialize(path, login, password, server)`` starts the terminal if it is not
running and logs in, so the control panel only needs those four values.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..data.io import normalize
from .base import AccountInfo, Broker, BrokerPosition, ClosedTrade, OrderResult, SymbolSpec, Tick

RETCODE_HELP = {
    10004: "requote",
    10006: "request rejected",
    10013: "invalid request",
    10014: "invalid volume",
    10015: "invalid price",
    10016: "invalid stops (SL/TP too close to price or on the wrong side)",
    10017: "trading disabled for this symbol/account",
    10018: "market closed",
    10019: "not enough money",
    10020: "price changed",
    10021: "no quotes",
    10024: "too many requests",
    10026: "autotrading disabled by the broker server",
    10027: "AutoTrading is OFF in the terminal - click the 'Algo Trading' button in the MT5 toolbar",
    10030: "unsupported filling mode",
    10031: "no connection to the trade server",
    10040: "position limit reached",
}
OK_CODES = {10008, 10009, 10010}
RETRY_PRICE = {10004, 10020, 10021}
CENT_CURRENCIES = {"USC", "USCENT", "USX"}


NY = "America/New_York"


class MT5Error(RuntimeError):
    pass


def _ny_offset_hours(ts_utc: pd.Timestamp) -> int:
    """New York's UTC offset at that moment: -5 in winter, -4 in summer."""
    return int(ts_utc.tz_convert(NY).utcoffset().total_seconds() // 3600)


def last_ny_close(now_utc: pd.Timestamp) -> pd.Timestamp:
    """Most recent 17:00 New York time (gold's daily/weekly close) at or before ``now_utc``."""
    ny = now_utc.tz_convert(NY)
    close = ny.replace(hour=17, minute=0, second=0, microsecond=0)
    if close > ny:
        close = (ny - pd.Timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
    while close.weekday() >= 5:  # no close on Saturday/Sunday
        close = (close - pd.Timedelta(days=1)).replace(hour=17)
    return close.tz_convert("UTC")


def import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on platform
        raise MT5Error(
            "The MetaTrader5 package is not installed. It only exists for Windows: run "
            "`pip install MetaTrader5` on the PC where MT5 is installed."
        ) from exc
    return mt5


def _c(mt5, name: str, default: int) -> int:
    return getattr(mt5, name, default)


class MT5Broker(Broker):
    def __init__(self, cfg: Config, login: Any = None, password: str | None = None, server: str | None = None,
                 path: str | None = None, mt5_module: Any = None):
        self.cfg = cfg
        self.mt5 = mt5_module or import_mt5()
        self.login, self.password, self.server, self.path = login, password, server, path
        self.symbol = ""
        self.offset = timedelta(0)
        self.offset_note = ""
        self.dst_mode = False
        self._lock = threading.RLock()
        self._spec: Optional[SymbolSpec] = None
        self._is_demo = True

    # ------------------------------------------------------------------ session
    def connect(self) -> AccountInfo:
        mt5 = self.mt5
        kwargs: dict[str, Any] = {"timeout": 60000}
        if self.path:
            kwargs["path"] = str(self.path)
        if self.login:
            kwargs["login"] = int(self.login)
        if self.password:
            kwargs["password"] = str(self.password)
        if self.server:
            kwargs["server"] = str(self.server)
        with self._lock:
            if not mt5.initialize(**kwargs):
                err = mt5.last_error()
                mt5.shutdown()
                raise MT5Error(
                    f"Could not start/connect MetaTrader 5 ({err}). Check that MT5 is installed (or set the "
                    "terminal path), and that login, password and server are exactly as in your broker email."
                )
            info = mt5.account_info()
            if info is None:
                raise MT5Error(f"MT5 started but no account is logged in ({mt5.last_error()}). Enter login/password/server.")
            self.symbol = self._resolve_symbol(self.cfg.symbol.name, info.currency)
            if not mt5.symbol_select(self.symbol, True):
                raise MT5Error(f"Could not add {self.symbol} to Market Watch ({mt5.last_error()})")
            self._spec = None
            self._detect_offset()
            acct = self.account()
            self._is_demo = acct.is_demo
            return acct

    def shutdown(self) -> None:
        with self._lock:
            self.mt5.shutdown()

    def preflight(self) -> list[str]:
        """Human-readable problems that would stop the bot from trading."""
        problems = []
        with self._lock:
            ti = self.mt5.terminal_info()
            acct = self.account()
            si = self.mt5.symbol_info(self.symbol)
        if ti is not None and not getattr(ti, "trade_allowed", True):
            problems.append("Algo Trading is OFF in MT5 - click the 'Algo Trading' button in the toolbar.")
        if not acct.is_demo and not self.cfg.account.allow_real_account:
            problems.append(
                f"Account {acct.login} is a REAL account (cent accounts are real money). Orders are blocked "
                "until you set account.allow_real_account: true in your config."
            )
        if si is not None and getattr(si, "trade_mode", 4) == _c(self.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0):
            problems.append(f"Trading is disabled for {self.symbol} on this account.")
        return problems

    def _resolve_symbol(self, wanted: str, currency: str) -> str:
        mt5 = self.mt5
        if wanted and wanted.lower() != "auto":
            if mt5.symbol_info(wanted) is None:
                golds = [s.name for s in (mt5.symbols_get("*XAU*") or [])] + [s.name for s in (mt5.symbols_get("*GOLD*") or [])]
                raise MT5Error(f"Symbol {wanted!r} not found. Gold symbols on this server: {golds or 'none'}")
            return wanted
        cent = currency.upper() in CENT_CURRENCIES
        best, best_score = None, -1e9
        for s in mt5.symbols_get() or []:
            name = s.name
            up = name.upper()
            if not (("XAU" in up and "USD" in up) or up.startswith("GOLD")):
                continue
            score = 10.0 if up.startswith("XAUUSD") else 5.0
            suffix = name[6:] if up.startswith("XAUUSD") else name[4:]
            if cent and suffix.lower().strip(".-_").startswith("c"):
                score += 5
            if not cent and suffix == "":
                score += 3
            if getattr(s, "trade_mode", 4) == _c(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0):
                score -= 50
            if getattr(s, "visible", False):
                score += 1
            score -= 0.1 * len(name)
            if score > best_score:
                best, best_score = name, score
        if best is None:
            raise MT5Error("No gold symbol (XAUUSD*/GOLD*) found on this server; set symbol.name in the config.")
        return best

    def _detect_offset(self, now: pd.Timestamp | None = None) -> None:
        """Work out how the broker's server clock relates to UTC.

        MT5 reports every time in server time. Most brokers run either a fixed offset
        (e.g. UTC+0) or "New York close + 7h" (UTC+2 in winter, UTC+3 in summer, switching
        with US daylight saving). Live, the latest tick gives the offset directly. With the
        market closed (weekends, daily break) the last tick is anchored to the last 17:00
        New York close instead.
        """
        setting = self.cfg.symbol.server_utc_offset_hours
        self.dst_mode = False
        if isinstance(setting, str) and setting.lower().replace(" ", "") in ("ny+7", "ny7"):
            self.dst_mode = True
            self.offset = timedelta(hours=_ny_offset_hours(now or pd.Timestamp.now(tz="UTC")) + 7)
            self.offset_note = "configured: New York time + 7h (UTC+2 winter / UTC+3 summer)"
            return
        if setting != "auto":
            self.offset = timedelta(hours=float(setting))
            self.offset_note = f"configured fixed UTC{float(setting):+g}"
            return
        now = now or pd.Timestamp.now(tz="UTC")
        hours, how = None, ""
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is not None and getattr(tick, "time", 0):
            diff = float(tick.time) - now.timestamp()
            h = round(diff / 3600)
            if abs(diff - h * 3600) < 120:
                hours, how = h, "live tick"
            else:
                close = last_ny_close(now)
                diff = float(tick.time) - close.timestamp()
                h = round(diff / 3600)
                if abs(diff - h * 3600) < 20 * 60 and -12 <= h <= 14:
                    hours, how = h, f"market closed; last tick matched the {close:%a %H:%M} UTC close"
        if hours is None:
            self.offset = timedelta(hours=2)
            self.offset_note = ("WARNING: could not detect the server time zone, assuming UTC+2. Set "
                                "symbol.server_utc_offset_hours (e.g. 0, 2, 3 or 'ny+7') in config/config.yaml")
            return
        self.offset = timedelta(hours=hours)
        if hours == _ny_offset_hours(now) + 7:
            self.dst_mode = True
            self.offset_note = (f"auto-detected UTC{hours:+d} ({how}); server follows New York daylight saving "
                                "(UTC+2 winter / UTC+3 summer)")
        else:
            self.offset_note = f"auto-detected fixed UTC{hours:+d} ({how})"

    # ------------------------------------------------------------------ data
    def _to_utc(self, server_seconds) -> pd.DatetimeIndex:
        srv = pd.DatetimeIndex(pd.to_datetime(np.asarray(server_seconds, dtype=np.int64), unit="s", utc=True))
        if not getattr(self, "dst_mode", False):
            return srv - self.offset
        approx = srv - pd.Timedelta(hours=3)
        ny_off = approx.tz_convert(NY).tz_localize(None) - approx.tz_localize(None)
        return srv - (ny_off + pd.Timedelta(hours=7))

    def _ts(self, server_seconds: float) -> pd.Timestamp:
        return self._to_utc([int(server_seconds)])[0]

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def account(self) -> AccountInfo:
        mt5 = self.mt5
        with self._lock:
            a = mt5.account_info()
        if a is None:
            raise MT5Error(f"Lost connection to MT5 ({mt5.last_error()})")
        demo = a.trade_mode == _c(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
        return AccountInfo(int(a.login), float(a.balance), float(a.equity), float(a.margin_free), str(a.currency),
                           int(a.leverage), bool(demo), str(getattr(a, "server", "")), str(getattr(a, "company", "")),
                           str(getattr(a, "name", "")))

    def spec(self) -> SymbolSpec:
        if self._spec is None:
            with self._lock:
                s = self.mt5.symbol_info(self.symbol)
            if s is None:
                raise MT5Error(f"No symbol info for {self.symbol}")
            self._spec = SymbolSpec(
                name=s.name, digits=int(s.digits), point=float(s.point),
                contract_size=float(s.trade_contract_size), volume_min=float(s.volume_min),
                volume_step=float(s.volume_step), volume_max=float(s.volume_max),
                stops_level=float(s.trade_stops_level) * float(s.point),
                currency_profit=str(getattr(s, "currency_profit", "USD")),
            )
        return self._spec

    def tick(self) -> Tick:
        with self._lock:
            t = self.mt5.symbol_info_tick(self.symbol)
        if t is None:
            raise MT5Error(f"No tick for {self.symbol} ({self.mt5.last_error()})")
        return Tick(self._ts(t.time), float(t.bid), float(t.ask))

    def _tf(self, timeframe: str) -> int:
        tf = getattr(self.mt5, f"TIMEFRAME_{timeframe.upper()}", None)
        if tf is None:
            raise MT5Error(f"Unsupported timeframe {timeframe}")
        return tf

    def _rates_df(self, rates, point: float) -> pd.DataFrame:
        r = pd.DataFrame(rates)
        idx = self._to_utc(r["time"].to_numpy())
        out = pd.DataFrame({"open": r["open"].to_numpy(), "high": r["high"].to_numpy(), "low": r["low"].to_numpy(),
                            "close": r["close"].to_numpy(), "volume": r["tick_volume"].to_numpy()}, index=idx)
        if "spread" in r.columns and point > 0:
            out["spread"] = r["spread"].to_numpy(dtype=float) * point
        return normalize(out)

    def candles(self, count: int) -> pd.DataFrame:
        with self._lock:
            rates = self.mt5.copy_rates_from_pos(self.symbol, self._tf(self.cfg.symbol.timeframe), 1, int(count))
        if rates is None or len(rates) == 0:
            raise MT5Error(f"No candles for {self.symbol} ({self.mt5.last_error()})")
        return self._rates_df(rates, self.spec().point)

    def history(self, start: pd.Timestamp, end: pd.Timestamp, timeframe: str | None = None) -> pd.DataFrame:
        """Long history for backtests, fetched in chunks (MT5 limits bars per request)."""
        tf = self._tf(timeframe or self.cfg.symbol.timeframe)
        frames, cur = [], start
        step = pd.Timedelta(days=60)
        while cur < end:
            nxt = min(cur + step, end)
            rates = None
            for attempt in range(3):  # the terminal may still be downloading old history
                with self._lock:
                    rates = self.mt5.copy_rates_range(self.symbol, tf, (cur + self.offset).to_pydatetime(),
                                                      (nxt + self.offset).to_pydatetime())
                if rates is not None and len(rates):
                    break
                time.sleep(1.0 + attempt)
            if rates is not None and len(rates):
                frames.append(self._rates_df(rates, self.spec().point))
            cur = nxt
        if not frames:
            raise MT5Error("MT5 returned no history; scroll the chart back or raise 'Max bars in chart' in MT5 options")
        return normalize(pd.concat(frames))

    def candles_for(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        try:
            with self._lock:
                if self.mt5.symbol_info(symbol) is None or not self.mt5.symbol_select(symbol, True):
                    return None
                rates = self.mt5.copy_rates_from_pos(symbol, self._tf(timeframe), 1, int(count))
            if rates is None or len(rates) == 0:
                return None
            return self._rates_df(rates, 0.0)
        except Exception:
            return None

    # ------------------------------------------------------------------ positions
    def positions(self) -> list[BrokerPosition]:
        mt5 = self.mt5
        with self._lock:
            ps = mt5.positions_get(symbol=self.symbol) or []
        buy = _c(mt5, "POSITION_TYPE_BUY", 0)
        out = []
        for p in ps:
            if int(p.magic) != self.cfg.account.magic:
                continue
            out.append(BrokerPosition(int(p.ticket), 1 if p.type == buy else -1, float(p.volume), float(p.price_open),
                                      float(p.sl), float(p.tp), self._ts(p.time), float(p.profit), str(p.comment),
                                      int(p.magic)))
        return out

    def loss_per_lot(self, side: int, entry: float, sl: float) -> float:
        mt5 = self.mt5
        otype = _c(mt5, "ORDER_TYPE_BUY", 0) if side > 0 else _c(mt5, "ORDER_TYPE_SELL", 1)
        with self._lock:
            p = mt5.order_calc_profit(otype, self.symbol, 1.0, float(entry), float(sl))
        if p is None:
            with self._lock:
                s = mt5.symbol_info(self.symbol)
            p = -abs(entry - sl) / s.trade_tick_size * s.trade_tick_value
        return abs(float(p)) + self.cfg.backtest.commission_per_lot

    # ------------------------------------------------------------------ orders
    def _fillings(self) -> list[int]:
        mt5 = self.mt5
        with self._lock:
            s = mt5.symbol_info(self.symbol)
        mask = int(getattr(s, "filling_mode", 0)) if s is not None else 0
        fok, ioc, ret = (_c(mt5, "ORDER_FILLING_FOK", 0), _c(mt5, "ORDER_FILLING_IOC", 1),
                         _c(mt5, "ORDER_FILLING_RETURN", 2))
        order = []
        if mask & _c(mt5, "SYMBOL_FILLING_FOK", 1):
            order.append(fok)
        if mask & _c(mt5, "SYMBOL_FILLING_IOC", 2):
            order.append(ioc)
        order.append(ret)
        return order + [f for f in (fok, ioc, ret) if f not in order]

    def _send(self, request: dict, refresh_price: bool = False) -> OrderResult:
        mt5 = self.mt5
        last_msg = ""
        attempts = 0
        for filling in self._fillings():
            for _ in range(3):
                attempts += 1
                req = dict(request, type_filling=filling)
                if refresh_price:
                    t = self.tick()
                    req["price"] = t.ask if req["type"] == _c(mt5, "ORDER_TYPE_BUY", 0) else t.bid
                with self._lock:
                    res = mt5.order_send(req)
                if res is None:
                    last_msg = f"order_send failed: {mt5.last_error()}"
                    time.sleep(0.5)
                    continue
                code = int(res.retcode)
                if code in OK_CODES or code == 10025:  # 10025 = no changes (SL/TP already there)
                    return OrderResult(True, int(getattr(res, "order", 0) or 0), float(getattr(res, "price", 0.0)),
                                       float(getattr(res, "volume", 0.0)), code, str(getattr(res, "comment", "")))
                last_msg = f"{code}: {RETCODE_HELP.get(code, getattr(res, 'comment', 'error'))}"
                if code == 10030:
                    break  # try the next filling mode
                if code in RETRY_PRICE:
                    refresh_price = "price" in request
                    time.sleep(0.3)
                    continue
                return OrderResult(False, retcode=code, message=last_msg)
        return OrderResult(False, message=last_msg or f"order failed after {attempts} attempts")

    def _guard_real(self) -> Optional[OrderResult]:
        if not self._is_demo and not self.cfg.account.allow_real_account:
            return OrderResult(False, message="refused: real account and account.allow_real_account is false")
        return None

    def open_market(self, side: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult:
        refused = self._guard_real()
        if refused:
            return refused
        mt5, spec = self.mt5, self.spec()
        t = self.tick()
        price = t.ask if side > 0 else t.bid
        request = {
            "action": _c(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": self.symbol,
            "volume": float(volume),
            "type": _c(mt5, "ORDER_TYPE_BUY", 0) if side > 0 else _c(mt5, "ORDER_TYPE_SELL", 1),
            "price": price,
            "sl": round(float(sl), spec.digits),
            "tp": round(float(tp), spec.digits) if tp else 0.0,
            "deviation": int(self.cfg.account.deviation_points),
            "magic": int(self.cfg.account.magic),
            "comment": comment[:31],
            "type_time": _c(mt5, "ORDER_TIME_GTC", 0),
        }
        res = self._send(request)
        if res.ok:
            res.ticket = self._position_ticket(res.ticket) or res.ticket
        return res

    def _position_ticket(self, order_ticket: int) -> int:
        with self._lock:
            ps = self.mt5.positions_get(ticket=order_ticket)
        if ps:
            return int(ps[0].ticket)
        mine = sorted(self.positions(), key=lambda p: p.open_time)
        return mine[-1].ticket if mine else 0

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> OrderResult:
        spec = self.spec()
        request = {
            "action": _c(self.mt5, "TRADE_ACTION_SLTP", 6),
            "symbol": self.symbol,
            "position": int(ticket),
            "sl": round(float(sl), spec.digits),
            "tp": round(float(tp), spec.digits) if tp else 0.0,
            "magic": int(self.cfg.account.magic),
        }
        return self._send(request)

    def close_position(self, ticket: int) -> OrderResult:
        mt5 = self.mt5
        with self._lock:
            ps = mt5.positions_get(ticket=int(ticket))
        if not ps:
            return OrderResult(False, ticket, message="position not found")
        p = ps[0]
        is_buy = p.type == _c(mt5, "POSITION_TYPE_BUY", 0)
        t = self.tick()
        request = {
            "action": _c(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": self.symbol,
            "volume": float(p.volume),
            "type": _c(mt5, "ORDER_TYPE_SELL", 1) if is_buy else _c(mt5, "ORDER_TYPE_BUY", 0),
            "position": int(ticket),
            "price": t.bid if is_buy else t.ask,
            "deviation": int(self.cfg.account.deviation_points),
            "magic": int(self.cfg.account.magic),
            "comment": "TulipAI close",
            "type_time": _c(mt5, "ORDER_TIME_GTC", 0),
        }
        res = self._send(request)
        res.ticket = int(ticket)
        return res

    def swap_per_lot_night(self) -> tuple[Optional[float], Optional[float], str]:
        """Broker's overnight swap for 1.0 lot (long, short) in account currency, if the
        symbol's swap mode can be converted; otherwise (None, None, reason)."""
        mt5 = self.mt5
        with self._lock:
            si = mt5.symbol_info(self.symbol)
        if si is None:
            return None, None, "no symbol info"
        mode = int(getattr(si, "swap_mode", -1))
        sl, ss = float(getattr(si, "swap_long", 0.0)), float(getattr(si, "swap_short", 0.0))
        if mode == _c(mt5, "SYMBOL_SWAP_MODE_DISABLED", 0):
            return 0.0, 0.0, "swap disabled (swap-free)"
        price = self.tick().bid

        def price_move_money(dist: float) -> float:  # account-currency value of a price move on 1 lot
            if dist == 0:
                return 0.0
            return self.loss_per_lot(1, price, price - abs(dist)) - self.cfg.backtest.commission_per_lot

        if mode == _c(mt5, "SYMBOL_SWAP_MODE_POINTS", 1):
            return (np.sign(sl) * price_move_money(sl * si.point), np.sign(ss) * price_move_money(ss * si.point),
                    f"{sl} / {ss} points")
        if mode == _c(mt5, "SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT", 4):
            return sl, ss, f"{sl} / {ss} in account currency"
        if mode in (_c(mt5, "SYMBOL_SWAP_MODE_INTEREST_CURRENT", 5), _c(mt5, "SYMBOL_SWAP_MODE_INTEREST_OPEN", 6)):
            day = price / 360.0  # annual % of the position value, per day
            return (np.sign(sl) * price_move_money(day * abs(sl) / 100), np.sign(ss) * price_move_money(day * abs(ss) / 100),
                    f"{sl}% / {ss}% a year")
        return None, None, f"swap mode {mode} not converted (raw {sl} / {ss})"

    def close_all(self) -> list[OrderResult]:
        return [self.close_position(p.ticket) for p in self.positions()]

    def closed_trade(self, ticket: int) -> Optional[ClosedTrade]:
        mt5 = self.mt5
        with self._lock:
            deals = mt5.history_deals_get(position=int(ticket))
            if not deals:
                now = datetime.now(timezone.utc) + self.offset + timedelta(days=1)
                deals = [d for d in (mt5.history_deals_get(now - timedelta(days=30), now) or [])
                         if int(getattr(d, "position_id", 0)) == int(ticket)]
        if not deals:
            return None
        out_entries = {_c(mt5, "DEAL_ENTRY_OUT", 1), _c(mt5, "DEAL_ENTRY_INOUT", 2), _c(mt5, "DEAL_ENTRY_OUT_BY", 3)}
        outs = [d for d in deals if int(d.entry) in out_entries]
        if not outs:
            return None
        last = max(outs, key=lambda d: d.time)
        pnl = sum(float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0)) for d in deals)
        reasons = {_c(mt5, "DEAL_REASON_SL", 4): "sl", _c(mt5, "DEAL_REASON_TP", 5): "tp",
                   _c(mt5, "DEAL_REASON_SO", 6): "stopout", _c(mt5, "DEAL_REASON_EXPERT", 3): "bot"}
        return ClosedTrade(int(ticket), float(last.price), self._ts(last.time), pnl,
                           reasons.get(int(getattr(last, "reason", -1)), "manual"))
