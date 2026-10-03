"""The live trading engine.

Polls the broker; whenever a new bar closes it: syncs closed trades, updates the risk
state, manages open positions, scores old shadow trades, then evaluates an entry:

    strategy signal -> risk pre-checks -> ML filter -> Claude (filter/autonomous) ->
    level validation -> position sizing -> order -> journal

The exact same sequence runs in replay mode on historical data (ReplayBroker), which is how
the test-suite proves live logic matches the backtester.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .ai.brain import AIDecision, ClaudeBrain
from .ai.context import build_context
from .broker.base import AccountInfo, Broker, BrokerPosition, Tick
from .config import Config
from .execution import simulate_trade, spread_array
from .indicators import Features
from .journal import Journal
from .management import manage_position
from .risk import RiskManager, RiskState
from .strategies import build_strategy

log = logging.getLogger("tulipai")


class LiveEngine:
    def __init__(
        self,
        cfg: Config,
        broker: Broker,
        journal: Journal,
        brain: Optional[ClaudeBrain] = None,
        calendar=None,
        headlines=None,
        ml=None,
        mode_label: str = "mt5",
    ):
        self.cfg = cfg
        self.broker = broker
        self.journal = journal
        self.brain = brain
        self.calendar = calendar
        self.headlines = headlines
        self.ml = ml
        self.mode_label = mode_label
        self.strategy = build_strategy(cfg.strategy)
        self.bar = pd.Timedelta(minutes=cfg.tf_minutes)
        saved = journal.get_kv(f"risk_state:{mode_label}") or {}
        self.risk = RiskManager(cfg.risk, cfg.tf_minutes, RiskState.from_dict(saved) if saved else None)
        self.last_bar: Optional[pd.Timestamp] = None
        self.bar_count = 0
        self.meta: dict[int, dict] = {}  # ticket -> journal trade row
        self.pending_close: dict[int, int] = {}  # ticket -> attempts to fetch the closing deal
        self.close_reasons: dict[int, str] = {}  # ticket -> why the engine closed it (weekend, time, ...)
        self.pause_entries = False
        self.account_info: Optional[AccountInfo] = None
        self.status: dict = {"state": "created", "last_bar": None, "last_decision": None, "last_ai": None,
                             "error": "", "warnings": []}
        self._ctx_symbols: Optional[list[str]] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> AccountInfo:
        acct = self.broker.connect()
        self.account_info = acct
        warnings = []
        if hasattr(self.broker, "preflight"):
            warnings = self.broker.preflight()
        if not acct.is_demo and not self.cfg.account.allow_real_account and self.mode_label == "mt5":
            warnings.append("REAL account detected and allow_real_account is false: entries are disabled.")
            self.pause_entries = True
        if self.brain is not None:
            ok, why = ClaudeBrain.available()
            if not ok:
                warnings.append(f"AI disabled: {why}")
                self.brain = None
        self.status.update(state="running", warnings=warnings, symbol=self.broker.symbol,
                           server_time=getattr(self.broker, "offset_note", ""),
                           ai_mode=self.cfg.ai.mode if self.brain is not None else "off")
        for w in warnings:
            log.warning(w)
        self._reconcile()
        log.info("Connected: account %s (%s, %s), symbol %s, AI mode %s", acct.login, acct.currency,
                 "demo" if acct.is_demo else "REAL", self.broker.symbol, self.cfg.ai.mode if self.brain else "off")
        return acct

    def _reconcile(self) -> None:
        """Match the journal with the broker after a (re)start."""
        positions = {p.ticket: p for p in self.broker.positions()}
        for row in self.journal.open_trades(self.mode_label):
            t = int(row["ticket"])
            self.meta[t] = row
            if t not in positions:
                self.pending_close[t] = 0
        for t, p in positions.items():
            if t not in self.meta:
                sl_dist = abs(p.entry - p.sl) if p.sl else 0.0
                lpl = self.broker.loss_per_lot(p.side, p.entry, p.sl) if p.sl else 0.0
                self.journal.open_trade(t, self.mode_label, self.broker.symbol, p.side, p.volume, p.open_time, p.entry,
                                        p.sl, p.tp, sl_dist, "adopted", None, None, lpl * p.volume)
                self.meta[t] = dict(self.journal.trade_by_ticket(t))
                log.info("Adopted existing position %s", t)

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        stop_event = stop_event or threading.Event()
        delay = self.cfg.live.poll_seconds
        while not stop_event.is_set():
            try:
                self.step()
                delay = self.cfg.live.poll_seconds
                self.status["error"] = ""
            except Exception as exc:  # keep the bot alive; report and back off
                log.exception("step failed")
                self.status["error"] = f"{exc.__class__.__name__}: {exc}"
                delay = min(max(delay * 2, 10), 300)
                try:
                    self.broker.connect()
                except Exception:
                    pass
            stop_event.wait(delay)
        self.status["state"] = "stopped"

    # ------------------------------------------------------------------ main step
    def step(self) -> bool:
        cfg = self.cfg
        df = self.broker.candles(cfg.symbol.history_bars)
        self._sync_closed()
        if len(df) < self.strategy.warmup + 10:
            self.status["error"] = f"only {len(df)} bars of history; need {self.strategy.warmup + 10}"
            return False
        last_t = df.index[-1]
        if self.last_bar is not None and last_t <= self.last_bar:
            return False
        self.last_bar = last_t
        self.bar_count += 1
        now = last_t + self.bar
        self.status["last_bar"] = str(last_t)

        acct = self.broker.account()
        self.account_info = acct
        tick = self.broker.tick()
        self.risk.update(now, acct.equity)
        self.journal.log_equity(now, acct.balance, acct.equity)

        feats = Features(df, cfg.symbol.timeframe)
        sig = self.strategy.generate(feats)
        atr = float(feats.atr(14).iloc[-1])
        bar_spread = float(df["spread"].iloc[-1]) if "spread" in df.columns else tick.spread

        for p in self.broker.positions():  # trade management always runs first
            self._manage(p, df, atr, now, bar_spread)
        self._sync_closed()
        self._save_risk()
        self._resolve_shadows(df)
        if self.calendar is not None:
            try:
                self.calendar.refresh()  # rate-limited inside; never raises
            except Exception as exc:  # belt and braces: news must never stop the bot
                self.calendar.error = f"news calendar error: {exc}"
            self.status["news_warning"] = self.calendar.error

        if self.pause_entries or Path(cfg.live.stop_file).exists():
            self._decide(last_t, final="PAUSED", reason="entries paused (STOP file or panel)")
            return True
        self._maybe_enter(df, feats, sig.iloc[-1], atr, now, acct, tick)
        self._save_risk()
        return True

    # ------------------------------------------------------------------ helpers
    def _save_risk(self) -> None:
        self.journal.set_kv(f"risk_state:{self.mode_label}", self.risk.state.to_dict())

    def _decide(self, bar_time, final: str = "HOLD", reason: str = "", **kw) -> None:
        self.journal.log_decision(bar_time, final_action=final, reason=reason, **kw)
        self.status["last_decision"] = {"bar": str(bar_time), "action": final, "reason": reason,
                                        **{k: v for k, v in kw.items() if k != "extra"}}

    def _sync_closed(self) -> None:
        open_now = {p.ticket for p in self.broker.positions()}
        for t in list(self.meta):
            if t not in open_now and t not in self.pending_close:
                self.pending_close[t] = 0
        for t in list(self.pending_close):
            info = self.broker.closed_trade(t)
            if info is None:
                self.pending_close[t] += 1
                if self.pending_close[t] < 20:
                    continue
                log.warning("No closing deal found for %s; recording pnl 0", t)
                self.journal.close_trade(t, self.broker.now(), 0.0, 0.0, "unknown")
                pnl, when = 0.0, self.broker.now()
            else:
                reason = self.close_reasons.pop(t, info.reason)
                self.journal.close_trade(t, info.exit_time, info.exit_price, info.pnl, reason)
                pnl, when = info.pnl, info.exit_time
                log.info("Closed %s: %s pnl %.2f", t, reason, info.pnl)
            self.risk.on_trade_closed(pnl, when)
            self.meta.pop(t, None)
            self.pending_close.pop(t, None)

    def _manage(self, p: BrokerPosition, df: pd.DataFrame, atr: float, now: pd.Timestamp, spread: float) -> None:
        meta = self.meta.get(p.ticket, {})
        init = float(meta.get("initial_sl_dist") or abs(p.entry - p.sl) or 0.0)
        bars_held = int(round((now - p.open_time) / self.bar))
        close = float(df["close"].iloc[-1])
        price = close if p.side > 0 else close + spread
        act = manage_position(p.side, p.entry, p.sl, init, bars_held, price, atr, now, self.cfg.management)
        if act.kind == "close":
            res = self.broker.close_position(p.ticket)
            log.info("Closing %s (%s): %s", p.ticket, act.reason, "ok" if res.ok else res.message)
            if res.ok:
                self.close_reasons[p.ticket] = act.reason
                self.pending_close.setdefault(p.ticket, 0)
        elif act.kind == "modify" and act.new_sl is not None:
            tick = self.broker.tick()
            gap = self.broker.spec().stops_level
            valid = act.new_sl < tick.bid - gap if p.side > 0 else act.new_sl > tick.ask + gap
            if not valid:
                return
            res = self.broker.modify_sl_tp(p.ticket, act.new_sl, p.tp)
            if res.ok:
                self.journal.update_trade_sl(p.ticket, act.new_sl)
            else:
                log.warning("SL update for %s failed: %s", p.ticket, res.message)

    def _resolve_shadows(self, df: pd.DataFrame) -> None:
        shadows = self.journal.unresolved_shadows()
        if not shadows:
            return
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        spread = spread_array(df, self.cfg.backtest.spread)
        max_bars = self.cfg.management.max_bars_in_trade or 96
        for s in shadows:
            t = pd.Timestamp(s["bar_time"])
            pos = df.index.searchsorted(t)
            if pos >= len(df) or df.index[pos] != t:
                if t < df.index[0]:
                    self.journal.resolve_shadow(s["id"], float("nan"), "expired")
                continue
            res = simulate_trade(o, h, l, c, spread, pos + 1, int(s["side"]), float(s["sl_dist"]), float(s["tp_dist"]),
                                 self.cfg.backtest.slippage, max_bars)
            if res is not None:
                self.journal.resolve_shadow(s["id"], res[4], res[3])

    def _other_markets(self) -> dict:
        if not self.cfg.ai.context_symbols:
            return {}
        if self._ctx_symbols is None:
            suffix = self.broker.symbol[6:] if self.broker.symbol.upper().startswith("XAUUSD") else ""
            found = []
            for name in self.cfg.ai.context_symbols:
                for cand in (name, name + suffix):
                    if self.broker.candles_for(cand, "H1", 30) is not None:
                        found.append(cand)
                        break
            self._ctx_symbols = found
        return {s: self.broker.candles_for(s, "H1", 200) for s in self._ctx_symbols}

    def _maybe_enter(self, df: pd.DataFrame, feats: Features, row: pd.Series, atr: float, now: pd.Timestamp,
                     acct: AccountInfo, tick: Tick) -> None:
        cfg = self.cfg
        last_t = df.index[-1]
        signal = int(row["signal"])
        ai_mode = cfg.ai.mode if self.brain is not None else "off"
        autonomous_due = ai_mode == "autonomous" and self.bar_count % max(1, cfg.ai.autonomous_every_bars) == 0
        if signal == 0 and not autonomous_due:
            self._decide(last_t, reason="no signal")
            return

        positions = self.broker.positions()
        news = None
        if self.calendar is not None:
            news = self.calendar.blackout(now, cfg.risk.news_blackout_before_min, cfg.risk.news_blackout_after_min)
        ok, why = self.risk.can_open(now, acct.equity, len(positions), tick.spread, atr, news)
        strategy = str(row["strategy"]) if signal else ""
        if not ok:
            self._decide(last_t, signal=signal, strategy=strategy, reason=why)
            return

        intent = None
        if signal:
            intent = {"signal": signal, "sl_dist": float(row["sl_dist"]), "tp_dist": float(row["tp_dist"]),
                      "strength": float(row["strength"]), "strategy": strategy}

        ml_prob = None
        if intent and self.ml is not None:
            from .ml.meta_label import feature_frame

            ml_prob = self.ml.predict(feature_frame(feats).iloc[-1], signal, intent["sl_dist"], intent["tp_dist"], atr, strategy)
            threshold = cfg.ml.threshold or self.ml.threshold
            if np.isfinite(ml_prob) and ml_prob < threshold:
                self.journal.add_shadow(last_t, signal, intent["sl_dist"], intent["tp_dist"], "ml_veto", strategy)
                self._decide(last_t, signal=signal, strategy=strategy, ml_prob=ml_prob,
                             reason=f"ML filter: p={ml_prob:.2f} < {threshold:.2f}")
                return

        risk_mult = 1.0
        ai_conf, ai_reason, ai_action = None, "", ""
        if ai_mode != "off":
            dec = self._ask_ai(df, feats, intent, ml_prob, atr, now, acct, tick)
            ai_conf, ai_reason, ai_action = dec.confidence, dec.reasoning, dec.action
            if dec.error:
                if not (intent and cfg.ai.on_error == "quant"):
                    self._decide(last_t, signal=signal, strategy=strategy, ml_prob=ml_prob,
                                 reason=f"AI unavailable: {dec.error}")
                    return
                ai_reason = f"(AI error, trading quant signal) {dec.error}"
            elif ai_mode == "filter" or (intent and dec.side == intent["signal"]):
                if not intent:
                    return
                if dec.side != intent["signal"] or dec.confidence < cfg.ai.min_confidence:
                    self.journal.add_shadow(last_t, signal, intent["sl_dist"], intent["tp_dist"], "ai_veto", strategy)
                    self._decide(last_t, signal=signal, strategy=strategy, ml_prob=ml_prob, ai_action=dec.action,
                                 ai_confidence=dec.confidence, ai_reasoning=dec.reasoning, reason="AI veto")
                    return
                intent["sl_dist"] = dec.sl_atr * atr if dec.sl_atr > 0 else intent["sl_dist"]
                intent["tp_dist"] = dec.tp_atr * atr if dec.tp_atr > 0 else intent["tp_dist"]
                risk_mult = dec.risk_multiplier
            else:  # autonomous: Claude's call is final
                if dec.side == 0 or dec.confidence < cfg.ai.min_confidence:
                    if intent:
                        self.journal.add_shadow(last_t, signal, intent["sl_dist"], intent["tp_dist"], "ai_veto", strategy)
                    self._decide(last_t, signal=signal, strategy=strategy, ml_prob=ml_prob, ai_action=dec.action,
                                 ai_confidence=dec.confidence, ai_reasoning=dec.reasoning, reason="AI hold")
                    return
                intent = {"signal": dec.side, "sl_dist": dec.sl_atr * atr, "tp_dist": dec.tp_atr * atr,
                          "strength": dec.confidence, "strategy": "claude"}
                signal, strategy = dec.side, "claude"
                risk_mult = dec.risk_multiplier

        if intent is None:
            self._decide(last_t, reason="no trade")
            return
        self._execute(last_t, now, intent, acct, tick, atr, risk_mult, ml_prob, ai_conf, ai_reason, ai_action)

    def _ask_ai(self, df, feats, intent, ml_prob, atr, now, acct, tick) -> AIDecision:
        cfg = self.cfg
        if self.headlines is not None:
            self.headlines.refresh()
        events = self.calendar.upcoming(now, 24, all_impacts=True) if self.calendar is not None else None
        heads = self.headlines.relevant(now) if self.headlines is not None else None
        recent = self.journal.recent_closed_trades(8)
        positions = [p.__dict__ for p in self.broker.positions()]
        from .strategies.session_breakout import SessionBreakout

        bo = {**SessionBreakout.default_params, **cfg.strategy.params.get("session_breakout", {})}
        context = build_context(
            now=now, symbol=self.broker.symbol, timeframe=cfg.symbol.timeframe, candles=df, feats=feats,
            mode=cfg.ai.mode, signal=intent, ml_prob=ml_prob, spread=tick.spread, account=acct.__dict__,
            positions=positions, risk_state=self.risk.state.to_dict(), events=events, headlines=heads,
            recent_trades=recent, other_markets=self._other_markets(),
            session_tz=bo["session_tz"], range_hours=(bo["range_start_hour"], bo["range_end_hour"]),
        )
        dec = self.brain.decide(context)
        self.journal.log_ai_call(dec.model or cfg.ai.model, cfg.ai.mode, dec.action, dec.confidence, dec.usage,
                                 dec.latency_s, dec.error, dec.reasoning)
        self.status["last_ai"] = {"time": str(now), "action": dec.action, "confidence": dec.confidence,
                                  "reasoning": dec.reasoning, "risks": dec.key_risks, "bias": dec.market_bias,
                                  "error": dec.error}
        log.info("AI %s conf %.2f: %s%s", dec.action, dec.confidence, dec.reasoning[:160],
                 f" [error: {dec.error}]" if dec.error else "")
        return dec

    def _execute(self, last_t, now, intent, acct, tick, atr, risk_mult, ml_prob, ai_conf, ai_reason, ai_action) -> None:
        side, strategy = int(intent["signal"]), intent["strategy"]
        ok, why, sl_d, tp_d = self.risk.validate_levels(intent["sl_dist"], intent["tp_dist"], atr)
        spec = self.broker.spec()
        if ok and sl_d < spec.stops_level + tick.spread:
            ok, why = False, f"stop {sl_d:.2f} closer than broker minimum {spec.stops_level + tick.spread:.2f}"
        common = dict(signal=side, strategy=strategy, ml_prob=ml_prob, ai_action=ai_action, ai_confidence=ai_conf,
                      ai_reasoning=ai_reason)
        if not ok:
            self._decide(last_t, reason=why, **common)
            return
        ref = tick.ask if side > 0 else tick.bid
        sl = round(ref - side * sl_d, spec.digits)
        tp = round(ref + side * tp_d, spec.digits)
        loss_per_lot = self.broker.loss_per_lot(side, ref, sl)
        lots, why = self.risk.position_size(acct.equity, loss_per_lot, spec.volume_min, spec.volume_step,
                                            spec.volume_max, risk_mult)
        if lots <= 0:
            self._decide(last_t, reason=f"size: {why}", **common)
            return
        res = self.broker.open_market(side, lots, sl, tp, f"{self.cfg.account.comment} {strategy[:12]}")
        if not res.ok:
            self._decide(last_t, reason=f"order failed: {res.message}", **common)
            log.error("Order failed: %s", res.message)
            return
        self.risk.on_trade_opened(now)
        entry = res.price or ref
        self.journal.open_trade(res.ticket, self.mode_label, self.broker.symbol, side, lots, now, entry, sl, tp, sl_d,
                                strategy, ai_conf, ml_prob, lots * loss_per_lot,
                                extra={"atr": atr, "risk_mult": risk_mult, "ai_reasoning": ai_reason})
        self.meta[res.ticket] = dict(self.journal.trade_by_ticket(res.ticket))
        action = "BUY" if side > 0 else "SELL"
        self._decide(last_t, final=action, reason=f"{lots} lots, SL {sl}, TP {tp}", **common)
        log.info("%s %s lots @ %.2f SL %.2f TP %.2f (%s)", action, lots, entry, sl, tp, strategy)

    # ------------------------------------------------------------------ manual controls
    def close_all(self) -> int:
        n = 0
        for p in self.broker.positions():
            if self.broker.close_position(p.ticket).ok:
                self.close_reasons[p.ticket] = "panel close-all"
                n += 1
        self._sync_closed()
        return n


def run_replay(df: pd.DataFrame, cfg: Config, journal_path: str | Path, start: Optional[int] = None,
               brain: Optional[ClaudeBrain] = None, ml=None, progress: bool = False) -> LiveEngine:
    """Drive the live engine bar-by-bar over historical data with simulated fills."""
    from .broker.replay import ReplayBroker

    broker = ReplayBroker(df, cfg, start_index=start if start is not None else cfg.backtest.warmup_bars)
    engine = LiveEngine(cfg, broker, Journal(journal_path), brain=brain, ml=ml, mode_label="replay")
    engine.start()
    t0, n = time.time(), 0
    while broker.has_more():
        engine.step()
        broker.advance()
        n += 1
        if progress and n % 2000 == 0:
            print(f"  replay: {n} bars ({time.time() - t0:.0f}s)", flush=True)
    broker.flatten_at_close()  # same convention as the backtester
    engine._sync_closed()
    return engine
