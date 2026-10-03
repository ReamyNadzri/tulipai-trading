"""Risk manager: the non-negotiable guardrails.

Neither a strategy nor the AI can bypass these checks. The same object is used by the
backtester, the replay engine and live trading, so a backtest obeys exactly the limits the
live bot will obey.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import RiskConfig


@dataclass
class RiskState:
    day: Optional[str] = None
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    cooldown_until: Optional[str] = None  # ISO timestamp
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RiskState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def volume_decimals(step: float) -> int:
    return max(0, int(round(-math.log10(step)))) if step < 1 else 0


def in_blackout(events: np.ndarray, now: pd.Timestamp, before_min: int, after_min: int) -> bool:
    """``events`` is a sorted numpy datetime64[ns] array of high-impact event times (UTC)."""
    if events is None or len(events) == 0:
        return False
    now64 = np.datetime64(now.tz_convert("UTC").tz_localize(None), "ns")
    lo = now64 - np.timedelta64(int(after_min), "m")
    hi = now64 + np.timedelta64(int(before_min), "m")
    k = np.searchsorted(events, lo, side="left")
    return bool(k < len(events) and events[k] <= hi)


class RiskManager:
    def __init__(self, cfg: RiskConfig, tf_minutes: int, state: RiskState | None = None):
        self.cfg = cfg
        self.bar = pd.Timedelta(minutes=tf_minutes)
        self.state = state or RiskState()

    # ----------------------------------------------------------------- state updates
    def update(self, now: pd.Timestamp, equity: float) -> None:
        st = self.state
        day = now.strftime("%Y-%m-%d")
        if st.day != day:
            st.day = day
            st.day_start_equity = equity
            st.trades_today = 0
        if equity > st.peak_equity:
            st.peak_equity = equity
        if not st.halted and st.peak_equity > 0 and equity <= st.peak_equity * (1 - self.cfg.max_drawdown_pct / 100):
            st.halted = True
            st.halt_reason = (
                f"max drawdown {self.cfg.max_drawdown_pct}% hit (equity {equity:.2f} vs peak {st.peak_equity:.2f}) "
                "- trading halted until you reset it"
            )

    def on_trade_opened(self, now: pd.Timestamp) -> None:
        self.state.trades_today += 1

    def on_trade_closed(self, pnl: float, now: pd.Timestamp) -> None:
        st = self.state
        if pnl < 0:
            st.consecutive_losses += 1
            if self.cfg.loss_streak_pause and st.consecutive_losses >= self.cfg.loss_streak_pause:
                st.cooldown_until = (now + self.cfg.cooldown_bars * self.bar).isoformat()
                st.consecutive_losses = 0
        elif pnl > 0:
            st.consecutive_losses = 0

    def reset_halt(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.peak_equity = 0.0

    # ----------------------------------------------------------------- checks
    def daily_loss_hit(self, equity: float) -> bool:
        st = self.state
        return st.day_start_equity > 0 and equity <= st.day_start_equity * (1 - self.cfg.max_daily_loss_pct / 100)

    def in_session(self, now: pd.Timestamp) -> bool:
        c = self.cfg
        if now.weekday() not in c.trade_days:
            return False
        if now.weekday() == 4 and now.hour >= c.friday_cutoff_hour_utc:
            return False
        return any(start <= now.hour < end for start, end in c.trade_hours_utc)

    def can_open(
        self,
        now: pd.Timestamp,
        equity: float,
        open_positions: int,
        spread: float,
        atr: float,
        news_block: str | None = None,
    ) -> tuple[bool, str]:
        c, st = self.cfg, self.state
        if st.halted:
            return False, "halted: " + st.halt_reason
        if self.daily_loss_hit(equity):
            return False, f"daily loss limit {c.max_daily_loss_pct}% reached"
        if not self.in_session(now):
            return False, "outside trading session"
        if open_positions >= c.max_open_positions:
            return False, "max open positions"
        if st.trades_today >= c.max_trades_per_day:
            return False, "max trades per day"
        if st.cooldown_until and now < pd.Timestamp(st.cooldown_until):
            return False, f"cooldown after {c.loss_streak_pause} losses"
        if not (atr > 0 and np.isfinite(atr)):
            return False, "ATR unavailable"
        if spread > c.max_spread or spread > c.max_spread_atr * atr:
            return False, f"spread too wide ({spread:.2f})"
        if news_block:
            return False, f"news blackout: {news_block}"
        return True, "ok"

    def validate_levels(self, sl_dist: float, tp_dist: float, atr: float) -> tuple[bool, str, float, float]:
        """Clamp the stop into [min_sl_atr, max_sl_atr] x ATR (keeping the reward:risk) and
        reject trades whose reward:risk is below ``min_rr``."""
        c = self.cfg
        if not (sl_dist > 0 and tp_dist > 0 and atr > 0):
            return False, "invalid SL/TP", sl_dist, tp_dist
        rr = tp_dist / sl_dist
        sl = min(max(sl_dist, c.min_sl_atr * atr), c.max_sl_atr * atr)
        tp = sl * rr
        if rr < c.min_rr - 1e-9:
            return False, f"reward:risk {rr:.2f} < {c.min_rr}", sl, tp
        return True, "ok", sl, tp

    def position_size(
        self,
        equity: float,
        loss_per_lot: float,
        vol_min: float,
        vol_step: float,
        vol_max: float,
        risk_multiplier: float = 1.0,
    ) -> tuple[float, str]:
        """Lots such that hitting the stop loses ``risk_per_trade_pct`` of equity.

        ``loss_per_lot`` is the account-currency loss of 1.0 lot at the stop (including
        commission). For an MT5 cent account this comes from ``order_calc_profit`` and is in
        USC, so a 10,000 USC balance with 1% risk budgets 100 USC (= 1 USD) per trade.
        """
        c = self.cfg
        mult = min(max(risk_multiplier, 0.0), 1.0)
        budget = equity * c.risk_per_trade_pct / 100.0 * mult
        if loss_per_lot <= 0 or not np.isfinite(loss_per_lot):
            return 0.0, "invalid loss per lot"
        if budget <= 0:
            return 0.0, "no risk budget"
        dec = volume_decimals(vol_step)
        cap = min(c.max_lot, vol_max)
        lots = math.floor(min(budget / loss_per_lot, cap) / vol_step + 1e-9) * vol_step
        lots = round(lots, dec)
        if lots < vol_min:
            if vol_min * loss_per_lot <= budget * c.min_lot_risk_overshoot and vol_min <= cap:
                return round(vol_min, dec), "ok (minimum lot)"
            return 0.0, f"min lot {vol_min} risks {vol_min * loss_per_lot:.2f} > budget {budget:.2f}"
        return lots, "ok"
