"""Claude as the trading decision layer.

Claude reads the market briefing and returns a structured decision (JSON schema enforced
by the API). The decision is advisory: position size is always computed by the risk
manager, and every limit in risk.py still applies to whatever Claude proposes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import pandas as pd

from ..config import AIConfig, ManagementConfig, RiskConfig

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "number"},
        "sl_atr": {"type": "number"},
        "tp_atr": {"type": "number"},
        "risk_multiplier": {"type": "number"},
        "market_bias": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "reasoning": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "confidence", "sl_atr", "tp_atr", "risk_multiplier", "market_bias", "reasoning", "key_risks"],
    "additionalProperties": False,
}

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AIDecision:
    action: str = "HOLD"
    confidence: float = 0.0
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    risk_multiplier: float = 1.0
    market_bias: str = "neutral"
    reasoning: str = ""
    key_risks: list = field(default_factory=list)
    error: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    model: str = ""

    @property
    def side(self) -> int:
        return {"BUY": 1, "SELL": -1}.get(self.action, 0)


def system_prompt(ai: AIConfig, risk: RiskConfig, mg: ManagementConfig, timeframe: str) -> str:
    hours = ", ".join(f"{a:02d}:00-{b:02d}:00" for a, b in risk.trade_hours_utc)
    return f"""You are the decision layer of TulipAI, an autonomous trading system for spot gold (XAUUSD) on a MetaTrader 5 demo or cent account. Each request is a market briefing taken at the close of a {timeframe} bar. Return one decision for the next bar: BUY, SELL or HOLD, with stop-loss and take-profit distances as multiples of the current ATR(14) on {timeframe}.

How the system around you works:
- Code computes indicators and runs rule-based strategies: Asian-range breakout at the London open, trend-following pullbacks, and range mean reversion.
- A risk manager you cannot override sizes every position from the stop distance ({risk.risk_per_trade_pct}% of equity at risk per trade), clamps stops to {risk.min_sl_atr}-{risk.max_sl_atr} ATR, rejects reward:risk below {risk.min_rr}, enforces a {risk.max_daily_loss_pct}% daily loss limit, max {risk.max_open_positions} open position(s), trading hours {hours} UTC, and a news blackout around high-impact USD releases.
- Code manages open trades: stop to break-even at +{mg.breakeven_at_r}R, ATR trailing stop from +{mg.trail_start_r}R, time stop, flat before the weekend.

Modes:
- FILTER: a strategy proposed a trade. Approve it by returning the same direction (you may adjust sl_atr/tp_atr and lower risk_multiplier), or veto it with HOLD. The opposite direction is treated as HOLD.
- AUTONOMOUS: no strategy signal is needed. Propose a trade only when trend, levels and catalysts line up; otherwise HOLD.

What drives gold intraday:
- Real yields and the US dollar are the dominant macro drivers: rising real yields or a stronger dollar weigh on gold. Fed expectations, inflation prints and geopolitical risk move them.
- Trade with the H4/H1 trend unless price is at a clear range extreme. The London open and the London/New York overlap carry most of the volume; Asian-session breakouts often fail.
- Around NFP, CPI, PCE and FOMC spreads blow out and price whipsaws. If one is due within about an hour, HOLD even if the blackout has not started. After a news spike, wait for structure rather than chasing.

Calibrate confidence honestly: 0.5 is a coin flip; 0.7 means you expect to win clearly more often than lose at the proposed reward:risk. The system only acts when confidence >= {ai.min_confidence}. HOLD is a good answer when evidence conflicts: a skipped trade costs nothing. Use the recent trade results as feedback; if a setup keeps failing in current conditions, lower risk_multiplier or HOLD.

Fields: sl_atr in [{risk.min_sl_atr}, {risk.max_sl_atr}]; tp_atr / sl_atr >= {risk.min_rr}; risk_multiplier in [0.25, 1.0] (1.0 = normal risk); for HOLD use sl_atr = tp_atr = 0. reasoning: at most 80 words, concrete (trend, levels, catalysts). key_risks: up to 3 short items."""


class ClaudeBrain:
    def __init__(self, ai: AIConfig, risk: RiskConfig, mg: ManagementConfig, timeframe: str, client=None):
        self.cfg = ai
        self.risk = risk
        self.system = system_prompt(ai, risk, mg, timeframe)
        self._client = client
        self._fallbacks_ok = ai.use_fallbacks
        self._day = ""
        self.calls_today = 0

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic package not installed (pip install anthropic)"
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False, "ANTHROPIC_API_KEY is not set (put it in .env or the control panel)"
        return True, "ok"

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.cfg.timeout_s, max_retries=2)
        return self._client

    def _budget_ok(self) -> bool:
        day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        if day != self._day:
            self._day, self.calls_today = day, 0
        return self.calls_today < self.cfg.max_calls_per_day

    def _create(self, context: str):
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": context}],
            output_config={"effort": self.cfg.effort, "format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
        )
        if self._fallbacks_ok:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def decide(self, context: str) -> AIDecision:
        if not self._budget_ok():
            return AIDecision(error=f"daily AI call budget ({self.cfg.max_calls_per_day}) reached")
        self.calls_today += 1
        import anthropic

        t0 = time.time()
        try:
            try:
                resp = self._create(context)
            except anthropic.BadRequestError as exc:
                if self._fallbacks_ok and "fallback" in str(exc).lower():
                    self._fallbacks_ok = False  # account/model without server-side fallbacks
                    resp = self._create(context)
                else:
                    raise
        except anthropic.AuthenticationError:
            return AIDecision(error="invalid Anthropic API key", latency_s=time.time() - t0)
        except anthropic.RateLimitError:
            return AIDecision(error="rate limited by the Claude API", latency_s=time.time() - t0)
        except anthropic.APIStatusError as exc:
            return AIDecision(error=f"Claude API error {exc.status_code}: {exc.message}", latency_s=time.time() - t0)
        except anthropic.APIConnectionError:
            return AIDecision(error="cannot reach the Claude API (network)", latency_s=time.time() - t0)
        return self.parse(resp, time.time() - t0)

    def parse(self, resp, latency: float) -> AIDecision:
        usage = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                             "cache_creation_input_tokens")}
        model = str(getattr(resp, "model", self.cfg.model))
        if getattr(resp, "stop_reason", "") == "refusal":
            return AIDecision(error="model declined this request", usage=usage, latency_s=latency, model=model)
        if getattr(resp, "stop_reason", "") == "max_tokens":
            return AIDecision(error="response hit max_tokens", usage=usage, latency_s=latency, model=model)
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return AIDecision(error=f"unparseable AI output: {text[:200]!r}", usage=usage, latency_s=latency, model=model)
        return self.sanitize(data, usage, latency, model)

    def sanitize(self, data: dict, usage: dict | None = None, latency: float = 0.0, model: str = "") -> AIDecision:
        r = self.risk

        def num(key, lo, hi, default):
            try:
                v = float(data.get(key, default))
            except (TypeError, ValueError):
                v = default
            return min(max(v, lo), hi)

        action = str(data.get("action", "HOLD")).upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        d = AIDecision(
            action=action,
            confidence=num("confidence", 0.0, 1.0, 0.0),
            sl_atr=num("sl_atr", r.min_sl_atr, r.max_sl_atr, r.min_sl_atr) if action != "HOLD" else 0.0,
            tp_atr=num("tp_atr", 0.0, 20.0, 0.0) if action != "HOLD" else 0.0,
            risk_multiplier=num("risk_multiplier", 0.25, 1.0, 1.0),
            market_bias=str(data.get("market_bias", "neutral")),
            reasoning=str(data.get("reasoning", ""))[:800],
            key_risks=[str(k)[:120] for k in (data.get("key_risks") or [])][:3],
            usage=usage or {},
            latency_s=latency,
            model=model or self.cfg.model,
        )
        return d

    def review(self, report_text: str) -> str:
        """Free-form performance review used by `tulip review` (not part of the trading loop)."""
        prompt = (
            "Below is the performance record of the TulipAI gold trading system (backtest and/or live journal). "
            "Write a concise review in Markdown: what is working, what is not (by strategy, session, exit reason, "
            "AI/ML veto value), whether the edge is statistically credible given the random-entry benchmark, and up "
            "to 5 concrete, testable parameter or rule changes ranked by expected impact. Be candid; do not "
            "recommend going live with real money unless the out-of-sample evidence supports it.\n\n" + report_text
        )
        resp = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=16000,
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return "The model declined to write this review."
        return "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
