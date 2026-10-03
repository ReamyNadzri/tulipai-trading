"""Live performance statistics from the journal (used by the panel and `tulip report`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest.metrics import compute_metrics
from .config import AIConfig
from .journal import Journal


def journal_trades(j: Journal, mode: str | None = None) -> pd.DataFrame:
    t = j.frame("trades")
    if mode and len(t):
        t = t[t["mode"] == mode]
    return t


def journal_stats(j: Journal, mode: str | None = None, ai_cfg: AIConfig | None = None) -> dict:
    trades = journal_trades(j, mode)
    closed = trades[trades["close_time"].notna()].copy() if len(trades) else trades
    eq = j.frame("equity")
    out: dict = {"trades_total": int(len(closed)), "open_trades": int(len(trades) - len(closed)) if len(trades) else 0}

    if len(eq):
        eq_series = pd.Series(eq["equity"].to_numpy(dtype=float), index=pd.to_datetime(eq["ts"], utc=True, format="ISO8601"))
        eq_series = eq_series[~eq_series.index.duplicated(keep="last")]
        initial = float(eq["balance"].iloc[0])
    else:
        eq_series, initial = pd.Series(dtype=float), 0.0

    if len(closed):
        closed["side"] = closed["side"].astype(int)
        closed["bars_held"] = 0
        m = compute_metrics(closed.rename(columns={"exit": "exit_price"}), eq_series if len(eq_series) else
                            pd.Series([initial], index=[pd.Timestamp.now(tz="UTC")]), initial or 1.0)
        out.update({k: m[k] for k in ("win_rate", "profit_factor", "avg_r", "total_r", "max_dd_pct", "by_strategy",
                                      "exit_reasons", "max_consec_losses")})
        out["net_pnl"] = float(closed["pnl"].sum())
        today = pd.Timestamp.now(tz="UTC").normalize()
        ct = pd.to_datetime(closed["close_time"], utc=True, format="ISO8601")
        out["today_pnl"] = float(closed.loc[ct >= today, "pnl"].sum())
        out["today_trades"] = int((ct >= today).sum())
    else:
        out.update({"win_rate": 0.0, "profit_factor": 0.0, "avg_r": 0.0, "total_r": 0.0, "max_dd_pct": 0.0,
                    "net_pnl": 0.0, "today_pnl": 0.0, "today_trades": 0, "by_strategy": {}, "exit_reasons": {},
                    "max_consec_losses": 0})
    if out["profit_factor"] == float("inf"):
        out["profit_factor"] = None

    ai = j.frame("ai_calls")
    if len(ai):
        ts = pd.to_datetime(ai["ts"], utc=True, format="ISO8601")
        today = ts >= pd.Timestamp.now(tz="UTC").normalize()
        pin = (ai_cfg.price_input_per_mtok if ai_cfg else 4.0) / 1e6
        pout = (ai_cfg.price_output_per_mtok if ai_cfg else 20.0) / 1e6
        cost = ai["input_tokens"] * pin + ai["output_tokens"] * pout + ai["cache_read_tokens"] * pin * 0.1
        out["ai"] = {
            "calls_total": int(len(ai)), "calls_today": int(today.sum()),
            "errors": int((ai["error"].fillna("") != "").sum()),
            "est_cost_total_usd": round(float(cost.sum()), 2), "est_cost_today_usd": round(float(cost[today].sum()), 2),
            "avg_latency_s": round(float(ai["latency_s"].mean()), 1),
        }
    else:
        out["ai"] = {"calls_total": 0, "calls_today": 0, "errors": 0, "est_cost_total_usd": 0.0,
                     "est_cost_today_usd": 0.0, "avg_latency_s": 0.0}

    sh = j.frame("shadows")
    vetoes = {}
    if len(sh):
        for src, g in sh.groupby("source"):
            res = g[g["resolved"] == 1]["r_multiple"].dropna()
            vetoes[str(src)] = {"count": int(len(g)), "resolved": int(len(res)),
                                "avg_r_if_taken": float(res.mean()) if len(res) else None,
                                "total_r_avoided": float(-res.sum()) if len(res) else 0.0}
    out["vetoes"] = vetoes
    out["equity_curve"] = [[int(t.value // 1_000_000), float(v)] for t, v in _downsample(eq_series, 600).items()]
    return out


def _downsample(s: pd.Series, n: int) -> pd.Series:
    if len(s) <= n:
        return s
    idx = np.unique(np.linspace(0, len(s) - 1, n).astype(int))
    return s.iloc[idx]
