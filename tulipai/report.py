"""Self-contained HTML performance report (backtest, walk-forward or live journal)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .backtest.engine import BacktestResult
from .backtest.metrics import compute_metrics

WEB = Path(__file__).resolve().parent / "web"


def _points(s: pd.Series, n: int = 800) -> list:
    if s is None or len(s) == 0:
        return []
    s = s.dropna()
    if len(s) > n:
        s = s.iloc[np.unique(np.linspace(0, len(s) - 1, n).astype(int))]
    return [[int(pd.Timestamp(t).value // 1_000_000), round(float(v), 4)] for t, v in s.items()]


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def monthly_returns(equity: pd.Series, initial: float) -> list[dict]:
    if equity is None or len(equity) == 0:
        return []
    m = equity.resample("ME").last().dropna()
    prev = pd.concat([pd.Series([initial]), m.iloc[:-1].reset_index(drop=True)]).to_numpy()
    return [{"month": t.strftime("%Y-%m"), "return_pct": (v / p - 1) * 100} for t, v, p in zip(m.index, m.to_numpy(), prev)]


def build_report(
    result: BacktestResult,
    title: str,
    subtitle: str = "",
    buy_hold: Optional[pd.Series] = None,
    random_bench: Optional[dict] = None,
    windows: Optional[list] = None,
    notes: Optional[list[str]] = None,
    currency: str = "",
) -> str:
    m = compute_metrics(result.trades, result.equity, result.initial_balance)
    eq = result.equity.dropna()
    peak = np.maximum.accumulate(np.concatenate([[result.initial_balance], eq.to_numpy()]))[1:] if len(eq) else []
    dd = pd.Series(eq.to_numpy() / peak * 100 - 100, index=eq.index) if len(eq) else pd.Series(dtype=float)
    trades = result.trades.copy()
    for c in ("signal_time", "entry_time", "exit_time"):
        if c in trades:
            trades[c] = trades[c].astype(str).str.slice(0, 16)
    payload = {
        "title": title, "subtitle": subtitle, "currency": currency, "metrics": m,
        "equity": _points(eq), "buy_hold": _points(buy_hold) if buy_hold is not None else [],
        "drawdown": _points(dd), "random": random_bench or {}, "windows": windows or [],
        "monthly": monthly_returns(eq, result.initial_balance), "blocked": result.blocked,
        "trades": trades.tail(300).drop(columns=["signal_idx"], errors="ignore").to_dict("records"),
        "notes": notes or [],
    }
    data = json.dumps(_clean(payload), default=str).replace("</", "<\\/")
    page = (WEB / "report.html").read_text(encoding="utf-8")
    return (page.replace("/*__THEME__*/", (WEB / "theme.css").read_text(encoding="utf-8"))
                .replace("/*__CHARTS__*/", (WEB / "charts.js").read_text(encoding="utf-8"))
                .replace("__TITLE__", html.escape(title))
                .replace("__DATA__", data))


def write_report(path: str | Path, page: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    return p


def journal_result(journal, mode: Optional[str] = None) -> BacktestResult:
    """Turn the live journal into a BacktestResult so live trading gets the same report."""
    t = journal.frame("trades")
    if mode and len(t):
        t = t[t["mode"] == mode]
    closed = t[t["close_time"].notna()].copy() if len(t) else t
    eqf = journal.frame("equity")
    if len(eqf):
        eq = pd.Series(eqf["equity"].to_numpy(dtype=float), index=pd.to_datetime(eqf["ts"], utc=True, format="ISO8601"))
        eq = eq[~eq.index.duplicated(keep="last")]
        initial = float(eqf["balance"].iloc[0])
    else:
        eq, initial = pd.Series(dtype=float), 0.0
    if len(closed):
        trades = pd.DataFrame({
            "id": closed["id"], "side": closed["side"].astype(int), "strategy": closed["strategy"],
            "signal_time": closed["open_time"], "entry_time": closed["open_time"], "exit_time": closed["close_time"],
            "entry": closed["entry"], "exit": closed["exit"], "sl_initial": closed["entry"] - closed["side"] * closed["initial_sl_dist"],
            "tp": closed["tp"], "lots": closed["volume"], "sl_dist": closed["initial_sl_dist"], "tp_dist": np.nan,
            "pnl": closed["pnl"], "r_multiple": closed["r_multiple"], "exit_reason": closed["exit_reason"],
            "bars_held": 0, "signal_idx": 0, "ml_prob": closed["ml_prob"],
        })
    else:
        trades = pd.DataFrame(columns=["id", "side", "strategy", "pnl", "r_multiple", "exit_reason", "bars_held"])
    return BacktestResult(trades.reset_index(drop=True), eq, initial or 1.0, label="live journal")
