"""Performance metrics for backtests, replays and the live journal."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def equity_stats(equity: pd.Series, initial: float) -> dict:
    if equity is None or len(equity) == 0:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "cagr_pct": 0.0, "calmar": 0.0}
    eq = equity.astype(float).ffill().fillna(initial)
    final = float(eq.iloc[-1])
    peak = np.maximum.accumulate(np.concatenate([[initial], eq.to_numpy()]))[1:]
    dd = eq.to_numpy() / peak - 1.0
    max_dd = float(-dd.min()) if len(dd) else 0.0
    daily = eq.resample("1D").last().dropna()
    daily = pd.concat([pd.Series([initial], index=[daily.index[0] - pd.Timedelta(days=1)]), daily]) if len(daily) else daily
    rets = daily.pct_change().dropna()
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    dsd = float(np.sqrt((downside ** 2).mean())) if len(downside) else 0.0
    sharpe = float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    sortino = float(rets.mean() / dsd * math.sqrt(252)) if dsd > 0 else 0.0
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    cagr = (final / initial) ** (1 / years) - 1 if final > 0 and years > 0.08 else (final / initial - 1)
    return {
        "return_pct": (final / initial - 1) * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr_pct": cagr * 100,
        "calmar": (cagr / max_dd) if max_dd > 0 else 0.0,
    }


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, initial_balance: float) -> dict:
    m: dict = {"initial_balance": initial_balance}
    final = float(equity.iloc[-1]) if equity is not None and len(equity) else initial_balance
    m["final_balance"] = final
    m["net_profit"] = final - initial_balance
    m.update(equity_stats(equity, initial_balance))
    n = 0 if trades is None else len(trades)
    m["trades"] = n
    if n == 0:
        m.update({"win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "avg_r": 0.0, "total_r": 0.0,
                  "avg_win": 0.0, "avg_loss": 0.0, "max_consec_losses": 0, "avg_bars_held": 0.0,
                  "long_trades": 0, "short_trades": 0, "by_strategy": {}, "exit_reasons": {}})
        return m
    pnl = trades["pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    m["win_rate"] = len(wins) / n * 100
    m["profit_factor"] = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    m["expectancy"] = float(pnl.mean())
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    r = trades["r_multiple"].astype(float)
    m["avg_r"] = float(r.mean())
    m["total_r"] = float(r.sum())
    m["max_consec_losses"] = _max_consecutive((pnl < 0).to_numpy())
    m["avg_bars_held"] = float(trades["bars_held"].mean()) if "bars_held" in trades else 0.0
    m["long_trades"] = int((trades["side"] > 0).sum())
    m["short_trades"] = int((trades["side"] < 0).sum())
    m["long_pnl"] = float(pnl[trades["side"] > 0].sum())
    m["short_pnl"] = float(pnl[trades["side"] < 0].sum())
    by = {}
    for name, g in trades.groupby("strategy"):
        gp_ = g["pnl"]
        by[str(name)] = {
            "trades": len(g),
            "win_rate": float((gp_ > 0).mean() * 100),
            "pnl": float(gp_.sum()),
            "avg_r": float(g["r_multiple"].mean()),
        }
    m["by_strategy"] = by
    m["exit_reasons"] = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades else {}
    return m


def format_metrics(m: dict, currency: str = "") -> str:
    cur = f" {currency}" if currency else ""
    pf = m.get("profit_factor", 0.0)
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    lines = [
        f"Net profit      : {m['net_profit']:,.2f}{cur} ({m['return_pct']:+.2f}%)",
        f"Final balance   : {m['final_balance']:,.2f}{cur}",
        f"Trades          : {m['trades']} (long {m.get('long_trades', 0)}, short {m.get('short_trades', 0)})",
        f"Win rate        : {m['win_rate']:.1f}%",
        f"Profit factor   : {pf_s}",
        f"Avg R / total R : {m['avg_r']:+.3f} / {m['total_r']:+.1f}",
        f"Max drawdown    : {m['max_dd_pct']:.2f}%",
        f"Sharpe / Sortino: {m['sharpe']:.2f} / {m['sortino']:.2f}",
        f"CAGR / Calmar   : {m['cagr_pct']:+.2f}% / {m['calmar']:.2f}",
        f"Max loss streak : {m['max_consec_losses']}",
    ]
    for name, s in (m.get("by_strategy") or {}).items():
        lines.append(f"  - {name:<18} {s['trades']:>4} trades  win {s['win_rate']:5.1f}%  pnl {s['pnl']:>10,.2f}  avgR {s['avg_r']:+.2f}")
    return "\n".join(lines)
