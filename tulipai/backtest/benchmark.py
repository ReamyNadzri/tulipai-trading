"""Benchmarks: is the strategy actually better than doing nothing clever?

1. Buy & hold gold over the same period (unlevered).
2. Random-entry Monte Carlo: the SAME engine, risk rules, sessions, sizing and
   stop/target geometry as the strategy, but random entry times and random direction.
   If the strategy does not beat most random runs, its profit is luck, not skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..indicators import Features
from .engine import Backtester, BacktestResult
from .metrics import equity_stats


def buy_and_hold(df: pd.DataFrame, initial_balance: float, start: int = 0, end: int | None = None,
                 bar_minutes: int = 15) -> tuple[pd.Series, dict]:
    sub = df.iloc[start:end]
    eq = initial_balance * sub["close"] / sub["open"].iloc[0]
    eq.index = sub.index + pd.Timedelta(minutes=bar_minutes)
    stats = equity_stats(eq, initial_balance)
    stats["net_profit"] = float(eq.iloc[-1] - initial_balance)
    return eq.rename("buy_and_hold"), stats


def random_entry_benchmark(
    df: pd.DataFrame,
    cfg: Config,
    result: BacktestResult,
    n_sims: int = 100,
    seed: int = 0,
    start: int = 0,
    end: int | None = None,
    features: Features | None = None,
    progress: bool = False,
) -> dict:
    trades = result.trades
    if len(trades) < 5:
        return {"n_sims": 0, "note": "not enough strategy trades for a meaningful benchmark"}
    feats = features or Features(df, cfg.symbol.timeframe)
    atr = feats.atr(14).to_numpy()
    n = len(df)
    end = n if end is None else end
    rng = np.random.default_rng(seed)

    # Stop/target geometry of the real trades, in ATR units at the signal bar.
    sig_idx = trades["signal_idx"].to_numpy(dtype=int)
    sl_atr = trades["sl_dist"].to_numpy() / np.where(atr[sig_idx] > 0, atr[sig_idx], np.nan)
    rr = trades["tp_dist"].to_numpy() / trades["sl_dist"].to_numpy()
    geom = np.column_stack([sl_atr, rr])
    geom = geom[np.isfinite(geom).all(axis=1)]

    # Candidate bars: inside the trading session with a valid ATR.
    bar = pd.Timedelta(minutes=cfg.tf_minutes)
    close_t = df.index + bar
    hours = close_t.hour.to_numpy()
    wd = close_t.weekday.to_numpy()
    in_sess = np.zeros(n, dtype=bool)
    for a, b in cfg.risk.trade_hours_utc:
        in_sess |= (hours >= a) & (hours < b)
    in_sess &= np.isin(wd, cfg.risk.trade_days) & ~((wd == 4) & (hours >= cfg.risk.friday_cutoff_hour_utc))
    in_sess &= np.isfinite(atr) & (atr > 0)
    cand = np.flatnonzero(in_sess[start:end]) + start
    if len(cand) < len(trades):
        return {"n_sims": 0, "note": "not enough candidate bars"}

    # Random runs fire more often than the strategy takes trades (positions block some),
    # so oversample signals to land on a similar number of executed trades.
    n_signals = min(len(cand), int(len(trades) * 1.6) + 5)
    bt = Backtester(cfg)
    nets, rets, dds, ntr = [], [], [], []
    for k in range(n_sims):
        if progress and k and k % 20 == 0:
            print(f"  random-entry run {k}/{n_sims}", flush=True)
        pick = np.sort(rng.choice(cand, size=n_signals, replace=False))
        g = geom[rng.integers(0, len(geom), size=n_signals)]
        sig = pd.DataFrame(index=df.index, data={"signal": 0, "sl_dist": 0.0, "tp_dist": 0.0, "strength": 0.0, "strategy": ""})
        sides = rng.choice([-1, 1], size=n_signals)
        sl = g[:, 0] * atr[pick]
        sig.iloc[pick, 0] = sides
        sig.iloc[pick, 1] = sl
        sig.iloc[pick, 2] = sl * g[:, 1]
        sig.iloc[pick, 4] = "random"
        r = bt.run(df, sig, start=start, end=end, initial_balance=result.initial_balance, features=feats)
        m = equity_stats(r.equity, r.initial_balance)
        nets.append(r.final_balance - r.initial_balance)
        rets.append(m["return_pct"])
        dds.append(m["max_dd_pct"])
        ntr.append(len(r.trades))

    nets = np.array(nets)
    strat_net = result.final_balance - result.initial_balance
    p_value = (np.sum(nets >= strat_net) + 1) / (len(nets) + 1)
    return {
        "n_sims": n_sims,
        "strategy_net": float(strat_net),
        "random_net_mean": float(nets.mean()),
        "random_net_median": float(np.median(nets)),
        "random_net_p5": float(np.percentile(nets, 5)),
        "random_net_p95": float(np.percentile(nets, 95)),
        "random_return_mean_pct": float(np.mean(rets)),
        "random_maxdd_mean_pct": float(np.mean(dds)),
        "random_trades_mean": float(np.mean(ntr)),
        "strategy_percentile": float((nets < strat_net).mean() * 100),
        "p_value": float(p_value),
        "random_nets": nets.tolist(),
        "verdict": _verdict(p_value, strat_net, n_sims),
    }


def _verdict(p: float, strat_net: float, n_sims: int) -> str:
    if strat_net <= 0:
        # Beating random entries is meaningless if the strategy still loses money: costs are
        # what make random entries lose, and a smaller loss is not an edge.
        return "NO EDGE: the strategy lost money after costs (beating random entries does not change that)"
    if n_sims < 100:
        return f"INCONCLUSIVE: only {n_sims} random runs - use at least 200 for a verdict"
    if p <= 0.05:
        return "EDGE: beats >=95% of random-entry runs with identical risk rules"
    if p <= 0.20:
        return "WEAK EDGE: better than most random runs, not yet statistically convincing"
    return "NO EDGE: results are within what random entries achieve - do not trust the profit"
