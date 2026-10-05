"""Walk-forward optimisation - the honest way to "train" strategy parameters.

The history is cut into rolling windows. In each window the parameters are chosen on the
TRAIN part only and then traded, untouched, on the following TEST part. Only the stitched
test results are reported, so the numbers are out-of-sample.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..indicators import Features
from ..strategies import STRATEGIES, Strategy
from ..strategies.ensemble import combine_frames
from .engine import Backtester, BacktestResult
from .metrics import compute_metrics


def score(m: dict, min_trades: int) -> float:
    """Robust objective: total R scaled by consistency, punished for drawdown and too few trades."""
    if m["trades"] < min_trades:
        return -1e9 + m["trades"]
    sqn_like = m["avg_r"] * math.sqrt(m["trades"])
    return sqn_like - 0.05 * m["max_dd_pct"]


def _grid(strategy_cls: type[Strategy], base: dict, max_combos: int, rng: np.random.Generator) -> list[dict]:
    keys = list(strategy_cls.param_grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(strategy_cls.param_grid[k] for k in keys))]
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), size=max_combos, replace=False)]
    return [{**base, **c} for c in combos]


@dataclass
class WalkForwardResult:
    windows: list = field(default_factory=list)
    oos: BacktestResult | None = None
    best_params: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        return self.oos.metrics() if self.oos else {}


def walk_forward(
    df: pd.DataFrame,
    cfg: Config,
    train_months: int = 6,
    test_months: int = 1,
    max_combos: int = 27,
    min_trades: int = 15,
    seed: int = 0,
    progress: bool = True,
) -> WalkForwardResult:
    rng = np.random.default_rng(seed)
    feats = Features(df, cfg.symbol.timeframe)
    members = cfg.strategy.members if cfg.strategy.name == "ensemble" else [cfg.strategy.name]
    base_params = {m: dict(cfg.strategy.params.get(m, {})) for m in members}
    bt = Backtester(cfg)
    idx = df.index
    warm = cfg.backtest.warmup_bars

    # Signals for every candidate parameter set are computed once on the full history (all
    # indicators are causal, so slicing afterwards cannot leak future data).
    cache: dict = {}

    def signals_for(name: str, params: dict) -> pd.DataFrame:
        key = (name, tuple(sorted(params.items())))
        if key not in cache:
            cache[key] = STRATEGIES[name](**params).generate(feats)
        return cache[key]

    def combined(param_map: dict) -> pd.DataFrame:
        if len(members) == 1:
            return signals_for(members[0], param_map[members[0]])
        frames = [signals_for(m, param_map[m]) for m in members]
        return combine_frames(frames)

    t0 = idx[0] + pd.DateOffset(months=train_months)
    windows = []
    balance = cfg.backtest.initial_balance
    oos_trades, oos_equity = [], []
    current = {m: dict(base_params[m]) for m in members}
    while True:
        test_start, test_end = t0, t0 + pd.DateOffset(months=test_months)
        if test_start >= idx[-1] or idx[-1] - test_start < pd.Timedelta(days=10):
            break  # a final stub of a few days says nothing and only adds noise
        train_start = test_start - pd.DateOffset(months=train_months)
        i_tr0 = max(int(idx.searchsorted(train_start)), warm)
        i_te0 = int(idx.searchsorted(test_start))
        i_te1 = int(idx.searchsorted(test_end))
        if i_te0 - i_tr0 < 500 or i_te1 - i_te0 < 50:
            t0 = test_end
            continue

        # Coordinate-wise search: optimise one member at a time, others held at current best.
        chosen = {m: dict(current[m]) for m in members}
        for m in members:
            best_s, best_p = -np.inf, chosen[m]
            for params in _grid(STRATEGIES[m], base_params[m], max_combos, rng):
                trial = {**chosen, m: params}
                r = bt.run(df, combined(trial), start=i_tr0, end=i_te0, features=feats)
                s = score(r.metrics(), min_trades)
                if s > best_s:
                    best_s, best_p = s, params
            chosen[m] = best_p
        train_res = bt.run(df, combined(chosen), start=i_tr0, end=i_te0, features=feats)
        test_res = bt.run(df, combined(chosen), start=i_te0, end=i_te1, initial_balance=balance, features=feats)
        tm, sm = test_res.metrics(), train_res.metrics()
        windows.append({
            "train": f"{idx[i_tr0].date()} -> {idx[i_te0 - 1].date()}",
            "test": f"{idx[i_te0].date()} -> {idx[i_te1 - 1].date()}",
            "params": chosen,
            "train_trades": sm["trades"], "train_avg_r": sm["avg_r"], "train_return_pct": sm["return_pct"],
            "test_trades": tm["trades"], "test_avg_r": tm["avg_r"], "test_return_pct": tm["return_pct"],
            "test_max_dd_pct": tm["max_dd_pct"],
        })
        if progress:
            w = windows[-1]
            print(f"  WF {w['test']}: train {w['train_trades']:>3} tr avgR {w['train_avg_r']:+.2f} | "
                  f"TEST {w['test_trades']:>3} tr avgR {w['test_avg_r']:+.2f} ret {w['test_return_pct']:+.2f}%", flush=True)
        oos_trades.append(test_res.trades)
        oos_equity.append(test_res.equity)
        balance = test_res.final_balance
        current = chosen
        t0 = test_end

    if not windows:
        raise ValueError("Not enough data for walk-forward; use more history or shorter windows")
    trades = pd.concat(oos_trades, ignore_index=True)
    equity = pd.concat(oos_equity)
    oos = BacktestResult(trades, equity, cfg.backtest.initial_balance, label="walk-forward OOS")
    return WalkForwardResult(windows=windows, oos=oos, best_params=current)


def summarize_windows(result: WalkForwardResult) -> pd.DataFrame:
    rows = [{k: v for k, v in w.items() if k != "params"} for w in result.windows]
    return pd.DataFrame(rows)


__all__ = ["walk_forward", "WalkForwardResult", "summarize_windows", "compute_metrics"]
