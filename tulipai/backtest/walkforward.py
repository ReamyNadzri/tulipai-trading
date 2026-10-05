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


def score_metrics(res: BacktestResult) -> dict:
    """Just what ``score`` needs - same numbers as compute_metrics, without the costly extras
    (daily resampling, per-strategy tables) that the optimiser would compute ~1,500 times."""
    n = len(res.trades)
    avg_r = float(res.trades["r_multiple"].astype(float).mean()) if n else 0.0
    max_dd = 0.0
    if len(res.equity):
        eq = res.equity.astype(float).ffill().fillna(res.initial_balance).to_numpy()
        peak = np.maximum.accumulate(np.concatenate([[res.initial_balance], eq]))[1:]
        max_dd = float(-(eq / peak - 1.0).min()) * 100
    return {"trades": n, "avg_r": avg_r, "max_dd_pct": max_dd}


def _grid(strategy_cls: type[Strategy], base: dict, max_combos: int, rng: np.random.Generator) -> list[dict]:
    keys = list(strategy_cls.param_grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(strategy_cls.param_grid[k] for k in keys))]
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), size=max_combos, replace=False)]
    return [{**base, **c} for c in combos]


class WindowOptimiser:
    """Chooses strategy parameters on one training slice. Shared by the walk-forward test and
    by the live bot's monthly re-tune, so the bot re-tunes exactly the way it was tested."""

    def __init__(self, df: pd.DataFrame, cfg: Config, max_combos: int = 27, min_trades: int = 15, seed: int = 0,
                 features: Features | None = None):
        self.df, self.cfg = df, cfg
        self.max_combos, self.min_trades = max_combos, min_trades
        self.rng = np.random.default_rng(seed)
        self.feats = features or Features(df, cfg.symbol.timeframe)
        self.members = cfg.strategy.members if cfg.strategy.name == "ensemble" else [cfg.strategy.name]
        self.base_params = {m: dict(cfg.strategy.params.get(m, {})) for m in self.members}
        self.bt = Backtester(cfg)
        # Signals for every candidate parameter set are computed once on the full history (all
        # indicators are causal, so slicing afterwards cannot leak future data).
        self._signals: dict = {}
        self._combined: dict = {}

    def signals_for(self, name: str, params: dict) -> pd.DataFrame:
        key = (name, tuple(sorted(params.items())))
        if key not in self._signals:
            self._signals[key] = STRATEGIES[name](**params).generate(self.feats)
        return self._signals[key]

    def combined(self, param_map: dict) -> pd.DataFrame:
        if len(self.members) == 1:
            return self.signals_for(self.members[0], param_map[self.members[0]])
        key = tuple((m, tuple(sorted(param_map[m].items()))) for m in self.members)
        if key not in self._combined:
            if len(self._combined) >= 4:  # tiny cache: only the train/test re-runs of a choice repeat
                self._combined.clear()
            self._combined[key] = combine_frames([self.signals_for(m, param_map[m]) for m in self.members])
        return self._combined[key]

    def run(self, param_map: dict, i0: int, i1: int, initial_balance: float | None = None) -> BacktestResult:
        return self.bt.run(self.df, self.combined(param_map), start=i0, end=i1, initial_balance=initial_balance,
                           features=self.feats)

    def optimise(self, i0: int, i1: int, start: dict | None = None) -> dict:
        """Coordinate-wise search on bars [i0, i1): optimise one member at a time, the others
        held at their current best. ``start`` is where the search begins (the previous choice)."""
        chosen = {m: dict((start or self.base_params)[m]) for m in self.members}
        for m in self.members:
            best_s, best_p = -np.inf, chosen[m]
            for params in _grid(STRATEGIES[m], self.base_params[m], self.max_combos, self.rng):
                s = score(score_metrics(self.run({**chosen, m: params}, i0, i1)), self.min_trades)
                if s > best_s:
                    best_s, best_p = s, params
            chosen[m] = best_p
        return chosen


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
    opt = WindowOptimiser(df, cfg, max_combos, min_trades, seed)
    members = opt.members
    idx = df.index
    warm = cfg.backtest.warmup_bars

    t0 = idx[0] + pd.DateOffset(months=train_months)
    windows = []
    balance = cfg.backtest.initial_balance
    oos_trades, oos_equity = [], []
    current = {m: dict(opt.base_params[m]) for m in members}
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

        chosen = opt.optimise(i_tr0, i_te0, current)
        train_res = opt.run(chosen, i_tr0, i_te0)
        test_res = opt.run(chosen, i_te0, i_te1, initial_balance=balance)
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


@dataclass
class RetuneResult:
    params: dict
    train_range: str
    train_metrics: dict
    trades: int


def retune(df: pd.DataFrame, cfg: Config, train_months: int = 6, max_combos: int = 27, min_trades: int = 15,
           seed: int = 0) -> RetuneResult:
    """One walk-forward step done "live": choose the parameters on the most recent
    ``train_months`` of ``df`` (starting from the current settings), as the walk-forward test
    did at the start of every test month."""
    opt = WindowOptimiser(df, cfg, max_combos, min_trades, seed)
    idx = df.index
    end_time = idx[-1] + pd.Timedelta(minutes=cfg.tf_minutes)
    train_start = end_time - pd.DateOffset(months=train_months)
    if idx[0] > train_start + pd.Timedelta(days=7):
        raise ValueError(f"history starts {idx[0]:%Y-%m-%d}, after the {train_months}-month training start "
                         f"{train_start:%Y-%m-%d}; in MT5 set Tools > Options > Charts > Max bars in chart to Unlimited")
    i0 = max(int(idx.searchsorted(train_start)), cfg.backtest.warmup_bars)
    i1 = len(df)
    if i1 - i0 < 500:
        raise ValueError(f"only {i1 - i0} bars in the last {train_months} months after warm-up; need 500+")
    chosen = opt.optimise(i0, i1, opt.base_params)
    m = opt.run(chosen, i0, i1).metrics()
    return RetuneResult(chosen, f"{idx[i0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}", m, int(m["trades"]))


def summarize_windows(result: WalkForwardResult) -> pd.DataFrame:
    rows = [{k: v for k, v in w.items() if k != "params"} for w in result.windows]
    return pd.DataFrame(rows)


__all__ = ["walk_forward", "retune", "WindowOptimiser", "WalkForwardResult", "RetuneResult", "summarize_windows",
           "compute_metrics"]
