"""Backtester mechanics, the no-edge sanity check, benchmarks and live/backtest parity."""

import numpy as np
import pandas as pd
import pytest

from tulipai.backtest import Backtester
from tulipai.backtest.benchmark import buy_and_hold, random_entry_benchmark
from tulipai.data.synthetic import synthetic_gold
from tulipai.execution import check_exit, simulate_trade
from tulipai.indicators import Features
from tulipai.journal import Journal
from tulipai.live import run_replay
from tulipai.strategies import build_strategy


def _flat_df(n=400, price=2000.0):
    idx = pd.date_range("2025-03-03 00:00", periods=n, freq="15min", tz="UTC")  # Monday
    df = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0}, index=idx)
    return df


def _one_signal(df, i, side, sl, tp):
    sig = pd.DataFrame({"signal": 0, "sl_dist": 0.0, "tp_dist": 0.0, "strength": 0.0, "strategy": ""}, index=df.index)
    sig.iloc[i, 0], sig.iloc[i, 1], sig.iloc[i, 2], sig.iloc[i, 4] = side, sl, tp, "test"
    return sig


def test_long_hits_take_profit_with_spread(cfg):
    df = _flat_df()
    i = 40  # signal at 10:00 bar, decision at 10:15 (London session)
    df.iloc[i + 3, df.columns.get_loc("high")] = 2020.0  # spike up later
    cfg.backtest.slippage = 0.0
    cfg.backtest.spread = 0.5
    res = Backtester(cfg).run(df, _one_signal(df, i, 1, 2.0, 4.0), start=20)
    t = res.trades.iloc[0]
    assert t.entry == pytest.approx(2000.5)  # bought at ask = bid + spread
    assert t.exit_reason == "tp" and t.exit == pytest.approx(2004.5)
    assert t.entry_time == df.index[i + 1]


def test_stop_first_when_bar_hits_both(cfg):
    df = _flat_df()
    i = 40
    df.iloc[i + 2, df.columns.get_loc("high")] = 2050.0
    df.iloc[i + 2, df.columns.get_loc("low")] = 1950.0
    cfg.backtest.slippage = 0.0
    res = Backtester(cfg).run(df, _one_signal(df, i, 1, 2.0, 4.0), start=20)
    assert res.trades.iloc[0].exit_reason == "sl"
    assert res.trades.iloc[0].r_multiple == pytest.approx(-1.0, abs=0.2)


def test_gap_through_stop_fills_at_open():
    assert check_exit(1, 2000, 1990, 2020, o=1980, h=1985, l=1975, spread=0.3, slippage=0.0) == (1980, "sl")
    px, reason = check_exit(-1, 2000, 2010, 1980, o=2015, h=2016, l=2014, spread=0.5, slippage=0.0)
    assert reason == "sl" and px == pytest.approx(2015.5)  # shorts exit at the ask


def test_no_trades_outside_session(cfg):
    df = _flat_df()
    res = Backtester(cfg).run(df, _one_signal(df, 8, 1, 2.0, 4.0), start=0)  # 02:00 UTC
    assert len(res.trades) == 0 and res.blocked.get("outside trading session") == 1


def test_simulate_trade_reports_r():
    o = np.array([100.0, 100, 100, 100])
    h = np.array([100.5, 101, 104, 100])
    l = np.array([99.5, 99.5, 99.5, 99.5])
    c = o.copy()
    res = simulate_trade(o, h, l, c, np.zeros(4), 1, 1, 1.0, 3.0, 0.0, 10)
    assert res[3] == "tp" and res[4] == pytest.approx(3.0)


def test_random_walk_has_no_edge(cfg):
    """On driftless random walks the strategies must NOT make money after costs.
    A positive average here would mean look-ahead bias in the engine."""
    rs = []
    for seed in range(4):
        df = synthetic_gold(days=150, seed=100 + seed, random_walk=True)
        f = Features(df)
        res = Backtester(cfg).run(df, build_strategy(cfg.strategy).generate(f), start=300, features=f)
        rs.append(res.metrics()["avg_r"])
    assert np.mean(rs) < 0.05, rs


def test_buy_and_hold(gold_df):
    eq, stats = buy_and_hold(gold_df, 10000)
    assert eq.iloc[-1] == pytest.approx(10000 * gold_df.close.iloc[-1] / gold_df.open.iloc[0])
    assert stats["max_dd_pct"] >= 0


def test_random_benchmark_runs(gold_df, cfg):
    f = Features(gold_df)
    res = Backtester(cfg).run(gold_df, build_strategy(cfg.strategy).generate(f), start=300, features=f)
    rb = random_entry_benchmark(gold_df, cfg, res, n_sims=8, start=300, features=f)
    assert rb["n_sims"] == 8 and 0 < rb["p_value"] <= 1 and "EDGE" in rb["verdict"]


def test_live_engine_replay_matches_backtester(cfg, tmp_path):
    """The live engine (driven by ReplayBroker) must produce exactly the backtester's trades."""
    df = synthetic_gold(days=14, seed=11)
    cfg.symbol.history_bars = len(df) + 10
    f = Features(df)
    start = cfg.backtest.warmup_bars
    bt = Backtester(cfg).run(df, build_strategy(cfg.strategy).generate(f), start=start - 1, features=f)
    eng = run_replay(df, cfg, tmp_path / "replay.db", start=start)
    live = eng.journal.frame("trades")
    assert len(bt.trades) == len(live) > 3
    assert list(pd.to_datetime(live.open_time, utc=True)) == list(bt.trades.entry_time)
    assert list(live.side) == list(bt.trades.side)
    assert list(live.exit_reason) == list(bt.trades.exit_reason)
    assert live.pnl.sum() == pytest.approx(bt.trades.pnl.sum(), abs=1e-6)


def test_journal_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.open_trade(7, "mt5", "XAUUSDc", 1, 0.1, pd.Timestamp("2025-01-02", tz="UTC"), 2000, 1990, 2020, 10, "x",
                 0.7, None, 100.0)
    j.close_trade(7, pd.Timestamp("2025-01-02 05:00", tz="UTC"), 2020, 200.0, "tp")
    row = j.trade_by_ticket(7)
    assert row["r_multiple"] == pytest.approx(2.0) and row["exit_reason"] == "tp"
    j.set_kv("k", {"a": 1})
    assert j.get_kv("k") == {"a": 1}
