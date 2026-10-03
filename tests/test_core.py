"""Indicators, strategies (incl. no look-ahead), risk sizing and config."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tulipai.config import Config, load_config
from tulipai.indicators import Features, atr, ema, rsi
from tulipai.risk import RiskManager, in_blackout
from tulipai.strategies import STRATEGIES, build_strategy


def test_ema_matches_recursive_definition():
    s = pd.Series(np.arange(1, 101, dtype=float))
    e = ema(s, 10)
    alpha = 2 / 11
    ref = s.iloc[0]
    for v in s.iloc[1:]:
        ref = alpha * v + (1 - alpha) * ref
    assert e.iloc[-1] == pytest.approx(ref)
    assert e.iloc[:9].isna().all()


def test_rsi_extremes():
    up = pd.Series(np.arange(100, dtype=float))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(100, 0, -1, dtype=float))
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)


def test_atr_constant_range():
    n = 60
    close = pd.Series(np.full(n, 100.0))
    high, low = close + 1, close - 1
    assert atr(high, low, close, 14).iloc[-1] == pytest.approx(2.0)


@pytest.mark.parametrize("name", ["ensemble", *STRATEGIES])
def test_strategies_have_no_lookahead(gold_df, name):
    """A signal at bar i must not change when future bars are appended."""
    cfg = Config()
    cfg.strategy.name = name
    strat = build_strategy(cfg.strategy)
    full = strat.generate(Features(gold_df))
    for cut in (3000, 5555, 8000):
        part = strat.generate(Features(gold_df.iloc[:cut]))
        assert (part["signal"].to_numpy() == full["signal"].iloc[:cut].to_numpy()).all(), cut
        np.testing.assert_allclose(part["sl_dist"].to_numpy(), full["sl_dist"].iloc[:cut].to_numpy(), rtol=1e-9)


def test_strategies_emit_valid_levels(gold_df):
    sig = build_strategy(Config().strategy).generate(Features(gold_df))
    s = sig[sig.signal != 0]
    assert len(s) > 20
    assert (s.sl_dist > 0).all() and (s.tp_dist > 0).all()
    assert set(s.strategy) <= set(STRATEGIES)


def test_unknown_strategy_param_rejected():
    with pytest.raises(ValueError):
        STRATEGIES["trend_pullback"](not_a_param=1)


def test_config_file_matches_defaults():
    """config/config.yaml documents every default; it must not drift from the code."""
    path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    assert load_config(path).to_dict() == Config().to_dict()


def test_config_rejects_typos(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk:\n  risk_per_trade: 2\n")
    with pytest.raises(ValueError, match="risk_per_trade"):
        load_config(p)


def test_cent_account_position_size():
    """10,000 USC (= $100) at 1% risk budgets 100 USC. A $8 stop on 1.0 lot of XAUUSDc loses 800 USC,
    so the bot must trade 0.12 lots (100/800 = 0.125 -> rounded DOWN to the 0.01 step)."""
    rm = RiskManager(Config().risk, 15)
    lots, why = rm.position_size(10000, 800.0, 0.01, 0.01, 200.0)
    assert lots == pytest.approx(0.12) and why == "ok"
    assert lots * 800 <= 100


def test_min_lot_overshoot_rule():
    rm = RiskManager(Config().risk, 15)
    lots, _ = rm.position_size(1000, 1200.0, 0.01, 0.01, 100)  # budget 10, min lot risks 12 (<=1.5x) -> allowed
    assert lots == 0.01
    lots, why = rm.position_size(1000, 3000.0, 0.01, 0.01, 100)  # min lot risks 30 -> refused
    assert lots == 0 and "min lot" in why


def test_max_lot_cap():
    cfg = Config()
    cfg.risk.max_lot = 0.5
    lots, _ = RiskManager(cfg.risk, 15).position_size(1_000_000, 10.0, 0.01, 0.01, 100)
    assert lots == 0.5


def test_risk_guards():
    cfg = Config()
    rm = RiskManager(cfg.risk, 15)
    t = pd.Timestamp("2025-03-04 10:00", tz="UTC")  # Tuesday, London session
    rm.update(t, 10000)
    assert rm.can_open(t, 10000, 0, 0.3, 5.0) == (True, "ok")
    assert not rm.can_open(pd.Timestamp("2025-03-04 03:00", tz="UTC"), 10000, 0, 0.3, 5.0)[0]  # Asia
    assert not rm.can_open(pd.Timestamp("2025-03-07 19:00", tz="UTC"), 10000, 0, 0.3, 5.0)[0]  # Friday late
    assert not rm.can_open(t, 10000, 1, 0.3, 5.0)[0]  # max positions
    assert not rm.can_open(t, 10000, 0, 3.0, 5.0)[0]  # spread
    assert not rm.can_open(t, 10000, 0, 0.3, 5.0, news_block="CPI")[0]
    assert not rm.can_open(t, 9690, 0, 0.3, 5.0)[0]  # -3.1% today
    for _ in range(3):
        rm.on_trade_closed(-10, t)
    assert "cooldown" in rm.can_open(t, 10000, 0, 0.3, 5.0)[1]
    assert rm.can_open(t + pd.Timedelta(hours=3), 10000, 0, 0.3, 5.0)[0]
    rm.update(t, 7900)  # -21% from peak -> kill switch
    assert rm.state.halted and not rm.can_open(t, 7900, 0, 0.3, 5.0)[0]


def test_validate_levels_clamps_stop_and_keeps_rr():
    rm = RiskManager(Config().risk, 15)
    ok, _, sl, tp = rm.validate_levels(1.0, 2.0, 5.0)  # SL 0.2 ATR -> widened to 0.8 ATR
    assert ok and sl == pytest.approx(4.0) and tp == pytest.approx(8.0)
    ok, why, _, _ = rm.validate_levels(5.0, 2.0, 5.0)
    assert not ok and "reward:risk" in why


def test_news_blackout_window():
    ev = np.array([np.datetime64("2025-03-12T12:30")], dtype="datetime64[ns]")
    assert in_blackout(ev, pd.Timestamp("2025-03-12 12:10", tz="UTC"), 30, 30)
    assert in_blackout(ev, pd.Timestamp("2025-03-12 12:55", tz="UTC"), 30, 30)
    assert not in_blackout(ev, pd.Timestamp("2025-03-12 13:05", tz="UTC"), 30, 30)
    assert not in_blackout(ev, pd.Timestamp("2025-03-12 11:55", tz="UTC"), 30, 30)
