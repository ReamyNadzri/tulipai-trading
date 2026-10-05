"""Monthly re-tune: the live bot must re-choose its settings exactly as the walk-forward test did."""

import yaml

import pandas as pd
import pytest

from fake_mt5 import FakeMT5
from tulipai.backtest.walkforward import retune, walk_forward
from tulipai.broker.mt5 import MT5Broker
from tulipai.config import apply_params_file, params_file_info, save_params_file
from tulipai.data.synthetic import synthetic_gold
from tulipai.journal import Journal
from tulipai.live import LiveEngine


@pytest.fixture(scope="module")
def year_df():
    return synthetic_gold(start="2024-01-01", days=300, seed=11)


def test_retune_picks_the_walk_forward_window_choice(cfg, year_df):
    wf = walk_forward(year_df, cfg, train_months=6, test_months=1, max_combos=4, progress=False)
    first_test = wf.windows[0]["test"].split(" -> ")[0]
    cut = int(year_df.index.searchsorted(pd.Timestamp(first_test, tz="UTC")))
    res = retune(year_df.iloc[:cut], cfg, train_months=6, max_combos=4)
    assert res.params == wf.windows[0]["params"]
    assert res.train_range.split(" -> ")[0] == wf.windows[0]["train"].split(" -> ")[0]


def test_retune_refuses_too_little_history(cfg, year_df):
    with pytest.raises(ValueError, match="Max bars in chart"):
        retune(year_df.iloc[:3000], cfg, train_months=6)


def test_reapplying_params_file_keeps_manual_overrides(cfg, tmp_path):
    cfg.strategy.params_file = str(tmp_path / "p.yaml")
    cfg.strategy.params = {"trend_pullback": {"rr": 3.0}}
    save_params_file(cfg.strategy.params_file, {"trend_pullback": {"adx_min": 22.0, "rr": 1.5}}, "wf", "x")
    apply_params_file(cfg)
    save_params_file(cfg.strategy.params_file, {"trend_pullback": {"adx_min": 26.0, "rr": 2.0}}, "retune", "y")
    apply_params_file(cfg)
    assert cfg.strategy.params["trend_pullback"] == {"adx_min": 26.0, "rr": 3.0}


def test_live_engine_retunes_in_background_when_month_changes(cfg, tmp_path):
    df = synthetic_gold(start="2024-01-01", days=80, seed=21)
    cfg.symbol.server_utc_offset_hours = 2
    cfg.symbol.history_bars = 2500
    cfg.strategy.auto_retune = True
    cfg.strategy.retune_train_months = 1
    cfg.strategy.params_file = str(tmp_path / "optimized_params.yaml")
    save_params_file(cfg.strategy.params_file, {"trend_pullback": {"adx_min": 26.0}}, "walk-forward", "old")
    text = yaml.safe_load((tmp_path / "optimized_params.yaml").read_text())
    text["generated"] = "2024-02-03"  # last month's settings
    (tmp_path / "optimized_params.yaml").write_text(yaml.safe_dump(text))
    cfg.strategy_source = apply_params_file(cfg)

    fake = FakeMT5(df, start_index=len(df) - 10)
    broker = MT5Broker(cfg, login="12345678", password="pw", server="Fake-MT5Trial", path=None, mt5_module=fake)
    eng = LiveEngine(cfg, broker, Journal(tmp_path / "j.db"), mode_label="mt5")
    eng.start()
    assert eng.status["auto_retune"] is True

    now = pd.Timestamp("2024-03-01 09:00", tz="UTC")
    assert eng.retune_due(now)[0]
    eng._retune_tick(now)
    assert eng.status["retune"]["state"] == "running"
    eng._retune_thread.join(timeout=120)
    eng._retune_tick(now)  # applies the result on the engine thread

    st = eng.status["retune"]
    assert st["state"] == "done", st
    info = params_file_info(cfg.strategy.params_file)
    assert info["generated"][:7] != "2024-02" and "auto re-tune" in yaml.safe_load(
        (tmp_path / "optimized_params.yaml").read_text())["method"]
    assert eng.strategy.members[1].params == {**eng.strategy.members[1].default_params,
                                              **info["params"]["trend_pullback"]}
    assert eng.journal.get_kv("last_retune:mt5")["train"] == st["train"]
    # broker's own swap was used (fake: swap_mode 1, -650 points long)
    assert not eng.retune_due(pd.Timestamp.now(tz="UTC"))[0]


def test_failed_retune_keeps_settings_and_backs_off(cfg, tmp_path):
    df = synthetic_gold(start="2024-01-01", days=20, seed=3)  # far too short for 6 months
    cfg.symbol.server_utc_offset_hours = 2
    cfg.strategy.auto_retune = True
    cfg.strategy.params_file = str(tmp_path / "none.yaml")
    fake = FakeMT5(df, start_index=len(df) - 10)
    broker = MT5Broker(cfg, login="12345678", password="pw", server="Fake-MT5Trial", path=None, mt5_module=fake)
    eng = LiveEngine(cfg, broker, Journal(tmp_path / "j.db"), mode_label="paper")
    eng.start()
    before = dict(cfg.strategy.params)
    now = pd.Timestamp("2024-01-25 09:00", tz="UTC")
    eng._retune_tick(now)
    eng._retune_thread.join(timeout=60)
    eng._retune_tick(now)
    assert eng.status["retune"]["state"] == "failed" and "Max bars in chart" in eng.status["retune"]["error"]
    assert cfg.strategy.params == before and not (tmp_path / "none.yaml").exists()
    eng._retune_tick(now + pd.Timedelta(hours=1))
    assert eng._retune_thread is None  # waits 6 hours before trying again


def test_replay_never_retunes(cfg, tmp_path):
    cfg.strategy.auto_retune = True
    from tulipai.broker.replay import ReplayBroker

    df = synthetic_gold(start="2024-01-01", days=20, seed=3)
    eng = LiveEngine(cfg, ReplayBroker(df, cfg, start_index=500), Journal(tmp_path / "j.db"), mode_label="replay")
    assert eng.auto_retune is False


def test_retune_month_check_uses_utc_dates(cfg, tmp_path, monkeypatch):
    """A re-tune just after 00:00 UTC on the 1st must not repeat every 10 minutes on PCs west of UTC."""
    import datetime as dt

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 11, 1, 0, 5, tzinfo=dt.timezone.utc) if tz else dt.datetime(2026, 10, 31, 19, 5)

    monkeypatch.setattr(dt, "datetime", Clock)  # config imports datetime inside the function
    path = tmp_path / "p.yaml"
    save_params_file(path, {"trend_pullback": {"rr": 2.0}}, "retune", "x")
    assert params_file_info(path)["generated"].startswith("2026-11-01")
