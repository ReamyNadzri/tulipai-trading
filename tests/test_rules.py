"""Rules-only mode and the fixes from the AI-free audit: news calendar, verdict, swap, DST sessions, panel."""

import json
import time

import pandas as pd
import pytest
import requests

from fake_mt5 import FakeMT5
from tulipai.backtest import Backtester
from tulipai.backtest.benchmark import _verdict
from tulipai.broker.mt5 import MT5Broker
from tulipai.config import Config, NewsConfig, load_config
from tulipai.data.synthetic import synthetic_gold
from tulipai.execution import rollover_nights, swap_money
from tulipai.indicators import Features
from tulipai.news.calendar import EconomicCalendar, builtin_schedule, parse_forexfactory
from tulipai.strategies import STRATEGIES

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


# ---------------------------------------------------------------------------- defaults
def test_rules_only_is_the_default():
    assert Config().ai.mode == "off"


def test_unquoted_yaml_off_means_off(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("ai:\n  mode: off\n")  # YAML 1.1 reads this as boolean false
    assert load_config(p).ai.mode == "off"


# ---------------------------------------------------------------------------- news calendar
def test_empty_or_failed_calendar_never_crashes(tmp_path, monkeypatch):
    cfg = NewsConfig(cache_dir=str(tmp_path))

    def boom(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", boom)
    cal = EconomicCalendar(cfg, use_builtin=False)
    cal.refresh(force=True)
    assert cal.blackout(T("2026-10-14 12:10"), 30, 30) is None
    assert "download failed" in cal.error
    assert len(cal.event_times()) == 0
    assert str(parse_forexfactory([]).dtypes["time"]).startswith("datetime64")


def test_builtin_schedule_keeps_blackout_when_feed_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow")))
    cal = EconomicCalendar(NewsConfig(cache_dir=str(tmp_path)))
    cal.refresh(force=True)
    assert "built-in" in cal.error
    assert "CPI" in cal.blackout(T("2026-10-14 12:10"), 30, 30)          # CPI 08:30 New York = 12:30 UTC (EDT)
    assert "payrolls" in cal.blackout(T("2026-11-06 13:20"), 30, 30)     # after US DST ends: 13:30 UTC
    assert "FOMC" in cal.blackout(T("2026-10-28 18:20"), 30, 30)
    assert cal.blackout(T("2026-10-14 15:00"), 30, 30) is None


def test_stale_cache_is_ignored(tmp_path, monkeypatch):
    cache = tmp_path / "calendar.json"
    cache.write_text(json.dumps([{"title": "Old NFP", "country": "USD", "date": "2026-09-04T08:30:00-04:00",
                                  "impact": "High"}]))
    old = time.time() - 10 * 86400
    import os

    os.utime(cache, (old, old))
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    cal = EconomicCalendar(NewsConfig(cache_dir=str(tmp_path)), use_builtin=False)
    cal.refresh(force=True)
    assert cal.feed_events.empty and "cached" not in cal.error


def test_feed_and_builtin_are_merged_without_duplicate_names(tmp_path, monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"title": "CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00", "impact": "High"}]

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    cal = EconomicCalendar(NewsConfig(cache_dir=str(tmp_path)))
    cal.refresh(force=True)
    assert cal.error == ""
    assert cal.blackout(T("2026-10-14 12:20"), 30, 30).count("12:30Z") == 1


def test_builtin_schedule_is_utc_and_sorted():
    b = builtin_schedule()
    assert b["time"].is_monotonic_increasing and str(b["time"].dt.tz) == "UTC"
    assert set(b["currency"]) == {"USD"} and set(b["impact"]) == {"High"}


# ---------------------------------------------------------------------------- benchmark verdict
def test_losing_strategy_is_never_an_edge():
    assert _verdict(0.001, -50.0, 500).startswith("NO EDGE")
    assert _verdict(0.03, 120.0, 500).startswith("EDGE")
    assert _verdict(0.03, 120.0, 20).startswith("INCONCLUSIVE")


# ---------------------------------------------------------------------------- swap
def test_rollover_counting():
    assert rollover_nights(T("2026-10-05 10:00"), T("2026-10-05 20:00")) == 0       # same session
    assert rollover_nights(T("2026-10-05 19:00"), T("2026-10-06 03:00")) == 1       # Mon 21:00 UTC rollover
    assert rollover_nights(T("2026-10-07 19:00"), T("2026-10-08 03:00")) == 3       # Wednesday = triple
    assert rollover_nights(T("2026-11-02 21:30"), T("2026-11-02 23:00")) == 1       # winter: 22:00 UTC
    assert swap_money(1, 0.1, T("2026-10-05 19:00"), T("2026-10-06 03:00"), -60, -10) == pytest.approx(-6.0)
    assert swap_money(-1, 0.1, T("2026-10-05 19:00"), T("2026-10-06 03:00"), -60, -10) == pytest.approx(-1.0)


def test_backtest_charges_swap_on_overnight_trades(cfg):
    df = synthetic_gold(start="2025-03-03", days=40, seed=13)
    f = Features(df)
    sig = STRATEGIES["trend_pullback"]().generate(f)
    cfg.strategy.name = "trend_pullback"
    base = Backtester(cfg).run(df, sig, start=300, features=f)
    cfg.backtest.swap_long, cfg.backtest.swap_short = -500.0, -500.0
    costly = Backtester(cfg).run(df, sig, start=300, features=f)
    overnight = [rollover_nights(r.entry_time, r.exit_time) > 0 for r in base.trades.itertuples()]
    assert any(overnight), "test data should contain at least one overnight trade"
    assert costly.trades["pnl"].sum() < base.trades["pnl"].sum()


# ---------------------------------------------------------------------------- DST-aware sessions
def test_breakout_window_follows_london_daylight_saving():
    df = synthetic_gold(start="2025-01-01", days=330, seed=5)
    sig = STRATEGIES["session_breakout"]().generate(Features(df))
    s = sig[sig.signal != 0]
    summer = set(s[(s.index.month >= 5) & (s.index.month <= 9)].index.hour)
    winter = set(s[(s.index.month <= 2) | (s.index.month == 12)].index.hour)
    assert min(summer) == 7 and max(summer) <= 12      # London 08:00-14:00 = 07:00-13:00 UTC (BST)
    assert min(winter) == 8 and max(winter) <= 13      # London 08:00-14:00 = 08:00-14:00 UTC (GMT)


# ---------------------------------------------------------------------------- broker swap + panel
def test_mt5_swap_conversion(cfg):
    df = synthetic_gold(days=20, seed=1)
    cfg.symbol.server_utc_offset_hours = 2
    b = MT5Broker(cfg, mt5_module=FakeMT5(df, start_index=900))
    b.connect()
    long_, short_, how = b.swap_per_lot_night()
    assert long_ == pytest.approx(-65.0) and short_ == pytest.approx(12.0) and "points" in how


def test_panel_defaults_to_config_mode_and_reports_effective_mode(cfg, tmp_path, monkeypatch):
    from tulipai.panel import server

    monkeypatch.setattr(server, "SETTINGS", tmp_path / "none.json")
    cfg.live.journal_path = str(tmp_path / "p.db")
    app = server.PanelApp(cfg, 8799)
    page = server._page(app).decode()
    assert '"ai_mode": "off"' in page and 'value="off">Rules only' in page
    assert app.status()["ai_mode"] == "off"


def test_cli_ai_override(tmp_path, monkeypatch):
    from tulipai import cli

    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["live", "--ai", "filter"])
    assert cli._cfg(args).ai.mode == "filter"
    args = cli.build_parser().parse_args(["live"])
    assert cli._cfg(args).ai.mode == "off"
