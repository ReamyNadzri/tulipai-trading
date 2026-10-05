"""MT5 adapter (via a fake MetaTrader5 module), Claude layer (fake client), news parsing, panel."""

import json
import threading
import urllib.request
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from fake_mt5 import FakeMT5
from tulipai.ai.brain import ClaudeBrain
from tulipai.broker.mt5 import MT5Broker
from tulipai.data.synthetic import synthetic_gold
from tulipai.journal import Journal
from tulipai.live import LiveEngine
from tulipai.news import parse_feed, parse_forexfactory


@pytest.fixture(scope="module")
def mdf():
    return synthetic_gold(days=70, seed=21)


def _broker(cfg, mdf, **kw):
    cfg.symbol.server_utc_offset_hours = 2
    fake = FakeMT5(mdf, start_index=kw.pop("start", 3000), **kw)
    return MT5Broker(cfg, login="12345678", password="pw", server="Fake-MT5Trial", path=None, mt5_module=fake), fake


def test_connect_resolves_cent_symbol_and_utc(cfg, mdf):
    b, fake = _broker(cfg, mdf)
    acct = b.connect()
    assert fake.initialized_with["login"] == 12345678 and fake.initialized_with["server"] == "Fake-MT5Trial"
    assert b.symbol == "XAUUSDc" and acct.currency == "USC" and acct.is_demo
    candles = b.candles(500)
    assert candles.index[-1] == mdf.index[2999]  # closed bars only, converted back to UTC
    assert candles["spread"].iloc[-1] == pytest.approx(0.3)


def test_login_failure_is_explained(cfg, mdf):
    b, _ = _broker(cfg, mdf, fail_login=True)
    with pytest.raises(Exception, match="login, password and server"):
        b.connect()


def test_order_uses_supported_filling_and_magic(cfg, mdf):
    b, fake = _broker(cfg, mdf, filling_mask=2)
    b.connect()
    t = b.tick()
    res = b.open_market(1, 0.12, t.ask - 8, t.ask + 16, "TulipAI test")
    assert res.ok and res.ticket in fake.positions
    req = fake.requests[-1]
    assert req["type_filling"] == fake.ORDER_FILLING_IOC and req["magic"] == cfg.account.magic
    assert b.loss_per_lot(1, t.ask, t.ask - 8) == pytest.approx(800.0)
    assert b.modify_sl_tp(res.ticket, t.ask - 4, t.ask + 16).ok
    assert fake.positions[res.ticket].sl == pytest.approx(round(t.ask - 4, 3))
    assert b.close_position(res.ticket).ok
    assert b.closed_trade(res.ticket) is not None


def test_filling_fallback_on_10030(cfg, mdf):
    b, fake = _broker(cfg, mdf, filling_mask=1)  # FOK only
    b.connect()
    t = b.tick()
    assert b.open_market(-1, 0.05, t.bid + 8, t.bid - 16, "x").ok
    assert fake.requests[-1]["type_filling"] == fake.ORDER_FILLING_FOK


def test_real_account_is_refused_by_default(cfg, mdf):
    b, fake = _broker(cfg, mdf, demo=False)
    b.connect()
    t = b.tick()
    res = b.open_market(1, 0.01, t.ask - 8, t.ask + 16, "x")
    assert not res.ok and "allow_real_account" in res.message and not fake.requests
    assert any("REAL" in p for p in b.preflight())


def test_algo_trading_off_message(cfg, mdf):
    b, fake = _broker(cfg, mdf)
    b.connect()
    fake.algo_trading = False
    t = b.tick()
    res = b.open_market(1, 0.01, t.ask - 8, t.ask + 16, "x")
    assert not res.ok and "Algo Trading" in res.message


def test_live_engine_trades_fake_mt5_end_to_end(cfg, mdf, tmp_path):
    cfg.symbol.history_bars = 2500
    b, fake = _broker(cfg, mdf, start=2600)
    eng = LiveEngine(cfg, b, Journal(tmp_path / "j.db"), mode_label="mt5")
    eng.start()
    for _ in range(700):
        eng.step()
        if not fake.advance():
            break
    trades = eng.journal.frame("trades")
    assert len(trades) >= 3
    closed = trades[trades.close_time.notna()]
    assert len(closed) >= 2 and set(closed.exit_reason) <= {"sl", "tp", "bot", "manual"}
    assert all(r["magic"] == cfg.account.magic for r in fake.requests if "magic" in r)
    # every entry carried a stop loss and respected the 1% risk budget
    entries = [r for r in fake.requests if r["action"] == fake.TRADE_ACTION_DEAL and "position" not in r]
    assert entries and all(r["sl"] > 0 for r in entries)
    assert (closed.pnl >= -closed.risk_money * 1.3).all()


# ---------------------------------------------------------------------------- Claude layer
class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=json.dumps(self.payload))],
                  stop_reason=self.stop_reason, model=kw["model"],
                  usage=NS(input_tokens=1200, output_tokens=300, cache_read_input_tokens=900, cache_creation_input_tokens=0))


def _brain(cfg, client):
    return ClaudeBrain(cfg.ai, cfg.risk, cfg.management, "M15", client=client)


def test_brain_request_shape_and_parsing(cfg):
    client = FakeClient({"action": "BUY", "confidence": 0.72, "sl_atr": 9.0, "tp_atr": 3.0, "risk_multiplier": 3,
                         "market_bias": "bullish", "reasoning": "trend + London breakout", "key_risks": ["CPI"]})
    d = _brain(cfg, client).decide("snapshot")
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert d.action == "BUY" and d.side == 1
    assert d.sl_atr == cfg.risk.max_sl_atr  # clamped
    assert d.risk_multiplier == 1.0  # clamped, AI can never raise risk
    assert d.usage["cache_read_input_tokens"] == 900


def test_brain_refusal_and_garbage_become_hold(cfg):
    assert _brain(cfg, FakeClient({}, stop_reason="refusal")).decide("x").action == "HOLD"
    bad = FakeClient({})
    bad._create = lambda **kw: NS(content=[NS(type="text", text="not json")], stop_reason="end_turn", model="m", usage=None)
    bad.beta = NS(messages=NS(create=bad._create))
    d = _brain(cfg, bad).decide("x")
    assert d.action == "HOLD" and d.error


def test_brain_daily_budget(cfg):
    cfg.ai.max_calls_per_day = 1
    b = _brain(cfg, FakeClient({"action": "HOLD", "confidence": 0.5, "sl_atr": 0, "tp_atr": 0, "risk_multiplier": 1,
                                "market_bias": "neutral", "reasoning": "", "key_risks": []}))
    assert not b.decide("a").error
    assert "budget" in b.decide("b").error


class StubBrain:
    """Deterministic brain for engine tests: vetoes every other proposal."""

    def __init__(self):
        self.n = 0

    def decide(self, context):
        from tulipai.ai.brain import AIDecision

        self.n += 1
        assert "Quant strategy proposal" in context and "Multi-timeframe" in context
        side = "BUY" if "BUY |" in context else "SELL"
        if self.n % 2 == 0:
            return AIDecision(action="HOLD", confidence=0.4, reasoning="veto")
        return AIDecision(action=side, confidence=0.8, sl_atr=1.5, tp_atr=3.0, risk_multiplier=0.5, reasoning="ok")


def test_filter_mode_vetoes_are_shadowed(cfg, mdf, tmp_path):
    cfg.ai.mode = "filter"
    cfg.symbol.history_bars = 2500
    b, fake = _broker(cfg, mdf, start=2600)
    eng = LiveEngine(cfg, b, Journal(tmp_path / "j.db"), brain=StubBrain(), mode_label="mt5")
    eng.start()
    eng.brain = StubBrain()
    for _ in range(600):
        eng.step()
        if not fake.advance():
            break
    dec = eng.journal.frame("decisions")
    assert (dec.reason == "AI veto").sum() >= 1
    assert len(eng.journal.frame("shadows")) == (dec.reason == "AI veto").sum()
    trades = eng.journal.frame("trades")
    assert len(trades) >= 1
    # risk_multiplier 0.5 -> about half the normal 1% risk
    first = trades.iloc[0]
    assert first.risk_money <= 0.0055 * 10000 * 1.2


# ---------------------------------------------------------------------------- news
def test_forexfactory_parsing():
    ev = parse_forexfactory([
        {"title": "CPI m/m", "country": "USD", "date": "2025-03-12T08:30:00-04:00", "impact": "High", "forecast": "0.3%"},
        {"title": "bad", "country": "EUR"},
    ])
    assert len(ev) == 1 and ev.time.iloc[0] == pd.Timestamp("2025-03-12 12:30", tz="UTC")


def test_rss_and_atom_parsing():
    rss = """<rss><channel><item><title>Gold rises as Fed &amp; yields fall</title>
             <pubDate>Tue, 11 Mar 2025 14:00:00 GMT</pubDate></item></channel></rss>"""
    atom = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Dollar slips</title>
              <updated>2025-03-11T15:00:00Z</updated></entry></feed>"""
    a, b = parse_feed(rss, "x"), parse_feed(atom, "y")
    assert a[0]["title"] == "Gold rises as Fed & yields fall" and a[0]["time"].hour == 14
    assert b[0]["title"] == "Dollar slips"


# ---------------------------------------------------------------------------- panel
def test_panel_requires_token_and_localhost(cfg, tmp_path):
    from http.server import ThreadingHTTPServer

    from tulipai.panel.server import PanelApp, make_handler

    cfg.live.journal_path = str(tmp_path / "p.db")
    app = PanelApp(cfg, 0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    port = httpd.server_address[1]
    app.port = port
    httpd.RequestHandlerClass = make_handler(app)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert app.token in page and "Connect MetaTrader 5" in page and "TulipCharts" in page
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/status")
        assert e.value.code == 403
        req = urllib.request.Request(base + "/api/status", headers={"X-Tulip-Token": app.token})
        st = json.loads(urllib.request.urlopen(req).read())
        assert st["connected"] is False and "stats" in st
        bad_host = urllib.request.Request(base + "/", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad_host)
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------- server time zone
def test_offset_detected_on_weekend_and_dst_aware(cfg):
    """Saturday: no live ticks. The last tick (Friday's close) must still reveal the offset, and a
    UTC+3-in-summer broker must be recognised as 'New York + 7h' so winter bars use UTC+2."""
    df = synthetic_gold(start="2025-06-02", days=12, seed=3)
    last_friday_bar = df.index.get_loc(pd.Timestamp("2025-06-13 20:45", tz="UTC"))
    fake = FakeMT5(df, start_index=last_friday_bar, offset_hours=3)
    b = MT5Broker(cfg, mt5_module=fake)
    b.symbol = "XAUUSDc"
    b._detect_offset(now=pd.Timestamp("2025-06-14 12:00", tz="UTC"))
    assert b.offset.total_seconds() == 3 * 3600 and b.dst_mode, b.offset_note
    summer = int(pd.Timestamp("2025-06-13 23:45").timestamp())  # server time label
    winter = int(pd.Timestamp("2025-01-10 22:45").timestamp())
    assert b._ts(summer) == pd.Timestamp("2025-06-13 20:45", tz="UTC")
    assert b._ts(winter) == pd.Timestamp("2025-01-10 20:45", tz="UTC")


def test_offset_fixed_utc0_broker(cfg):
    df = synthetic_gold(start="2025-06-02", days=12, seed=3)
    i = df.index.get_loc(pd.Timestamp("2025-06-13 20:45", tz="UTC"))
    fake = FakeMT5(df, start_index=i, offset_hours=0)
    b = MT5Broker(cfg, mt5_module=fake)
    b.symbol = "XAUUSDc"
    b._detect_offset(now=pd.Timestamp("2025-06-14 12:00", tz="UTC"))
    assert b.offset.total_seconds() == 0 and not b.dst_mode


def test_research_command_end_to_end(cfg, tmp_path, monkeypatch):
    """`tulip research` against the fake MT5: download, backtest, benchmark, walk-forward, ML, summary."""
    import tulipai.broker.mt5 as mt5mod
    from tulipai import cli

    df = synthetic_gold(start="2024-01-01", days=300, seed=4)
    fake = FakeMT5(df, start_index=len(df) - 1, offset_hours=2)
    monkeypatch.setattr(mt5mod, "import_mt5", lambda: fake)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text("symbol:\n  server_utc_offset_hours: 2\nnews:\n  enabled: false\n")
    rc = cli.main(["research", "--start", "2024-01-01", "--end", "2024-10-25", "--mc", "4", "--max-combos", "2",
                   "--train-months", "3", "--test-months", "2"])
    assert rc == 0
    text = (tmp_path / "reports" / "research_summary.txt").read_text()
    for needle in ("== 1. Data ==", "Symbol: XAUUSDc", "BACKTEST VERDICT", "WALK-FORWARD VERDICT",
                   "Strategy settings:", "By direction", "ML filter", "Finished"):
        assert needle in text, needle
    assert "generated:" in (tmp_path / "config" / "optimized_params.yaml").read_text()
    assert "password" not in text.lower().replace("no passwords", "")
    assert (tmp_path / "reports" / "walkforward.html").exists() and (tmp_path / "data" / "xauusd_m15.csv").exists()
