"""Local control panel: log in to MT5 from the browser, start/stop the bot, watch it trade.

Runs only on 127.0.0.1. Every API call needs the random token embedded in the page and a
localhost Host header, so other websites open in your browser cannot drive the bot.
The MT5 password and Claude API key are kept in memory only - never written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from ..config import Config
from ..journal import Journal
from ..stats import journal_stats

log = logging.getLogger("tulipai")
WEB = Path(__file__).resolve().parent.parent / "web"
HERE = Path(__file__).resolve().parent
SETTINGS = Path("runs/panel_settings.json")


class PanelApp:
    def __init__(self, cfg: Config, port: int):
        self.cfg = cfg
        self.port = port
        self.token = secrets.token_urlsafe(24)
        self.journal = Journal(cfg.live.journal_path)
        self.engine = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.mode = "mt5"

    # ------------------------------------------------------------------ actions
    def saved_settings(self) -> dict:
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def connect(self, body: dict) -> dict:
        from ..ai.brain import ClaudeBrain
        from ..broker.mt5 import MT5Broker
        from ..broker.paper import PaperBroker
        from ..live import LiveEngine
        from ..news import EconomicCalendar, HeadlineFeed

        with self.lock:
            if self.running:
                return {"ok": False, "error": "Stop the bot before reconnecting."}
            if self.engine is not None:
                self._disconnect()
            cfg = self.cfg
            ai_mode = body.get("ai_mode") or cfg.ai.mode
            if ai_mode not in ("off", "filter", "autonomous"):
                return {"ok": False, "error": "bad AI mode"}
            cfg.ai.mode = ai_mode
            api_key = (body.get("api_key") or "").strip()
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key  # this process only, never saved
            self.mode = "paper" if body.get("broker") == "paper" else "mt5"
            login = (body.get("login") or "").strip() or os.environ.get("MT5_LOGIN") or None
            password = body.get("password") or os.environ.get("MT5_PASSWORD") or None
            server = (body.get("server") or "").strip() or os.environ.get("MT5_SERVER") or None
            path = (body.get("path") or "").strip() or os.environ.get("MT5_PATH") or None
            if login and not str(login).isdigit():
                return {"ok": False, "error": "MT5 login must be the account number (digits only)."}
            try:
                mt5b = MT5Broker(cfg, login=login, password=password, server=server, path=path)
                broker = PaperBroker(mt5b, cfg) if self.mode == "paper" else mt5b
                brain = None
                if ai_mode != "off":
                    brain = ClaudeBrain(cfg.ai, cfg.risk, cfg.management, cfg.symbol.timeframe)
                calendar = EconomicCalendar(cfg.news) if cfg.news.enabled else None
                headlines = HeadlineFeed(cfg.news) if cfg.news.enabled and brain is not None else None  # Claude only
                ml = _load_ml(cfg)
                engine = LiveEngine(cfg, broker, self.journal, brain=brain, calendar=calendar, headlines=headlines,
                                    ml=ml, mode_label=self.mode)
                acct = engine.start()
            except Exception as exc:
                log.exception("connect failed")
                return {"ok": False, "error": str(exc)}
            self.engine = engine
            if body.get("remember"):
                SETTINGS.parent.mkdir(parents=True, exist_ok=True)
                SETTINGS.write_text(json.dumps({"login": login or "", "server": server or "", "path": path or "",
                                                "broker": self.mode, "ai_mode": ai_mode}), encoding="utf-8")
            return {"ok": True, "account": acct.__dict__, "warnings": engine.status.get("warnings", [])}

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> dict:
        with self.lock:
            if self.engine is None:
                return {"ok": False, "error": "Connect first."}
            if self.running:
                return {"ok": True}
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self.engine.run, args=(self.stop_event,), daemon=True, name="tulip-engine")
            self.thread.start()
            return {"ok": True}

    def stop(self) -> dict:
        with self.lock:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=30)
            self.thread = None
            if self.engine is not None:
                self.engine.status["state"] = "stopped"
            return {"ok": True}

    def pause(self, paused: bool) -> dict:
        if self.engine is None:
            return {"ok": False, "error": "Connect first."}
        if not paused and self.engine.account_info and not self.engine.account_info.is_demo \
                and not self.cfg.account.allow_real_account and self.mode == "mt5":
            return {"ok": False, "error": "Real account: set account.allow_real_account: true in the config to trade it."}
        self.engine.pause_entries = bool(paused)
        return {"ok": True}

    def close_all(self) -> dict:
        if self.engine is None:
            return {"ok": False, "error": "Connect first."}
        n = self.engine.close_all()
        return {"ok": True, "closed": n}

    def reset_halt(self) -> dict:
        if self.engine is None:
            return {"ok": False, "error": "Connect first."}
        self.engine.risk.reset_halt()
        self.engine._save_risk()
        return {"ok": True}

    def _disconnect(self) -> None:
        self.stop()
        try:
            self.engine.broker.shutdown()
        except Exception:
            pass
        self.engine = None

    def disconnect(self) -> dict:
        with self.lock:
            if self.engine is not None:
                self._disconnect()
            return {"ok": True}

    def status(self) -> dict:
        eng = self.engine
        ai_mode = eng.status.get("ai_mode", self.cfg.ai.mode) if eng is not None else self.cfg.ai.mode  # what really runs
        out = {"connected": eng is not None, "running": self.running, "mode": self.mode, "ai_mode": ai_mode,
               "risk_pct": self.cfg.risk.risk_per_trade_pct, "timeframe": self.cfg.symbol.timeframe}
        if eng is not None:
            acct = eng.account_info
            try:
                if not self.running:
                    acct = eng.broker.account()
                    eng.account_info = acct
                positions = [p.__dict__ for p in eng.broker.positions()]
            except Exception as exc:
                positions = []
                out["broker_error"] = str(exc)
            out.update({
                "account": acct.__dict__ if acct else None,
                "symbol": eng.broker.symbol,
                "positions": positions,
                "engine": {k: v for k, v in eng.status.items()},
                "risk_state": eng.risk.state.to_dict(),
                "paused": eng.pause_entries,
            })
        out["stats"] = journal_stats(self.journal, self.mode, self.cfg.ai)
        dec = self.journal.recent_decisions(25)
        out["decisions"] = dec.drop(columns=["extra"], errors="ignore").fillna("").to_dict("records") if len(dec) else []
        tr = self.journal.frame("trades", limit=20)
        out["trades"] = tr.drop(columns=["extra"], errors="ignore").fillna("").to_dict("records") if len(tr) else []
        return out


def _load_ml(cfg: Config):
    if not cfg.ml.enabled:
        return None
    if not Path(cfg.ml.model_path).exists():
        log.warning("ml.enabled is true but %s does not exist; run `tulip train-ml` first", cfg.ml.model_path)
        return None
    from ..ml import MetaLabelModel

    return MetaLabelModel.load(cfg.ml.model_path)


def _page(app: PanelApp) -> bytes:
    html = (HERE / "index.html").read_text(encoding="utf-8")
    html = html.replace("/*__THEME__*/", (WEB / "theme.css").read_text(encoding="utf-8"))
    html = html.replace("/*__CHARTS__*/", (WEB / "charts.js").read_text(encoding="utf-8"))
    html = html.replace("__TOKEN__", app.token)
    settings = {"ai_mode": app.cfg.ai.mode, **app.saved_settings()}  # config default, unless the user saved a choice
    html = html.replace("__SETTINGS__", json.dumps(settings).replace("</", "<\\/"))
    return html.encode("utf-8")


def make_handler(app: PanelApp):
    allowed_hosts = {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            log.debug("panel: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data, default=str).encode("utf-8"))

        def _guard(self, api: bool) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(403, {"error": "bad host"})
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Tulip-Token", ""), app.token):
                self._json(403, {"error": "bad token - reload the page"})
                return False
            return True

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                if self._guard(api=False):
                    self._send(200, _page(app), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                if self._guard(api=True):
                    try:
                        self._json(200, app.status())
                    except Exception as exc:
                        log.exception("status failed")
                        self._json(500, {"error": str(exc)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard(api=True):
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0") or 0), 64_000)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "bad json"})
                return
            routes = {
                "/api/connect": lambda: app.connect(body),
                "/api/start": app.start,
                "/api/stop": app.stop,
                "/api/pause": lambda: app.pause(bool(body.get("paused", True))),
                "/api/close_all": app.close_all,
                "/api/reset_halt": app.reset_halt,
                "/api/disconnect": app.disconnect,
            }
            fn = routes.get(self.path)
            if fn is None:
                self._json(404, {"error": "not found"})
                return
            try:
                self._json(200, fn())
            except Exception as exc:
                log.exception("panel action failed")
                self._json(500, {"ok": False, "error": str(exc)})

    return Handler


def serve(cfg: Config, port: int | None = None, open_browser: bool = True) -> None:
    port = port or cfg.live.panel_port
    app = PanelApp(cfg, port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"TulipAI control panel: {url}  (Ctrl+C to quit)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        app.disconnect()
        httpd.server_close()
