"""SQLite trade journal - the live benchmark record.

Every bar decision, every trade, equity snapshots, AI calls (with token usage) and
"shadow" trades (signals the AI/ML vetoed, scored later to measure whether vetoes helped).
A new connection per call keeps it safe to use from the engine thread and the panel.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, bar_time TEXT, signal INTEGER, strategy TEXT, ml_prob REAL,
    ai_action TEXT, ai_confidence REAL, ai_reasoning TEXT, final_action TEXT, reason TEXT, extra TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket INTEGER, mode TEXT, symbol TEXT, side INTEGER, volume REAL,
    open_time TEXT, entry REAL, sl REAL, tp REAL, initial_sl_dist REAL,
    close_time TEXT, exit REAL, pnl REAL, r_multiple REAL, exit_reason TEXT,
    strategy TEXT, ai_confidence REAL, ml_prob REAL, risk_money REAL, extra TEXT
);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, balance REAL, equity REAL);
CREATE TABLE IF NOT EXISTS ai_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, model TEXT, mode TEXT, action TEXT, confidence REAL,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, latency_s REAL, error TEXT,
    reasoning TEXT
);
CREATE TABLE IF NOT EXISTS shadows (
    id INTEGER PRIMARY KEY AUTOINCREMENT, bar_time TEXT, side INTEGER, sl_dist REAL, tp_dist REAL,
    source TEXT, strategy TEXT, resolved INTEGER DEFAULT 0, r_multiple REAL, exit_reason TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS ix_trades_ticket ON trades(ticket);
CREATE INDEX IF NOT EXISTS ix_decisions_ts ON decisions(ts);
"""


def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    return pd.Timestamp(ts).isoformat()


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        # One persistent connection per thread (engine thread, panel threads).
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        with self._lock:
            yield conn
            conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._conn() as c:
            cur = c.execute(sql, args)
            return int(cur.lastrowid or 0)

    # ------------------------------------------------------------------ writes
    def log_decision(self, bar_time, signal: int = 0, strategy: str = "", ml_prob: float | None = None,
                     ai_action: str = "", ai_confidence: float | None = None, ai_reasoning: str = "",
                     final_action: str = "HOLD", reason: str = "", extra: dict | None = None) -> int:
        return self._exec(
            "INSERT INTO decisions (ts, bar_time, signal, strategy, ml_prob, ai_action, ai_confidence, ai_reasoning,"
            " final_action, reason, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_iso(pd.Timestamp.now(tz="UTC")), _iso(bar_time), int(signal), strategy, ml_prob, ai_action,
             ai_confidence, ai_reasoning, final_action, reason, json.dumps(extra or {}, default=str)),
        )

    def open_trade(self, ticket: int, mode: str, symbol: str, side: int, volume: float, open_time, entry: float,
                   sl: float, tp: float, initial_sl_dist: float, strategy: str, ai_confidence: float | None,
                   ml_prob: float | None, risk_money: float, extra: dict | None = None) -> int:
        return self._exec(
            "INSERT INTO trades (ticket, mode, symbol, side, volume, open_time, entry, sl, tp, initial_sl_dist, strategy,"
            " ai_confidence, ml_prob, risk_money, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(ticket), mode, symbol, int(side), float(volume), _iso(open_time), float(entry), float(sl), float(tp),
             float(initial_sl_dist), strategy, ai_confidence, ml_prob, float(risk_money),
             json.dumps(extra or {}, default=str)),
        )

    def update_trade_sl(self, ticket: int, sl: float) -> None:
        self._exec("UPDATE trades SET sl=? WHERE ticket=? AND close_time IS NULL", (float(sl), int(ticket)))

    def close_trade(self, ticket: int, close_time, exit_price: float, pnl: float, exit_reason: str) -> None:
        row = self.trade_by_ticket(ticket)
        risk_money = float(row["risk_money"]) if row and row["risk_money"] else 0.0
        r = pnl / risk_money if risk_money > 0 else 0.0
        self._exec(
            "UPDATE trades SET close_time=?, exit=?, pnl=?, r_multiple=?, exit_reason=? WHERE ticket=? AND close_time IS NULL",
            (_iso(close_time), float(exit_price), float(pnl), float(r), exit_reason, int(ticket)),
        )

    def log_equity(self, ts, balance: float, equity: float) -> None:
        self._exec("INSERT INTO equity (ts, balance, equity) VALUES (?,?,?)", (_iso(ts), float(balance), float(equity)))

    def log_ai_call(self, model: str, mode: str, action: str, confidence: float | None, usage: dict,
                    latency_s: float, error: str = "", reasoning: str = "") -> None:
        self._exec(
            "INSERT INTO ai_calls (ts, model, mode, action, confidence, input_tokens, output_tokens, cache_read_tokens,"
            " latency_s, error, reasoning) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_iso(pd.Timestamp.now(tz="UTC")), model, mode, action, confidence, int(usage.get("input_tokens", 0)),
             int(usage.get("output_tokens", 0)), int(usage.get("cache_read_input_tokens", 0)), float(latency_s),
             error, reasoning),
        )

    def add_shadow(self, bar_time, side: int, sl_dist: float, tp_dist: float, source: str, strategy: str) -> None:
        self._exec("INSERT INTO shadows (bar_time, side, sl_dist, tp_dist, source, strategy) VALUES (?,?,?,?,?,?)",
                   (_iso(bar_time), int(side), float(sl_dist), float(tp_dist), source, strategy))

    def resolve_shadow(self, shadow_id: int, r_multiple: float, exit_reason: str) -> None:
        self._exec("UPDATE shadows SET resolved=1, r_multiple=?, exit_reason=? WHERE id=?",
                   (float(r_multiple), exit_reason, int(shadow_id)))

    def set_kv(self, key: str, value: Any) -> None:
        self._exec("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, json.dumps(value, default=str)))

    # ------------------------------------------------------------------ reads
    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def trade_by_ticket(self, ticket: int):
        with self._conn() as c:
            return c.execute("SELECT * FROM trades WHERE ticket=? ORDER BY id DESC LIMIT 1", (int(ticket),)).fetchone()

    def open_trades(self, mode: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM trades WHERE close_time IS NULL", ()
        if mode:
            sql, args = sql + " AND mode=?", (mode,)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def unresolved_shadows(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM shadows WHERE resolved=0").fetchall()]

    def frame(self, table: str, limit: int | None = None, order: str = "id") -> pd.DataFrame:
        if table not in {"decisions", "trades", "equity", "ai_calls", "shadows"}:
            raise ValueError(table)
        order_col = "ts" if table == "equity" else order
        sql = f"SELECT * FROM {table} ORDER BY {order_col} DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._conn() as c:
            df = pd.read_sql_query(sql, c)
        return df.iloc[::-1].reset_index(drop=True)

    def recent_decisions(self, limit: int = 25) -> pd.DataFrame:
        """Decisions where something happened (signal, AI call or order) - skips idle bars."""
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT * FROM decisions WHERE signal != 0 OR ai_action != '' OR final_action != 'HOLD' "
                "ORDER BY id DESC LIMIT ?", c, params=(int(limit),))
        return df.iloc[::-1].reset_index(drop=True)

    def recent_closed_trades(self, n: int = 10) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE close_time IS NOT NULL ORDER BY close_time DESC LIMIT ?", c, params=(n,))
        return df
