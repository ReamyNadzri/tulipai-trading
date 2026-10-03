"""High-impact economic calendar (NFP, CPI, FOMC, ...) for the news blackout.

Live: the free weekly ForexFactory JSON feed, cached on disk. Backtests: an optional CSV
with columns ``time,currency,impact,title`` (time in UTC).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ..config import NewsConfig
from ..risk import in_blackout


def parse_forexfactory(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        try:
            ts = pd.Timestamp(it["date"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except (KeyError, ValueError):
            continue
        rows.append({"time": ts, "currency": str(it.get("country", "")).upper(), "impact": str(it.get("impact", "")),
                     "title": str(it.get("title", "")), "forecast": str(it.get("forecast", "") or ""),
                     "previous": str(it.get("previous", "") or "")})
    df = pd.DataFrame(rows, columns=["time", "currency", "impact", "title", "forecast", "previous"])
    return df.sort_values("time").reset_index(drop=True)


class EconomicCalendar:
    def __init__(self, cfg: NewsConfig):
        self.cfg = cfg
        self.events = pd.DataFrame(columns=["time", "currency", "impact", "title", "forecast", "previous"])
        self._fetched_at = 0.0
        self.error = ""
        self.cache_file = Path(cfg.cache_dir) / "calendar.json"

    def load_csv(self, path: str | Path) -> "EconomicCalendar":
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for col in ("currency", "impact", "title", "forecast", "previous"):
            if col not in df.columns:
                df[col] = ""
        self.events = df.sort_values("time").reset_index(drop=True)
        self._fetched_at = float("inf")
        return self

    def refresh(self, force: bool = False) -> None:
        if not self.cfg.enabled:
            return
        if not force and time.time() - self._fetched_at < self.cfg.refresh_minutes * 60:
            return
        items = None
        try:
            resp = requests.get(self.cfg.calendar_url, timeout=15, headers={"User-Agent": "TulipAI/0.1"})
            resp.raise_for_status()
            items = resp.json()
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(items), encoding="utf-8")
            self.error = ""
        except (requests.RequestException, ValueError) as exc:
            self.error = f"calendar fetch failed: {exc}"
            if self.cache_file.exists():
                items = json.loads(self.cache_file.read_text(encoding="utf-8"))
        if items is not None:
            self.events = parse_forexfactory(items)
        self._fetched_at = time.time()

    def _relevant(self) -> pd.DataFrame:
        ev = self.events
        if ev.empty:
            return ev
        cur = {c.upper() for c in self.cfg.currencies}
        imp = {i.lower() for i in self.cfg.impacts}
        return ev[ev["currency"].isin(cur) & ev["impact"].str.lower().isin(imp)]

    def event_times(self) -> np.ndarray:
        ev = self._relevant()
        return np.sort(ev["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"))

    def blackout(self, now: pd.Timestamp, before_min: int, after_min: int) -> Optional[str]:
        if not in_blackout(self.event_times(), now, before_min, after_min):
            return None
        ev = self._relevant()
        near = ev[(ev["time"] >= now - pd.Timedelta(minutes=after_min)) & (ev["time"] <= now + pd.Timedelta(minutes=before_min))]
        return ", ".join(f"{r.title} {r.time:%H:%M}Z" for r in near.itertuples()) or "high-impact event"

    def upcoming(self, now: pd.Timestamp, hours: int = 24, all_impacts: bool = False) -> pd.DataFrame:
        ev = self.events if all_impacts else self._relevant()
        if ev.empty:
            return ev
        ev = ev[ev["currency"].isin({c.upper() for c in self.cfg.currencies})] if all_impacts else ev
        return ev[(ev["time"] >= now - pd.Timedelta(hours=1)) & (ev["time"] <= now + pd.Timedelta(hours=hours))]
