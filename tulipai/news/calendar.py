"""High-impact economic calendar (NFP, CPI, FOMC, ...) for the news blackout.

Live: the free weekly ForexFactory JSON feed, cached on disk, MERGED with a built-in copy of
the official US schedules (BLS jobs report and CPI, Fed FOMC decisions). The built-in list
means a feed outage never silently removes the blackout around the biggest gold movers.
Backtests: an optional CSV with columns ``time,currency,impact,title`` (time in UTC).
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

COLUMNS = ["time", "currency", "impact", "title", "forecast", "previous"]
CACHE_MAX_AGE_S = 7 * 86400  # the feed covers one week; an older cache is useless

# Official release dates (US Eastern time). Sources: federalreserve.gov FOMC calendar,
# bls.gov Employment Situation and CPI schedules. 2027 FOMC dates are the Fed's tentative
# schedule; 2027 jobs-report dates fall back to the usual first-Friday rule below.
_FOMC = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28",
         "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15",
         "2027-10-27", "2027-12-08"]
_NFP = ["2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05", "2026-07-02",
        "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"]
_CPI = ["2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14",
        "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10"]


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    df["time"] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


def estimated_nfp_date(year: int, month: int) -> pd.Timestamp:
    """Estimated jobs-report date for the release in ``year``/``month`` (data for the previous month).

    BLS normally publishes on the third Friday after the end of the reference week (the week
    containing the 12th). This rule reproduces the published 2026 schedule except shutdown and
    holiday shifts; a release that would land on New Year's Day moves a week later, and one on the
    Independence Day holiday moves to the day before.
    """
    ref = pd.Timestamp(year=year, month=month, day=1) - pd.offsets.MonthBegin(1)  # first day of reference month
    twelfth = ref + pd.Timedelta(days=11)
    week_end = twelfth + pd.Timedelta(days=(5 - twelfth.weekday()) % 7)  # Saturday ending the reference week
    first_friday_after = week_end + pd.Timedelta(days=6)
    release = first_friday_after + pd.Timedelta(days=14)
    if release.month == 1 and release.day <= 2:
        release += pd.Timedelta(days=7)
    if release.month == 7 and release.day in (3, 4):
        release -= pd.Timedelta(days=1)
    return release


def builtin_schedule() -> pd.DataFrame:
    """Official US high-impact releases as UTC timestamps (DST handled via New York time)."""
    rows = []

    def add(day: str | pd.Timestamp, hh: int, mm: int, title: str) -> None:
        local = pd.Timestamp(day).replace(hour=hh, minute=mm).tz_localize("America/New_York")
        rows.append({"time": local.tz_convert("UTC"), "currency": "USD", "impact": "High", "title": title,
                     "forecast": "", "previous": ""})

    for d in _FOMC:
        add(d, 14, 0, "FOMC rate decision (built-in)")
    for d in _NFP:
        add(d, 8, 30, "Non-farm payrolls (built-in)")
    for d in _CPI:
        add(d, 8, 30, "CPI (built-in)")
    for month in range(1, 13):  # 2027 jobs reports are not published yet: BLS rule estimate
        add(estimated_nfp_date(2027, month), 8, 30, "Non-farm payrolls (built-in, estimated)")
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values("time").reset_index(drop=True)


def parse_forexfactory(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items or []:
        try:
            ts = pd.Timestamp(it["date"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        rows.append({"time": ts, "currency": str(it.get("country", "")).upper(), "impact": str(it.get("impact", "")),
                     "title": str(it.get("title", "")), "forecast": str(it.get("forecast", "") or ""),
                     "previous": str(it.get("previous", "") or "")})
    if not rows:
        return _empty()
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


class EconomicCalendar:
    def __init__(self, cfg: NewsConfig, use_builtin: bool = True):
        self.cfg = cfg
        self.use_builtin = use_builtin
        self.feed_events = _empty()
        self.events = builtin_schedule() if use_builtin else _empty()
        self._fetched_at = 0.0
        self.error = ""
        self.cache_file = Path(cfg.cache_dir) / "calendar.json"

    def load_csv(self, path: str | Path) -> "EconomicCalendar":
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for col in COLUMNS[1:]:
            if col not in df.columns:
                df[col] = ""
        self.events = df[COLUMNS].sort_values("time").reset_index(drop=True)
        self.use_builtin = False
        self._fetched_at = float("inf")
        return self

    def _merge(self) -> None:
        parts = [self.feed_events] + ([builtin_schedule()] if self.use_builtin else [])
        parts = [p for p in parts if not p.empty]
        self.events = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True) if parts else _empty()

    def builtin_coverage(self, now: pd.Timestamp | None = None) -> str:
        """Which built-in release types still have future dates (they run out over time)."""
        if not self.use_builtin:
            return "no news blackout"
        now = now or pd.Timestamp.now(tz="UTC")
        b = builtin_schedule()
        future = b[b["time"] >= now]["title"]
        kinds = [k for k, key in (("NFP", "payrolls"), ("CPI", "CPI"), ("FOMC", "FOMC")) if future.str.contains(key).any()]
        if not kinds:
            return "the built-in US schedule has expired - NO news blackout until the feed works (update TulipAI)"
        missing = [k for k in ("NFP", "CPI", "FOMC") if k not in kinds]
        return (f"built-in US schedule still covers {', '.join(kinds)}"
                + (f" (not {', '.join(missing)})" if missing else ""))

    @staticmethod
    def _valid_feed(items) -> bool:
        if not isinstance(items, list):
            return False
        return not items or not parse_forexfactory(items).empty

    def refresh(self, force: bool = False) -> None:
        """Download the weekly feed. Never raises: any problem becomes ``self.error`` and the
        last good data (or the built-in schedule) keeps the blackout working."""
        if not self.cfg.enabled:
            return
        if not force and time.time() - self._fetched_at < self.cfg.refresh_minutes * 60:
            return
        self._fetched_at = time.time()  # also rate-limits retries after a failure
        items, problem = None, ""
        try:
            resp = requests.get(self.cfg.calendar_url, timeout=15, headers={"User-Agent": "TulipAI/0.1"})
            resp.raise_for_status()
            payload = resp.json()
            if self._valid_feed(payload):
                items = payload
            else:
                problem = "unexpected reply from the calendar site"
        except (requests.RequestException, ValueError) as exc:
            problem = f"download failed ({exc.__class__.__name__})"
        except Exception as exc:  # never let the news feed break the trading loop
            problem = f"download failed ({exc.__class__.__name__}: {exc})"
        if items is not None:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.cache_file.write_text(json.dumps(items), encoding="utf-8")
            except OSError:
                pass  # the cache is only a convenience
            self.error = ""
        else:
            self.error = f"news calendar {problem}; {self.builtin_coverage()}"
            try:
                if self.cache_file.exists() and time.time() - self.cache_file.stat().st_mtime < CACHE_MAX_AGE_S:
                    cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
                    if self._valid_feed(cached):
                        items = cached
                        self.error += "; using this week's cached calendar"
            except (OSError, ValueError):
                pass
        if items is not None:
            self.feed_events = parse_forexfactory(items)
        self._merge()

    def _relevant(self) -> pd.DataFrame:
        ev = self.events
        if ev.empty:
            return ev
        cur = {c.upper() for c in self.cfg.currencies}
        imp = {i.lower() for i in self.cfg.impacts}
        return ev[ev["currency"].isin(cur) & ev["impact"].str.lower().isin(imp)]

    def event_times(self) -> np.ndarray:
        ev = self._relevant()
        if ev.empty:
            return np.array([], dtype="datetime64[ns]")
        t = pd.to_datetime(ev["time"], utc=True)
        return np.sort(t.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"))

    def blackout(self, now: pd.Timestamp, before_min: int, after_min: int) -> Optional[str]:
        if not in_blackout(self.event_times(), now, before_min, after_min):
            return None
        ev = self._relevant()
        near = ev[(ev["time"] >= now - pd.Timedelta(minutes=after_min)) & (ev["time"] <= now + pd.Timedelta(minutes=before_min))]
        near = near.drop_duplicates(subset="time")  # the feed and the built-in list often name the same release
        return ", ".join(f"{r.title} {r.time:%H:%M}Z" for r in near.itertuples()) or "high-impact event"

    def upcoming(self, now: pd.Timestamp, hours: int = 24, all_impacts: bool = False) -> pd.DataFrame:
        ev = self.events if all_impacts else self._relevant()
        if ev.empty:
            return ev
        ev = ev[ev["currency"].isin({c.upper() for c in self.cfg.currencies})] if all_impacts else ev
        return ev[(ev["time"] >= now - pd.Timedelta(hours=1)) & (ev["time"] <= now + pd.Timedelta(hours=hours))]
