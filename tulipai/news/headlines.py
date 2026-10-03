"""Market headlines from public RSS/Atom feeds, filtered for what moves gold."""

from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Optional

import pandas as pd
import requests

from ..config import NewsConfig

_TAG = re.compile(r"<[^>]+>")


def _text(el: Optional[ET.Element]) -> str:
    if el is None or el.text is None:
        return ""
    return html.unescape(_TAG.sub("", el.text)).strip()


def _when(s: str) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        return pd.Timestamp(parsedate_to_datetime(s)).tz_convert("UTC")
    except (TypeError, ValueError, IndexError):
        try:
            ts = pd.Timestamp(s)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except ValueError:
            return None


def parse_feed(xml_text: str, source: str = "") -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        children = {c.tag.split("}")[-1]: c for c in it}
        title = _text(children.get("title"))
        stamp = _when(_text(children.get("pubDate")) or _text(children.get("published")) or _text(children.get("updated")))
        if title:
            items.append({"time": stamp, "title": title, "source": source})
    return items


class HeadlineFeed:
    def __init__(self, cfg: NewsConfig):
        self.cfg = cfg
        self.items: list[dict] = []
        self.errors: list[str] = []
        self._fetched_at = 0.0

    def refresh(self, force: bool = False) -> None:
        if not self.cfg.enabled or (not force and time.time() - self._fetched_at < self.cfg.refresh_minutes * 60):
            return
        items, errors = [], []
        for url in self.cfg.feeds:
            try:
                resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 TulipAI/0.1"})
                resp.raise_for_status()
                items.extend(parse_feed(resp.text, source=re.sub(r"^https?://(www\.)?", "", url).split("/")[0]))
            except (requests.RequestException, ET.ParseError) as exc:
                errors.append(f"{url}: {exc.__class__.__name__}")
        if items:
            self.items = items
        self.errors = errors
        self._fetched_at = time.time()

    def relevant(self, now: pd.Timestamp, hours: int = 12) -> list[dict]:
        kws = [k.lower() for k in self.cfg.keywords]
        seen, out = set(), []
        for it in sorted(self.items, key=lambda x: x["time"] or pd.Timestamp(0, tz="UTC"), reverse=True):
            t = it["title"]
            key = t.lower()[:80]
            if key in seen:
                continue
            if it["time"] is not None and it["time"] < now - pd.Timedelta(hours=hours):
                continue
            if any(k in key or k in t.lower() for k in kws):
                seen.add(key)
                out.append(it)
            if len(out) >= self.cfg.max_headlines:
                break
        return out
