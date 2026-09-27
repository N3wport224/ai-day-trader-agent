#!/usr/bin/env python3
"""
Event-risk filter: no new entries around earnings or scheduled macro events.

A day-trading model learns from ordinary sessions. Earnings releases (and
Fed decisions, CPI and similar) produce gaps and whipsaws it has never seen,
and a gap can jump straight past a stop-loss. So:

- Earnings: no new entries in a stock from EARNINGS_BLACKOUT_DAYS_BEFORE
  trading days before its earnings date through EARNINGS_BLACKOUT_DAYS_AFTER
  days after. The defaults are 0 before / 1 after in day-trading mode (an
  after-close report gaps the next morning), and 1 before in swing mode, so a
  position is never carried into the report.
- Market-wide events: EVENT_BLACKOUT lists dates or time windows (ET), e.g.
  "2026-10-28 13:45-15:00; 2026-11-12". Every symbol is paused inside them.

Earnings dates come from Yahoo Finance (yfinance), cached for 12 hours in
data/earnings_cache.json. If they can't be fetched the filter fails open
(entries allowed) and logs it: a data outage must not freeze trading. Exits,
stops and the EOD flatten are never affected. EVENT_FILTER=false turns it off.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("America/New_York")
_WINDOW = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2}))?$")


def event_filter_enabled() -> bool:
    return os.getenv("EVENT_FILTER", "true").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class BlackoutWindow:
    day: date
    start: Optional[time] = None   # None = the whole day
    end: Optional[time] = None
    text: str = ""

    def contains(self, local: datetime) -> bool:
        if local.date() != self.day:
            return False
        return self.start is None or self.start <= local.time() < self.end


def parse_blackouts(spec: str) -> List[BlackoutWindow]:
    """ "2026-10-28 13:45-15:00; 2026-11-12" -> windows (bad entries are skipped with a warning)."""
    windows = []
    for raw in re.split(r"[;\n]", spec or ""):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        match = _WINDOW.match(text)
        try:
            if not match:
                raise ValueError("expected YYYY-MM-DD or YYYY-MM-DD HH:MM-HH:MM")
            day = date.fromisoformat(match.group(1))
            if match.group(2):
                start = time.fromisoformat(match.group(2).zfill(5))
                end = time.fromisoformat(match.group(3).zfill(5))
                if end <= start:
                    raise ValueError("the window ends before it starts")
                windows.append(BlackoutWindow(day, start, end, text))
            else:
                windows.append(BlackoutWindow(day, text=text))
        except ValueError as exc:
            logger.warning(f"Ignoring EVENT_BLACKOUT entry {text!r}: {exc}")
    return windows


def yfinance_earnings(symbol: str) -> List[datetime]:
    """Past and upcoming earnings timestamps from Yahoo Finance ([] for ETFs)."""
    import yfinance as yf

    frame = yf.Ticker(symbol).get_earnings_dates(limit=12)
    if frame is None or len(frame) == 0:
        return []
    return [ts.to_pydatetime() for ts in frame.index]


def _trading_days_between(start: date, end: date) -> int:
    """Signed weekday count from start to end (holidays ignored: slightly conservative)."""
    return int(np.busday_count(start, end)) if start <= end else -int(np.busday_count(end, start))


class EventCalendar:
    def __init__(
        self,
        cache_path: Path = Path("data/earnings_cache.json"),
        fetch: Optional[Callable[[str], List[datetime]]] = None,
        days_before: int = 0,
        days_after: int = 1,
        blackouts: Optional[List[BlackoutWindow]] = None,
        ttl_hours: float = 12.0,
        retry_hours: float = 1.0,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.cache_path = Path(cache_path)
        self.fetch = fetch or (lambda symbol: yfinance_earnings(symbol))
        self.days_before, self.days_after = max(0, days_before), max(0, days_after)
        self.blackouts = blackouts or []
        self.ttl, self.retry = timedelta(hours=ttl_hours), timedelta(hours=retry_hours)
        self.now_fn = now_fn
        self._cache: Dict[str, Dict] = self._load()

    @classmethod
    def from_env(cls, day_trading: bool = True, cache_path: Optional[Path] = None) -> "EventCalendar":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            cache_path=Path(cache_path or os.getenv("EARNINGS_CACHE_PATH", "data/earnings_cache.json")),
            days_before=_int("EARNINGS_BLACKOUT_DAYS_BEFORE", 0 if day_trading else 1),
            days_after=_int("EARNINGS_BLACKOUT_DAYS_AFTER", 1),
            blackouts=parse_blackouts(os.getenv("EVENT_BLACKOUT", "")),
        )

    # -- cache -----------------------------------------------------------------

    def _load(self) -> Dict[str, Dict]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(self.cache_path)  # atomic: the paper and live bots share this file
        except OSError as exc:
            logger.warning(f"Could not save the earnings cache: {exc}")

    def earnings_dates(self, symbol: str) -> Optional[List[date]]:
        """Earnings dates (ET) for ``symbol``; None when unknown (fetch failed)."""
        symbol = symbol.upper()
        now = self.now_fn()
        entry = self._cache.get(symbol) or {}
        fetched = entry.get("fetched_at")
        age = now - datetime.fromisoformat(fetched) if fetched else None
        fresh = age is not None and age < (self.ttl if entry.get("ok") else self.retry)
        if not fresh:
            try:
                stamps = self.fetch(symbol)
                entry = {"ok": True, "fetched_at": now.isoformat(),
                         "dates": sorted({(ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
                                          .astimezone(MARKET_TZ).date().isoformat() for ts in stamps})}
            except Exception as exc:  # network, parsing, rate limits
                logger.warning(f"Earnings dates for {symbol} unavailable ({exc}); not filtering it")
                entry = {"ok": False, "fetched_at": now.isoformat(), "dates": entry.get("dates") or []}
            self._cache[symbol] = entry
            self._save()
        if not entry.get("ok") and not entry.get("dates"):
            return None
        return [date.fromisoformat(d) for d in entry.get("dates") or []]

    # -- decisions ---------------------------------------------------------------

    def block_reason(self, symbol: str, now: Optional[datetime] = None) -> Optional[str]:
        """Why no new entry in ``symbol`` right now, or None."""
        local = (now or self.now_fn()).astimezone(MARKET_TZ)
        for window in self.blackouts:
            if window.contains(local):
                return f"Event blackout ({window.text}): no new entries"
        today = local.date()
        for day in self.earnings_dates(symbol) or []:
            offset = _trading_days_between(day, today)   # >0: after earnings, <0: before
            if -self.days_before <= offset <= self.days_after:
                when = "today" if offset == 0 else (f"{day:%a %b %d}")
                return (f"Earnings for {symbol} {when}: no new entries from {self.days_before} trading day(s) "
                        f"before to {self.days_after} after (EARNINGS_BLACKOUT_DAYS_BEFORE/AFTER)")
        return None

    def upcoming(self, symbols: List[str], days: int = 14) -> List[Dict[str, str]]:
        """Earnings in the next ``days`` calendar days, soonest first (dashboard)."""
        today = self.now_fn().astimezone(MARKET_TZ).date()
        rows = []
        for symbol in symbols:
            for day in self.earnings_dates(symbol) or []:
                if today <= day <= today + timedelta(days=days):
                    rows.append({"symbol": symbol.upper(), "date": day.isoformat()})
        return sorted(rows, key=lambda r: r["date"])
