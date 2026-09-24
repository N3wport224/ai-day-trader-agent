#!/usr/bin/env python3
"""
Intraday session clock for US equities (America/New_York).

Phases of a regular session (defaults shown for a normal 9:30-16:00 day):

    CLOSED            before 9:30, after the close, weekends
    OPENING_LOCKOUT   9:30 - 9:45   no new entries (opening-range whiplash)
    OPEN              9:45 - 15:45  entries allowed
    ENTRY_CUTOFF      15:45 - 15:50 no new entries; exits allowed
    FLATTEN           15:50 - close cancel working orders, close every position

Cutoff and flatten are defined as minutes *before the close*, so they move
with early-close days (e.g. 13:00 half-days) when the actual close is known:
live, from Alpaca's clock (``next_close`` while the market is open); in
backtests, from the last regular bar of each day.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class SessionPhase(str, Enum):
    CLOSED = "CLOSED"
    OPENING_LOCKOUT = "OPENING_LOCKOUT"
    OPEN = "OPEN"
    ENTRY_CUTOFF = "ENTRY_CUTOFF"
    FLATTEN = "FLATTEN"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SessionConfig:
    opening_lockout_minutes: int = 15
    entry_cutoff_minutes: int = 15      # before the close
    flatten_minutes: int = 10           # before the close
    no_overnight: bool = True

    def __post_init__(self) -> None:
        if min(self.opening_lockout_minutes, self.entry_cutoff_minutes, self.flatten_minutes) < 0:
            raise ValueError("Session minute settings must be >= 0")
        if self.flatten_minutes > self.entry_cutoff_minutes:
            raise ValueError("Flatten must not start before the entry cutoff")

    @classmethod
    def from_env(cls, no_overnight: Optional[bool] = None) -> "SessionConfig":
        return cls(
            opening_lockout_minutes=_env_int("OPENING_LOCKOUT_MINUTES", 15),
            entry_cutoff_minutes=_env_int("ENTRY_CUTOFF_MINUTES_BEFORE_CLOSE", 15),
            flatten_minutes=_env_int("FLATTEN_MINUTES_BEFORE_CLOSE", 10),
            no_overnight=_env_bool("NO_OVERNIGHT", True) if no_overnight is None else no_overnight,
        )


def to_market_time(ts: Any) -> datetime:
    """Any timestamp (aware, naive-UTC, or ISO string) -> aware America/New_York."""
    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    return stamp.tz_convert(MARKET_TZ).to_pydatetime()


@dataclass(frozen=True)
class SessionTimes:
    open: datetime
    lockout_end: datetime
    entry_cutoff: datetime
    flatten: datetime
    close: datetime


class SessionClock:
    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        self.config = config or SessionConfig.from_env()

    def times(self, day: date, close: Optional[Any] = None) -> SessionTimes:
        cfg = self.config
        open_ = datetime.combine(day, REGULAR_OPEN, tzinfo=MARKET_TZ)
        close_dt = to_market_time(close) if close is not None else datetime.combine(day, REGULAR_CLOSE, tzinfo=MARKET_TZ)
        return SessionTimes(
            open=open_,
            lockout_end=open_ + timedelta(minutes=cfg.opening_lockout_minutes),
            entry_cutoff=close_dt - timedelta(minutes=cfg.entry_cutoff_minutes),
            flatten=close_dt - timedelta(minutes=cfg.flatten_minutes),
            close=close_dt,
        )

    def phase(self, now: Any, close: Optional[Any] = None) -> SessionPhase:
        """Phase at ``now``. Pass the session's actual ``close`` on early-close days."""
        local = to_market_time(now)
        if local.weekday() >= 5:
            return SessionPhase.CLOSED
        t = self.times(local.date(), close)
        if local < t.open or local >= t.close:
            return SessionPhase.CLOSED
        if local < t.lockout_end:
            return SessionPhase.OPENING_LOCKOUT
        if local < t.entry_cutoff:
            return SessionPhase.OPEN
        if local < t.flatten or not self.config.no_overnight:
            # Without no-overnight there is nothing to liquidate; keep blocking entries.
            return SessionPhase.ENTRY_CUTOFF
        return SessionPhase.FLATTEN

    def can_enter(self, now: Any, close: Optional[Any] = None) -> bool:
        return self.phase(now, close) is SessionPhase.OPEN

    def must_flatten(self, now: Any, close: Optional[Any] = None) -> bool:
        return self.phase(now, close) is SessionPhase.FLATTEN

    def phase_from_alpaca_clock(self, clock: Dict[str, Any]) -> SessionPhase:
        """Use Alpaca's /v2/clock payload (timestamp, is_open, next_close)."""
        if not clock.get("is_open"):
            return SessionPhase.CLOSED
        return self.phase(clock.get("timestamp") or datetime.now(MARKET_TZ), clock.get("next_close"))

    def seconds_until_flatten(self, now: Any, close: Optional[Any] = None) -> Optional[float]:
        """Seconds until today's flatten time, or None if it has passed / market closed."""
        local = to_market_time(now)
        if not self.config.no_overnight or self.phase(local, close) in {SessionPhase.CLOSED, SessionPhase.FLATTEN}:
            return None
        return (self.times(local.date(), close).flatten - local).total_seconds()
