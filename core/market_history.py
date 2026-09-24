#!/usr/bin/env python3
"""
Historical OHLCV bars for feature engineering, shared by training and live
inference so both see identically-shaped data.

Alpaca Market Data (paginated) is the primary source; Yahoo Finance is the
no-key fallback. The still-forming bar is always dropped: features must only
use candles that have closed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from core.features import OHLCV_COLUMNS

load_dotenv()
logger = logging.getLogger(__name__)

TIMEFRAMES = {
    # name: (bar length, Yahoo interval, Yahoo max history in days)
    "1Min": (timedelta(minutes=1), "1m", 7),
    "5Min": (timedelta(minutes=5), "5m", 59),
    "15Min": (timedelta(minutes=15), "15m", 59),
    "1Hour": (timedelta(hours=1), "1h", 729),
    "1Day": (timedelta(days=1), "1d", 3650),
}
TIMEFRAME_ALIASES = {
    "1m": "1Min", "1min": "1Min",
    "5m": "5Min", "5min": "5Min",
    "15m": "15Min", "15min": "15Min",
    "1h": "1Hour", "60m": "1Hour", "1hour": "1Hour",
    "1d": "1Day", "1day": "1Day", "d": "1Day",
}
MARKET_TZ = ZoneInfo("America/New_York")
INTRADAY_TIMEFRAMES = ("1Min", "5Min", "15Min")
# History needed per timeframe: EMA200 warm-up plus ~20 sessions for relative volume.
DEFAULT_LOOKBACK_DAYS = {"1Min": 30, "5Min": 45, "15Min": 60, "1Hour": 60, "1Day": 400}


def normalize_timeframe(timeframe: str) -> str:
    """Accept "5m", "5Min", "1h", "1Hour", ... and return the canonical name."""
    if timeframe in TIMEFRAMES:
        return timeframe
    canonical = TIMEFRAME_ALIASES.get(str(timeframe).strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; use one of {list(TIMEFRAMES)} or {list(TIMEFRAME_ALIASES)}"
        )
    return canonical


def is_intraday(timeframe: str) -> bool:
    return normalize_timeframe(timeframe) in INTRADAY_TIMEFRAMES


def bar_length(timeframe: str) -> timedelta:
    return TIMEFRAMES[normalize_timeframe(timeframe)][0]


def timeframe_from_length(length: timedelta) -> str:
    for name, (bar_len, _, _) in TIMEFRAMES.items():
        if bar_len == length:
            return name
    raise ValueError(f"Unsupported bar length {length}")


def drop_incomplete_bar(bars: pd.DataFrame, timeframe: str, now: Optional[datetime] = None) -> pd.DataFrame:
    """Remove bars whose period hasn't ended yet (their close isn't final)."""
    if bars.empty:
        return bars
    now = now or datetime.now(timezone.utc)
    closes_at = bars.index + bar_length(timeframe)
    return bars[closes_at <= now]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))


def fetch_alpaca_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    *,
    http_get: Callable = requests.get,
) -> pd.DataFrame:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID") or os.getenv("ALPACA_LIVE_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_LIVE_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Alpaca credentials not configured")

    base = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
    params = {
        "timeframe": timeframe,
        "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": 10000,
        "adjustment": "all",
        "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
        "sort": "asc",
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    rows = []
    while True:
        resp = http_get(f"{base}/v2/stocks/{symbol}/bars", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("bars") or [])
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    if not rows:
        return _empty()
    frame = pd.DataFrame(rows).rename(
        columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame.index = pd.to_datetime(frame.pop("time"), utc=True)
    return frame[OHLCV_COLUMNS].astype(float).sort_index()


def fetch_yahoo_bars(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf

    _, interval, max_days = TIMEFRAMES[normalize_timeframe(timeframe)]
    start = max(start, end - timedelta(days=max_days))
    hist = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=True)
    if hist.empty:
        return _empty()
    hist = hist.rename(columns=str.lower)[OHLCV_COLUMNS]
    hist.index = pd.to_datetime(hist.index, utc=True)
    return hist.astype(float).sort_index()


def fetch_bars(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Raw bars in [start, end] from Alpaca, falling back to Yahoo Finance."""
    timeframe = normalize_timeframe(timeframe)
    bars = _empty()
    try:
        bars = fetch_alpaca_bars(symbol, timeframe, start, end)
    except Exception as exc:
        logger.info(f"Alpaca bars unavailable for {symbol} ({exc}); trying Yahoo Finance")
    if bars.empty:
        try:
            bars = fetch_yahoo_bars(symbol, timeframe, start, end)
        except Exception as exc:
            logger.warning(f"Yahoo bars unavailable for {symbol}: {exc}")
    return bars[~bars.index.duplicated(keep="last")].dropna()


def get_history(
    symbol: str,
    timeframe: str = "1Hour",
    lookback_days: int = 60,
    *,
    end: Optional[datetime] = None,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Oldest-first OHLCV bars, completed candles only unless asked otherwise."""
    timeframe = normalize_timeframe(timeframe)
    end = end or datetime.now(timezone.utc)
    bars = fetch_bars(symbol, timeframe, end - timedelta(days=lookback_days), end)
    return bars if include_incomplete else drop_incomplete_bar(bars, timeframe, end)


@dataclass
class _CacheEntry:
    bars: pd.DataFrame
    full_at: datetime


class BarCache:
    """Per-symbol bar history for the live loop, fetched incrementally.

    The first request per symbol loads the whole lookback window; later ones
    fetch only from a few bars before the newest cached bar and replace that
    overlap (so a bar that was still forming, or a late correction, is
    overwritten). A full reload happens on each new market day and after
    ``full_refresh_hours``, so split/dividend-adjusted history never mixes
    old and new adjustments for long. Returned frames hold completed bars only.
    """

    def __init__(
        self,
        timeframe: str,
        lookback_days: int,
        *,
        fetch: Callable[[str, str, datetime, datetime], pd.DataFrame] = None,
        overlap_bars: int = 3,
        full_refresh_hours: float = 6.0,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.timeframe = normalize_timeframe(timeframe)
        self.lookback = timedelta(days=lookback_days)
        self.fetch = fetch or fetch_bars
        self.overlap = bar_length(self.timeframe) * max(1, overlap_bars)
        self.full_refresh = timedelta(hours=full_refresh_hours)
        self.now_fn = now_fn
        self._entries: Dict[str, _CacheEntry] = {}
        self.stats = {"full": 0, "incremental": 0, "failed_incremental": 0}

    def _needs_full(self, entry: Optional[_CacheEntry], now: datetime) -> bool:
        if entry is None or entry.bars.empty:
            return True
        if now - entry.full_at >= self.full_refresh:
            return True
        return entry.full_at.astimezone(MARKET_TZ).date() != now.astimezone(MARKET_TZ).date()

    def get(self, symbol: str) -> pd.DataFrame:
        now = self.now_fn()
        entry = self._entries.get(symbol)
        if self._needs_full(entry, now):
            bars = self.fetch(symbol, self.timeframe, now - self.lookback, now)
            self.stats["full"] += 1
            if bars is None or bars.empty:
                if entry is None:
                    return _empty()
                bars = entry.bars  # keep serving what we have; retry next call
            else:
                entry = self._entries[symbol] = _CacheEntry(bars.sort_index(), now)
        else:
            start = entry.bars.index[-1] - self.overlap
            fresh = self.fetch(symbol, self.timeframe, start.to_pydatetime(), now)
            if fresh is None or fresh.empty:
                self.stats["failed_incremental"] += 1
            else:
                self.stats["incremental"] += 1
                fresh = fresh.sort_index()
                kept = entry.bars[entry.bars.index < fresh.index[0]]
                combined = pd.concat([kept, fresh])
                combined = combined[~combined.index.duplicated(keep="last")]
                entry.bars = combined[combined.index >= now - self.lookback]
        return drop_incomplete_bar(entry.bars, self.timeframe, now)

    def invalidate(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._entries.clear()
        else:
            self._entries.pop(symbol, None)
