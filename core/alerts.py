#!/usr/bin/env python3
"""
Operational alerts to a Discord or Slack incoming webhook.

Every execution-telemetry event passes through ``AlertSink.notify``; the ones
worth waking someone for (fills, rejections, EOD flatten results, breaker
trips, reconciliation mismatches, bot start/stop, the end-of-session report)
are formatted as one short line and POSTed to ALERT_WEBHOOK_URL.

- Sending happens on a background thread with a short timeout: a slow or
  broken webhook never delays trading.
- Repeated identical alerts are throttled (ALERT_MIN_INTERVAL_SECONDS per
  event+symbol) and capped per hour (ALERT_MAX_PER_HOUR).
- The payload carries both ``content`` (Discord) and ``text`` (Slack).

Treat the webhook URL as a secret: anyone with it can post to the channel.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_EVENTS = (
    "order_submitted",
    "order_rejected",
    "order_recovered",
    "order_filled",
    "flatten",
    "breaker_tripped",
    "reconciliation",
    "trailing_stop_failed",
    "session_report",
    "edge_gate",
    "edge_decay",
    "bot_started",
    "bot_stopped",
    "bot_error",
    "bot_restarted",
    "bot_restart_blocked",
    "revalidation",
    "not_flat",
    "flat_confirmed",
    "streak_lockout_active",
    "unprotected_position",
)


def format_alert(event: Dict[str, Any]) -> Optional[str]:
    """One human-readable line per event, or None if it shouldn't alert."""
    name = event.get("event")
    sym = event.get("symbol", "")
    if name == "order_submitted":
        return (
            f"🟢 {event.get('side', '').upper()} {event.get('qty')} {sym} submitted"
            + (f" (stop {event['stop_loss']}, target {event['take_profit']})" if event.get("stop_loss") else "")
        )
    if name == "order_rejected":
        return f"🔴 {sym} {event.get('side', '')} rejected [{event.get('category')}]: {event.get('message')}"
    if name == "order_recovered":
        return f"🟡 {sym} recovered after {event.get('category')}: {event.get('plan')}"
    if name == "order_filled":
        if not event.get("adverse"):
            return None  # only fills worse than SLIPPAGE_ALERT_BPS
        return (f"🐌 {event.get('side', '').upper()} {sym} filled {event.get('fill_price')} vs "
                f"{event.get('reference')} {event.get('reference_price')}: {event.get('slippage_bps'):+.1f} bps slippage")
    if name == "flatten":
        closed = ", ".join(c["symbol"] for c in event.get("closed") or []) or "nothing to close"
        failures = event.get("failures") or []
        text = f"🌙 EOD flatten: cancelled {event.get('cancelled_orders', 0)} orders, closed {closed}"
        if failures:
            text += "; ⚠️ FAILED: " + ", ".join(f"{f.get('symbol')} ({f.get('category')})" for f in failures)
        return text
    if name == "breaker_tripped":
        return f"🛑 {event.get('reason')}"
    if name == "reconciliation":
        discrepancies = event.get("discrepancies") or []
        if not discrepancies:
            return None  # clean reconciliations are routine
        kinds = ", ".join(f"{d['kind']} {d['symbol']}" for d in discrepancies)
        return f"⚠️ Broker/local mismatch ({event.get('mode')}): {kinds}"
    if name == "trailing_stop_failed":
        return f"⚠️ {sym} trailing stop not raised ({event.get('category')}): {event.get('message')}"
    if name == "session_report":
        return (
            f"📊 Session {event.get('date')}: P&L {event.get('pnl', 0):+,.2f} ({event.get('pnl_pct', 0):+.2f}%), "
            f"equity {event.get('equity', 0):,.2f}, {event.get('fills', 0)} fills, "
            f"{event.get('open_positions', 0)} open positions"
            + (f", slippage {q['mean_bps']:+.1f} bps avg" if (q := event.get("fill_quality") or {}).get("measured")
               else "")
            + (" ⚠️ NOT FLAT" if event.get("open_positions") and event.get("no_overnight") else "")
        )
    if name == "edge_gate":
        return None if event.get("passed") else f"🚧 Edge gate: live entries blocked: {event.get('reason')}"
    if name == "edge_decay":
        return f"📉 Live edge decayed, entries paused: {event.get('reason')}"
    if name == "bot_started":
        return f"▶️ Bot started: {event.get('mode')}, {event.get('timeframe')}, {event.get('symbols')}"
    if name == "bot_stopped":
        return f"⏹️ Bot stopped ({event.get('reason')})"
    if name == "not_flat":
        held = ", ".join(f"{p.get('symbol')} {p.get('qty')}" for p in event.get("positions") or []) or "open orders"
        return (f"🚨 NOT FLAT {event.get('minutes_to_close')} min before the close: {held}. "
                "Close them in the dashboard (Close everything) or at alpaca.markets.")
    if name == "streak_lockout_active":
        return (f"🧊 {event.get('losses')} losing trades in a row ({', '.join(event.get('symbols') or [])}): "
                f"new entries paused until {event.get('until_et')} ET")
    if name == "slippage_timeout":
        return f"💨 {sym} entry cancelled: {event.get('reason')}"
    if name == "unprotected_position":
        return (f"🚨 {sym}: {event.get('qty')} shares have NO stop-loss ({event.get('error')}). "
                "Set one or close it at alpaca.markets.")
    if name == "flat_confirmed":
        return f"✅ Flat for the day at {event.get('at_et')} ET"
    if name == "revalidation":
        result = event.get("result")
        if result == "validated":
            return "✅ Weekly re-validation passed: the strategy still shows an edge; model retrained."
        if result == "no_edge":
            return "🚧 Re-validation found NO edge any more: bots will stop opening new trades. Review on Get Started."
        return "⚠️ Re-validation could not run (see the dashboard log)."
    if name == "bot_restarted":
        return f"🔁 Bot restarted automatically ({event.get('reason')})"
    if name == "bot_restart_blocked":
        return f"⚠️ Bot is down and was NOT restarted: {event.get('reason')}"
    if name == "bot_error":
        return f"❗ Bot error: {event.get('message')}"
    return None


class AlertSink:
    def __init__(
        self,
        webhook_url: str,
        *,
        events: Iterable[str] = DEFAULT_EVENTS,
        min_interval_seconds: float = 60.0,
        max_per_hour: int = 60,
        post: Callable[..., Any] = requests.post,
        asynchronous: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.webhook_url = webhook_url
        self.events = set(events)
        self.min_interval = min_interval_seconds
        self.max_per_hour = max_per_hour
        self.post = post
        self.asynchronous = asynchronous
        self.clock = clock
        self._last_sent: Dict[str, float] = {}
        self._recent: Deque[float] = deque()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Optional["AlertSink"]:
        url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        if not url:
            return None
        events = [e.strip() for e in os.getenv("ALERT_EVENTS", ",".join(DEFAULT_EVENTS)).split(",") if e.strip()]
        return cls(
            url,
            events=events,
            min_interval_seconds=float(os.getenv("ALERT_MIN_INTERVAL_SECONDS", "60")),
            max_per_hour=int(os.getenv("ALERT_MAX_PER_HOUR", "60")),
        )

    def _allowed(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            while self._recent and now - self._recent[0] > 3600:
                self._recent.popleft()
            if len(self._recent) >= self.max_per_hour:
                return False
            last = self._last_sent.get(key)
            if last is not None and now - last < self.min_interval:
                return False
            self._last_sent[key] = now
            self._recent.append(now)
            return True

    def notify(self, event: Dict[str, Any]) -> bool:
        """Send if the event is selected, formattable and not throttled."""
        if event.get("event") not in self.events:
            return False
        text = format_alert(event)
        if not text:
            return False
        key = f"{event.get('event')}:{event.get('symbol', '')}:{event.get('category', '')}"
        if not self._allowed(key):
            return False
        if self.asynchronous:
            threading.Thread(target=self._send, args=(text,), daemon=True).start()
        else:
            self._send(text)
        return True

    def _send(self, text: str) -> None:
        try:
            resp = self.post(self.webhook_url, json={"content": text[:1900], "text": text}, timeout=5)
            if getattr(resp, "status_code", 200) >= 400:
                logger.warning(f"Alert webhook returned HTTP {resp.status_code}")
        except Exception as exc:  # alerts must never break trading
            logger.warning(f"Alert webhook failed: {exc}")
