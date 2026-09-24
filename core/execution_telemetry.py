#!/usr/bin/env python3
"""
Structured execution telemetry and broker-rejection classification.

Every order rejection, recovery attempt, reconciliation finding and stop
adjustment is written as one JSON object per line (JSONL) to
EXECUTION_LOG_PATH (default logs/execution_events.jsonl) and mirrored to the
Python logger, so incidents can be grepped, charted or fed to alerting.

Note on "wash" rejections: Alpaca rejects orders that could trade against
your own open orders ("potential wash trade detected"), e.g. a market sell
while bracket legs are working. That is self-trade prevention, not the IRS
wash-sale rule, which is a tax treatment and never blocks an order.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("execution")

# category -> (root cause, remediation)
REJECTION_CATEGORIES: Dict[str, tuple] = {
    "buying_power": (
        "Insufficient buying power: pending orders or open positions already use the cash.",
        "Retried once at the size current buying power allows; otherwise reduce size or wait for fills.",
    ),
    "margin_or_pdt": (
        "Margin / pattern-day-trader restriction on the account.",
        "Check day-trade count and account equity (PDT needs $25k); avoid same-day round trips.",
    ),
    "wash_trade": (
        "Potential wash trade: the order could cross your own open orders on this symbol (e.g. bracket legs).",
        "For exits the symbol's open orders are cancelled and the sell retried once; entries are not retried.",
    ),
    "qty_held": (
        "Shares are reserved by open orders (bracket stop/target legs) or the position is smaller than requested.",
        "Open orders for the symbol are cancelled and the sell retried once with the available quantity.",
    ),
    "short_restriction": (
        "Short sale not allowed (asset not shortable or account not permitted).",
        "The bot is long-only; this indicates a sell larger than the position. Reconcile positions.",
    ),
    "invalid_price": (
        "Bracket stop/target invalid relative to the current price (moved since sizing) or bad tick size.",
        "Retried once with levels recomputed from a fresh quote.",
    ),
    "not_tradable": (
        "Asset not tradable/active, or the market is closed for this order type.",
        "Skip the symbol; check Alpaca asset status and market hours.",
    ),
    "account_restricted": (
        "Account blocked or not authorised for this action (HTTP 403).",
        "Check the Alpaca dashboard for account status and API key permissions.",
    ),
    "rate_limited": (
        "Alpaca API rate limit reached (HTTP 429).",
        "Back off; the next cycle will retry. Reduce watchlist size or scan frequency.",
    ),
    "unknown": (
        "Unclassified broker rejection.",
        "Inspect the raw broker message in the telemetry event.",
    ),
}

_PATTERNS = [
    ("wash_trade", ("wash trade",)),
    ("buying_power", ("insufficient buying power", "buying power")),
    ("margin_or_pdt", ("pattern day", "day trading", "pdt", "margin", "daytrade")),
    ("qty_held", ("insufficient qty", "held for orders", "qty available", "exceeds position")),
    ("short_restriction", ("shortable", "short sale", "cannot be sold short")),
    ("invalid_price", ("stop_price", "limit_price", "base_price", "take_profit", "stop_loss", "sub-penny", "tick")),
    ("not_tradable", ("not tradable", "not active", "market is closed", "asset")),
]


@dataclass(frozen=True)
class Rejection:
    category: str
    root_cause: str
    remediation: str
    http_status: Optional[int]
    broker_code: Optional[int]
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_rejection(http_status: Optional[int], body: str) -> Rejection:
    """Map an Alpaca error response to a category, root cause and remediation."""
    message, code = body or "", None
    try:
        payload = json.loads(body)
        message = str(payload.get("message") or body)
        code = payload.get("code")
    except (ValueError, TypeError, AttributeError):
        pass
    text = message.lower()

    category = "unknown"
    for name, needles in _PATTERNS:
        if any(n in text for n in needles):
            category = name
            break
    if category == "unknown":
        if http_status == 429:
            category = "rate_limited"
        elif http_status == 403:
            category = "account_restricted"
    root_cause, remediation = REJECTION_CATEGORIES[category]
    return Rejection(category, root_cause, remediation, http_status, code, message)


class EventLog:
    """Append-only JSONL telemetry. ``path=None`` keeps events in memory only."""

    def __init__(self, path: Optional[str] = None, keep_last: int = 500) -> None:
        self.path = Path(path) if path else None
        self.keep_last = keep_last
        self.events: List[Dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> "EventLog":
        return cls(os.getenv("EXECUTION_LOG_PATH", "logs/execution_events.jsonl") or None)

    def record(self, event: str, level: int = logging.INFO, **fields: Any) -> Dict[str, Any]:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self.events.append(entry)
        del self.events[: -self.keep_last]
        line = json.dumps(entry, default=str)
        logger.log(level, line)
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(line + "\n")
            except OSError as exc:  # telemetry must never break trading
                logger.error(f"Could not write telemetry to {self.path}: {exc}")
        return entry
