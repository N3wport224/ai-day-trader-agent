#!/usr/bin/env python3
"""
Live fill quality: how far real fills land from the price we expected.

Each cycle the bot hands today's Alpaca orders (bracket legs nested) to
``FillTracker.update``. Every newly filled order is compared with its
reference price and written to telemetry as ``order_filled``:

  market entries / exits   the live quote (or last price) used when submitting,
                           from AlpacaExecutor.expected_prices
  stop legs                the stop price (slippage = how far through the stop)
  limit / take-profit      the limit price (usually zero or price improvement)

Slippage is signed so positive always means worse for us, in basis points:
buys (fill - ref) / ref, sells (ref - fill) / ref. Compare the session mean
with the backtest's --slippage-bps / --spread-bps assumptions; if live is
consistently worse, the backtest is flattering the strategy.

Processed order ids are persisted (FILL_STATE_PATH) so a restart neither
double-reports nor misses fills from earlier in the session.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.execution_telemetry import EventLog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillRecord:
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    fill_price: float
    reference_price: Optional[float]
    reference: str                 # quote | stop | limit | none
    slippage_bps: Optional[float]  # positive = adverse
    filled_at: str


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def slippage_bps(side: str, fill: float, reference: float) -> Optional[float]:
    if fill <= 0 or reference <= 0:
        return None
    move = (fill - reference) if side == "buy" else (reference - fill)
    return round(move / reference * 10_000, 2)


def _walk(orders: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for order in orders or []:
        yield order
        yield from _walk(order.get("legs") or [])


class FillTracker:
    def __init__(
        self,
        expected_prices: Optional[Mapping[str, float]] = None,
        *,
        state_path: Optional[str] = None,
        telemetry: Optional[EventLog] = None,
        alert_bps: Optional[float] = None,
    ) -> None:
        self.expected_prices = expected_prices if expected_prices is not None else {}
        self.state_path = Path(state_path or os.getenv("FILL_STATE_PATH", "data/fill_state.json"))
        self.telemetry = telemetry or EventLog.from_env()
        # A single fill this much worse than expected is logged as a warning.
        self.alert_bps = alert_bps if alert_bps is not None else float(os.getenv("SLIPPAGE_ALERT_BPS", "25"))
        self._day: Optional[str] = None
        self._seen: set = set()
        self.records: List[FillRecord] = []
        self._saved_expected: Dict[str, float] = {}  # survives restarts
        self._load()

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        self._day = state.get("day")
        self._seen = set(state.get("seen") or [])
        self.records = [FillRecord(**r) for r in state.get("records") or []]
        self._saved_expected = {k: float(v) for k, v in (state.get("expected") or {}).items()}

    def _save(self) -> None:
        pending = {**self._saved_expected, **self.expected_prices}
        state = {
            "day": self._day,
            "seen": sorted(self._seen),
            "records": [asdict(r) for r in self.records],
            "expected": {k: v for k, v in pending.items() if k not in self._seen},
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(self.state_path)
        except OSError as exc:
            logger.error(f"Could not persist fill state: {exc}")

    def _roll(self, session_date: date) -> None:
        day = str(session_date)
        if self._day != day:
            self._day, self._seen, self.records = day, set(), []
            self._saved_expected = {}

    # ------------------------------------------------------------------

    def _reference(self, order: Dict[str, Any]) -> tuple[Optional[float], str]:
        order_id = order.get("id") or ""
        expected = self.expected_prices.get(order_id) or self._saved_expected.get(order_id)
        if expected:
            return float(expected), "quote"
        kind = str(order.get("type", order.get("order_type", ""))).lower()
        if kind in {"stop", "stop_limit", "trailing_stop"} and _num(order.get("stop_price")) > 0:
            return _num(order.get("stop_price")), "stop"
        if kind == "limit" and _num(order.get("limit_price")) > 0:
            return _num(order.get("limit_price")), "limit"
        return None, "none"

    def update(self, orders: Iterable[Dict[str, Any]], session_date: date) -> List[FillRecord]:
        """Record fills not seen before; returns the new ones."""
        self._roll(session_date)
        new: List[FillRecord] = []
        for order in _walk(orders):
            order_id = order.get("id")
            if not order_id or order_id in self._seen or str(order.get("status", "")).lower() != "filled":
                continue
            fill = _num(order.get("filled_avg_price"))
            if fill <= 0:
                continue
            side = str(order.get("side", "")).lower()
            reference, source = self._reference(order)
            record = FillRecord(
                order_id=order_id,
                symbol=str(order.get("symbol") or ""),
                side=side,
                order_type=str(order.get("type", order.get("order_type", ""))).lower(),
                qty=_num(order.get("filled_qty") or order.get("qty")),
                fill_price=fill,
                reference_price=reference,
                reference=source,
                slippage_bps=slippage_bps(side, fill, reference) if reference else None,
                filled_at=str(order.get("filled_at") or ""),
            )
            self._seen.add(order_id)
            self.records.append(record)
            new.append(record)
            adverse = record.slippage_bps is not None and record.slippage_bps >= self.alert_bps
            self.telemetry.record(
                "order_filled", logging.WARNING if adverse else logging.INFO, **asdict(record), adverse=adverse
            )
        if new or any(k not in self._saved_expected for k in self.expected_prices):
            self._saved_expected.update(self.expected_prices)
            self._save()
        return new

    def summary(self) -> Dict[str, Any]:
        """Session fill-quality stats (for the end-of-session report)."""
        measured = [r for r in self.records if r.slippage_bps is not None]

        def stats(rows: List[FillRecord]) -> Dict[str, Any]:
            if not rows:
                return {"fills": 0}
            values = [r.slippage_bps for r in rows]
            notional = sum(r.qty * r.fill_price for r in rows)
            weighted = sum(r.slippage_bps * r.qty * r.fill_price for r in rows) / notional if notional else 0.0
            worst = max(rows, key=lambda r: r.slippage_bps)
            return {
                "fills": len(rows),
                "mean_bps": round(sum(values) / len(values), 2),
                "notional_weighted_bps": round(weighted, 2),
                "worst_bps": worst.slippage_bps,
                "worst": f"{worst.side} {worst.symbol} ({worst.order_type})",
                "cost_usd": round(sum(r.slippage_bps / 10_000 * r.qty * r.fill_price for r in rows), 2),
            }

        by_ref = {src: stats([r for r in measured if r.reference == src]) for src in ("quote", "stop", "limit")}
        return {
            "fills": len(self.records),
            "measured": len(measured),
            **{k: v for k, v in stats(measured).items() if k != "fills"},
            "by_reference": {k: v for k, v in by_ref.items() if v["fills"]},
        }
