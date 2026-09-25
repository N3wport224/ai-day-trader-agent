#!/usr/bin/env python3
"""
Smart limit entries: never cross the spread blind, never chase a runaway.

A BUY goes out as a bracket order with a limit at the current ask plus a
small buffer (ENTRY_LIMIT_BUFFER_BPS, at least ENTRY_LIMIT_MIN_BUFFER dollars),
instead of a raw market order. Then ``SmartLimitChaser`` watches it:

1. If it hasn't filled within ENTRY_FILL_TIMEOUT_SECONDS (10), check the
   latest quote.
   - Ask still within ENTRY_MAX_SLIPPAGE_BPS (15 = 0.15%) of the ask when the
     entry was decided (the arrival price): re-peg the limit to the new ask
     (PATCH /v2/orders/{id}; if Alpaca refuses to replace it, cancel and
     resubmit). The limit is never raised past the slippage cap.
   - Ask beyond that: cancel the entry and log ``slippage_timeout``.
   - After ENTRY_MAX_REPEGS (3) re-pegs without a fill: cancel
     (``entry_unfilled``).
2. When it fills, the bracket legs are moved so the stop-loss and take-profit
   sit exactly the planned distances from the actual fill price
   (``bracket_reanchored``).
3. A partial fill stops the chase: the unfilled rest is cancelled.
4. After any fill the position must have a working stop at the broker: the
   bracket's legs, or, if a partial fill or a replace left none, a new OCO
   stop/target for the filled shares (``partial_fill_protected``). If even that
   fails, ``unprotected_position`` is raised as an alert.

Every cancel re-reads the order, so a fill that races the cancel is still
handled (legs re-anchored), never lost. The chase blocks the calling cycle for
at most (ENTRY_MAX_REPEGS + 1) x ENTRY_FILL_TIMEOUT_SECONDS.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import requests

logger = logging.getLogger(__name__)

FILLED = "filled"
PARTIAL = "partially_filled"
DONE_STATES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
_STOP_TYPES = {"stop", "stop_limit", "trailing_stop"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _num(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _cents(price: float) -> float:
    return round(price + 1e-9, 2)


@dataclass(frozen=True)
class ChaseConfig:
    enabled: bool = True
    buffer_bps: float = 5.0            # limit = ask + max(buffer_bps of price, min_buffer)
    min_buffer: float = 0.01
    fill_timeout_seconds: float = 10.0
    max_slippage_bps: float = 15.0     # vs the arrival ask; beyond this the entry is cancelled
    max_repegs: int = 3
    poll_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "ChaseConfig":
        enabled = os.getenv("ENTRY_CHASE", "true").strip().lower() not in {"0", "false", "no", "off"}
        return cls(
            enabled=enabled,
            buffer_bps=max(0.0, _env_float("ENTRY_LIMIT_BUFFER_BPS", cls.buffer_bps)),
            min_buffer=max(0.0, _env_float("ENTRY_LIMIT_MIN_BUFFER", cls.min_buffer)),
            fill_timeout_seconds=max(0.0, _env_float("ENTRY_FILL_TIMEOUT_SECONDS", cls.fill_timeout_seconds)),
            max_slippage_bps=max(0.0, _env_float("ENTRY_MAX_SLIPPAGE_BPS", cls.max_slippage_bps)),
            max_repegs=max(0, int(_env_float("ENTRY_MAX_REPEGS", cls.max_repegs))),
        )

    def limit_price(self, ask: float) -> float:
        return _cents(ask + max(ask * self.buffer_bps / 10_000, self.min_buffer))

    def slippage_cap(self, arrival_ask: float) -> float:
        return _cents(arrival_ask * (1 + self.max_slippage_bps / 10_000))


class ChaseBroker(Protocol):
    def place_entry(self, symbol: str, qty: int, stop: float, target: float, limit_price: float) -> Dict: ...
    def get_order(self, order_id: str, nested: bool = False) -> Dict: ...
    def replace_order(self, order_id: str, **fields: Any) -> Dict: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def quote(self, symbol: str) -> Optional[Dict[str, float]]: ...
    def replace_stop_price(self, order_id: str, stop_price: float) -> Dict: ...
    def place_exit_oco(self, symbol: str, qty: int, stop: float, target: float) -> Dict: ...
    def open_exit_orders(self, symbol: str) -> List[Dict]: ...


@dataclass
class ChaseResult:
    status: str                         # filled | partial | slippage_timeout | unfilled | rejected | canceled
    order: Dict[str, Any] = field(default_factory=dict)
    order_id: Optional[str] = None      # the live order id (changes on every re-peg)
    fill_price: Optional[float] = None
    filled_qty: float = 0.0
    repegs: int = 0
    stop: Optional[float] = None        # final (re-anchored) bracket levels
    target: Optional[float] = None
    detail: str = ""

    @property
    def filled(self) -> bool:
        return self.filled_qty > 0


class SmartLimitChaser:
    def __init__(self, broker: ChaseBroker, config: Optional[ChaseConfig] = None, telemetry: Any = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 fill_cache: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None):
        self.broker = broker
        self.config = config or ChaseConfig.from_env()
        self.telemetry = telemetry
        self.sleep = sleep
        self.clock = clock
        # Optional: order updates already pushed by the trade_updates stream
        # (core.stream_listener), checked before polling REST.
        self.fill_cache = fill_cache

    def _record(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.record(event, level, **fields)

    # -- entry -------------------------------------------------------------

    def enter(self, symbol: str, qty: int, stop: float, target: float, arrival_ask: float,
              order: Optional[Dict[str, Any]] = None) -> ChaseResult:
        """Submit (unless ``order`` was already submitted) and chase one entry."""
        cfg = self.config
        stop_distance, target_distance = arrival_ask - stop, target - arrival_ask
        cap = cfg.slippage_cap(arrival_ask)
        limit = min(cfg.limit_price(arrival_ask), max(cap, _cents(arrival_ask)))
        if order is None:
            order = self.broker.place_entry(symbol, qty, stop, target, limit)
        result = ChaseResult(status="unfilled", order=order, order_id=order.get("id"))
        current_stop, current_target = stop, target

        while True:
            state = self._await_fill(result.order_id)
            status = str(state.get("status", "")).lower()
            if status == FILLED:
                return self._finish(result, state, symbol, stop_distance, target_distance)
            if status in {"canceled", "expired", "rejected", "done_for_day"}:
                result.status = "rejected" if status == "rejected" else "canceled"
                result.order, result.detail = state, f"entry order {status} at the broker"
                self._record("entry_unfilled", logging.WARNING, symbol=symbol, order_id=result.order_id,
                             reason=result.detail, repegs=result.repegs)
                return result
            if _num(state.get("filled_qty")) > 0:
                # Partial fill: stop chasing, keep the filled shares protected.
                return self._cancel_and_settle(result, symbol, stop_distance, target_distance, "partial fill")

            quote = self.broker.quote(symbol)
            ask = _num((quote or {}).get("ask"))
            if ask <= 0:
                return self._abort(result, symbol, stop_distance, target_distance, "entry_unfilled",
                                   "no quote to re-peg against", arrival_ask)
            if ask > cap:
                return self._abort(result, symbol, stop_distance, target_distance, "slippage_timeout",
                                   f"ask {ask:.2f} ran past the {cfg.max_slippage_bps:g} bps cap {cap:.2f} "
                                   f"(arrival {arrival_ask:.2f})", arrival_ask, ask=ask)
            if result.repegs >= cfg.max_repegs:
                return self._abort(result, symbol, stop_distance, target_distance, "entry_unfilled",
                                   f"not filled after {result.repegs} re-pegs", arrival_ask, ask=ask)

            result.repegs += 1
            new_limit = min(cfg.limit_price(ask), cap)
            if new_limit <= limit:
                continue  # the ask didn't move up: our limit is still marketable, keep waiting
            shift = new_limit - limit
            try:
                replaced = self._repeg(result, symbol, qty, new_limit, current_stop + shift,
                                       current_target + shift, stop_distance, target_distance)
            except requests.exceptions.RequestException as exc:  # network trouble: stop chasing
                return self._abort(result, symbol, stop_distance, target_distance, "entry_unfilled",
                                   f"re-peg failed: {exc}", arrival_ask, ask=ask)
            if replaced is None:
                return result  # settled while re-pegging (filled, or could not resubmit)
            limit = new_limit
            current_stop, current_target = current_stop + shift, current_target + shift

    def _await_fill(self, order_id: Optional[str]) -> Dict[str, Any]:
        deadline = self.clock() + self.config.fill_timeout_seconds
        state: Dict[str, Any] = {}
        while True:
            cached = self.fill_cache(order_id) if self.fill_cache and order_id else None
            if cached and str(cached.get("status", "")).lower() == FILLED:
                return cached
            try:
                state = self.broker.get_order(order_id)
            except (requests.exceptions.RequestException, ValueError) as exc:
                logger.warning(f"Could not read entry order {order_id}: {exc}")
            status = str(state.get("status", "")).lower()
            if status in DONE_STATES - {"replaced"} or _num(state.get("filled_qty")) > 0:
                return state
            if self.clock() >= deadline:
                return state
            self.sleep(self.config.poll_seconds)

    def _repeg(self, result: ChaseResult, symbol: str, qty: int, limit: float, stop: float, target: float,
               stop_distance: float, target_distance: float) -> Optional[Dict[str, Any]]:
        old_id = result.order_id
        try:
            order = self.broker.replace_order(old_id, limit_price=str(limit))
            how = "replace"
        except requests.exceptions.HTTPError as exc:
            # Some order states/classes can't be replaced: cancel, confirm
            # nothing filled, then resubmit at the new price.
            logger.info(f"Replace of {old_id} refused ({exc}); cancelling and resubmitting")
            settled = self._cancel(old_id)
            if _num(settled.get("filled_qty")) > 0:  # filled while we cancelled
                self._settle_filled(result, settled, symbol, stop_distance, target_distance)
                return None
            try:
                order = self.broker.place_entry(symbol, qty, _cents(stop), _cents(target), limit)
            except requests.exceptions.RequestException as resubmit_exc:
                result.status, result.detail = "canceled", f"resubmit after cancel failed: {resubmit_exc}"
                self._record("entry_unfilled", logging.WARNING, symbol=symbol, order_id=old_id,
                             reason=result.detail, repegs=result.repegs)
                return None
            how = "cancel_resubmit"
        result.order, result.order_id = order, order.get("id") or old_id
        self._record("entry_repegged", symbol=symbol, old_order_id=old_id, order_id=result.order_id,
                     limit_price=limit, repeg=result.repegs, method=how)
        return order

    # -- settling ----------------------------------------------------------

    def _cancel(self, order_id: Optional[str]) -> Dict[str, Any]:
        """Cancel, then re-read: a fill can land between our decision and the cancel."""
        try:
            self.broker.cancel_order(order_id)
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Cancel of {order_id} failed: {exc}")
        for _ in range(5):
            try:
                state = self.broker.get_order(order_id)
            except (requests.exceptions.RequestException, ValueError):
                state = {}
            if str(state.get("status", "")).lower() in DONE_STATES:
                return state
            self.sleep(self.config.poll_seconds)
        return state

    def _abort(self, result: ChaseResult, symbol: str, stop_distance: float, target_distance: float,
               event: str, reason: str, arrival_ask: float, ask: Optional[float] = None) -> ChaseResult:
        settled = self._cancel(result.order_id)
        if _num(settled.get("filled_qty")) > 0:  # it filled after all
            return self._settle_filled(result, settled, symbol, stop_distance, target_distance)
        result.status, result.order, result.detail = event, settled or result.order, reason
        self._record(event, logging.WARNING, symbol=symbol, order_id=result.order_id, reason=reason,
                     arrival_ask=arrival_ask, ask=ask, repegs=result.repegs,
                     max_slippage_bps=self.config.max_slippage_bps)
        logger.warning(f"Entry {symbol} cancelled: {reason}")
        return result

    def _cancel_and_settle(self, result: ChaseResult, symbol: str, stop_distance: float,
                           target_distance: float, why: str) -> ChaseResult:
        settled = self._cancel(result.order_id)
        logger.info(f"Entry {symbol}: {why}; cancelled the unfilled rest")
        return self._settle_filled(result, settled, symbol, stop_distance, target_distance)

    def _settle_filled(self, result: ChaseResult, state: Dict[str, Any], symbol: str,
                       stop_distance: float, target_distance: float) -> ChaseResult:
        result = self._finish(result, state, symbol, stop_distance, target_distance)
        if str(state.get("status", "")).lower() != FILLED:
            result.status = "partial"
        return result

    def _finish(self, result: ChaseResult, state: Dict[str, Any], symbol: str,
                stop_distance: float, target_distance: float) -> ChaseResult:
        fill = _num(state.get("filled_avg_price"))
        result.order, result.status = state, "filled" if str(state.get("status", "")).lower() == FILLED else "partial"
        result.filled_qty = _num(state.get("filled_qty"))
        result.fill_price = fill or None
        if fill > 0 and stop_distance > 0 and target_distance > 0:
            result.stop, result.target = _cents(fill - stop_distance), _cents(fill + target_distance)
            legs = self._legs(result.order_id, symbol)
            self._reanchor(result, symbol, legs)
            self._ensure_protected(result, symbol, legs)
        return result

    def _legs(self, order_id: Optional[str], symbol: str) -> List[Dict[str, Any]]:
        """The entry's working exit orders: its bracket legs, or, if the order
        (e.g. a replacement) carries none, the symbol's open sell orders."""
        try:
            legs = list(self.broker.get_order(order_id, nested=True).get("legs") or [])
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning(f"Could not read bracket legs of {order_id}: {exc}")
            legs = []
        live = [leg for leg in legs if str(leg.get("status", "")).lower() not in DONE_STATES]
        if live:
            return live
        try:
            return [o for o in self.broker.open_exit_orders(symbol)
                    if str(o.get("status", "")).lower() not in DONE_STATES]
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning(f"Could not read open {symbol} orders: {exc}")
            return []

    def _reanchor(self, result: ChaseResult, symbol: str, legs: List[Dict[str, Any]]) -> None:
        """Move the bracket legs to the planned distances from the real fill."""
        moved: Dict[str, Any] = {}
        for leg in legs:
            kind = str(leg.get("type") or leg.get("order_type") or "").lower()
            try:
                if kind in _STOP_TYPES and abs(_num(leg.get("stop_price")) - result.stop) >= 0.01:
                    self.broker.replace_stop_price(leg["id"], result.stop)
                    moved["stop"] = (leg.get("stop_price"), result.stop)
                elif kind == "limit" and abs(_num(leg.get("limit_price")) - result.target) >= 0.01:
                    self.broker.replace_order(leg["id"], limit_price=str(result.target))
                    moved["target"] = (leg.get("limit_price"), result.target)
            except requests.exceptions.RequestException as exc:
                # The original legs still protect the position.
                self._record("bracket_reanchor_failed", logging.WARNING, symbol=symbol,
                             order_id=result.order_id, leg_id=leg.get("id"), error=str(exc))
        if moved:
            self._record("bracket_reanchored", symbol=symbol, order_id=result.order_id,
                         fill_price=result.fill_price, stop=result.stop, target=result.target,
                         moved={k: {"from": v[0], "to": v[1]} for k, v in moved.items()})

    def _ensure_protected(self, result: ChaseResult, symbol: str, legs: List[Dict[str, Any]]) -> None:
        """Filled shares must always have a working stop at the broker (the
        bracket's, or a new OCO if a partial fill or a replace left none)."""
        open_stop = any(str(leg.get("type") or leg.get("order_type") or "").lower() in _STOP_TYPES
                        for leg in legs)
        if open_stop or result.stop is None or result.filled_qty <= 0:
            return
        qty = int(result.filled_qty)
        try:
            order = self.broker.place_exit_oco(symbol, qty, result.stop, result.target)
            self._record("partial_fill_protected", symbol=symbol, qty=qty, stop=result.stop,
                         target=result.target, order_id=order.get("id"))
        except requests.exceptions.RequestException as exc:
            self._record("unprotected_position", logging.ERROR, symbol=symbol, qty=qty, error=str(exc))
            logger.error(f"{symbol}: {qty} shares filled but no stop could be placed: {exc}")
