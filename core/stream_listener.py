#!/usr/bin/env python3
"""
Real-time order updates from Alpaca's ``trade_updates`` WebSocket stream.

The bot normally learns about fills by polling REST once per cycle (every
bar). With the stream it hears about them within a second:

- every fill is classified (entry fill, stop-loss hit, take-profit hit, other
  exit) and logged as ``stream_fill``;
- listeners are notified; the bot wakes up and immediately re-syncs with the
  broker (local book, fill quality, losing-streak check, trailing-stop
  anchors) instead of waiting for the next bar;
- the latest state of each order is cached, so the smart limit chaser sees an
  entry fill without another REST call.

It is only an accelerator: the REST reconciliation loop keeps running
unchanged. If the connection drops it reconnects with exponential backoff
(1 s .. 60 s), logging ``stream_disconnected`` once per outage and
``stream_connected`` when it's back. Bad credentials stop the listener (no
retry storm); the bot carries on with REST only. STREAM_TRADE_UPDATES=false
turns it off.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

STOP_TYPES = {"stop", "stop_limit", "trailing_stop"}
FILL_EVENTS = {"fill", "partial_fill"}


def stream_enabled() -> bool:
    return os.getenv("STREAM_TRADE_UPDATES", "true").strip().lower() not in {"0", "false", "no", "off"}


def stream_url(base_url: str) -> str:
    """https://paper-api.alpaca.markets/v2 -> wss://paper-api.alpaca.markets/stream"""
    root = base_url.rstrip("/")
    if root.endswith("/v2"):
        root = root[:-3]
    return root.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/stream"


def classify_fill(order: Dict[str, Any]) -> str:
    """entry_fill | stop_hit | target_hit | exit_fill"""
    side = str(order.get("side", "")).lower()
    kind = str(order.get("type") or order.get("order_type") or "").lower()
    if side == "buy":
        return "entry_fill"
    if kind in STOP_TYPES:
        return "stop_hit"
    if kind == "limit" and str(order.get("order_class") or "").lower() in {"bracket", "oco"}:
        return "target_hit"
    return "exit_fill"


class AuthError(Exception):
    pass


def _default_connect(url: str):
    from websockets.sync.client import connect

    return connect(url, open_timeout=10, close_timeout=2)


class TradeUpdateStream:
    def __init__(self, url: str, key: str, secret: str, telemetry: Any = None,
                 connect: Callable[[str], Any] = _default_connect, sleep: Optional[Callable[[float], Any]] = None,
                 max_backoff: float = 60.0, recv_timeout: float = 1.0, cache_size: int = 500):
        self.url, self._key, self._secret = url, key, secret
        self.telemetry = telemetry
        self.connect = connect
        self.max_backoff, self.recv_timeout, self.cache_size = max_backoff, recv_timeout, cache_size
        self._stop = threading.Event()
        self.sleep = sleep or self._stop.wait
        self.connected = False
        self.failed: Optional[str] = None           # set when the stream gave up (bad credentials)
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._orders: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._outage_logged = False

    @classmethod
    def for_executor(cls, executor: Any, telemetry: Any = None, **kwargs: Any) -> "TradeUpdateStream":
        return cls(stream_url(executor.base_url), executor.api_key, executor.secret_key,
                   telemetry=telemetry or getattr(executor, "telemetry", None), **kwargs)

    # -- public ------------------------------------------------------------

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        return "connected" if self.connected else ("stopped" if self._stop.is_set() else "reconnecting")

    def start(self) -> "TradeUpdateStream":
        self._thread = threading.Thread(target=self.run, name="trade-updates", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def order_update(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Latest streamed state of an order (None if not seen)."""
        with self._lock:
            return self._orders.get(order_id)

    # -- connection loop ---------------------------------------------------

    def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                with self.connect(self.url) as ws:
                    self._ws = ws
                    self._authenticate(ws)
                    self.connected, backoff = True, 1.0
                    if self._outage_logged:
                        self._record("stream_connected")
                        self._outage_logged = False
                    logger.info("Order stream connected (trade_updates)")
                    self._pump(ws)
            except AuthError as exc:
                self.connected, self.failed = False, str(exc)
                self._record("stream_failed", logging.ERROR, reason=str(exc))
                logger.error(f"Order stream disabled: {exc}; using REST checks only")
                return
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._disconnected(exc, backoff)
                self.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
            finally:
                self._ws = None
                self.connected = False

    def _authenticate(self, ws: Any) -> None:
        ws.send(json.dumps({"action": "auth", "key": self._key, "secret": self._secret}))
        reply = self._decode(ws.recv(timeout=10))
        data = (reply or {}).get("data") or {}
        if (reply or {}).get("stream") != "authorization" or str(data.get("status")).lower() != "authorized":
            raise AuthError(f"authentication refused ({data.get('error') or data.get('status') or reply})")
        ws.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))

    def _pump(self, ws: Any) -> None:
        while not self._stop.is_set():
            try:
                raw = ws.recv(timeout=self.recv_timeout)
            except TimeoutError:
                continue
            message = self._decode(raw)
            if message and message.get("stream") == "trade_updates":
                try:
                    self.handle(message.get("data") or {})
                except Exception as exc:  # one odd payload must not drop the connection
                    logger.warning(f"Could not process an order update: {exc}")

    def _disconnected(self, exc: Exception, retry_in: float) -> None:
        if not self._outage_logged:
            self._outage_logged = True
            self._record("stream_disconnected", logging.WARNING, error=str(exc) or type(exc).__name__,
                         retry_in_seconds=retry_in)
        logger.warning(f"Order stream down ({exc}); REST checks continue; retrying in {retry_in:g}s")

    @staticmethod
    def _decode(raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, (bytes, bytearray)):  # the paper endpoint sends binary frames
            raw = raw.decode("utf-8", errors="replace")
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return message if isinstance(message, dict) else None

    # -- updates -----------------------------------------------------------

    def handle(self, update: Dict[str, Any]) -> None:
        """Ingest one trade_updates payload: cache the order, log fills, notify."""
        order = update.get("order") or {}
        order_id = order.get("id")
        if not order_id:
            return
        with self._lock:
            self._orders[order_id] = order
            self._orders.move_to_end(order_id)
            while len(self._orders) > self.cache_size:
                self._orders.popitem(last=False)
        event = str(update.get("event", "")).lower()
        if event in FILL_EVENTS:
            kind = classify_fill(order)
            self._record("stream_fill", kind=kind, update=event, symbol=order.get("symbol"), order_id=order_id,
                         side=order.get("side"), price=_num(update.get("price")), qty=_num(update.get("qty")),
                         position_qty=_num(update.get("position_qty")), at=update.get("timestamp"))
            update = {**update, "kind": kind}
        for listener in list(self.listeners):
            try:
                listener(update)
            except Exception as exc:  # a listener must never kill the stream
                logger.warning(f"Order stream listener failed: {exc}")

    def _record(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.record(event, level, **fields)


def _num(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
