#!/usr/bin/env python3
"""
Paper self-test: prove the whole order path works before trusting the bot.

Quick test (no trades):
  1. Account      the paper keys work and the account can trade
  2. Market clock Alpaca's clock is readable (order steps need market hours)
  3. Quote        a live bid/ask for the test symbol (the smart limit needs it)
  4. Order stream the trade_updates WebSocket logs in (real-time fills)
  During market hours, also:
  5. Place        a 1-share buy with stop-loss/take-profit, priced 5% under the
                  market so it cannot fill
  6. Re-price     PATCH the limit, as the smart limit chaser does, and check
                  whether the stop/take-profit stay attached (Alpaca behaviour
                  the bot relies on, verified on your account)
  7. Cancel       and confirm the cancel
  8. Stream saw   the order updates arrived over the stream

Full test (adds a real paper round trip, 1 share):
  9. Buy          through the smart limit chaser (ask + buffer, re-peg)
 10. Protection   the stop/take-profit sit at the planned distances from the fill
 11. Stream fill  the fill arrived over the stream
 12. Sell         cancel that entry's stop/take-profit (confirmed first, so
                  nothing can sell twice), then sell the 1 share

Paper only; it refuses to run against the live account. Every order it makes
is cancelled or closed before it returns, even if a step fails.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from core.execution_router import DONE_STATES, SmartLimitChaser, _cents, _num
from core.health import ORDER, Check

logger = logging.getLogger(__name__)
STOP_TYPES = {"stop", "stop_limit", "trailing_stop"}


NETWORK_FIX = ("Check the internet connection. A VPN, proxy or firewall may be blocking alpaca.markets; "
               "allow it and run the test again.")


def _http_detail(exc: Exception) -> str:
    """A short, readable reason (no stack of proxy/urllib3 internals)."""
    response = getattr(exc, "response", None)
    if response is not None:
        return f"Alpaca answered HTTP {response.status_code}: {(getattr(response, 'text', '') or '')[:200]}"
    if isinstance(exc, requests.exceptions.Timeout):
        return "Alpaca did not answer in time"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Could not reach Alpaca (no connection)"
    return str(exc)[:200]


def _is_network(exc: Exception) -> bool:
    return getattr(exc, "response", None) is None and isinstance(
        exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


class SelfTest:
    def __init__(self, executor: Any, symbol: str = "SPY", full: bool = False,
                 stream_factory: Optional[Callable[[Any], Any]] = None,
                 sleep: Callable[[float], None] = time.sleep, stream_wait: float = 10.0):
        if getattr(executor, "mode", "paper") != "paper":
            raise ValueError("The self-test only runs against the paper account")
        self.ex = executor
        self.symbol = symbol.upper()
        self.full = full
        self.stream_factory = stream_factory
        self.sleep = sleep
        self.stream_wait = stream_wait
        self.steps: List[Check] = []
        self.created: List[str] = []   # every order id we made, for cleanup
        self.stream: Any = None

    # -- helpers -------------------------------------------------------------

    def _add(self, id: str, label: str, status: str, detail: str, fix: str = "") -> Check:
        check = Check(id, label, status, detail, fix)
        self.steps.append(check)
        return check

    def _skip(self, id: str, label: str, why: str) -> None:
        self._add(id, label, "info", f"Skipped: {why}")

    def _wait_status(self, order_id: str, wanted: set, seconds: float = 5.0) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for _ in range(max(1, int(seconds))):
            try:
                state = self.ex.get_order(order_id)
            except (requests.exceptions.RequestException, ValueError):
                state = {}
            if str(state.get("status", "")).lower() in wanted:
                return state
            self.sleep(1.0)
        return state

    def _live_legs(self, order_id: str) -> List[Dict[str, Any]]:
        legs = self.ex.get_order(order_id, nested=True).get("legs") or []
        return [leg for leg in legs if str(leg.get("status", "")).lower() not in DONE_STATES]

    def _stream_saw(self, order_id: str, status: Optional[str] = None, seconds: float = 3.0) -> bool:
        if self.stream is None or self.stream.status != "connected":
            return False
        for _ in range(max(1, int(seconds))):
            update = self.stream.order_update(order_id)
            if update and (status is None or str(update.get("status", "")).lower() == status):
                return True
            self.sleep(1.0)
        return False

    # -- run -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        try:
            self._run()
        except Exception as exc:  # report, never crash the dashboard
            logger.exception("Self-test failed")
            self._add("error", "Unexpected error", "fail", str(exc), "Try again; if it repeats, see the dashboard log.")
        finally:
            self._cleanup()
            if self.stream is not None:
                self.stream.stop()
        overall = min((c.status for c in self.steps), key=lambda s: ORDER.get(s, 3), default="ok")
        overall = "ok" if overall == "info" else overall
        result = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "full": self.full,
            "overall": overall,
            "steps": [asdict(c) for c in self.steps],
        }
        telemetry = getattr(self.ex, "telemetry", None)
        if telemetry is not None:
            telemetry.record("selftest", logging.INFO if overall == "ok" else logging.WARNING,
                             overall=overall, full=self.full,
                             failed=[c.label for c in self.steps if c.status == "fail"])
        return result

    def _run(self) -> None:
        try:
            account = self.ex.get_account()
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            fix = (NETWORK_FIX if _is_network(exc) else
                   "Alpaca rejected the keys: re-enter the paper API key and secret on the API Keys tab "
                   "(Test keys shows whether they work)." if status in (401, 403) else
                   "Try again in a minute; if it repeats, check status.alpaca.markets.")
            self._add("account", "Paper account", "fail", _http_detail(exc), fix)
            return
        status = str(account.get("status") or "").upper()
        if status != "ACTIVE" or account.get("trading_blocked") or account.get("account_blocked"):
            self._add("account", "Paper account", "fail", f"Account status {status or 'unknown'}; trading blocked",
                      "Open alpaca.markets and check the paper account; you can reset it there.")
            return
        self._add("account", "Paper account", "ok",
                  f"Active; equity ${_num(account.get('equity')):,.2f}, buying power "
                  f"${_num(account.get('buying_power')):,.2f}")

        try:
            clock = self.ex.get_clock()
            market_open = bool(clock.get("is_open"))
            self._add("clock", "Market clock", "ok", "Market is open" if market_open else
                      f"Market is closed (next open {clock.get('next_open', '?')})")
        except requests.exceptions.RequestException as exc:
            self._add("clock", "Market clock", "fail", _http_detail(exc), NETWORK_FIX)
            return

        quote = None
        try:
            quote = self.ex.get_quote(self.symbol)
        except requests.exceptions.RequestException as exc:
            self._add("quote", f"{self.symbol} quote", "fail", _http_detail(exc),
                      NETWORK_FIX if _is_network(exc) else
                      "Market data uses the same Alpaca keys; check them on the API Keys tab.")
        if quote is not None:
            bid, ask = _num(quote.get("bid")), _num(quote.get("ask"))
            if bid > 0 and ask > 0:
                self._add("quote", f"{self.symbol} quote", "ok", f"bid ${bid:.2f} / ask ${ask:.2f}")
            elif market_open:
                self._add("quote", f"{self.symbol} quote", "warn", "No bid/ask right now",
                          "Entries fall back to a plain marketable limit without a quote.")
            else:
                self._add("quote", f"{self.symbol} quote", "info", "No live quote outside market hours")

        self._check_stream()

        if not market_open:
            for id, label in (("order_place", "Test order"), ("order_replace", "Re-price order"),
                              ("order_cancel", "Cancel order")):
                self._skip(id, label, "order tests run during market hours (9:30 to 4:00 ET)")
            if self.full:
                self._skip("round_trip", "1-share buy and sell", "needs market hours")
            return
        if not quote or _num(quote.get("bid")) <= 0 or _num(quote.get("ask")) <= 0:
            self._skip("order_place", "Test order", "no quote to price the test order")
            return
        self._order_path(_num(quote.get("bid")))
        if self.full:
            self._round_trip()

    def _check_stream(self) -> None:
        if self.stream_factory is None:
            self._skip("stream", "Live order stream", "turned off (STREAM_TRADE_UPDATES=false)")
            return
        self.stream = self.stream_factory(self.ex)
        self.stream.start()
        waited = 0.0
        while self.stream.status not in {"connected", "failed"} and waited < self.stream_wait:
            self.sleep(0.5)
            waited += 0.5
        if self.stream.status == "connected":
            self._add("stream", "Live order stream", "ok", "Logged in to trade_updates; fills arrive in real time")
        elif self.stream.status == "failed":
            self._add("stream", "Live order stream", "fail", f"Login refused: {self.stream.failed}",
                      "Check the paper keys. The bot still works without it (it checks every cycle).")
        else:
            self._add("stream", "Live order stream", "warn", f"No connection within {self.stream_wait:g} s",
                      "A firewall or proxy may block WebSockets (wss://paper-api.alpaca.markets/stream). "
                      "The bot still works, just reacts to fills at the next cycle.")

    # -- safe order path: never fills ------------------------------------------

    def _order_path(self, bid: float) -> None:
        price = _cents(bid * 0.95)
        stop, target = _cents(price * 0.98), _cents(price * 1.04)
        try:
            order = self.ex._place_bracket_order(self.symbol, 1, stop, target, limit_price=price)
        except requests.exceptions.RequestException as exc:
            self._add("order_place", "Test order", "fail", _http_detail(exc),
                      "Alpaca refused a simple bracket order; check the account on alpaca.markets.")
            return
        order_id = order.get("id")
        self.created.append(order_id)
        legs_before = len(self._live_legs(order_id))
        self._add("order_place", "Test order", "ok" if legs_before >= 2 else "warn",
                  f"1-share {self.symbol} buy at ${price:.2f} (5% under the market, will not fill) with "
                  f"{legs_before} protective leg(s)",
                  "" if legs_before >= 2 else "Expected a stop-loss and a take-profit leg.")

        current = order_id
        new_price = _cents(price - max(0.05, price * 0.001))
        try:
            replaced = self.ex.replace_order(order_id, limit_price=new_price)
            current = replaced.get("id") or order_id
            self.created.append(current)
            self.sleep(1.0)
            legs_after = self._live_legs(current)
            if len(legs_after) >= 2:
                self._add("order_replace", "Re-price order", "ok",
                          f"Limit moved to ${new_price:.2f}; stop-loss and take-profit stayed attached")
            else:
                self._add("order_replace", "Re-price order", "warn",
                          f"Limit moved, but only {len(legs_after)} protective leg(s) carried over",
                          "Handled automatically: after each fill the bot adds its own stop if the order has none.")
        except requests.exceptions.HTTPError as exc:
            self._add("order_replace", "Re-price order", "warn", f"Alpaca refused re-pricing ({_http_detail(exc)})",
                      "Handled automatically: the bot cancels and resubmits instead (a little slower).")

        try:
            self.ex.cancel_order(current)
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Self-test cancel failed: {exc}")
        state = self._wait_status(current, {"canceled"})
        if str(state.get("status", "")).lower() == "canceled":
            self._add("order_cancel", "Cancel order", "ok", "Cancelled and confirmed (its legs go with it)")
        else:
            self._add("order_cancel", "Cancel order", "fail", f"Status still {state.get('status', 'unknown')}",
                      f"Cancel the {self.symbol} test order under Orders at alpaca.markets.")

        if self.stream is not None and self.stream.status == "connected":
            if self._stream_saw(current) or self._stream_saw(order_id):
                self._add("stream_updates", "Order updates over the stream", "ok",
                          "The stream reported the test order's changes")
            else:
                self._add("stream_updates", "Order updates over the stream", "warn",
                          "Connected, but no update arrived for the test order",
                          "The bot falls back to checking every cycle.")

    # -- full: a real 1-share paper round trip ----------------------------------

    def _round_trip(self) -> None:
        quote = self.ex.get_quote(self.symbol) or {}
        ask = _num(quote.get("ask"))
        if ask <= 0:
            self._skip("round_trip", "1-share buy and sell", "no ask price")
            return
        stop, target = _cents(ask * 0.99), _cents(ask * 1.02)
        chaser = SmartLimitChaser(self.ex._chase_broker(), self.ex.chase, telemetry=self.ex.telemetry,
                                  sleep=self.sleep,
                                  fill_cache=self.stream.order_update if self.stream is not None else None)
        try:
            order = self.ex._place_bracket_order(self.symbol, 1, stop, target,
                                                 limit_price=self.ex.chase.limit_price(ask))
        except requests.exceptions.RequestException as exc:
            self._add("buy", "Smart limit buy", "fail", _http_detail(exc), "Check the account on alpaca.markets.")
            return
        self.created.append(order.get("id"))
        result = chaser.enter(self.symbol, 1, stop, target, ask, order=order)
        if result.order_id:
            self.created.append(result.order_id)
        if not result.filled:
            self._add("buy", "Smart limit buy", "warn", f"Did not fill ({result.status}: {result.detail})",
                      "Normal in a fast market: the chaser refuses to overpay. Try again in a calmer minute.")
            return
        self._add("buy", "Smart limit buy", "ok",
                  f"Bought 1 {self.symbol} at ${result.fill_price:.2f} (ask was ${ask:.2f}) after "
                  f"{result.repegs} re-peg(s)")

        legs = self._live_legs(result.order_id)
        stops = [leg for leg in legs if str(leg.get("type", "")).lower() in STOP_TYPES]
        if stops and abs(_num(stops[0].get("stop_price")) - (result.stop or 0)) < 0.011:
            self._add("protection", "Stop-loss and take-profit", "ok",
                      f"Stop ${result.stop:.2f}, target ${result.target:.2f}: planned distances from the fill")
        elif stops:
            self._add("protection", "Stop-loss and take-profit", "warn",
                      f"Stop at ${_num(stops[0].get('stop_price')):.2f}, expected ${result.stop:.2f}",
                      "The position is protected, but the stop wasn't moved to the fill price.")
        else:
            self._add("protection", "Stop-loss and take-profit", "warn",
                      "No stop leg on the entry (the bot placed a separate stop order if possible)")

        if self.stream is not None and self.stream.status == "connected":
            seen = self._stream_saw(result.order_id, "filled")
            self._add("stream_fill", "Fill over the stream", "ok" if seen else "warn",
                      "The fill arrived in real time" if seen else "The fill did not arrive over the stream",
                      "" if seen else "The bot falls back to checking every cycle.")

        self._sell_test_share(result.order_id)

    def _sell_test_share(self, entry_id: str) -> None:
        # Release the shares from this entry's own stop/take-profit first, and
        # confirm it: a leg still working could also sell and leave you short.
        pending = []
        for leg in self._live_legs(entry_id):
            try:
                self.ex.cancel_order(leg["id"])
            except requests.exceptions.RequestException:
                pass
            pending.append(leg["id"])
        still_open = [leg_id for leg_id in pending
                      if str(self._wait_status(leg_id, DONE_STATES).get("status", "")).lower() not in DONE_STATES]
        if still_open:
            self._add("sell", "Sell the test share", "fail", "Could not confirm the protective orders were cancelled",
                      f"The share is still protected by its stop. Sell 1 {self.symbol} on the Paper tab or at "
                      "alpaca.markets.")
            return
        try:
            sell = self.ex._place_order(self.symbol, 1, "sell")
        except requests.exceptions.RequestException as exc:
            self._add("sell", "Sell the test share", "fail", _http_detail(exc),
                      f"Sell 1 {self.symbol} on the Paper tab or at alpaca.markets.")
            return
        state = self._wait_status(sell.get("id"), {"filled"}, seconds=10)
        if str(state.get("status", "")).lower() == "filled":
            self._add("sell", "Sell the test share", "ok", f"Sold at ${_num(state.get('filled_avg_price')):.2f}")
        else:
            self._add("sell", "Sell the test share", "warn", f"Sell order status {state.get('status', 'unknown')}",
                      "It should fill shortly; check Open positions.")

    # -- cleanup ---------------------------------------------------------------

    def _cleanup(self) -> None:
        """Cancel any test order still working (buy side only: a filled test
        share is handled by _sell_test_share, never here)."""
        seen: set = set()
        for order_id in dict.fromkeys(i for i in self.created if i):
            # Follow re-prices we may not have recorded (replaced_by chain).
            for _ in range(10):
                if not order_id or order_id in seen:
                    break
                seen.add(order_id)
                try:
                    state = self.ex.get_order(order_id)
                    status = str(state.get("status", "")).lower()
                    if status not in DONE_STATES:
                        self.ex.cancel_order(order_id)
                except requests.exceptions.RequestException as exc:
                    logger.warning(f"Self-test cleanup of {order_id} failed: {exc}")
                    break
                order_id = state.get("replaced_by") if status == "replaced" else None


def run_selftest(executor: Any, **kwargs: Any) -> Dict[str, Any]:
    return SelfTest(executor, **kwargs).run()
