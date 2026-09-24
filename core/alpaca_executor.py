#!/usr/bin/env python3
"""
Alpaca Executor - Sends real orders to Alpaca paper trading account.
Sits between pipeline.py (signals) and portfolio_manager.py (recording).

Usage:
    Set these in your .env file:
        ALPACA_API_KEY=your_key_here
        ALPACA_SECRET_KEY=your_secret_here
        ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from core.execution_telemetry import EventLog, Rejection, classify_rejection
from core.risk_manager import RiskManager
from core.session_clock import SessionClock, SessionPhase

load_dotenv()
logger = logging.getLogger(__name__)


MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of submitting a signal: the broker order, or why it was skipped."""

    order: Optional[Dict[str, Any]] = None
    skipped_reason: Optional[str] = None
    rejection: Optional[Rejection] = None   # set when the broker refused the order
    recovered: bool = False                 # True when a retry after a rejection succeeded


@dataclass
class FlattenReport:
    reason: str
    cancelled: int = 0
    closed: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class BrokerSnapshot:
    positions: List[Dict[str, Any]]
    open_orders: List[Dict[str, Any]]


def flatten_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Include nested bracket legs as top-level entries (deduplicated by id)."""
    flat: Dict[str, Dict[str, Any]] = {}
    for order in orders or []:
        flat[order.get("id") or str(len(flat))] = order
        for leg in order.get("legs") or []:
            if leg.get("status") not in {"filled", "canceled", "expired", "replaced", "rejected"}:
                flat[leg.get("id") or str(len(flat))] = {**leg, "parent_id": order.get("id")}
    return list(flat.values())


def protective_stop_orders(open_orders: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
    """Working sell-side stop orders for ``symbol``."""
    return [
        o for o in open_orders
        if o.get("symbol") == symbol
        and str(o.get("side", "")).lower() == "sell"
        and str(o.get("type", o.get("order_type", ""))).lower() in {"stop", "stop_limit", "trailing_stop"}
    ]


def _default_price_lookup(symbol: str) -> float:
    from core.candle_fetcher_provider import get_candlestick_fetcher

    quote = get_candlestick_fetcher().fetch_realtime_quote(symbol)
    return float(quote.get("current_price") or 0)


class AlpacaExecutor:
    """
    Sends orders to Alpaca and returns results.
    Uses Alpaca paper trading by default.

    Every order passes through a RiskManager first, and every BUY is sent as
    a bracket order so its stop-loss and take-profit live at the broker.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        risk_manager: Optional[RiskManager] = None,
        price_lookup: Optional[Callable[[str], float]] = None,
        telemetry: Optional[EventLog] = None,
        session_clock: Optional[SessionClock] = None,
    ):
        self.risk_manager = risk_manager or RiskManager()
        # Optional intraday session rules (opening lockout / EOD cutoff) for entries.
        self.session_clock = session_clock
        self.telemetry = telemetry or EventLog.from_env()
        self.price_lookup = price_lookup or _default_price_lookup
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = self._normalize_base_url(
            base_url
            or os.getenv("ALPACA_TRADING_BASE_URL")
            or os.getenv("ALPACA_BASE_URL")
            or "https://paper-api.alpaca.markets/v2"
        )

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file"
            )

        if "paper-api.alpaca.markets" not in self.base_url:
            raise ValueError(
                "Paper trading requires ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2"
            )

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }
        logger.info("AlpacaExecutor ready in PAPER mode")

    def _normalize_base_url(self, base_url: str) -> str:
        """Accept either the Alpaca root URL or the versioned v2 URL."""
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v2"):
            normalized = f"{normalized}/v2"
        return normalized

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Return account details (buying power, equity, etc.)."""
        resp = requests.get(
            f"{self.base_url}/account", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> list:
        """Return all open positions."""
        resp = requests.get(
            f"{self.base_url}/positions", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return a single open position, or None if not held."""
        try:
            resp = requests.get(
                f"{self.base_url}/positions/{symbol}",
                headers=self.headers,
                timeout=10,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            return None

    def get_orders_today(self) -> List[Dict]:
        """Return all orders submitted since midnight US/Eastern."""
        start = datetime.combine(datetime.now(MARKET_TZ).date(), time.min, tzinfo=MARKET_TZ)
        resp = requests.get(
            f"{self.base_url}/orders",
            headers=self.headers,
            params={
                "status": "all",
                "after": start.isoformat(),
                "limit": 500,
                "direction": "desc",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_clock(self) -> Dict:
        """Return Alpaca's market clock (is_open, next_open, next_close)."""
        resp = requests.get(
            f"{self.base_url}/clock", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def is_market_open(self) -> bool:
        """Return True if the US market is currently open."""
        return bool(self.get_clock().get("is_open", False))

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """Submit a signal and return the Alpaca order, or None if skipped."""
        return self.submit(signal).order

    def submit(self, signal: Dict) -> ExecutionResult:
        """
        Main entry point.  Pass the dict that pipeline.py returns and this
        will run risk checks and place the order when conditions are right.

        Recognised keys: symbol, recommendation/signal, quantity, and
        optionally price and risk_parameters (stop_loss/take_profit, or
        stop_distance/target_distance to place ATR levels around the live price).
        """
        action = str(signal.get("recommendation") or signal.get("signal") or "HOLD").upper()
        symbol = str(signal.get("symbol") or "").upper()
        quantity = int(signal.get("quantity") or 0)

        if action not in {"BUY", "SELL"} or quantity <= 0 or not symbol:
            logger.info(f"Skipping execution: {action} {quantity} {symbol}")
            return ExecutionResult(skipped_reason="No actionable BUY/SELL signal")

        # Don't queue market orders while the market is closed; they would
        # fill at an unknown price at the next open.
        if not self.is_market_open():
            logger.warning(f"Market is closed. Skipping {action} {quantity} {symbol}.")
            return ExecutionResult(skipped_reason="Market is closed")
        if action == "BUY" and self.session_clock is not None:
            phase = self.session_clock.phase_from_alpaca_clock(self.get_clock())
            if phase is not SessionPhase.OPEN:
                logger.info(f"Session phase {phase.value}: skipping entry {quantity} {symbol}")
                return ExecutionResult(skipped_reason=f"Entries not allowed during {phase.value}")

        risk = signal.get("risk_parameters") or {}
        position = self.get_position(symbol)
        price = float(signal.get("price") or 0)
        if action == "BUY":
            # Analysis prices can be an hour old; size and set stops off a
            # live quote when one is available.
            try:
                live_price = float(self.price_lookup(symbol) or 0)
            except Exception as exc:
                logger.warning(f"Live quote for {symbol} failed, using signal price: {exc}")
                live_price = 0.0
            price = live_price or price

        stop_loss, take_profit = risk.get("stop_loss"), risk.get("take_profit")
        stop_distance = float(risk.get("stop_distance") or 0)
        target_distance = float(risk.get("target_distance") or 0)
        if action == "BUY" and price > 0 and stop_distance > 0 and target_distance > 0:
            # ATR-based distances re-centred on the live entry price.
            stop_loss, take_profit = price - stop_distance, price + target_distance

        decision = self.risk_manager.check_order(
            side=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            account=self.get_account(),
            position=position,
            orders_today=self.get_orders_today() if action == "BUY" else (),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if not decision.approved:
            logger.warning(f"Risk check blocked {action} {quantity} {symbol}: {decision.reason}")
            return ExecutionResult(skipped_reason=decision.reason)

        try:
            if action == "SELL":
                # Bracket stop/target legs reserve the shares; release them first.
                self.cancel_open_orders(symbol)
                return ExecutionResult(order=self._place_order(symbol, decision.quantity, "sell"))

            order = self._place_bracket_order(
                symbol, decision.quantity, decision.stop_loss, decision.take_profit
            )
            return ExecutionResult(order=order)
        except requests.exceptions.HTTPError as exc:
            return self._recover_from_rejection(exc, action, symbol, decision.quantity, price)

    # ------------------------------------------------------------------
    # Rejection handling
    # ------------------------------------------------------------------

    def _rejection(self, exc: requests.exceptions.HTTPError, **context: Any) -> Rejection:
        response = exc.response
        status = getattr(response, "status_code", None) if response is not None else None
        body = getattr(response, "text", "") if response is not None else str(exc)
        rejection = classify_rejection(status, body or str(exc))
        self.telemetry.record("order_rejected", logging.WARNING, **context, **rejection.as_dict())
        return rejection

    def _recover_from_rejection(
        self,
        exc: requests.exceptions.HTTPError,
        action: str,
        symbol: str,
        quantity: int,
        price: float,
    ) -> ExecutionResult:
        """Classify the rejection, log it, and retry once when a safe fix exists."""
        rejection = self._rejection(exc, symbol=symbol, side=action, qty=quantity, attempt=1)
        retry: Optional[Callable[[], Dict]] = None
        plan = ""

        if action == "BUY" and rejection.category == "buying_power":
            buying_power = float(self.get_account().get("buying_power") or 0)
            affordable = int(buying_power * 0.97 // price) if price > 0 else 0
            if 0 < affordable < quantity:
                stop, target = self.risk_manager.bracket_prices(price)
                if stop is not None:
                    plan = f"retry with {affordable} shares (buying power ${buying_power:,.2f})"
                    retry = lambda: self._place_bracket_order(symbol, affordable, stop, target)  # noqa: E731
        elif action == "BUY" and rejection.category == "invalid_price":
            try:
                fresh = float(self.price_lookup(symbol) or 0)
            except Exception:
                fresh = 0.0
            stop, target = self.risk_manager.bracket_prices(fresh) if fresh > 0 else (None, None)
            if stop is not None:
                plan = f"retry with levels from fresh quote ${fresh:.2f}: stop {stop}, target {target}"
                retry = lambda: self._place_bracket_order(symbol, quantity, stop, target)  # noqa: E731
        elif action == "SELL" and rejection.category in {"wash_trade", "qty_held"}:
            self.cancel_open_orders(symbol)
            position = self.get_position(symbol) or {}
            available = int(float(position.get("qty_available") or position.get("qty") or 0))
            sell_qty = min(quantity, available)
            if sell_qty > 0:
                plan = f"cancelled open {symbol} orders; retry selling {sell_qty}"
                retry = lambda: self._place_order(symbol, sell_qty, "sell")  # noqa: E731

        if retry is None:
            return ExecutionResult(
                skipped_reason=f"Alpaca rejected the order ({rejection.category}): {rejection.message}",
                rejection=rejection,
            )

        self.telemetry.record("order_retry", symbol=symbol, side=action, category=rejection.category, plan=plan)
        try:
            order = retry()
        except requests.exceptions.HTTPError as retry_exc:
            second = self._rejection(retry_exc, symbol=symbol, side=action, qty=quantity, attempt=2)
            return ExecutionResult(
                skipped_reason=(
                    f"Alpaca rejected the order ({rejection.category}) and the retry ({second.category}): "
                    f"{second.message}"
                ),
                rejection=second,
            )
        self.telemetry.record("order_recovered", symbol=symbol, side=action, category=rejection.category,
                              order_id=order.get("id"), plan=plan)
        return ExecutionResult(order=order, rejection=rejection, recovered=True)

    def _place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Dict:
        """Place a market order and return the Alpaca response."""
        payload = {
            "symbol":        symbol,
            "qty":           str(quantity),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }

        logger.info(f"Placing order: {side.upper()} {quantity} {symbol}")
        resp = requests.post(
            f"{self.base_url}/orders",
            headers=self.headers,
            json=payload,
            timeout=10,
        )

        if not resp.ok:
            logger.error(f"Order failed: {resp.status_code} {resp.text}")
            resp.raise_for_status()

        order = resp.json()
        logger.info(
            f"Order placed — ID: {order['id']} | "
            f"{order['side'].upper()} {order['qty']} {order['symbol']} "
            f"@ {order.get('filled_avg_price', 'pending')}"
        )
        return order

    def _place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        stop_loss: float,
        take_profit: float,
    ) -> Dict:
        """Buy at market with a broker-side stop-loss and take-profit attached."""
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "take_profit": {"limit_price": str(take_profit)},
            "stop_loss": {"stop_price": str(stop_loss)},
        }

        logger.info(
            f"Placing bracket order: BUY {quantity} {symbol} "
            f"stop ${stop_loss} target ${take_profit}"
        )
        resp = requests.post(
            f"{self.base_url}/orders",
            headers=self.headers,
            json=payload,
            timeout=10,
        )
        if not resp.ok:
            logger.error(f"Order failed: {resp.status_code} {resp.text}")
            resp.raise_for_status()

        order = resp.json()
        logger.info(f"Bracket order placed — ID: {order['id']} | BUY {order['qty']} {order['symbol']}")
        return order

    # ------------------------------------------------------------------
    # Trailing stop helper (from video concept)
    # ------------------------------------------------------------------

    def place_trailing_stop(
        self, symbol: str, quantity: int, trail_percent: float = 5.0
    ) -> Dict:
        """
        Buy shares and immediately attach a trailing stop order.
        trail_percent: how far below peak to set the floor (default 5 %).
        """
        buy_order = self._place_order(symbol, quantity, "buy")

        stop_payload = {
            "symbol":           symbol,
            "qty":              str(quantity),
            "side":             "sell",
            "type":             "trailing_stop",
            "trail_percent":    str(trail_percent),
            "time_in_force":    "gtc",
        }

        resp = requests.post(
            f"{self.base_url}/orders",
            headers=self.headers,
            json=stop_payload,
            timeout=10,
        )
        resp.raise_for_status()
        stop_order = resp.json()
        logger.info(
            f"Trailing stop set: {trail_percent}% below peak for {symbol}"
        )
        return {"buy_order": buy_order, "stop_order": stop_order}

    # ------------------------------------------------------------------
    # Cancel helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # End-of-day liquidation
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, qty: float, current_price: float = 0.0) -> Dict:
        """Close one position: market order (DELETE /v2/positions/{symbol}) or,
        with FLATTEN_ORDER_TYPE=limit, a marketable limit priced
        FLATTEN_LIMIT_OFFSET_BPS through the last price."""
        if os.getenv("FLATTEN_ORDER_TYPE", "market").lower() == "limit" and current_price > 0:
            offset = float(os.getenv("FLATTEN_LIMIT_OFFSET_BPS", "20")) / 10_000
            side = "sell" if qty > 0 else "buy"
            limit = current_price * (1 - offset) if side == "sell" else current_price * (1 + offset)
            resp = requests.post(
                f"{self.base_url}/orders",
                headers=self.headers,
                json={"symbol": symbol, "qty": str(abs(int(qty))), "side": side, "type": "limit",
                      "limit_price": str(round(limit, 2)), "time_in_force": "day"},
                timeout=10,
            )
        else:
            resp = requests.delete(f"{self.base_url}/positions/{symbol}", headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def flatten_all(self, reason: str = "eod") -> "FlattenReport":
        """Cancel every working order (bracket legs included), then close every
        position. Failures are classified and logged; the bot retries on the
        next cycle while the session is still in its FLATTEN phase."""
        report = FlattenReport(reason=reason)
        try:
            report.cancelled = len(self.cancel_all_orders() or [])
        except requests.exceptions.HTTPError as exc:
            report.failures.append({"symbol": "*", **self._rejection(exc, action="cancel_all").as_dict()})

        for position in self.get_positions():
            symbol, qty = position["symbol"], float(position.get("qty") or 0)
            if qty == 0:
                continue
            try:
                order = self.close_position(symbol, qty, float(position.get("current_price") or 0))
                report.closed.append({"symbol": symbol, "qty": qty, "order_id": order.get("id")})
            except requests.exceptions.HTTPError as exc:
                rejection = self._rejection(exc, symbol=symbol, side="close", qty=qty, attempt=1)
                report.failures.append({"symbol": symbol, **rejection.as_dict()})

        self.telemetry.record(
            "flatten",
            logging.WARNING if report.failures else logging.INFO,
            reason=reason,
            cancelled_orders=report.cancelled,
            closed=report.closed,
            failures=report.failures,
        )
        return report

    def cancel_all_orders(self) -> list:
        """Cancel every open order — useful for end-of-day cleanup."""
        resp = requests.delete(
            f"{self.base_url}/orders", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        logger.info("All open orders cancelled")
        return resp.json() if resp.text else []

    # ------------------------------------------------------------------
    # Broker state (reconciliation / trailing stops)
    # ------------------------------------------------------------------

    def get_open_orders(self) -> List[Dict]:
        """All open orders with bracket legs flattened into the list."""
        resp = requests.get(
            f"{self.base_url}/orders",
            headers=self.headers,
            params={"status": "open", "nested": "true", "limit": 500},
            timeout=10,
        )
        resp.raise_for_status()
        return flatten_orders(resp.json())

    def get_snapshot(self) -> "BrokerSnapshot":
        return BrokerSnapshot(positions=self.get_positions(), open_orders=self.get_open_orders())

    def replace_stop_price(self, order_id: str, stop_price: float) -> Dict:
        """Move a working stop order (e.g. a bracket stop leg) to a new price."""
        resp = requests.patch(
            f"{self.base_url}/orders/{order_id}",
            headers=self.headers,
            json={"stop_price": str(stop_price)},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_open_orders(self, symbol: str) -> int:
        """Cancel open orders for one symbol (e.g. bracket legs) before an exit."""
        resp = requests.get(
            f"{self.base_url}/orders",
            headers=self.headers,
            params={"status": "open", "symbols": symbol, "nested": "false"},
            timeout=10,
        )
        resp.raise_for_status()
        cancelled = 0
        for order in resp.json():
            if order.get("symbol") == symbol and self.cancel_order(order["id"]):
                cancelled += 1
        if cancelled:
            logger.info(f"Cancelled {cancelled} open order(s) for {symbol} before exit")
        return cancelled

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID."""
        resp = requests.delete(
            f"{self.base_url}/orders/{order_id}",
            headers=self.headers,
            timeout=10,
        )
        return resp.status_code == 204


# ------------------------------------------------------------------
# Quick connection test — run this file directly to verify keys work
# python core/alpaca_executor.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    executor = AlpacaExecutor()

    print("\n--- Account ---")
    account = executor.get_account()
    print(f"Status:       {account['status']}")
    print(f"Equity:       ${float(account['equity']):,.2f}")
    print(f"Buying Power: ${float(account['buying_power']):,.2f}")
    print(f"Market open:  {executor.is_market_open()}")

    print("\n--- Open Positions ---")
    positions = executor.get_positions()
    if positions:
        for p in positions:
            print(
                f"  {p['symbol']:6} {p['qty']:>6} shares  "
                f"avg ${float(p['avg_entry_price']):.2f}  "
                f"P&L ${float(p['unrealized_pl']):.2f}"
            )
    else:
        print("  No open positions")

    print("\nConnection OK ✓")
