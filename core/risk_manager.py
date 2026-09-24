#!/usr/bin/env python3
"""
Pre-trade risk checks for orders sent to the broker.

Every order goes through ``RiskManager.check_order`` before it is submitted.
Checks only ever block or shrink an order; they never enlarge one. Exits
(SELL of shares already held) are always allowed except by the kill switch,
so the bot can still reduce risk after a loss limit trips.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RiskLimits:
    """Account-level guardrails, configured through environment variables."""

    trading_enabled: bool = True
    max_daily_trades: int = 3
    max_daily_loss_pct: float = 3.0
    max_position_pct: float = 0.25
    min_price: float = 5.0
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 6.0

    @classmethod
    def from_env(cls) -> "RiskLimits":
        stop_loss_pct = _env_float("STOP_LOSS_PCT", cls.stop_loss_pct)
        return cls(
            trading_enabled=_env_bool("TRADING_ENABLED", cls.trading_enabled),
            max_daily_trades=_env_int("MAX_DAILY_TRADES", cls.max_daily_trades),
            max_daily_loss_pct=_env_float("MAX_DAILY_LOSS_PCT", cls.max_daily_loss_pct),
            max_position_pct=_env_float("MAX_PORTFOLIO_ALLOCATION", cls.max_position_pct),
            min_price=_env_float("MIN_PRICE", cls.min_price),
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=_env_float("TAKE_PROFIT_PCT", stop_loss_pct * 2),
        )


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    quantity: int = 0
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def round_price(price: float) -> float:
    """Round to a price increment Alpaca accepts (sub-penny only under $1)."""
    return round(price, 4 if price < 1 else 2)


class RiskManager:
    """Decides whether an order may be sent and how large it may be."""

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits.from_env()

    def check_order(
        self,
        *,
        side: str,
        symbol: str,
        quantity: int,
        price: float,
        account: Dict[str, Any],
        position: Optional[Dict[str, Any]] = None,
        orders_today: Iterable[Dict[str, Any]] = (),
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> RiskDecision:
        limits = self.limits
        side = side.lower()

        if not limits.trading_enabled:
            return RiskDecision(False, reason="Trading disabled (TRADING_ENABLED=false)")
        if account.get("trading_blocked") or account.get("account_blocked"):
            return RiskDecision(False, reason="Broker account is blocked from trading")
        if quantity <= 0:
            return RiskDecision(False, reason="Quantity must be positive")

        if side == "sell":
            held = int(_to_float((position or {}).get("qty")))
            sell_qty = min(quantity, held)
            if sell_qty <= 0:
                return RiskDecision(False, reason=f"No {symbol} position to sell")
            return RiskDecision(True, quantity=sell_qty, reason="Exit approved")

        if side != "buy":
            return RiskDecision(False, reason=f"Unsupported side: {side}")

        if price <= 0:
            return RiskDecision(False, reason="No usable price for sizing and stops")
        if price < limits.min_price:
            return RiskDecision(
                False, reason=f"{symbol} price ${price:.2f} is below MIN_PRICE ${limits.min_price:.2f}"
            )

        equity = _to_float(account.get("equity"))
        last_equity = _to_float(account.get("last_equity"))
        if equity <= 0:
            return RiskDecision(False, reason="Account equity unavailable")

        if last_equity > 0:
            day_change_pct = (equity - last_equity) / last_equity * 100
            if day_change_pct <= -limits.max_daily_loss_pct:
                return RiskDecision(
                    False,
                    reason=(
                        f"Daily loss limit hit ({day_change_pct:.2f}% vs "
                        f"-{limits.max_daily_loss_pct:.2f}%); no new entries today"
                    ),
                )

        entries_today = sum(
            1 for order in orders_today if str(order.get("side", "")).lower() == "buy"
        )
        if entries_today >= limits.max_daily_trades:
            return RiskDecision(
                False,
                reason=f"Daily entry limit reached ({entries_today}/{limits.max_daily_trades})",
            )

        stop_loss, take_profit = self._bracket_prices(price, stop_loss, take_profit)
        if stop_loss is None:
            return RiskDecision(False, reason="No valid stop-loss below the entry price")

        # Cap the total position (existing + new) at max_position_pct of equity.
        held_value = abs(_to_float((position or {}).get("market_value")))
        room = equity * limits.max_position_pct - held_value
        buying_power = _to_float(account.get("buying_power"))
        affordable = min(room, buying_power if buying_power > 0 else 0.0)
        max_qty = int(affordable // price)
        approved_qty = min(quantity, max_qty)
        if approved_qty <= 0:
            return RiskDecision(
                False,
                reason=(
                    f"No room for {symbol}: position cap ${equity * limits.max_position_pct:,.2f}, "
                    f"held ${held_value:,.2f}, buying power ${buying_power:,.2f}"
                ),
            )

        reason = "Entry approved"
        if approved_qty < quantity:
            reason = f"Entry approved, reduced from {quantity} to {approved_qty} shares by position/buying-power limits"
        return RiskDecision(
            True,
            quantity=approved_qty,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _bracket_prices(
        self,
        price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Use the strategy's stop/target as a pair when both fit the entry
        price, otherwise the configured percentages for both."""
        limits = self.limits
        tick = 0.0001 if price < 1 else 0.01

        stop = _to_float(stop_loss)
        target = _to_float(take_profit)
        if not (0 < stop <= price - tick and target >= price + tick):
            stop = price * (1 - limits.stop_loss_pct / 100)
            target = price * (1 + limits.take_profit_pct / 100)

        stop = round_price(stop)
        target = round_price(target)
        if not (0 < stop <= price - tick) or target < price + tick:
            return None, None
        return stop, target
