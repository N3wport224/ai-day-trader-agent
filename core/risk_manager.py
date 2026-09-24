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
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")


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
    # Day-trading mode: every entry is expected to be closed the same session.
    day_trading: bool = False
    pdt_min_equity: float = 25_000.0
    pdt_max_day_trades: int = 3
    pdt_buffer: int = 0
    max_intraday_drawdown_pct: float = 2.0   # 0 disables the breaker
    # No re-entry into a symbol this soon after its last exit (0 disables);
    # with reentry_cooldown_stops_only only stop-outs start the cooldown.
    reentry_cooldown_minutes: float = 30.0
    reentry_cooldown_stops_only: bool = True

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
            day_trading=_env_bool("DAY_TRADING_MODE", cls.day_trading),
            pdt_min_equity=_env_float("PDT_MIN_EQUITY", cls.pdt_min_equity),
            pdt_max_day_trades=_env_int("PDT_MAX_DAY_TRADES", cls.pdt_max_day_trades),
            pdt_buffer=_env_int("PDT_DAYTRADE_BUFFER", cls.pdt_buffer),
            max_intraday_drawdown_pct=_env_float("MAX_INTRADAY_DRAWDOWN_PCT", cls.max_intraday_drawdown_pct),
            reentry_cooldown_minutes=_env_float("REENTRY_COOLDOWN_MINUTES", cls.reentry_cooldown_minutes),
            reentry_cooldown_stops_only=_env_bool("REENTRY_COOLDOWN_STOPS_ONLY", cls.reentry_cooldown_stops_only),
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


@dataclass(frozen=True)
class ExitFill:
    """The most recent filled sell for a symbol."""

    filled_at: datetime
    stopped_out: bool
    price: float = 0.0


_STOP_TYPES = {"stop", "stop_limit", "trailing_stop"}


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def last_exit_fill(orders: Iterable[Dict[str, Any]], symbol: str) -> Optional[ExitFill]:
    """Latest filled sell of ``symbol`` in an Alpaca order list, bracket legs
    included (a stop leg that fired is how most stop-outs appear)."""
    latest: Optional[ExitFill] = None
    stack = list(orders or [])
    while stack:
        order = stack.pop()
        stack.extend(order.get("legs") or [])
        if order.get("symbol") != symbol or str(order.get("side", "")).lower() != "sell":
            continue
        if str(order.get("status", "")).lower() != "filled":
            continue
        filled_at = _parse_ts(order.get("filled_at"))
        if filled_at is None:
            continue
        if latest is None or filled_at > latest.filled_at:
            kind = str(order.get("type", order.get("order_type", ""))).lower()
            latest = ExitFill(filled_at, kind in _STOP_TYPES, _to_float(order.get("filled_avg_price")))
    return latest


class RiskManager:
    """Decides whether an order may be sent and how large it may be."""

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits.from_env()
        # Intraday drawdown breaker: once tripped, stays tripped for that session.
        self._breaker_session: Optional[date] = None
        self._breaker_reason: Optional[str] = None

    @property
    def breaker_tripped(self) -> bool:
        return self._breaker_reason is not None

    def update_breaker(self, account: Dict[str, Any], session_date: Optional[date] = None) -> Optional[str]:
        """Evaluate the intraday drawdown breaker; returns the halt reason if tripped.

        Drawdown is (equity - start-of-day equity) / start-of-day equity, where
        Alpaca's ``last_equity`` is the previous close, so realized and
        unrealized P&L are both included. Call every cycle, not only on entries.
        """
        session_date = session_date or datetime.now(MARKET_TZ).date()
        if session_date != self._breaker_session:
            self._breaker_session, self._breaker_reason = session_date, None
        limit = self.limits.max_intraday_drawdown_pct
        if self._breaker_reason or limit <= 0:
            return self._breaker_reason
        equity, start = _to_float(account.get("equity")), _to_float(account.get("last_equity"))
        if equity > 0 and start > 0:
            drawdown = (equity - start) / start * 100
            if drawdown <= -limit:
                self._breaker_reason = (
                    f"Intraday drawdown breaker tripped ({drawdown:.2f}% vs -{limit:.2f}% of starting equity); "
                    f"no new entries for the rest of the {session_date} session"
                )
        return self._breaker_reason

    def _pdt_block(self, account: Dict[str, Any], equity: float) -> Optional[str]:
        """Reject intraday entries that would trip (or trade through) the PDT rule."""
        limits = self.limits
        if not limits.day_trading or equity >= limits.pdt_min_equity:
            return None
        if account.get("pattern_day_trader") in (True, "true", "True", 1):
            return (
                f"PDT: account is flagged as a pattern day trader with equity ${equity:,.2f} "
                f"< ${limits.pdt_min_equity:,.0f}; day trading is restricted"
            )
        count = int(_to_float(account.get("daytrade_count")))
        allowed = limits.pdt_max_day_trades - limits.pdt_buffer
        if count >= allowed:
            return (
                f"PDT: {count} day trades in the last 5 business days with equity ${equity:,.2f} "
                f"< ${limits.pdt_min_equity:,.0f}; another intraday round trip would flag the account"
            )
        return None

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
        session_date: Optional[date] = None,
        last_exit: Optional["ExitFill"] = None,
        now: Optional[datetime] = None,
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

        # Latching breaker (tighter by default); evaluated after the daily loss
        # limit so each rule reports its own reason.
        breaker = self.update_breaker(account, session_date)
        if breaker:
            return RiskDecision(False, reason=breaker)
        pdt = self._pdt_block(account, equity)
        if pdt:
            return RiskDecision(False, reason=pdt)

        cooldown = self._cooldown_block(symbol, last_exit, now)
        if cooldown:
            return RiskDecision(False, reason=cooldown)

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

    def _cooldown_block(self, symbol: str, last_exit: Optional["ExitFill"], now: Optional[datetime]) -> Optional[str]:
        minutes = self.limits.reentry_cooldown_minutes
        if minutes <= 0 or last_exit is None:
            return None
        if self.limits.reentry_cooldown_stops_only and not last_exit.stopped_out:
            return None
        now = now or datetime.now(timezone.utc)
        elapsed = (now - last_exit.filled_at).total_seconds() / 60
        if 0 <= elapsed < minutes:
            kind = "stop-out" if last_exit.stopped_out else "exit"
            return (f"Re-entry cooldown for {symbol}: {kind} {elapsed:.0f} min ago "
                    f"(REENTRY_COOLDOWN_MINUTES={minutes:g})")
        return None

    def bracket_prices(self, price: float) -> tuple[Optional[float], Optional[float]]:
        """Stop/target from the configured percentages around ``price``."""
        return self._bracket_prices(price, None, None)

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


# ---------------------------------------------------------------------------
# Position sizing (the size *requested*; check_order still caps it)
# ---------------------------------------------------------------------------

SIZING_METHODS = ("fixed", "volatility", "kelly")


@dataclass(frozen=True)
class SizingConfig:
    """POSITION_SIZING_METHOD: fixed | volatility | kelly.

    fixed       risk ``base_risk_pct`` of equity between entry and stop.
    volatility  scale that risk by median(ATR%) / current ATR%, clipped to
                [vol_scale_min, vol_scale_max]: less risk when the symbol is
                unusually volatile, more when it is unusually calm.
    kelly       fractional Kelly on the model's calibrated probability, capped
                at ``base_risk_pct`` (Kelly can only shrink the bet), then
                volatility-scaled. Uncalibrated (heuristic) probabilities fall
                back to ``volatility``.
    """

    method: str = "fixed"
    base_risk_pct: float = 1.0
    kelly_fraction: float = 0.25
    vol_scale_min: float = 0.5
    vol_scale_max: float = 1.5

    def __post_init__(self) -> None:
        if self.method not in SIZING_METHODS:
            raise ValueError(f"POSITION_SIZING_METHOD must be one of {SIZING_METHODS}")

    @classmethod
    def from_env(cls) -> "SizingConfig":
        return cls(
            method=os.getenv("POSITION_SIZING_METHOD", "fixed").strip().lower(),
            base_risk_pct=_env_float("RISK_PER_TRADE_PCT", 1.0),
            kelly_fraction=_env_float("KELLY_FRACTION", 0.25),
            vol_scale_min=_env_float("VOL_SCALE_MIN", 0.5),
            vol_scale_max=_env_float("VOL_SCALE_MAX", 1.5),
        )


def volatility_scale(atr_pct: Optional[float], atr_pct_median: Optional[float], lo: float = 0.5, hi: float = 1.5) -> float:
    """median / current ATR%, clipped. 1.0 when either value is unknown."""
    cur, med = _to_float(atr_pct), _to_float(atr_pct_median)
    if not (cur > 0 and med > 0) or cur != cur or med != med:
        return 1.0
    return max(lo, min(hi, med / cur))


def kelly_fraction_of_equity(probability: float, reward_risk: float) -> float:
    """Full-Kelly fraction f* = p - (1 - p) / b for a win of b R vs a loss of 1 R."""
    if reward_risk <= 0:
        return 0.0
    return probability - (1 - probability) / reward_risk


def risk_pct_for_trade(
    config: SizingConfig,
    *,
    probability: Optional[float],
    reward_risk: Optional[float],
    atr_pct: Optional[float],
    atr_pct_median: Optional[float],
    calibrated: bool,
) -> float:
    """Percent of equity to put at risk between entry and stop."""
    scale = volatility_scale(atr_pct, atr_pct_median, config.vol_scale_min, config.vol_scale_max)
    if config.method == "fixed":
        return config.base_risk_pct
    if config.method == "kelly" and calibrated and probability is not None and reward_risk:
        kelly_pct = config.kelly_fraction * kelly_fraction_of_equity(probability, reward_risk) * 100
        if kelly_pct <= 0:
            return 0.0  # the model's own odds say this bet has no edge
        return min(config.base_risk_pct, kelly_pct) * scale
    return config.base_risk_pct * scale


def position_size(equity: float, risk_pct: float, stop_distance: Optional[float]) -> int:
    """Shares such that a stop-out loses ``risk_pct`` of equity."""
    stop_distance = _to_float(stop_distance)
    if stop_distance <= 0 or equity <= 0 or risk_pct <= 0:
        return 0
    return max(0, int(equity * risk_pct / 100 // stop_distance))


# ---------------------------------------------------------------------------
# Trailing protection for winners
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrailingConfig:
    """Once price has moved ``trigger_r`` R in favour, raise the stop to
    entry + ``lock_r`` R (breakeven by default), then trail it ``distance_r``
    R below the highest price seen. Stops only ever move up."""

    enabled: bool = True
    trigger_r: float = 1.5
    lock_r: float = 0.0
    distance_r: Optional[float] = 1.5

    @classmethod
    def from_env(cls) -> "TrailingConfig":
        distance = os.getenv("TRAILING_STOP_DISTANCE_R", "1.5").strip()
        return cls(
            enabled=_env_bool("TRAILING_STOP_ENABLED", True),
            trigger_r=_env_float("TRAILING_STOP_TRIGGER_R", 1.5),
            lock_r=_env_float("TRAILING_STOP_LOCK_R", 0.0),
            distance_r=float(distance) if distance not in {"", "none", "off"} else None,
        )


def trailing_stop_price(
    *,
    entry: float,
    initial_stop: float,
    current_stop: float,
    high_water: float,
    config: TrailingConfig,
) -> float:
    """New stop for a long position (never lower than ``current_stop``)."""
    risk = entry - initial_stop
    if not config.enabled or risk <= 0:
        return current_stop
    if (high_water - entry) / risk < config.trigger_r:
        return current_stop
    new_stop = entry + config.lock_r * risk
    if config.distance_r is not None:
        new_stop = max(new_stop, high_water - config.distance_r * risk)
    return max(current_stop, round_price(new_stop))
