from __future__ import annotations

import pytest

from core.risk_manager import RiskLimits, RiskManager


ACCOUNT = {"equity": "10000", "last_equity": "10000", "buying_power": "10000"}


def _buy(manager: RiskManager, **overrides):
    kwargs = {
        "side": "BUY",
        "symbol": "AAPL",
        "quantity": 10,
        "price": 100.0,
        "account": ACCOUNT,
        "position": None,
        "orders_today": [],
        "stop_loss": 95.0,
        "take_profit": 110.0,
    }
    kwargs.update(overrides)
    return manager.check_order(**kwargs)


def test_buy_within_limits_keeps_strategy_levels() -> None:
    decision = _buy(RiskManager(RiskLimits()))

    assert decision.approved
    assert decision.quantity == 10
    assert decision.stop_loss == 95.0
    assert decision.take_profit == 110.0


def test_kill_switch_blocks_everything() -> None:
    manager = RiskManager(RiskLimits(trading_enabled=False))

    assert not _buy(manager).approved
    assert not manager.check_order(
        side="SELL", symbol="AAPL", quantity=1, price=100.0, account=ACCOUNT, position={"qty": "5"}
    ).approved


def test_daily_loss_limit_blocks_entries_but_allows_exits() -> None:
    manager = RiskManager(RiskLimits(max_daily_loss_pct=3.0))
    down_4pct = {"equity": "9600", "last_equity": "10000", "buying_power": "9600"}

    entry = _buy(manager, account=down_4pct)
    exit_ = manager.check_order(
        side="SELL", symbol="AAPL", quantity=5, price=100.0, account=down_4pct, position={"qty": "5"}
    )

    assert not entry.approved
    assert "Daily loss limit" in entry.reason
    assert exit_.approved and exit_.quantity == 5


def test_daily_entry_limit_counts_only_buys() -> None:
    manager = RiskManager(RiskLimits(max_daily_trades=2))

    assert _buy(manager, orders_today=[{"side": "buy"}, {"side": "sell"}]).approved
    blocked = _buy(manager, orders_today=[{"side": "buy"}, {"side": "buy"}])
    assert not blocked.approved
    assert "2/2" in blocked.reason


def test_position_cap_shrinks_order_including_existing_holding() -> None:
    manager = RiskManager(RiskLimits(max_position_pct=0.25))  # $2,500 of $10,000

    decision = _buy(manager, quantity=50, position={"qty": "10", "market_value": "1000"})

    assert decision.approved
    assert decision.quantity == 15  # ($2,500 - $1,000) / $100
    assert "reduced from 50 to 15" in decision.reason


def test_buying_power_limits_size() -> None:
    manager = RiskManager(RiskLimits(max_position_pct=1.0, margin_buffer_pct=0))

    decision = _buy(manager, quantity=50, account={**ACCOUNT, "buying_power": "450"})

    assert decision.quantity == 4


def test_rejects_penny_stocks_and_missing_price() -> None:
    manager = RiskManager(RiskLimits(min_price=5.0))

    assert "below MIN_PRICE" in _buy(manager, price=3.0, stop_loss=2.5, take_profit=4.0).reason
    assert not _buy(manager, price=0).approved


@pytest.mark.parametrize("stop_loss", [None, 0, 100.0, 120.0])
def test_invalid_strategy_stop_falls_back_to_configured_percent(stop_loss) -> None:
    manager = RiskManager(RiskLimits(stop_loss_pct=3.0, take_profit_pct=6.0))

    decision = _buy(manager, stop_loss=stop_loss, take_profit=110.0)

    assert decision.approved
    assert decision.stop_loss == 97.0
    assert decision.take_profit == 106.0


def test_sell_without_position_is_rejected_and_never_shorts() -> None:
    manager = RiskManager(RiskLimits())

    none_held = manager.check_order(side="SELL", symbol="AAPL", quantity=5, price=100.0, account=ACCOUNT)
    capped = manager.check_order(
        side="SELL", symbol="AAPL", quantity=50, price=100.0, account=ACCOUNT, position={"qty": "7"}
    )

    assert not none_held.approved
    assert capped.quantity == 7


def test_blocked_broker_account_is_rejected() -> None:
    decision = _buy(RiskManager(RiskLimits()), account={**ACCOUNT, "trading_blocked": True})

    assert not decision.approved
    assert "blocked" in decision.reason


def test_limits_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_ENABLED", "false")
    monkeypatch.setenv("MAX_DAILY_TRADES", "7")
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "1.5")
    monkeypatch.setenv("STOP_LOSS_PCT", "2")
    monkeypatch.delenv("TAKE_PROFIT_PCT", raising=False)

    limits = RiskLimits.from_env()

    assert limits.trading_enabled is False
    assert limits.max_daily_trades == 7
    assert limits.max_daily_loss_pct == 1.5
    assert limits.take_profit_pct == 4.0


# ---------------------------------------------------------------------------
# Dynamic sizing
# ---------------------------------------------------------------------------

from core.risk_manager import (  # noqa: E402
    SizingConfig,
    TrailingConfig,
    kelly_fraction_of_equity,
    position_size,
    risk_pct_for_trade,
    trailing_stop_price,
    volatility_scale,
)


def _risk(method, **kw):
    params = dict(probability=0.5, reward_risk=2.0, atr_pct=0.02, atr_pct_median=0.02, calibrated=True)
    params.update(kw)
    return risk_pct_for_trade(SizingConfig(method=method, base_risk_pct=1.0, kelly_fraction=0.25), **params)


def test_volatility_scale_is_inverse_and_clipped() -> None:
    assert volatility_scale(0.02, 0.02) == 1.0
    assert volatility_scale(0.04, 0.02) == 0.5          # twice as volatile -> half the risk
    assert volatility_scale(0.01, 0.02) == 1.5          # capped
    assert volatility_scale(0.10, 0.02) == 0.5          # floored
    assert volatility_scale(None, 0.02) == 1.0          # unknown -> neutral
    assert volatility_scale(float("nan"), 0.02) == 1.0


def test_fixed_volatility_and_kelly_risk_percentages() -> None:
    assert _risk("fixed", atr_pct=0.04) == 1.0
    assert _risk("volatility", atr_pct=0.04) == pytest.approx(0.5)
    # f* = 0.5 - 0.5/2 = 0.25; quarter Kelly = 6.25% -> capped at the 1% base risk
    assert kelly_fraction_of_equity(0.5, 2.0) == pytest.approx(0.25)
    assert _risk("kelly") == pytest.approx(1.0)
    # weak edge: f* = 0.35 - 0.65/2 = 0.025 -> 0.625% of equity
    assert _risk("kelly", probability=0.35) == pytest.approx(0.625)
    # no edge by the model's own odds -> no trade
    assert _risk("kelly", probability=0.3) == 0.0


def test_kelly_falls_back_to_volatility_for_uncalibrated_scores() -> None:
    assert _risk("kelly", probability=0.3, atr_pct=0.04, calibrated=False) == pytest.approx(0.5)


def test_position_size_scales_inversely_with_stop_distance() -> None:
    assert position_size(100_000, 1.0, 2.0) == 500
    assert position_size(100_000, 1.0, 4.0) == 250
    assert position_size(100_000, 0.0, 2.0) == 0
    assert position_size(100_000, 1.0, None) == 0


def test_invalid_sizing_method_rejected() -> None:
    with pytest.raises(ValueError):
        SizingConfig(method="martingale")


# ---------------------------------------------------------------------------
# Trailing stops
# ---------------------------------------------------------------------------

TRAIL = TrailingConfig(enabled=True, trigger_r=1.5, lock_r=0.0, distance_r=1.5)


def _trail(high_water, current_stop=95.0, config=TRAIL):
    return trailing_stop_price(entry=100.0, initial_stop=95.0, current_stop=current_stop,
                               high_water=high_water, config=config)


def test_trailing_waits_for_trigger_then_locks_breakeven() -> None:
    assert _trail(107.0) == 95.0     # +1.4R: untouched
    assert _trail(107.5) == 100.0    # +1.5R: breakeven (and trail 107.5 - 7.5 = 100)


def test_trailing_follows_high_and_never_lowers() -> None:
    assert _trail(112.0, current_stop=100.0) == 104.5   # 112 - 1.5R(7.5)
    assert _trail(108.0, current_stop=104.5) == 104.5   # pullback: stop holds


def test_trailing_lock_profit_and_disable() -> None:
    lock_half_r = TrailingConfig(enabled=True, trigger_r=1.5, lock_r=0.5, distance_r=None)
    assert _trail(107.5, config=lock_half_r) == 102.5
    assert _trail(130.0, config=TrailingConfig(enabled=False)) == 95.0
