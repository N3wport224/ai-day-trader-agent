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
    manager = RiskManager(RiskLimits(max_position_pct=1.0))

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
