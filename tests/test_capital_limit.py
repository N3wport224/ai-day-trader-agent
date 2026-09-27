"""'Start small': a hard cap on money in positions, with loss limits scaled to it."""
from __future__ import annotations

import os

from config.api import control
from core.risk_manager import PortfolioRisk, RiskLimits, RiskManager, portfolio_risk

ACCOUNT = {"equity": "30000", "last_equity": "30000", "buying_power": "60000"}


def _manager(**kw):
    return RiskManager(RiskLimits(min_price=1.0, max_position_pct=0.5, max_portfolio_heat_pct=0,
                                  margin_buffer_pct=0, reentry_cooldown_minutes=0, **kw))


def _buy(manager, account=ACCOUNT, portfolio=PortfolioRisk(), quantity=100, price=100.0):
    return manager.check_order(side="BUY", symbol="AAPL", quantity=quantity, price=price, account=account,
                               stop_loss=price * 0.98, take_profit=price * 1.04, portfolio=portfolio)


def test_no_limit_uses_the_whole_account():
    assert _buy(_manager()).quantity == 100              # $10,000 < 50% of $30,000


def test_capital_limit_caps_position_size_and_total_exposure():
    manager = _manager(max_capital=2000)
    assert _buy(manager).quantity == 10                  # 50% of $2,000 = $1,000 per position
    in_use = PortfolioRisk(open_positions=1, market_value=1500)
    decision = _buy(manager, portfolio=in_use)
    assert decision.quantity == 5                        # only $500 of the $2,000 left
    full = _buy(manager, portfolio=PortfolioRisk(open_positions=2, market_value=2000))
    assert not full.approved and "capital limit $2,000" in full.reason


def test_loss_limits_scale_to_the_capital_limit():
    manager = _manager(max_capital=1000, max_daily_loss_pct=3.0, max_intraday_drawdown_pct=2.0)
    down_25 = {**ACCOUNT, "equity": "29975"}             # -$25: 0.08% of the account, 2.5% of $1,000
    decision = _buy(manager, account=down_25, quantity=1)
    assert not decision.approved and "drawdown breaker" in decision.reason
    assert _buy(_manager(max_intraday_drawdown_pct=2.0), account=down_25, quantity=1).approved  # no cap: fine


def test_heat_budget_scales_to_the_capital_limit():
    manager = RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0, max_portfolio_heat_pct=4.0,
                                     margin_buffer_pct=0, reentry_cooldown_minutes=0, max_capital=5000))
    decision = _buy(manager, quantity=100, portfolio=PortfolioRisk())
    assert decision.quantity == 50                       # $5,000 cap; $200 heat budget / $2 risk = 100 -> cap 50


def test_portfolio_risk_reports_money_in_use():
    risk = portfolio_risk([{"symbol": "MSFT", "qty": "10", "current_price": "400", "market_value": "4000"},
                           {"symbol": "AMD", "qty": "5", "current_price": "150"}], [])
    assert risk.market_value == 4750.0


def test_bot_reads_the_limit_per_mode(monkeypatch):
    from bot import mode_capital_limit

    monkeypatch.setenv("LIVE_MAX_CAPITAL", "1500")
    monkeypatch.setenv("PAPER_MAX_CAPITAL", "junk")
    assert mode_capital_limit("live") == 1500.0 and mode_capital_limit("paper") == 0.0


def test_capital_api(client, manager):
    assert client.get("/api/control/live/capital").json()["max_capital"] == 0.0
    saved = client.post("/api/control/live/capital", json={"max_capital": 1000}).json()
    assert saved["max_capital"] == 1000 and "applies when the bot starts" in saved["message"]
    assert os.environ["LIVE_MAX_CAPITAL"] == "1000"
    assert client.post("/api/control/live/capital", json={"max_capital": -5}).status_code == 422
    cleared = client.post("/api/control/live/capital", json={"max_capital": 0}).json()
    assert cleared["max_capital"] == 0 and "LIVE_MAX_CAPITAL" not in os.environ
    assert control.capital_limit("live") == 0.0
