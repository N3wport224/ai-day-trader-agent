from __future__ import annotations

import pytest

from core.alpaca_executor import AlpacaExecutor
from core.risk_manager import RiskLimits, RiskManager


def test_alpaca_executor_uses_paper_v2_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "order-123",
                "side": "buy",
                "qty": "2",
                "symbol": "AAPL",
                "status": "accepted",
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("core.alpaca_executor.requests.post", fake_post)
    _open_market_with_account(monkeypatch)

    executor = AlpacaExecutor(risk_manager=RiskManager(RiskLimits()))
    order = executor.execute_signal(
        {
            "symbol": "AAPL",
            "recommendation": "BUY",
            "quantity": 2,
            "price": 100.0,
            "risk_parameters": {"stop_loss": 95.0, "take_profit": 110.0},
        }
    )

    assert captured["url"] == "https://paper-api.alpaca.markets/v2/orders"
    assert captured["headers"]["APCA-API-KEY-ID"] == "key"
    assert captured["headers"]["APCA-API-SECRET-KEY"] == "secret"
    assert captured["json"] == {
        "symbol": "AAPL",
        "qty": "2",
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": "110.0"},
        "stop_loss": {"stop_price": "95.0"},
    }
    assert order["id"] == "order-123"


def _open_market_with_account(monkeypatch, *, position=None, equity="100000"):
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(
        AlpacaExecutor,
        "get_account",
        lambda self: {"equity": equity, "last_equity": equity, "buying_power": equity},
    )
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, symbol: position)
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: [])


def test_alpaca_executor_reports_risk_block_without_ordering(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("blocked orders must not reach Alpaca")

    monkeypatch.setattr("core.alpaca_executor.requests.post", fail_if_called)
    _open_market_with_account(monkeypatch)

    executor = AlpacaExecutor(risk_manager=RiskManager(RiskLimits(trading_enabled=False)))
    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 2, "price": 100.0})

    assert result.order is None
    assert "TRADING_ENABLED" in result.skipped_reason


def test_alpaca_executor_sell_cancels_bracket_legs_and_caps_quantity(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")

    calls = []
    monkeypatch.setattr(
        AlpacaExecutor, "cancel_open_orders", lambda self, symbol: calls.append(("cancel", symbol)) or 2
    )
    monkeypatch.setattr(
        AlpacaExecutor,
        "_place_order",
        lambda self, symbol, qty, side: calls.append((side, symbol, qty)) or {"id": "sell-1", "qty": str(qty)},
    )
    _open_market_with_account(monkeypatch, position={"qty": "3", "market_value": "300"})

    executor = AlpacaExecutor(risk_manager=RiskManager(RiskLimits()))
    result = executor.submit({"symbol": "AAPL", "recommendation": "SELL", "quantity": 10})

    assert result.order["id"] == "sell-1"
    assert calls == [("cancel", "AAPL"), ("sell", "AAPL", 3)]


def test_alpaca_executor_rejects_non_paper_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets/v2")

    with pytest.raises(ValueError, match="Paper trading requires"):
        AlpacaExecutor()


def test_alpaca_executor_accepts_legacy_root_paper_url(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")

    executor = AlpacaExecutor()

    assert executor.base_url == "https://paper-api.alpaca.markets/v2"


def test_alpaca_executor_skips_orders_when_market_closed(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no order should be sent while the market is closed")

    monkeypatch.setattr("core.alpaca_executor.requests.post", fail_if_called)
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: False)

    executor = AlpacaExecutor()

    assert executor.execute_signal({"symbol": "AAPL", "recommendation": "BUY", "quantity": 2}) is None


def test_alpaca_executor_prefers_live_quote_and_reports_broker_rejection(monkeypatch):
    import requests

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    _open_market_with_account(monkeypatch)

    placed = {}

    class Rejected:
        text = '{"message":"stop_price must be less than base_price"}'

    def reject(self, symbol, qty, stop, target):
        placed.update(qty=qty, stop=stop, target=target)
        raise requests.exceptions.HTTPError(response=Rejected())

    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order", reject)

    executor = AlpacaExecutor(
        risk_manager=RiskManager(RiskLimits(stop_loss_pct=3.0, take_profit_pct=6.0)),
        price_lookup=lambda symbol: 50.0,
    )
    result = executor.submit(
        {
            "symbol": "AAPL",
            "recommendation": "BUY",
            "quantity": 2,
            "price": 100.0,  # stale analysis price
            "risk_parameters": {"stop_loss": 95.0, "take_profit": 110.0},
        }
    )

    # Strategy levels were set off the stale $100 price, so both fall back
    # to the configured percentages around the live $50 quote.
    assert placed == {"qty": 2, "stop": 48.5, "target": 53.0}
    assert result.order is None
    assert "stop_price must be less than base_price" in result.skipped_reason
