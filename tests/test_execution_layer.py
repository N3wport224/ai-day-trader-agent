from __future__ import annotations

import json

import pytest
import requests

from core.alpaca_executor import AlpacaExecutor, BrokerSnapshot, flatten_orders, protective_stop_orders
from core.execution_telemetry import EventLog, classify_rejection
from core.portfolio_manager import PortfolioManager
from core.reconciliation import reconcile
from core.risk_manager import RiskLimits, RiskManager, TrailingConfig
from core.trading_bot import TradingBot
from core.trading_workflow import WorkflowResult
from core.trailing_stops import TrailingStopManager


# ---------------------------------------------------------------------------
# Rejection classification and telemetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status, body, category",
    [
        (403, '{"code": 40310000, "message": "insufficient buying power"}', "buying_power"),
        (403, '{"code": 40310000, "message": "potential wash trade detected. use complex orders"}', "wash_trade"),
        (403, '{"message": "insufficient qty available for order (requested: 10, available: 0)"}', "qty_held"),
        (403, '{"message": "trade denied due to pattern day trading protection"}', "margin_or_pdt"),
        (422, '{"message": "stop_loss.stop_price must be <= base_price - 0.01"}', "invalid_price"),
        (422, '{"message": "asset XYZ is not tradable"}', "not_tradable"),
        (429, "rate limit exceeded", "rate_limited"),
        (403, '{"message": "forbidden"}', "account_restricted"),
        (500, "boom", "unknown"),
    ],
)
def test_classify_rejection(status, body, category) -> None:
    rejection = classify_rejection(status, body)

    assert rejection.category == category
    assert rejection.root_cause and rejection.remediation
    assert rejection.http_status == status


def test_event_log_writes_jsonl(tmp_path) -> None:
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.record("order_rejected", symbol="AAPL", category="buying_power")

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["event"] == "order_rejected"
    assert log.events[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Executor recovery
# ---------------------------------------------------------------------------

class Resp:
    def __init__(self, status, message):
        self.status_code = status
        self.text = json.dumps({"message": message})


def _http_error(status, message):
    return requests.exceptions.HTTPError(response=Resp(status, message))


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: [])
    monkeypatch.setattr(AlpacaExecutor, "cancel_open_orders", lambda self, symbol: 0)
    ex = AlpacaExecutor(
        risk_manager=RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0)),
        price_lookup=lambda symbol: 100.0,
        telemetry=EventLog(None),
    )
    ex.account = {"equity": "100000", "last_equity": "100000", "buying_power": "100000"}
    monkeypatch.setattr(AlpacaExecutor, "get_account", lambda self: self.account)
    return ex


def test_buying_power_rejection_retries_smaller(monkeypatch, executor) -> None:
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, symbol: None)
    attempts = []

    def bracket(self, symbol, qty, stop, target, **kw):
        attempts.append(qty)
        if len(attempts) == 1:
            self.account = {**self.account, "buying_power": "5000"}  # someone else used the cash
            raise _http_error(403, "insufficient buying power")
        return {"id": "o-2", "qty": str(qty), "symbol": symbol}

    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order", bracket)

    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 100})

    assert attempts == [100, 48]  # 5000 * 0.97 // 100
    assert result.recovered and result.order["id"] == "o-2"
    assert result.rejection.category == "buying_power"
    assert [e["event"] for e in executor.telemetry.events] == [
        "order_rejected", "order_retry", "order_recovered", "order_submitted",
    ]
    assert executor.telemetry.events[-1]["recovered"] is True and executor.telemetry.events[-1]["qty"] == 48


def test_wash_trade_on_exit_cancels_legs_and_retries(monkeypatch, executor) -> None:
    cancels, sells = [], []
    monkeypatch.setattr(AlpacaExecutor, "cancel_open_orders", lambda self, s: cancels.append(s) or 1)
    monkeypatch.setattr(AlpacaExecutor, "get_position",
                        lambda self, s: {"qty": "10", "qty_available": "10", "market_value": "1000"})

    def sell(self, symbol, qty, side):
        sells.append(qty)
        if len(sells) == 1:
            raise _http_error(403, "potential wash trade detected")
        return {"id": "s-2", "qty": str(qty)}

    monkeypatch.setattr(AlpacaExecutor, "_place_order", sell)

    result = executor.submit({"symbol": "AAPL", "recommendation": "SELL", "quantity": 10})

    assert result.recovered and result.rejection.category == "wash_trade"
    assert sells == [10, 10] and cancels == ["AAPL", "AAPL"]


def test_unrecoverable_rejection_is_reported_with_root_cause(monkeypatch, executor) -> None:
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, s: None)

    def bracket(self, *args, **kw):
        raise _http_error(403, "trade denied due to pattern day trading protection")

    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order", bracket)

    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 5})

    assert result.order is None and not result.recovered
    assert result.rejection.category == "margin_or_pdt"
    assert "(margin_or_pdt)" in result.skipped_reason
    assert [e["event"] for e in executor.telemetry.events] == ["order_rejected"]


def test_failed_retry_reports_second_rejection(monkeypatch, executor) -> None:
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, s: None)
    calls = []

    def bracket(self, symbol, qty, stop, target, **kw):
        calls.append(qty)
        if len(calls) == 1:
            self.account = {**self.account, "buying_power": "5000"}
            raise _http_error(403, "insufficient buying power")
        raise _http_error(422, "asset AAPL is not tradable")

    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order", bracket)

    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 100})

    assert result.order is None and result.rejection.category == "not_tradable"
    assert "and the retry (not_tradable)" in result.skipped_reason


# ---------------------------------------------------------------------------
# Broker state helpers
# ---------------------------------------------------------------------------

def test_flatten_orders_includes_working_legs_once() -> None:
    orders = [
        {"id": "p1", "symbol": "AAPL", "side": "buy", "type": "market", "legs": [
            {"id": "l1", "symbol": "AAPL", "side": "sell", "type": "limit", "status": "new"},
            {"id": "l2", "symbol": "AAPL", "side": "sell", "type": "stop", "status": "new", "stop_price": "95"},
            {"id": "l3", "symbol": "AAPL", "side": "sell", "type": "stop", "status": "canceled"},
        ]},
        {"id": "l2", "symbol": "AAPL", "side": "sell", "type": "stop", "status": "new", "stop_price": "95"},
    ]

    flat = flatten_orders(orders)

    assert sorted(o["id"] for o in flat) == ["l1", "l2", "p1"]
    assert [o["id"] for o in protective_stop_orders(flat, "AAPL")] == ["l2"]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _snapshot(positions, orders=()):
    return BrokerSnapshot(positions=list(positions), open_orders=list(orders))


def test_reconciliation_detects_and_syncs(portfolio_manager: PortfolioManager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    portfolio_manager.update_holding("paper", "MSFT", 5, 300.0)   # broker has 8
    portfolio_manager.update_holding("paper", "TSLA", 3, 200.0)   # broker has none
    snapshot = _snapshot(
        [
            {"symbol": "MSFT", "qty": "8", "avg_entry_price": "310"},
            {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150"},
        ],
        [{"id": "s1", "symbol": "MSFT", "side": "sell", "type": "stop", "qty": "8", "stop_price": "290"}],
    )
    log = EventLog(None)

    report = reconcile(snapshot, portfolio_manager, "paper", mode="sync", telemetry=log)

    assert report.kinds() == {"quantity_mismatch": 1, "unprotected": 1, "missing_locally": 1, "missing_at_broker": 1}
    unprotected = [d for d in report.discrepancies if d.kind == "unprotected"]
    assert unprotected[0].symbol == "AAPL"
    holdings = {h["symbol"]: h["quantity"] for h in portfolio_manager.get_holdings("paper")}
    assert holdings == {"MSFT": 8, "AAPL": 10}
    assert log.events[0]["event"] == "reconciliation"


def test_reconciliation_report_mode_does_not_modify(portfolio_manager: PortfolioManager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    snapshot = _snapshot([{"symbol": "AAPL", "qty": "10", "avg_entry_price": "150"}])

    report = reconcile(snapshot, portfolio_manager, "paper", mode="report", telemetry=EventLog(None))

    assert not report.clean and report.synced == []
    assert portfolio_manager.get_holdings("paper") == []


def test_clean_reconciliation(portfolio_manager: PortfolioManager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    portfolio_manager.update_holding("paper", "AAPL", 10, 150.0)
    snapshot = _snapshot(
        [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "150"}],
        [{"id": "s", "symbol": "AAPL", "side": "sell", "type": "stop", "qty": "10", "stop_price": "140"}],
    )

    assert reconcile(snapshot, portfolio_manager, "paper", telemetry=EventLog(None)).clean


# ---------------------------------------------------------------------------
# Live trailing stops
# ---------------------------------------------------------------------------

class FakeBroker:
    def __init__(self, fail=False):
        self.replaced = []
        self.fail = fail

    def replace_stop_price(self, order_id, stop_price):
        if self.fail:
            raise _http_error(422, "stop_price must be less than current price")
        self.replaced.append((order_id, stop_price))
        return {"id": order_id + "-r", "stop_price": str(stop_price)}


def _position_snapshot(price, stop):
    return _snapshot(
        [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "100", "current_price": str(price)}],
        [{"id": "stop-1", "symbol": "AAPL", "side": "sell", "type": "stop", "qty": "10", "stop_price": str(stop)}],
    )


def _manager(tmp_path, broker, **config):
    cfg = TrailingConfig(**{"enabled": True, "trigger_r": 1.5, "lock_r": 0.0, "distance_r": None, **config})
    return TrailingStopManager(broker, config=cfg, state_path=str(tmp_path / "trail.json"), telemetry=EventLog(None))


def test_trailing_manager_moves_stop_to_breakeven_after_trigger(tmp_path) -> None:
    broker = FakeBroker()
    manager = _manager(tmp_path, broker)

    assert manager.update(_position_snapshot(105.0, 95.0)) == []       # +1R: nothing
    adjustments = manager.update(_position_snapshot(107.6, 95.0))      # +1.52R: breakeven

    assert broker.replaced == [("stop-1", 100.0)]
    assert adjustments[0].ok and "locks +0.00R" in adjustments[0].detail
    assert manager.telemetry.events[-1]["event"] == "trailing_stop_raised"


def test_trailing_manager_remembers_initial_stop_across_restarts(tmp_path) -> None:
    broker = FakeBroker()
    _manager(tmp_path, broker, distance_r=1.5).update(_position_snapshot(104.0, 95.0))  # records R = 5

    restarted = _manager(tmp_path, broker, distance_r=1.5)
    # The broker stop is already at breakeven, but R must still be 5, not 0.
    restarted.update(_position_snapshot(112.0, 100.0))

    assert broker.replaced == [("stop-1", 104.5)]  # 112 - 1.5 * 5


def test_trailing_manager_logs_failures_and_forgets_closed_positions(tmp_path) -> None:
    manager = _manager(tmp_path, FakeBroker(fail=True))
    adjustments = manager.update(_position_snapshot(108.0, 95.0))

    assert not adjustments[0].ok
    assert manager.telemetry.events[-1]["event"] == "trailing_stop_failed"
    manager.update(_snapshot([]))
    assert manager.state == {}


# ---------------------------------------------------------------------------
# Bot cycle integration
# ---------------------------------------------------------------------------

class StubWorkflow:
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.calls = 0

    def run(self, symbol, portfolio_name, *, record_paper_trade, submit_alpaca_paper_order):
        self.calls += 1
        return WorkflowResult(symbol=symbol, portfolio_name=portfolio_name, analysis={"recommendation": "HOLD"})


class StubBroker:
    def __init__(self, fail=False):
        self.fail = fail

    def get_snapshot(self):
        if self.fail:
            raise ConnectionError("broker down")
        return _snapshot([])


class StubTrailing:
    def __init__(self):
        self.calls = 0

    def update(self, snapshot):
        self.calls += 1
        return []


def test_bot_reconciles_and_trails_each_cycle_in_execute_mode(portfolio_manager) -> None:
    modes, trailing = [], StubTrailing()

    def fake_reconcile(snapshot, pm, name, mode=None):
        modes.append(mode)
        return reconcile(snapshot, pm, name, mode="report", telemetry=EventLog(None))

    bot = TradingBot(StubWorkflow(portfolio_manager), ["AAPL"], execute=True,
                     broker=StubBroker(), trailing_manager=trailing, reconcile_fn=fake_reconcile)
    report = bot.run_cycle()

    assert report.reconciliation is not None and trailing.calls == 1
    assert modes == [None]  # execute mode uses RECONCILE_MODE (default sync)


def test_bot_dry_run_reconciles_read_only_and_never_trails(portfolio_manager) -> None:
    modes, trailing = [], StubTrailing()

    def fake_reconcile(snapshot, pm, name, mode=None):
        modes.append(mode)
        return reconcile(snapshot, pm, name, mode=mode, telemetry=EventLog(None))

    TradingBot(StubWorkflow(portfolio_manager), ["AAPL"], broker=StubBroker(),
               trailing_manager=trailing, reconcile_fn=fake_reconcile).run_cycle()

    assert modes == ["report"] and trailing.calls == 0


def test_bot_skips_cycle_when_broker_state_unknown(portfolio_manager) -> None:
    workflow = StubWorkflow(portfolio_manager)
    bot = TradingBot(workflow, ["AAPL"], execute=True, broker=StubBroker(fail=True))

    report = bot.run_cycle()

    assert report.broker_error == "broker down"
    assert workflow.calls == 0


def test_bot_logs_clean_reconciliation_and_mtf_state(portfolio_manager, caplog) -> None:
    import logging

    class MLWorkflow(StubWorkflow):
        def run(self, symbol, portfolio_name, *, record_paper_trade, submit_alpaca_paper_order):
            ml = {"mode": "model", "probability_up": 0.7, "regime": "TRENDING_BULL", "macro_aligned": 0.0,
                  "gated_by": "mtf", "sentiment_available": False}
            return WorkflowResult(symbol=symbol, portfolio_name=portfolio_name,
                                  analysis={"recommendation": "HOLD", "all_signals": {"ml": ml}})

    with caplog.at_level(logging.INFO):
        TradingBot(MLWorkflow(portfolio_manager), ["AAPL"], broker=StubBroker()).run_cycle()

    text = caplog.text
    assert "Reconciliation clean: 0 positions, 0 open orders" in text
    assert "regime TRENDING_BULL daily trend not aligned vetoed by mtf" in text


def test_trailing_manager_logs_initialisation(tmp_path, caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO):
        _manager(tmp_path, FakeBroker())

    assert "Trailing stops enabled: trigger 1.5R" in caplog.text
    assert "(0 tracked position(s))" in caplog.text
