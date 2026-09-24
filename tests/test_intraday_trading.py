from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
import requests

from core.alpaca_executor import AlpacaExecutor, BrokerSnapshot
from core.backtester import Backtester, BacktestConfig
from core.execution_telemetry import EventLog
from core.ml_strategy import MLStrategy
from core.risk_manager import RiskLimits, RiskManager
from core.session_clock import SessionClock, SessionConfig
from core.trading_bot import TradingBot
from core.trading_workflow import TradingWorkflow, WorkflowResult

SMALL = {"equity": "10000", "last_equity": "10000", "buying_power": "10000"}


def _buy(manager, account, **kw):
    return manager.check_order(
        side="BUY", symbol="AAPL", quantity=5, price=100.0, account=account,
        stop_loss=98.0, take_profit=104.0, session_date=kw.pop("session_date", date(2026, 3, 2)), **kw,
    )


# ---------------------------------------------------------------------------
# PDT gate
# ---------------------------------------------------------------------------

def test_pdt_blocks_fourth_day_trade_under_25k() -> None:
    manager = RiskManager(RiskLimits(day_trading=True, min_price=1.0, max_position_pct=1.0))

    assert _buy(manager, {**SMALL, "daytrade_count": 2}).approved
    blocked = _buy(manager, {**SMALL, "daytrade_count": 3})
    assert not blocked.approved and blocked.reason.startswith("PDT: 3 day trades")


def test_pdt_blocks_flagged_accounts_and_honours_buffer() -> None:
    flagged = _buy(RiskManager(RiskLimits(day_trading=True, min_price=1.0)),
                   {**SMALL, "pattern_day_trader": True, "daytrade_count": 0})
    assert not flagged.approved and "flagged as a pattern day trader" in flagged.reason

    cautious = RiskManager(RiskLimits(day_trading=True, pdt_buffer=1, min_price=1.0, max_position_pct=1.0))
    assert not _buy(cautious, {**SMALL, "daytrade_count": 2}).approved


def test_pdt_does_not_apply_above_25k_swing_mode_or_to_exits() -> None:
    big = {"equity": "30000", "last_equity": "30000", "buying_power": "30000", "daytrade_count": 9}
    assert _buy(RiskManager(RiskLimits(day_trading=True, min_price=1.0)), big).approved
    assert _buy(RiskManager(RiskLimits(day_trading=False, min_price=1.0, max_position_pct=1.0)),
                {**SMALL, "daytrade_count": 5}).approved
    exit_ = RiskManager(RiskLimits(day_trading=True)).check_order(
        side="SELL", symbol="AAPL", quantity=5, price=100.0,
        account={**SMALL, "daytrade_count": 5, "pattern_day_trader": True}, position={"qty": "5"},
    )
    assert exit_.approved


# ---------------------------------------------------------------------------
# Intraday drawdown breaker
# ---------------------------------------------------------------------------

def test_breaker_latches_for_the_session_and_resets_next_day() -> None:
    manager = RiskManager(RiskLimits(max_intraday_drawdown_pct=2.0, max_daily_loss_pct=50, min_price=1.0,
                                     max_position_pct=1.0))
    day = date(2026, 3, 2)

    assert manager.update_breaker({"equity": "9850", "last_equity": "10000"}, day) is None       # -1.5%
    reason = manager.update_breaker({"equity": "9790", "last_equity": "10000"}, day)             # -2.1%
    assert reason and "rest of the 2026-03-02 session" in reason

    recovered = {"equity": "10100", "last_equity": "10000", "buying_power": "10000"}
    assert not _buy(manager, recovered, session_date=day).approved       # still latched
    assert _buy(manager, recovered, session_date=date(2026, 3, 3)).approved  # new session


def test_breaker_disabled_with_zero() -> None:
    manager = RiskManager(RiskLimits(max_intraday_drawdown_pct=0, max_daily_loss_pct=50))
    assert manager.update_breaker({"equity": "5000", "last_equity": "10000"}) is None


# ---------------------------------------------------------------------------
# Executor: session gating and EOD flatten
# ---------------------------------------------------------------------------

@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    return AlpacaExecutor(telemetry=EventLog(None), price_lookup=lambda s: 100.0,
                          risk_manager=RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0)))


def _clock(local_time: str, is_open: bool = True):
    ts = pd.Timestamp(f"2026-03-02 {local_time}", tz="America/New_York")
    return {"is_open": is_open, "timestamp": ts.isoformat(), "next_close": "2026-03-02T16:00:00-05:00"}


def test_executor_blocks_entries_outside_open_phase(monkeypatch, executor) -> None:
    executor.session_clock = SessionClock(SessionConfig())
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(AlpacaExecutor, "get_clock", lambda self: _clock("09:35"))

    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 1})

    assert result.order is None and result.skipped_reason == "Entries not allowed during OPENING_LOCKOUT"


def test_executor_still_allows_exits_during_cutoff(monkeypatch, executor) -> None:
    executor.session_clock = SessionClock(SessionConfig())
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(AlpacaExecutor, "get_clock", lambda self: _clock("15:47"))
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, s: {"qty": "3", "market_value": "300"})
    monkeypatch.setattr(AlpacaExecutor, "get_account", lambda self: SMALL)
    monkeypatch.setattr(AlpacaExecutor, "cancel_open_orders", lambda self, s: 0)
    monkeypatch.setattr(AlpacaExecutor, "_place_order", lambda self, s, q, side: {"id": "sell-1", "qty": str(q)})

    result = executor.submit({"symbol": "AAPL", "recommendation": "SELL", "quantity": 3})

    assert result.order["id"] == "sell-1"


class _Resp:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(response=self)


def test_flatten_all_cancels_orders_then_closes_every_position(monkeypatch, executor) -> None:
    calls = []
    monkeypatch.setattr(AlpacaExecutor, "get_positions", lambda self: [
        {"symbol": "AAPL", "qty": "10", "current_price": "100"},
        {"symbol": "MSFT", "qty": "5", "current_price": "300"},
    ])

    def fake_delete(url, headers, timeout):
        calls.append(url.rsplit("/v2", 1)[1])
        if url.endswith("/orders"):
            return _Resp(207, [{"id": "a"}, {"id": "b"}])
        if url.endswith("/MSFT"):
            return _Resp(403, {"message": "trade denied due to pattern day trading protection"})
        return _Resp(200, {"id": "close-" + url.rsplit("/", 1)[1]})

    monkeypatch.setattr("core.alpaca_executor.requests.delete", fake_delete)

    report = executor.flatten_all()

    assert calls == ["/orders", "/positions/AAPL", "/positions/MSFT"]  # orders cancelled first
    assert report.cancelled == 2
    assert report.closed == [{"symbol": "AAPL", "qty": 10.0, "order_id": "close-AAPL"}]
    assert report.failures[0]["symbol"] == "MSFT" and report.failures[0]["category"] == "margin_or_pdt"
    assert executor.telemetry.events[-1]["event"] == "flatten"


def test_flatten_can_use_marketable_limit_orders(monkeypatch, executor) -> None:
    monkeypatch.setenv("FLATTEN_ORDER_TYPE", "limit")
    monkeypatch.setenv("FLATTEN_LIMIT_OFFSET_BPS", "20")
    posted = {}

    def fake_post(url, headers, json, timeout):
        posted.update(json)
        return _Resp(200, {"id": "lim-1"})

    monkeypatch.setattr("core.alpaca_executor.requests.post", fake_post)

    executor.close_position("AAPL", 10, current_price=100.0)

    assert posted == {"symbol": "AAPL", "qty": "10", "side": "sell", "type": "limit",
                      "limit_price": "99.8", "time_in_force": "day"}


# ---------------------------------------------------------------------------
# Bot: session phases
# ---------------------------------------------------------------------------

class RecordingWorkflow:
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.blocks = []

    def run(self, symbol, portfolio_name, *, record_paper_trade, submit_alpaca_paper_order, entry_block_reason=None):
        self.blocks.append(entry_block_reason)
        return WorkflowResult(symbol=symbol, portfolio_name=portfolio_name, analysis={"recommendation": "BUY"})


class FlattenBroker:
    def __init__(self, positions=1):
        self.flattened = 0
        self.positions = positions

    def get_snapshot(self):
        return BrokerSnapshot(positions=[{"symbol": "AAPL", "qty": "1"}] * self.positions, open_orders=[])

    def flatten_all(self, reason):
        from core.alpaca_executor import FlattenReport

        self.flattened += 1
        return FlattenReport(reason=reason, cancelled=2, closed=[{"symbol": "AAPL", "qty": 1, "order_id": "x"}])


def _bot(pm, local_time, *, execute=True, broker=None, positions=1):
    workflow = RecordingWorkflow(pm)
    broker = broker or FlattenBroker(positions)
    bot = TradingBot(
        workflow, ["AAPL"], execute=execute, market_clock=lambda: _clock(local_time), broker=broker,
        reconcile_fn=lambda *a, **k: type("R", (), {"discrepancies": [], "positions": 0, "open_orders": 0})(),
        session_clock=SessionClock(SessionConfig()),
    )
    return bot, workflow, broker


@pytest.mark.parametrize("when, phase, block", [
    ("09:40", "OPENING_LOCKOUT", "opening lockout (opening range still forming)"),
    ("11:00", "OPEN", None),
    ("15:46", "ENTRY_CUTOFF", "end-of-day entry cutoff"),
])
def test_bot_blocks_entries_by_phase(portfolio_manager, when, phase, block) -> None:
    bot, workflow, broker = _bot(portfolio_manager, when)
    report = bot.run_cycle()

    assert report.phase == phase
    assert workflow.blocks == [block]
    assert broker.flattened == 0


def test_bot_flattens_in_flatten_phase_and_skips_scanning(portfolio_manager) -> None:
    bot, workflow, broker = _bot(portfolio_manager, "15:51")
    report = bot.run_cycle()

    assert report.phase == "FLATTEN" and broker.flattened == 1
    assert report.flatten.closed[0]["symbol"] == "AAPL"
    assert workflow.blocks == []


def test_bot_flatten_is_read_only_in_dry_run_and_skipped_when_flat(portfolio_manager) -> None:
    bot, _, broker = _bot(portfolio_manager, "15:51", execute=False)
    bot.run_cycle()
    assert broker.flattened == 0

    bot, _, broker = _bot(portfolio_manager, "15:51", positions=0)
    bot.run_cycle()
    assert broker.flattened == 0


def test_bot_wakes_up_exactly_at_flatten_time(portfolio_manager) -> None:
    bot, _, _ = _bot(portfolio_manager, "15:45")
    bot.interval_seconds = 900
    bot.now_fn = lambda: pd.Timestamp("2026-03-02 15:45", tz="America/New_York").to_pydatetime()
    bot.run_cycle()

    assert bot._next_sleep() == 301      # 15:50 flatten + 1s, not 16:00
    bot.now_fn = lambda: pd.Timestamp("2026-03-02 11:00", tz="America/New_York").to_pydatetime()
    assert bot._next_sleep() == 900


def test_bot_breaker_blocks_entries(portfolio_manager) -> None:
    class BreakerBroker(FlattenBroker):
        def __init__(self):
            super().__init__()
            self.risk_manager = RiskManager(RiskLimits(max_intraday_drawdown_pct=2.0))

        def get_account(self):
            return {"equity": "9700", "last_equity": "10000"}

    bot, workflow, _ = _bot(portfolio_manager, "11:00", broker=BreakerBroker())
    report = bot.run_cycle()

    assert report.entry_block.startswith("Intraday drawdown breaker tripped")
    assert workflow.blocks[0].startswith("Intraday drawdown breaker tripped")


def test_workflow_entry_block_vetoes_buys_but_not_sells(portfolio_manager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    portfolio_manager.update_holding("paper", "AAPL", 5, 10.0)

    def run(action):
        workflow = TradingWorkflow(
            portfolio_manager,
            analysis_runner=lambda s, k, p: {"recommendation": action, "quantity": 1,
                                             "all_signals": {"technical": {"current_price": 10.0}}},
            api_key_loader=lambda: {},
            paper_order_executor_factory=lambda: type("E", (), {"submit": lambda self, sig: None})(),
        )
        return workflow.run("AAPL", "paper", record_paper_trade=True, entry_block_reason="opening lockout")

    assert run("BUY").skipped_reason == "Entry blocked: opening lockout"
    sell = run("SELL")
    assert sell.skipped_reason is None and sell.recorded_trade_id is not None


# ---------------------------------------------------------------------------
# Backtester intraday rules
# ---------------------------------------------------------------------------

class ScheduledModel:
    def __init__(self, schedule):
        self.schedule = schedule

    def predict_proba(self, X):
        p = np.array([self.schedule.get(ts, 0.5) for ts in X.index])
        return np.column_stack([1 - p, p])


def _day_bars(day="2026-03-02", n=78, start="09:30", price=100.0, freq="5min"):
    idx = pd.date_range(pd.Timestamp(f"{day} {start}", tz="America/New_York"), periods=n, freq=freq)
    close = np.full(n, price)
    return pd.DataFrame({"open": close, "high": close + 0.25, "low": close - 0.25, "close": close,
                         "volume": 1e4}, index=idx.tz_convert("UTC"))


def _intraday_backtest(bars, schedule, *, capital=100_000, limits=None, bar_length=pd.Timedelta(minutes=5),
                       session=SessionConfig(), **cfg):
    strategy = MLStrategy({"pipeline": ScheduledModel(schedule), "label_params": {"stop_atr_mult": 4.0,
                                                                              "target_atr_mult": 8.0}},
                          confidence_threshold=0.6, regime_policy="off", exit_threshold=0.01)
    config = BacktestConfig(initial_capital=capital, bar_length=bar_length, slippage_bps=0,
                            session=session, **cfg)
    manager = RiskManager(limits or RiskLimits(min_price=1.0, max_position_pct=0.5, max_daily_trades=10))
    return Backtester(strategy, manager, config).run({"AAA": bars})


def _et(bars, hhmm, day="2026-03-02"):
    return pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").tz_convert("UTC")


def test_backtest_forces_eod_flatten_at_1550() -> None:
    bars = pd.concat([_day_bars("2026-02-27"), _day_bars("2026-03-02")])  # prior day warms up ATR
    result = _intraday_backtest(bars, {_et(bars, "14:00"): 0.9})

    trade = result.trades[0]
    assert trade.exit_reason == "eod_flatten"
    assert trade.exit_time == _et(bars, "15:50")
    assert trade.exit_time.date() == trade.entry_time.date()


def test_backtest_flattens_on_last_bar_when_no_bar_opens_after_1550() -> None:
    hourly = pd.concat([_day_bars("2026-02-2" + str(d), n=7, start="09:30", freq="1h") for d in (3, 4, 5, 6, 7)]
                       + [_day_bars("2026-03-02", n=7, start="09:30", freq="1h")])
    decide = _et(hourly, "10:30")
    result = _intraday_backtest(hourly, {decide: 0.9}, bar_length=pd.Timedelta(hours=1))

    trade = result.trades[0]
    assert trade.exit_reason == "eod_flatten"
    assert trade.exit_time == _et(hourly, "15:30")   # last regular bar of the day, at its close


def test_backtest_opening_lockout_blocks_early_entries() -> None:
    bars = pd.concat([_day_bars("2026-02-27"), _day_bars("2026-03-02")])
    result = _intraday_backtest(bars, {_et(bars, "09:30"): 0.9, _et(bars, "15:40"): 0.9})

    assert result.trades == []
    assert result.session_blocked == {"OPENING_LOCKOUT": 1, "ENTRY_CUTOFF": 1}


def test_backtest_charges_half_the_spread_per_fill() -> None:
    bars = pd.concat([_day_bars("2026-02-27"), _day_bars("2026-03-02")])
    trade = _intraday_backtest(bars, {_et(bars, "14:00"): 0.9}, spread_bps=10).trades[0]

    assert trade.entry_price == pytest.approx(100.0 * 1.0005)
    assert trade.exit_price == pytest.approx(100.0 * 0.9995)


def test_backtest_simulates_pdt_for_small_accounts() -> None:
    days = ["2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-03-02"]
    bars = pd.concat([_day_bars(d) for d in days])
    schedule = {_et(bars, "14:00", d): 0.9 for d in days[1:]}
    limits = RiskLimits(day_trading=True, min_price=1.0, max_position_pct=0.5, max_daily_trades=10)

    small = _intraday_backtest(bars, schedule, capital=10_000, limits=limits)
    large = _intraday_backtest(bars, schedule, capital=50_000, limits=limits)

    assert len(small.trades) == 3                  # 4th intraday round trip blocked
    assert any(k.startswith("PDT") for k in small.blocked)
    assert len(large.trades) == 4


def test_backtest_without_session_rules_is_unchanged() -> None:
    bars = pd.concat([_day_bars("2026-02-27"), _day_bars("2026-03-02")])
    result = _intraday_backtest(bars, {_et(bars, "14:00"): 0.9}, session=None)

    assert result.trades[0].exit_reason == "end_of_test"   # held; no EOD rule
