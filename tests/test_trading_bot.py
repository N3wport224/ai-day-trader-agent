from __future__ import annotations

import pytest

from core.trading_bot import TradingBot
from core.trading_workflow import WorkflowResult


class FakeWorkflow:
    def __init__(self, fail_on: str | None = None):
        self.calls = []
        self.fail_on = fail_on

    def run(self, symbol, portfolio_name, *, record_paper_trade, submit_alpaca_paper_order):
        self.calls.append((symbol, portfolio_name, record_paper_trade, submit_alpaca_paper_order))
        if symbol == self.fail_on:
            raise RuntimeError("provider down")
        order = {"id": f"order-{symbol}"} if submit_alpaca_paper_order else None
        return WorkflowResult(
            symbol=symbol,
            portfolio_name=portfolio_name,
            analysis={"recommendation": "BUY", "quantity": 1},
            alpaca_order=order,
        )


def test_dry_run_never_submits_orders() -> None:
    workflow = FakeWorkflow()
    bot = TradingBot(workflow, ["aapl", "msft", "AAPL"], "paper")

    report = bot.run_cycle()

    assert [c[0] for c in workflow.calls] == ["AAPL", "MSFT"]
    assert all(c[2:] == (False, False) for c in workflow.calls)
    assert report.orders == []


def test_execute_mode_submits_and_isolates_symbol_errors() -> None:
    workflow = FakeWorkflow(fail_on="MSFT")
    bot = TradingBot(workflow, ["AAPL", "MSFT", "NVDA"], "paper", execute=True)

    report = bot.run_cycle()

    assert [r.symbol for r in report.orders] == ["AAPL", "NVDA"]
    assert report.errors == {"MSFT": "provider down"}


def test_closed_market_or_clock_failure_skips_cycle() -> None:
    workflow = FakeWorkflow()

    closed = TradingBot(workflow, ["AAPL"], market_clock=lambda: {"is_open": False}, execute=True)
    assert closed.run_cycle().market_open is False

    def broken_clock():
        raise ConnectionError("timeout")

    broken = TradingBot(workflow, ["AAPL"], market_clock=broken_clock, execute=True)
    assert broken.run_cycle().market_open is False
    assert workflow.calls == []


def test_run_sleeps_between_cycles_only() -> None:
    sleeps = []
    bot = TradingBot(FakeWorkflow(), ["AAPL"], interval_seconds=300, sleep=sleeps.append)

    reports = bot.run(max_cycles=3)

    assert len(reports) == 3
    assert sleeps == [300, 300]


def test_empty_watchlist_is_rejected() -> None:
    with pytest.raises(ValueError, match="Watchlist is empty"):
        TradingBot(FakeWorkflow(), ["", " "])
