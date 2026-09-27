"""Event-risk filter: earnings and listed event blackouts pause new entries (never exits)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from config.api import control
from core.alpaca_executor import AlpacaExecutor
from core.event_calendar import EventCalendar, parse_blackouts
from core.execution_telemetry import EventLog
from core.risk_manager import RiskLimits, RiskManager

ET = ZoneInfo("America/New_York")


def _et(day: str, hhmm: str = "10:30") -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}").replace(tzinfo=ET)


class Fetcher:
    def __init__(self, dates=None, fail=False):
        self.dates, self.fail, self.calls = dates or {}, fail, []

    def __call__(self, symbol):
        self.calls.append(symbol)
        if self.fail:
            raise ConnectionError("yahoo unavailable")
        return [datetime.fromisoformat(d).replace(tzinfo=ET) for d in self.dates.get(symbol, [])]


def _calendar(tmp_path, fetch, now, **kw):
    return EventCalendar(cache_path=tmp_path / "earnings.json", fetch=fetch, now_fn=lambda: now, **kw)


def test_parse_blackouts():
    windows = parse_blackouts("2026-10-28 13:45-15:00; 2026-11-12\n# comment\nnot a date; 2026-10-01 15:00-14:00")
    assert [(w.day, w.start, w.end) for w in windows] == [
        (date(2026, 10, 28), datetime(1, 1, 1, 13, 45).time(), datetime(1, 1, 1, 15, 0).time()),
        (date(2026, 11, 12), None, None)]


def test_earnings_day_and_next_trading_day_are_blocked(tmp_path):
    # AAPL reports Thursday Oct 29 after the close: Thursday and Friday are paused.
    fetch = Fetcher({"AAPL": ["2026-10-29T16:30"]})
    now = _et("2026-10-29")
    cal = _calendar(tmp_path, fetch, now)
    assert "Earnings for AAPL today" in cal.block_reason("AAPL", now)
    assert cal.block_reason("AAPL", _et("2026-10-30")) is not None
    assert cal.block_reason("AAPL", _et("2026-11-02")) is None       # Monday: two trading days later
    assert cal.block_reason("AAPL", _et("2026-10-28")) is None       # day before: fine when day trading
    assert cal.block_reason("MSFT", now) is None                     # no earnings / ETF


def test_friday_report_pauses_monday_and_swing_mode_pauses_the_day_before(tmp_path, monkeypatch):
    fetch = Fetcher({"NVDA": ["2026-10-30T16:30"]})                   # Friday after close
    cal = _calendar(tmp_path, fetch, _et("2026-11-02"))
    assert cal.block_reason("NVDA", _et("2026-11-02")) is not None    # Monday is the next trading day
    monkeypatch.delenv("EARNINGS_BLACKOUT_DAYS_BEFORE", raising=False)
    swing = EventCalendar.from_env(day_trading=False, cache_path=tmp_path / "e2.json")
    swing.fetch, swing.now_fn = fetch, lambda: _et("2026-10-29")
    assert swing.days_before == 1 and swing.block_reason("NVDA", _et("2026-10-29")) is not None


def test_event_windows_pause_every_symbol(tmp_path):
    cal = _calendar(tmp_path, Fetcher(), _et("2026-10-28"),
                    blackouts=parse_blackouts("2026-10-28 13:45-15:00"))
    assert cal.block_reason("AAPL", _et("2026-10-28", "14:00")).startswith("Event blackout (2026-10-28 13:45-15:00)")
    assert cal.block_reason("AAPL", _et("2026-10-28", "15:00")) is None
    assert cal.block_reason("AAPL", _et("2026-10-27", "14:00")) is None


def test_cached_for_12_hours_and_shared_through_the_file(tmp_path):
    fetch = Fetcher({"AAPL": ["2026-10-29T16:30"]})
    now = _et("2026-10-20")
    cal = _calendar(tmp_path, fetch, now)
    cal.earnings_dates("AAPL")
    cal.earnings_dates("AAPL")
    assert fetch.calls == ["AAPL"]
    other = _calendar(tmp_path, fetch, now + timedelta(hours=11))    # e.g. the live bot's process
    assert other.earnings_dates("AAPL") == [date(2026, 10, 29)] and fetch.calls == ["AAPL"]
    later = _calendar(tmp_path, fetch, now + timedelta(hours=13))
    later.earnings_dates("AAPL")
    assert fetch.calls == ["AAPL", "AAPL"]


def test_fetch_failure_fails_open_and_retries_hourly_keeping_known_dates(tmp_path):
    down = Fetcher(fail=True)
    now = _et("2026-10-29")
    cal = _calendar(tmp_path, down, now)
    assert cal.earnings_dates("AAPL") is None and cal.block_reason("AAPL", now) is None
    cal.block_reason("AAPL", now)
    assert down.calls == ["AAPL"]                                      # not hammering Yahoo
    cal.now_fn = lambda: now + timedelta(minutes=61)
    cal.earnings_dates("AAPL")
    assert down.calls == ["AAPL", "AAPL"]

    # A failed refresh keeps the dates we already had.
    good = _calendar(tmp_path / "b", Fetcher({"AAPL": ["2026-10-29T16:30"]}), now - timedelta(days=1))
    good.earnings_dates("AAPL")
    good.fetch, good.now_fn = Fetcher(fail=True), lambda: now
    assert good.block_reason("AAPL", now) is not None


def test_upcoming_lists_the_next_two_weeks(tmp_path):
    fetch = Fetcher({"AAPL": ["2026-07-30T16:30", "2026-10-29T16:30"], "MSFT": ["2026-10-22T16:05"],
                     "AMD": ["2026-12-01T16:05"]})
    cal = _calendar(tmp_path, fetch, _et("2026-10-20"))
    assert cal.upcoming(["AAPL", "MSFT", "AMD"]) == [{"symbol": "MSFT", "date": "2026-10-22"},
                                                     {"symbol": "AAPL", "date": "2026-10-29"}]


# ---------------------------------------------------------------------------
# Executor gate and API
# ---------------------------------------------------------------------------

@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, s: {"qty": "5", "market_value": "500"})
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: [])
    monkeypatch.setattr(AlpacaExecutor, "get_account", lambda self: {"equity": "100000", "last_equity": "100000",
                                                                    "buying_power": "100000"})
    placed = []
    monkeypatch.setattr(AlpacaExecutor, "cancel_open_orders", lambda self, s: 0)
    monkeypatch.setattr(AlpacaExecutor, "_place_order", lambda self, s, q, side, **k: placed.append(side) or
                        {"id": "x", "qty": str(q), "symbol": s})
    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order",
                        lambda self, s, q, stop, target, **k: placed.append("buy") or {"id": "b", "qty": str(q)})
    ex = AlpacaExecutor(risk_manager=RiskManager(RiskLimits(min_price=1.0)), price_lookup=lambda s: 100.0,
                        telemetry=EventLog(None))
    ex.placed = placed
    return ex


class Blocking:
    def block_reason(self, symbol, now=None):
        return f"Earnings for {symbol} today: no new entries"


def test_executor_skips_entries_but_not_exits_during_blackout(executor):
    executor.events = Blocking()
    buy = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 5})
    assert buy.skipped_reason.startswith("Earnings for AAPL")
    assert executor.telemetry.events[-1]["event"] == "entry_skipped_event"
    sell = executor.submit({"symbol": "AAPL", "recommendation": "SELL", "quantity": 5})
    assert sell.order is not None and executor.placed == ["sell"]


def test_a_broken_filter_never_blocks_trading(executor):
    class Broken:
        def block_reason(self, symbol, now=None):
            raise RuntimeError("bug")

    executor.events = Broken()
    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 5})
    assert result.order is not None and executor.placed == ["buy"]


def test_events_endpoint(client, monkeypatch):
    monkeypatch.setattr(control, "_upcoming_events", lambda symbols: {
        "enabled": True, "days_before": 0, "days_after": 1, "blackouts": [],
        "earnings": [{"symbol": s, "date": "2026-10-29"} for s in symbols]})
    body = client.get("/api/control/events?symbols=aapl,msft").json()
    assert [e["symbol"] for e in body["earnings"]] == ["AAPL", "MSFT"]
    assert client.get("/api/control/events?symbols=BAD;SYM").status_code == 400


def test_upcoming_events_reports_unknown_symbols_honestly(tmp_path, monkeypatch):
    from core import event_calendar

    monkeypatch.setattr(event_calendar, "yfinance_earnings", Fetcher(fail=True))
    body = control._upcoming_events(["AAPL"], root=tmp_path)
    assert body["earnings"] == [] and body["unavailable"] == ["AAPL"]
