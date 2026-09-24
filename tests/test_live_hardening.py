from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.alpaca_executor import AlpacaExecutor
from core.backtester import Backtester, BacktestConfig
from core.execution_telemetry import EventLog
from core.fill_quality import FillTracker, slippage_bps
from core.market_history import BarCache
from core.risk_manager import ExitFill, RiskLimits, RiskManager, last_exit_fill
from tests.test_backtester import DAY, _flat_bars, _strategy
from tests.test_operations import Broker, _bot

NOW = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _bracket(symbol, *, stop_filled_at=None, target_filled_at=None, entry_id="p1"):
    """An Alpaca nested bracket: filled parent buy with stop/limit legs."""
    return {
        "id": entry_id, "symbol": symbol, "side": "buy", "type": "market", "status": "filled",
        "filled_avg_price": "100.05", "filled_qty": "10", "filled_at": _iso(NOW - timedelta(hours=1)),
        "legs": [
            {"id": f"{entry_id}-stop", "symbol": symbol, "side": "sell", "type": "stop", "stop_price": "99.00",
             "status": "filled" if stop_filled_at else "canceled", "filled_avg_price": "98.90",
             "filled_qty": "10" if stop_filled_at else "0", "filled_at": _iso(stop_filled_at) if stop_filled_at else None},
            {"id": f"{entry_id}-tp", "symbol": symbol, "side": "sell", "type": "limit", "limit_price": "102.00",
             "status": "filled" if target_filled_at else "canceled", "filled_avg_price": "102.01",
             "filled_qty": "10" if target_filled_at else "0", "filled_at": _iso(target_filled_at) if target_filled_at else None},
        ],
    }


# ---------------------------------------------------------------------------
# Re-entry cooldown
# ---------------------------------------------------------------------------

def test_last_exit_fill_finds_latest_leg_and_kind() -> None:
    orders = [
        _bracket("AAPL", stop_filled_at=NOW - timedelta(minutes=50), entry_id="a"),
        _bracket("AAPL", target_filled_at=NOW - timedelta(minutes=5), entry_id="b"),
        _bracket("MSFT", stop_filled_at=NOW - timedelta(minutes=1), entry_id="c"),
    ]
    latest = last_exit_fill(orders, "AAPL")
    assert latest.filled_at == NOW - timedelta(minutes=5) and not latest.stopped_out
    assert latest.price == pytest.approx(102.01)
    assert last_exit_fill(orders, "MSFT").stopped_out
    assert last_exit_fill(orders, "NVDA") is None
    assert last_exit_fill([], "AAPL") is None


def _check(limits, last_exit, now=NOW):
    return RiskManager(limits).check_order(
        side="BUY", symbol="AAPL", quantity=10, price=100.0,
        account={"equity": "100000", "last_equity": "100000", "buying_power": "100000"},
        last_exit=last_exit, now=now,
    )


def test_cooldown_blocks_reentry_after_stop_out_only_within_window() -> None:
    limits = RiskLimits(min_price=1.0, max_position_pct=1.0, reentry_cooldown_minutes=30)
    stopped = ExitFill(NOW - timedelta(minutes=10), stopped_out=True)

    blocked = _check(limits, stopped)
    assert not blocked.approved and "Re-entry cooldown for AAPL: stop-out 10 min ago" in blocked.reason
    assert _check(limits, ExitFill(NOW - timedelta(minutes=31), True)).approved
    # Stops-only (default): a take-profit exit doesn't start a cooldown ...
    assert _check(limits, ExitFill(NOW - timedelta(minutes=1), False)).approved
    # ... unless configured to.
    assert not _check(replace(limits, reentry_cooldown_stops_only=False), ExitFill(NOW, False)).approved
    assert _check(replace(limits, reentry_cooldown_minutes=0), stopped).approved
    assert _check(limits, None).approved


def test_cooldown_env_settings(monkeypatch) -> None:
    monkeypatch.setenv("REENTRY_COOLDOWN_MINUTES", "45")
    monkeypatch.setenv("REENTRY_COOLDOWN_STOPS_ONLY", "false")
    limits = RiskLimits.from_env()
    assert limits.reentry_cooldown_minutes == 45 and limits.reentry_cooldown_stops_only is False


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, symbol: None)
    monkeypatch.setattr(AlpacaExecutor, "get_account", lambda self: {
        "equity": "100000", "last_equity": "100000", "buying_power": "100000"})
    ex = AlpacaExecutor(
        risk_manager=RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0, reentry_cooldown_minutes=30)),
        price_lookup=lambda symbol: 100.0,
        telemetry=EventLog(None),
    )
    ex.placed = []
    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order",
                        lambda self, symbol, qty, stop, target: self.placed.append(qty) or
                        {"id": f"o{len(self.placed)}", "qty": str(qty), "symbol": symbol})
    return ex


def test_executor_blocks_reentry_after_broker_stop_out(monkeypatch, executor) -> None:
    now = datetime.now(timezone.utc)
    orders = [_bracket("AAPL", stop_filled_at=now - timedelta(minutes=5))]
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: orders)

    result = executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 10})
    assert result.order is None and "Re-entry cooldown" in result.skipped_reason
    assert executor.placed == []

    orders[:] = [_bracket("AAPL", stop_filled_at=now - timedelta(minutes=45))]
    assert executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 10}).order["id"] == "o1"


def test_executor_records_expected_price_for_fills(monkeypatch, executor) -> None:
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: [])
    executor.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 10})
    assert executor.expected_prices == {"o1": 100.0}
    assert executor.telemetry.events[-1]["expected_price"] == 100.0


def test_backtester_applies_cooldown_after_stop_out() -> None:
    bars = _flat_bars()
    bars.loc[bars.index[32], ["open", "high", "low", "close"]] = [100.0, 100.2, 98.5, 99.0]
    schedule = {bars.index[30]: 0.9, bars.index[32]: 0.9}

    def run(cooldown_minutes):
        limits = RiskLimits(min_price=1.0, max_position_pct=1.0, reentry_cooldown_minutes=cooldown_minutes)
        config = BacktestConfig(initial_capital=10_000, slippage_bps=0, bar_length=DAY)
        return Backtester(_strategy(schedule), RiskManager(limits), config).run({"AAA": bars})

    no_cooldown, cooldown = run(0), run(30)
    assert no_cooldown.trades[0].exit_reason == "stop"
    assert len(no_cooldown.trades) == 2 and no_cooldown.trades[1].entry_time == bars.index[33]
    assert len(cooldown.trades) == 1
    assert cooldown.blocked == {"Re-entry cooldown": 1}


# ---------------------------------------------------------------------------
# Fill quality
# ---------------------------------------------------------------------------

def test_slippage_sign_convention() -> None:
    assert slippage_bps("buy", 100.10, 100.0) == pytest.approx(10.0)    # paid up: adverse
    assert slippage_bps("sell", 99.90, 100.0) == pytest.approx(10.0)    # sold lower: adverse
    assert slippage_bps("sell", 102.01, 102.0) == pytest.approx(-0.98)  # price improvement
    assert slippage_bps("buy", 100.0, 0) is None


def test_fill_tracker_measures_entries_stops_and_targets(tmp_path) -> None:
    telemetry = EventLog(None)
    tracker = FillTracker({"p1": 100.0}, state_path=str(tmp_path / "fills.json"), telemetry=telemetry, alert_bps=20)
    orders = [_bracket("AAPL", stop_filled_at=NOW)]

    new = tracker.update(orders, date(2026, 3, 2))
    by_id = {f.order_id: f for f in new}
    assert set(by_id) == {"p1", "p1-stop"}                         # cancelled target leg ignored
    assert by_id["p1"].reference == "quote" and by_id["p1"].slippage_bps == pytest.approx(5.0)
    assert by_id["p1-stop"].reference == "stop" and by_id["p1-stop"].slippage_bps == pytest.approx(10.1, abs=0.01)
    assert tracker.update(orders, date(2026, 3, 2)) == []            # no double counting

    summary = tracker.summary()
    assert summary["fills"] == 2 and summary["measured"] == 2
    assert summary["worst"] == "sell AAPL (stop)"
    assert set(summary["by_reference"]) == {"quote", "stop"}
    assert summary["cost_usd"] == pytest.approx(10 * 100.05 * 5 / 10_000 + 10 * 98.90 * 10.1 / 10_000, abs=0.01)
    assert [e["event"] for e in telemetry.events] == ["order_filled", "order_filled"]


def test_fill_tracker_flags_adverse_fills(tmp_path) -> None:
    telemetry = EventLog(None)
    tracker = FillTracker({}, state_path=str(tmp_path / "f.json"), telemetry=telemetry, alert_bps=10)
    tracker.update([_bracket("AAPL", stop_filled_at=NOW)], date(2026, 3, 2))
    stop_event = [e for e in telemetry.events if e["order_id"] == "p1-stop"][0]
    parent_event = [e for e in telemetry.events if e["order_id"] == "p1"][0]
    assert stop_event["adverse"] is True
    assert parent_event["reference"] == "none" and parent_event["slippage_bps"] is None


def test_fill_tracker_survives_restart_and_rolls_daily(tmp_path) -> None:
    path = str(tmp_path / "fills.json")
    first = FillTracker({"p1": 100.0, "p2": 50.0}, state_path=path, telemetry=EventLog(None))
    first.update([_bracket("AAPL")], date(2026, 3, 2))

    restarted = FillTracker({}, state_path=path, telemetry=EventLog(None))
    assert restarted.update([_bracket("AAPL")], date(2026, 3, 2)) == []   # already seen
    later = restarted.update([{**_bracket("MSFT", entry_id="p2"), "filled_avg_price": "50.10"}], date(2026, 3, 2))
    assert later[0].reference == "quote" and later[0].slippage_bps == pytest.approx(20.0)  # expected price persisted
    assert restarted.summary()["fills"] == 2

    restarted.update([], date(2026, 3, 3))
    assert restarted.summary() == {"fills": 0, "measured": 0, "by_reference": {}}
    state = json.loads((tmp_path / "fills.json").read_text())
    assert state["day"] == "2026-03-02"  # nothing new on the 3rd, nothing rewritten


def test_session_report_includes_fill_quality(portfolio_manager, tmp_path) -> None:
    class FillBroker(Broker):
        def get_orders_today(self):
            return [_bracket("AAPL", target_filled_at=NOW)]

    tracker = FillTracker({"p1": 100.0}, state_path=str(tmp_path / "f.json"), telemetry=EventLog(None))
    state = {"time": "15:00", "open": True}
    bot = _bot(portfolio_manager, FillBroker(), state, fill_tracker=tracker)

    bot.run_cycle()
    assert len(tracker.records) == 2  # entry + take-profit leg measured during the session
    state.update(time="16:05", open=False)
    report = bot.run_cycle().session_report
    assert report["fills"] == 2
    quality = report["fill_quality"]
    assert quality["measured"] == 2 and set(quality["by_reference"]) == {"quote", "limit"}


def test_fill_tracking_failure_never_breaks_the_cycle(portfolio_manager) -> None:
    class Exploding:
        records = []

        def update(self, orders, session_date):
            raise RuntimeError("boom")

    bot = _bot(portfolio_manager, Broker(), {"time": "15:00", "open": True}, fill_tracker=Exploding())
    report = bot.run_cycle()
    assert report.broker_error is None and not report.errors


# ---------------------------------------------------------------------------
# Incremental bar cache
# ---------------------------------------------------------------------------

class FakeSource:
    """5-minute bars from 09:30 ET on 2026-03-02, served by time range."""

    def __init__(self):
        idx = pd.date_range("2026-02-20 14:30", "2026-03-03 21:00", freq="5min", tz="UTC")
        self.frame = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": np.arange(len(idx), dtype=float), "volume": 1.0},
            index=idx,
        )
        self.calls = []

    def __call__(self, symbol, timeframe, start, end):
        self.calls.append((symbol, pd.Timestamp(start), pd.Timestamp(end)))
        return self.frame[(self.frame.index >= pd.Timestamp(start)) & (self.frame.index <= pd.Timestamp(end))]


def test_bar_cache_fetches_incrementally_and_matches_full_fetch() -> None:
    source = FakeSource()
    clock = {"now": pd.Timestamp("2026-03-02 15:02", tz="UTC").to_pydatetime()}
    cache = BarCache("5Min", 5, fetch=source, overlap_bars=3, now_fn=lambda: clock["now"])

    first = cache.get("AAPL")
    assert first.index[-1] == pd.Timestamp("2026-03-02 14:55", tz="UTC")  # 15:00 bar still forming
    assert cache.stats == {"full": 1, "incremental": 0, "failed_incremental": 0}

    source.frame.loc[pd.Timestamp("2026-03-02 15:00", tz="UTC"), "close"] = -1.0  # revised after first read
    clock["now"] = pd.Timestamp("2026-03-02 15:12", tz="UTC").to_pydatetime()
    second = cache.get("AAPL")
    _, start, _ = source.calls[-1]
    assert start == pd.Timestamp("2026-03-02 14:45", tz="UTC")  # newest cached bar minus 3 bars
    assert cache.stats["incremental"] == 1
    assert second.index[-1] == pd.Timestamp("2026-03-02 15:05", tz="UTC")
    assert second.loc[pd.Timestamp("2026-03-02 15:00", tz="UTC"), "close"] == -1.0

    # Identical to a fresh full fetch of the same window.
    fresh = BarCache("5Min", 5, fetch=source, now_fn=lambda: clock["now"])
    pd.testing.assert_frame_equal(second, fresh.get("AAPL"))
    assert second.index[0] >= clock["now"] - timedelta(days=5)


def test_bar_cache_full_reload_on_new_day_and_after_ttl() -> None:
    source = FakeSource()
    clock = {"now": pd.Timestamp("2026-03-02 15:00", tz="UTC").to_pydatetime()}
    cache = BarCache("5Min", 5, fetch=source, full_refresh_hours=6, now_fn=lambda: clock["now"])
    cache.get("AAPL")
    clock["now"] = pd.Timestamp("2026-03-02 20:00", tz="UTC").to_pydatetime()
    cache.get("AAPL")
    assert cache.stats["full"] == 1
    clock["now"] = pd.Timestamp("2026-03-02 21:05", tz="UTC").to_pydatetime()  # >6h since the full load
    cache.get("AAPL")
    assert cache.stats["full"] == 2
    clock["now"] = pd.Timestamp("2026-03-03 14:35", tz="UTC").to_pydatetime()  # next market day
    cache.get("AAPL")
    assert cache.stats["full"] == 3


def test_bar_cache_keeps_serving_when_fetch_fails() -> None:
    source = FakeSource()
    clock = {"now": pd.Timestamp("2026-03-02 15:00", tz="UTC").to_pydatetime()}
    cache = BarCache("5Min", 5, fetch=source, now_fn=lambda: clock["now"])
    before = cache.get("AAPL")

    cache.fetch = lambda *a: pd.DataFrame()
    clock["now"] = pd.Timestamp("2026-03-02 15:10", tz="UTC").to_pydatetime()
    after = cache.get("AAPL")
    assert cache.stats["failed_incremental"] == 1
    assert len(after) >= len(before)  # stale but usable; freshness is checked by the caller's bar time
    assert BarCache("5Min", 5, fetch=lambda *a: pd.DataFrame(), now_fn=lambda: clock["now"]).get("X").empty


def test_signal_engine_uses_bar_cache_by_default(monkeypatch, portfolio_manager) -> None:
    from core.ml_signal_engine import MLSignalEngine

    engine = MLSignalEngine(portfolio_manager, timeframe="5m", sentiment_loader=lambda s: None)
    assert engine._bar_cache is not None and engine.history_loader == engine._bar_cache.get
    monkeypatch.setenv("BAR_CACHE", "false")
    assert MLSignalEngine(portfolio_manager, timeframe="5m")._bar_cache is None


def test_signal_engine_refuses_stale_intraday_bars(monkeypatch, portfolio_manager) -> None:
    from core.ml_signal_engine import MLSignalEngine
    from core.ml_training import synthetic_intraday_bars

    monkeypatch.setenv("MAX_BAR_AGE_BARS", "3")
    bars = synthetic_intraday_bars(days=5, seed=1)
    last_close = bars.index[-1] + timedelta(minutes=5)

    def engine(now):
        return MLSignalEngine(portfolio_manager, timeframe="5m", history_loader=lambda s: bars,
                              sentiment_loader=lambda s: None, now_fn=lambda: now)

    # Mid-session with data 20 minutes old: no signal.
    stale_now = pd.Timestamp(last_close.tz_convert("America/New_York").date().isoformat() + " 12:00",
                             tz="America/New_York")
    stale_bars = bars[bars.index < stale_now.tz_convert("UTC") - timedelta(minutes=25)]
    stale = MLSignalEngine(portfolio_manager, timeframe="5m", history_loader=lambda s: stale_bars,
                           sentiment_loader=lambda s: None, now_fn=lambda: stale_now.to_pydatetime())(
        "AAPL", {}, "default")
    assert stale["error"] and stale["error_type"] == "stale_data" and "no signal on stale data" in stale["message"]

    # Fresh data mid-session, and any data after the close, are fine.
    fresh_now = stale_bars.index[-1] + timedelta(minutes=7)
    assert engine(fresh_now.to_pydatetime())._stale_reason(stale_bars) is None
    assert engine((last_close + timedelta(hours=3)).to_pydatetime())._stale_reason(bars) is None


def test_fill_alerts_only_for_adverse_fills_and_session_slippage() -> None:
    from core.alerts import format_alert

    base = {"event": "order_filled", "symbol": "AAPL", "side": "sell", "fill_price": 98.9,
            "reference": "stop", "reference_price": 99.0, "slippage_bps": 10.1}
    assert format_alert({**base, "adverse": False}) is None
    assert "+10.1 bps slippage" in format_alert({**base, "adverse": True})
    report = {"event": "session_report", "date": "2026-03-02", "pnl": 10.0, "pnl_pct": 0.1, "equity": 1e4,
              "fills": 2, "open_positions": 0, "fill_quality": {"measured": 2, "mean_bps": 3.25}}
    assert "slippage +3.2 bps avg" in format_alert(report)
