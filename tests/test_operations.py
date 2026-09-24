from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.alerts import AlertSink, format_alert
from core.alpaca_executor import BrokerSnapshot
from core.backtester import Backtester, BacktestConfig, Trade, attribution
from core.execution_telemetry import EventLog
from core.ml_strategy import MLStrategy
from core.ml_training import synthetic_bars
from core.risk_manager import RiskLimits, RiskManager
from core.session_clock import SessionClock, SessionConfig
from core.trading_bot import TradingBot
from core.trading_workflow import WorkflowResult


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Poster:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def __call__(self, url, json, timeout):
        if self.fail:
            raise ConnectionError("webhook down")
        self.sent.append(json)
        return type("R", (), {"status_code": 204})()


def _sink(poster, **kw):
    now = {"t": 0.0}
    sink = AlertSink("https://hooks.example/x", post=poster, asynchronous=False, clock=lambda: now["t"], **kw)
    return sink, now


def test_alert_formats_cover_key_events() -> None:
    assert "BUY 10 AAPL submitted (stop 95.0, target 110.0)" in format_alert(
        {"event": "order_submitted", "side": "buy", "qty": 10, "symbol": "AAPL", "stop_loss": 95.0, "take_profit": 110.0})
    assert "[buying_power]" in format_alert(
        {"event": "order_rejected", "symbol": "AAPL", "side": "BUY", "category": "buying_power", "message": "x"})
    flatten = format_alert({"event": "flatten", "cancelled_orders": 2, "closed": [{"symbol": "AAPL"}],
                            "failures": [{"symbol": "MSFT", "category": "margin_or_pdt"}]})
    assert "closed AAPL" in flatten and "FAILED: MSFT (margin_or_pdt)" in flatten
    assert format_alert({"event": "reconciliation", "discrepancies": []}) is None
    assert "unprotected AAPL" in format_alert(
        {"event": "reconciliation", "mode": "sync", "discrepancies": [{"kind": "unprotected", "symbol": "AAPL"}]})
    report = format_alert({"event": "session_report", "date": "2026-03-02", "pnl": -120.5, "pnl_pct": -0.12,
                           "equity": 99879.5, "fills": 6, "open_positions": 1, "no_overnight": True})
    assert "P&L -120.50" in report and "NOT FLAT" in report
    assert format_alert({"event": "order_retry"}) is None


def test_alert_sink_sends_payload_for_discord_and_slack() -> None:
    poster = Poster()
    sink, _ = _sink(poster)

    assert sink.notify({"event": "breaker_tripped", "reason": "Intraday drawdown breaker tripped"})
    assert poster.sent == [{"content": "🛑 Intraday drawdown breaker tripped",
                            "text": "🛑 Intraday drawdown breaker tripped"}]


def test_alert_sink_filters_throttles_and_caps() -> None:
    poster = Poster()
    sink, now = _sink(poster, events=["order_rejected"], min_interval_seconds=60, max_per_hour=3)
    rejected = {"event": "order_rejected", "symbol": "AAPL", "category": "buying_power", "message": "m"}

    assert not sink.notify({"event": "bot_started"})             # not selected
    assert sink.notify(rejected)
    assert not sink.notify(rejected)                               # same key within 60s
    now["t"] = 61
    assert sink.notify(rejected)
    assert sink.notify({**rejected, "symbol": "MSFT"})             # different key
    assert not sink.notify({**rejected, "symbol": "NVDA"})         # hourly cap (3) reached
    now["t"] = 3700
    assert sink.notify({**rejected, "symbol": "NVDA"})             # window rolled


def test_alert_failures_never_raise_and_env_config(monkeypatch) -> None:
    sink, _ = _sink(Poster(fail=True))
    assert sink.notify({"event": "bot_stopped", "reason": "x"})    # swallowed

    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    assert AlertSink.from_env() is None
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example/y")
    monkeypatch.setenv("ALERT_EVENTS", "flatten, session_report")
    assert AlertSink.from_env().events == {"flatten", "session_report"}


def test_event_log_offers_every_event_to_alerts(tmp_path) -> None:
    poster = Poster()
    sink, _ = _sink(poster)
    log = EventLog(str(tmp_path / "e.jsonl"), alerts=sink)

    log.record("bot_started", mode="DRY RUN", timeframe="5Min", symbols="AAPL")
    log.record("order_retry", symbol="AAPL")   # recorded, not alerted

    assert len((tmp_path / "e.jsonl").read_text().splitlines()) == 2
    assert [m["text"] for m in poster.sent] == ["▶️ Bot started: DRY RUN, 5Min, AAPL"]


def test_async_alerts_do_not_block_the_caller() -> None:
    release = threading.Event()

    def slow_post(url, json, timeout):
        release.wait(2)
        return type("R", (), {"status_code": 204})()

    sink = AlertSink("https://hooks.example/x", post=slow_post)
    started = time.monotonic()
    sink.notify({"event": "bot_stopped", "reason": "x"})
    assert time.monotonic() - started < 0.5
    release.set()


# ---------------------------------------------------------------------------
# Bot: session report, breaker alert, heartbeat, graceful stop
# ---------------------------------------------------------------------------

class Workflow:
    def __init__(self, pm, fail=False):
        self.portfolio_manager = pm
        self.fail = fail

    def run(self, symbol, portfolio_name, *, record_paper_trade, submit_alpaca_paper_order, entry_block_reason=None):
        if self.fail:
            raise RuntimeError("feature pipeline exploded")
        return WorkflowResult(symbol=symbol, portfolio_name=portfolio_name, analysis={"recommendation": "HOLD"})


class Broker:
    def __init__(self, positions=(), equity="10100", last_equity="10000"):
        self.positions = list(positions)
        self.account = {"equity": equity, "last_equity": last_equity}
        self.risk_manager = RiskManager(RiskLimits(max_intraday_drawdown_pct=2.0))

    def get_snapshot(self):
        return BrokerSnapshot(positions=self.positions, open_orders=[])

    def get_account(self):
        return self.account

    def get_positions(self):
        return self.positions

    def get_orders_today(self):
        return [{"status": "filled", "filled_qty": "5"}, {"status": "canceled", "filled_qty": "0"},
                {"status": "filled", "filled_qty": "5"}]


def _clock_at(state):
    def clock():
        ts = pd.Timestamp(f"2026-03-02 {state['time']}", tz="America/New_York")
        return {"is_open": state["open"], "timestamp": ts.isoformat(), "next_close": "2026-03-02T16:00:00-05:00"}
    return clock


def _bot(pm, broker, state, **kw):
    return TradingBot(
        kw.pop("workflow", Workflow(pm)), ["AAPL"], execute=True, broker=broker,
        market_clock=_clock_at(state), session_clock=SessionClock(SessionConfig()),
        reconcile_fn=lambda *a, **k: type("R", (), {"discrepancies": [], "positions": 0, "open_orders": 0})(),
        now_fn=lambda: pd.Timestamp(f"2026-03-02 {state['time']}", tz="America/New_York").to_pydatetime(),
        telemetry=EventLog(None), **kw,
    )


def test_session_report_once_after_the_close(portfolio_manager) -> None:
    state = {"time": "15:00", "open": True}
    bot = _bot(portfolio_manager, Broker(), state)

    assert bot.run_cycle().session_report is None
    state.update(time="16:05", open=False)
    report = bot.run_cycle().session_report
    assert report == {"date": "2026-03-02", "equity": 10100.0, "pnl": 100.0, "pnl_pct": 1.0, "fills": 2,
                      "open_positions": 0, "open_symbols": [], "no_overnight": True}
    assert bot.run_cycle().session_report is None                       # not repeated
    assert [e["event"] for e in bot.telemetry.events].count("session_report") == 1


def test_session_report_warns_when_not_flat(portfolio_manager) -> None:
    state = {"time": "15:00", "open": True}
    bot = _bot(portfolio_manager, Broker(positions=[{"symbol": "AAPL", "qty": "3"}]), state)
    bot.run_cycle()
    state.update(time="16:05", open=False)

    bot.run_cycle()

    event = [e for e in bot.telemetry.events if e["event"] == "session_report"][0]
    assert event["open_symbols"] == ["AAPL"] and "NOT FLAT" in format_alert(event)


def test_no_session_report_without_a_session(portfolio_manager) -> None:
    bot = _bot(portfolio_manager, Broker(), {"time": "18:00", "open": False})
    assert bot.run_cycle().session_report is None


def test_breaker_alert_fires_once_per_session(portfolio_manager) -> None:
    state = {"time": "11:00", "open": True}
    bot = _bot(portfolio_manager, Broker(equity="9700"), state)

    bot.run_cycle()
    bot.run_cycle()

    assert [e["event"] for e in bot.telemetry.events].count("breaker_tripped") == 1


def test_heartbeat_is_written_each_cycle(portfolio_manager, tmp_path) -> None:
    state = {"time": "11:00", "open": True}
    path = tmp_path / "hb.json"
    bot = _bot(portfolio_manager, Broker(positions=[{"symbol": "AAPL", "qty": "1"}]), state,
               heartbeat_path=str(path), timeframe="5Min")

    bot.run_cycle()

    beat = json.loads(path.read_text())
    assert beat["phase"] == "OPEN" and beat["positions"] == 1 and beat["timeframe"] == "5Min"
    assert beat["execute"] is True and beat["symbols"] == ["AAPL"]


def test_stop_interrupts_the_sleep_between_cycles(portfolio_manager) -> None:
    state = {"time": "11:00", "open": True}
    bot = _bot(portfolio_manager, Broker(), state)
    bot.interval_seconds = 3600
    threading.Timer(0.3, bot.stop, args=("SIGTERM",)).start()

    started = time.monotonic()
    reports = bot.run()

    assert time.monotonic() - started < 5 and len(reports) == 1
    events = [e["event"] for e in bot.telemetry.events]
    assert events[0] == "bot_started" and events[-1] == "bot_stopped"
    assert bot.telemetry.events[-1]["reason"] == "SIGTERM"


def test_a_failing_cycle_is_reported_but_does_not_kill_the_bot(portfolio_manager) -> None:
    state = {"time": "11:00", "open": True}
    bot = _bot(portfolio_manager, Broker(), state, workflow=Workflow(portfolio_manager, fail=True),
               sleep=lambda s: None)
    bot.run_cycle = lambda: (_ for _ in ()).throw(RuntimeError("boom"))

    bot.run(max_cycles=3)

    events = [e["event"] for e in bot.telemetry.events]
    assert events.count("bot_error") == 3 and events[-1] == "bot_stopped"


# ---------------------------------------------------------------------------
# Attribution and rolling walk-forward
# ---------------------------------------------------------------------------

def _trade(regime, pnl_per_share, hour=10):
    entry = pd.Timestamp(f"2026-03-02 {hour}:00", tz="America/New_York").tz_convert("UTC")
    t = Trade(symbol="AAA", entry_time=entry, entry_price=100.0, quantity=10, stop=99.0, target=102.0,
              probability_up=0.7, regime=regime)
    t.exit_time, t.exit_price = entry + pd.Timedelta(hours=1), 100.0 + pnl_per_share
    return t


def test_attribution_by_regime_and_hour() -> None:
    trades = [_trade("TRENDING_BULL", 2), _trade("TRENDING_BULL", -1, hour=14), _trade("CHOPPY", -1), _trade(None, 1)]

    by_regime = attribution(trades, "regime").set_index("regime")
    by_hour = attribution(trades, "entry_hour_et").set_index("entry_hour_et")

    assert by_regime.loc["TRENDING_BULL", "trades"] == 2
    assert by_regime.loc["TRENDING_BULL", "win_rate_pct"] == 50.0
    assert by_regime.loc["TRENDING_BULL", "avg_r"] == pytest.approx(0.5)
    assert by_regime.loc["CHOPPY", "total_pnl"] == -10.0
    assert by_regime.loc["n/a", "trades"] == 1
    assert by_regime["share_of_pnl_pct"].sum() == pytest.approx(100.0)
    assert by_hour.loc[10, "trades"] == 3 and by_hour.loc[14, "trades"] == 1
    assert attribution([], "regime").empty


def test_backtester_end_bound_limits_the_window() -> None:
    bars = synthetic_bars(n=600, seed=4)
    strategy = MLStrategy(artifact=None, model_path="/none", confidence_threshold=0.55, regime_policy="off")
    manager = RiskManager(RiskLimits(min_price=1.0, max_daily_trades=10))
    start, end = bars.index[300], bars.index[450]

    result = Backtester(strategy, manager, BacktestConfig(slippage_bps=0)).run({"X": bars}, start=start, end=end)

    assert result.equity.index.min() >= start and result.equity.index.max() < end
    assert all(start <= t.entry_time < end and t.exit_time < end for t in result.trades)


def test_fold_windows_are_contiguous_and_cover_the_test_period() -> None:
    from scripts.backtest import fold_windows

    timeline = list(pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"))
    windows = fold_windows(timeline, 0.6, 4)

    assert windows[0][0] == timeline[60]
    assert [w[1] for w in windows[:-1]] == [w[0] for w in windows[1:]]
    assert windows[-1][1] is None and len(windows) == 4
    with pytest.raises(ValueError):
        fold_windows(timeline[:3], 0.6, 5)


def test_backtest_cli_rolling_folds(tmp_path: Path) -> None:
    from scripts import backtest

    out = tmp_path / "wf"
    code = backtest.main(["--synthetic", "--timeframe", "5m", "--days", "60", "--symbols", "AAA,BBB",
                          "--threshold", "0.55", "--folds", "3", "--out", str(out)])

    assert code == 0
    folds = pd.read_csv(out / "folds.csv")
    assert folds["fold"].tolist() == ["fold 1/3", "fold 2/3", "fold 3/3"]
    assert (out / "fold_1" / "trades.csv").exists() and (out / "by_regime.csv").exists()
    pooled = json.loads((out / "summary.json").read_text())["pooled"]
    assert pooled["folds"] == 3 and pooled["trades"] == int(folds["trades"].sum())
