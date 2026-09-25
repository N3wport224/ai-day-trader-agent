"""Losing-streak lockout, trade_updates stream ingestion/fallback, and backtester alignment."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.alpaca_executor import BrokerSnapshot
from core.backtester import BacktestConfig, Trade, _chase_fill, _loss_streak
from core.execution_telemetry import EventLog
from core.risk_manager import LossStreak, RiskLimits, RiskManager, loss_streak
from core.session_clock import SessionClock, SessionConfig
from core.stream_listener import TradeUpdateStream, classify_fill, stream_url
from core.trading_bot import TradingBot
from tests.test_intraday_trading import RecordingWorkflow, _clock, _day_bars, _et, _intraday_backtest

T0 = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)  # 10:00 ET


def _fill(side, price, minute, qty=10, symbol="AAPL", kind="market", legs=None):
    return {"symbol": symbol, "side": side, "type": kind, "status": "filled", "filled_qty": str(qty),
            "filled_avg_price": str(price), "filled_at": (T0 + timedelta(minutes=minute)).isoformat(),
            "legs": legs or []}


def _round_trip(entry, exit_, start, symbol="AAPL", stop=True):
    """A bracket entry whose stop (or target) leg filled, nested as Alpaca returns it."""
    leg = _fill("sell", exit_, start + 10, symbol=symbol, kind="stop" if stop else "limit")
    return _fill("buy", entry, start, symbol=symbol, legs=[leg])


# ---------------------------------------------------------------------------
# Streak computation and lockout
# ---------------------------------------------------------------------------

def test_consecutive_stop_outs_counted_from_nested_bracket_legs():
    orders = [_round_trip(100, 98, 0), _round_trip(50, 49, 20, symbol="MSFT")]
    streak = loss_streak(orders)
    assert streak.count == 2
    assert streak.last_loss_at == T0 + timedelta(minutes=30)
    assert streak.losses == (("MSFT", -10.0), ("AAPL", -20.0))


def test_winning_close_resets_and_breakeven_is_neutral():
    orders = [_round_trip(100, 98, 0), _round_trip(100, 103, 20, stop=False), _round_trip(100, 99, 40)]
    assert loss_streak(orders).count == 1
    flat = [_round_trip(100, 98, 0), _round_trip(100, 100, 20), _round_trip(100, 99, 40)]
    assert loss_streak(flat).count == 2


def test_partially_filled_then_cancelled_entry_still_prices_the_exit():
    partial = {"symbol": "AAPL", "side": "buy", "type": "limit", "status": "canceled", "filled_qty": "4",
               "filled_avg_price": "100", "filled_at": None, "updated_at": T0.isoformat(),
               "legs": [_fill("sell", 98, 10, qty=4, kind="stop")]}
    assert loss_streak([partial, _round_trip(100, 99, 20)]).count == 2


def test_positions_bought_on_earlier_days_do_not_count():
    assert loss_streak([_fill("sell", 90, 5)]).count == 0


def _limits(**kw):
    return RiskLimits(min_price=1.0, max_position_pct=1.0, max_portfolio_heat_pct=0, margin_buffer_pct=0,
                      reentry_cooldown_minutes=0, **kw)


def test_lockout_blocks_entries_for_cooldown_then_expires():
    manager = RiskManager(_limits(max_consecutive_losses=2, streak_cooldown_minutes=45))
    streak = LossStreak(2, T0, (("AAPL", -20.0), ("MSFT", -10.0)))
    reason = manager.streak_block(streak, T0 + timedelta(minutes=44))
    assert reason and "paused until 10:45 ET" in reason
    assert manager.streak_block(streak, T0 + timedelta(minutes=45)) is None
    assert manager.streak_block(LossStreak(1, T0), T0) is None
    assert RiskManager(_limits(max_consecutive_losses=0)).streak_block(streak, T0) is None


def test_check_order_enforces_the_streak_on_buys_but_never_on_exits():
    manager = RiskManager(_limits())
    account = {"equity": "100000", "last_equity": "100000", "buying_power": "100000"}
    orders = [_round_trip(100, 98, 0), _round_trip(100, 98, 20)]
    now = T0 + timedelta(minutes=40)
    buy = manager.check_order(side="BUY", symbol="NVDA", quantity=10, price=100.0, account=account,
                              orders_today=orders, stop_loss=98.0, take_profit=104.0, now=now)
    assert not buy.approved and buy.reason.startswith("Loss streak: 2 losing trades")
    sell = manager.check_order(side="SELL", symbol="NVDA", quantity=5, price=100.0, account=account,
                               position={"qty": "5"}, orders_today=orders, now=now)
    assert sell.approved
    later = manager.check_order(side="BUY", symbol="NVDA", quantity=10, price=100.0, account=account,
                                orders_today=orders, stop_loss=98.0, take_profit=104.0,
                                now=T0 + timedelta(minutes=80))
    assert later.approved


def test_streak_limits_from_environment(monkeypatch):
    monkeypatch.setenv("MAX_CONSECUTIVE_LOSSES", "3")
    monkeypatch.setenv("COOLDOWN_MINUTES", "30")
    limits = RiskLimits.from_env()
    assert (limits.max_consecutive_losses, limits.streak_cooldown_minutes) == (3, 30.0)
    monkeypatch.setenv("STREAK_COOLDOWN_MINUTES", "60")  # the specific name wins
    assert RiskLimits.from_env().streak_cooldown_minutes == 60.0


class StreakBroker:
    def __init__(self, orders):
        self.orders = orders
        self.risk_manager = RiskManager(_limits())
        self.snapshots = 0

    def get_snapshot(self):
        self.snapshots += 1
        return BrokerSnapshot(positions=[], open_orders=[])

    def get_account(self):
        return {"equity": "100000", "last_equity": "100000", "buying_power": "100000"}

    def get_orders_today(self):
        return self.orders


def _bot(pm, broker, now, **kw):
    return TradingBot(
        RecordingWorkflow(pm), ["AAPL"], execute=True, broker=broker,
        market_clock=lambda: _clock("10:40"),
        reconcile_fn=lambda *a, **k: type("R", (), {"discrepancies": [], "positions": 0, "open_orders": 0})(),
        session_clock=SessionClock(SessionConfig()), telemetry=EventLog(None), now_fn=lambda: now, **kw,
    )


def test_bot_blocks_entries_and_alerts_once_per_lockout(portfolio_manager):
    broker = StreakBroker([_round_trip(100, 98, 0), _round_trip(100, 98, 20)])
    bot = _bot(portfolio_manager, broker, T0 + timedelta(minutes=40))

    first = bot.run_cycle()
    bot.run_cycle()

    assert first.entry_block.startswith("Loss streak")
    assert bot.workflow.blocks[0].startswith("Loss streak")
    alerts = [e for e in bot.telemetry.events if e["event"] == "streak_lockout_active"]
    assert len(alerts) == 1 and alerts[0]["losses"] == 2 and alerts[0]["until_et"] == "11:15"  # last loss 10:30 + 45

    # A third loss after the pause starts a new lockout and a new alert.
    broker.orders.append(_round_trip(100, 98, 80))
    bot.now_fn = lambda: T0 + timedelta(minutes=95)
    assert bot.run_cycle().entry_block.startswith("Loss streak: 3")
    assert len([e for e in bot.telemetry.events if e["event"] == "streak_lockout_active"]) == 2


def test_streak_alert_is_formatted_for_phone_alerts():
    from core.alerts import DEFAULT_EVENTS, format_alert

    assert "streak_lockout_active" in DEFAULT_EVENTS and "unprotected_position" in DEFAULT_EVENTS
    assert "slippage_timeout" not in DEFAULT_EVENTS   # logged and shown on the dashboard, not pushed
    text = format_alert({"event": "streak_lockout_active", "losses": 2, "symbols": ["AAPL", "MSFT"],
                         "until_et": "10:45"})
    assert "2 losing trades in a row" in text and "10:45" in text


# ---------------------------------------------------------------------------
# trade_updates stream
# ---------------------------------------------------------------------------

def test_stream_url_and_fill_classification():
    assert stream_url("https://paper-api.alpaca.markets/v2") == "wss://paper-api.alpaca.markets/stream"
    assert stream_url("https://api.alpaca.markets") == "wss://api.alpaca.markets/stream"
    assert classify_fill({"side": "buy", "type": "limit"}) == "entry_fill"
    assert classify_fill({"side": "sell", "type": "stop", "order_class": "bracket"}) == "stop_hit"
    assert classify_fill({"side": "sell", "type": "limit", "order_class": "bracket"}) == "target_hit"
    assert classify_fill({"side": "sell", "type": "market"}) == "exit_fill"


def _update(event="fill", **order):
    order = {"id": "leg-1", "symbol": "AAPL", "side": "sell", "type": "stop", "order_class": "bracket",
             "status": "filled", **order}
    return {"event": event, "price": "97.95", "qty": "10", "position_qty": "0",
            "timestamp": "2026-03-02T15:10:00Z", "order": order}


def test_fill_event_is_cached_logged_and_broadcast():
    stream = TradeUpdateStream("wss://x/stream", "k", "s", telemetry=EventLog(None))
    seen = []
    stream.listeners.append(seen.append)
    stream.listeners.append(lambda u: 1 / 0)  # a broken listener must not break the stream

    stream.handle(_update())

    assert stream.order_update("leg-1")["status"] == "filled"
    fill = [e for e in stream.telemetry.events if e["event"] == "stream_fill"][0]
    assert fill["kind"] == "stop_hit" and fill["price"] == 97.95 and fill["position_qty"] == 0
    assert seen[0]["kind"] == "stop_hit"


class FakeSocket:
    def __init__(self, frames, auth_ok=True):
        self.sent = []
        reply = {"stream": "authorization", "data": {"status": "authorized" if auth_ok else "unauthorized",
                                                     "action": "authenticate"}}
        self.frames = [json.dumps(reply)] + frames

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, timeout=None):
        if not self.frames:
            raise ConnectionError("connection dropped")
        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame

    def close(self):
        pass


def test_stream_ingests_binary_frames_and_reconnects_with_backoff():
    frame = json.dumps({"stream": "trade_updates", "data": _update()}).encode()   # paper sends binary
    attempts = []

    def connect(url):
        attempts.append(url)
        if len(attempts) == 1:
            return FakeSocket([TimeoutError(), frame])     # works, then drops
        if len(attempts) in (2, 3):
            raise OSError("network unreachable")             # outage continues
        stream._stop.set()                                   # 4th attempt: back up; end the test
        return FakeSocket([])

    stream = TradeUpdateStream("wss://x/stream", "key", "secret", telemetry=EventLog(None), connect=connect)
    waits = []
    stream.sleep = waits.append
    stream.run()

    kinds = [e["event"] for e in stream.telemetry.events]
    assert kinds == ["stream_fill", "stream_disconnected", "stream_connected"]   # one event per outage
    assert waits == [1.0, 2.0, 4.0]                                              # exponential backoff
    assert stream.order_update("leg-1") is not None and stream.status == "stopped"


def test_malformed_update_does_not_drop_the_connection():
    good = json.dumps({"stream": "trade_updates", "data": _update()})
    bad = json.dumps({"stream": "trade_updates", "data": {"event": "fill", "order": ["not", "a", "dict"]}})
    stream = TradeUpdateStream("wss://x/stream", "k", "s", telemetry=EventLog(None),
                               connect=lambda url: FakeSocket(["not json", bad, good]))
    stream.sleep = lambda s: stream._stop.set()
    stream.run()
    assert [e["event"] for e in stream.telemetry.events][:1] == ["stream_fill"]


def test_stream_sends_auth_then_listen():
    sock = FakeSocket([])
    stream = TradeUpdateStream("wss://x/stream", "key", "secret", telemetry=EventLog(None), connect=lambda u: sock)
    stream.sleep = lambda s: stream._stop.set()
    stream.run()
    assert sock.sent == [{"action": "auth", "key": "key", "secret": "secret"},
                         {"action": "listen", "data": {"streams": ["trade_updates"]}}]


def test_bad_credentials_stop_the_stream_without_retrying():
    attempts = []
    stream = TradeUpdateStream("wss://x/stream", "bad", "bad", telemetry=EventLog(None),
                               connect=lambda url: attempts.append(url) or FakeSocket([], auth_ok=False))
    stream.run()
    assert len(attempts) == 1 and stream.status == "failed"
    assert [e["event"] for e in stream.telemetry.events] == ["stream_failed"]


def test_streamed_fill_wakes_the_bot_for_an_immediate_sync(portfolio_manager):
    broker = StreakBroker([])
    bot = _bot(portfolio_manager, broker, T0)
    stream = TradeUpdateStream("wss://x/stream", "k", "s", telemetry=EventLog(None))
    bot.attach_stream(stream)

    stream.handle(_update(event="new"))      # not a fill: no wake-up
    assert not bot._wake.is_set()
    stream.handle(_update())                 # stop hit: wake up
    assert bot._wake.is_set()

    threading.Timer(0.05, bot.stop).start()  # end the idle wait shortly after the sync
    bot.sleep(5)

    assert bot._fast_syncs == 1 and broker.snapshots == 1


def test_rest_cycle_keeps_running_while_the_stream_is_down(portfolio_manager, tmp_path):
    broker = StreakBroker([])
    bot = _bot(portfolio_manager, broker, T0, heartbeat_path=str(tmp_path / "hb.json"))
    stream = TradeUpdateStream("wss://x/stream", "k", "s", telemetry=EventLog(None))
    bot.attach_stream(stream)            # never connected: status "reconnecting"

    bot.run_cycle()

    assert broker.snapshots == 1        # regular broker sync still happened
    assert json.loads((tmp_path / "hb.json").read_text())["order_stream"] == "reconnecting"


# ---------------------------------------------------------------------------
# Backtester: smart limit fills and the streak pause
# ---------------------------------------------------------------------------

CHASE = BacktestConfig(spread_bps=2.0, slippage_bps=0.0, entry_chase_slippage_bps=15.0)


def test_chase_fill_model():
    slip = CHASE.spread_bps / 2 / 10_000
    # Decision close 100.00 -> arrival ask 100.01; cap = 100.01 * 1.0015 = 100.16.
    assert _chase_fill(100.0, 100.05, 99.9, CHASE, slip) == pytest.approx(100.05 * (1 + slip))
    assert _chase_fill(100.0, 100.14, 100.0, CHASE, slip) == pytest.approx(100.14 * (1 + slip))  # re-pegged
    assert _chase_fill(100.0, 100.30, 99.0, CHASE, slip) is None     # gapped past cap: cancelled, even if it dips
    assert _chase_fill(100.0, 100.05, 100.10, CHASE, slip) is None   # bar never traded at the price paid


def test_backtest_counts_slippage_timeouts():
    bars = pd.concat([_day_bars("2026-02-27"), _day_bars("2026-03-02")])
    gap_at = _et(bars, "14:05")
    for col in ("open", "high", "low", "close"):
        bars.loc[gap_at:, col] = bars.loc[gap_at:, col] + 1.0   # +1% gap right after the signal
    result = _intraday_backtest(bars, {_et(bars, "14:00"): 0.9}, entry_chase_slippage_bps=15.0)
    assert not result.trades and result.session_blocked.get("slippage_timeout") == 1


def test_backtest_pauses_after_two_losses_in_a_row():
    day = _day_bars("2026-03-02")
    for hhmm in ("10:10", "10:40"):   # a flush through the stop after each entry
        day.loc[_et(day, hhmm), "low"] = 95.0
    bars = pd.concat([_day_bars("2026-02-27"), day])
    schedule = {_et(bars, t): 0.9 for t in ("10:00", "10:25", "11:00", "11:40")}
    limits = RiskLimits(min_price=1.0, max_position_pct=0.5, max_daily_trades=10, reentry_cooldown_minutes=0,
                        max_consecutive_losses=2, streak_cooldown_minutes=45)

    result = _intraday_backtest(bars, schedule, limits=limits)

    stops = [t for t in result.trades if t.exit_reason.startswith("stop")]
    assert len(stops) == 2
    assert result.blocked.get("Loss streak") == 1                 # 11:00 signal inside the 45-min pause
    later = [t for t in result.trades if t.entry_time >= _et(bars, "11:40")]
    assert len(later) == 1                                        # 11:40 is after 10:45 + 45 min


def test_backtest_streak_helper_matches_live_rule():
    t = pd.Timestamp("2026-03-02 15:00", tz="UTC")

    def trade(pnl, minutes):
        return Trade(symbol="A", entry_time=t, entry_price=100.0, quantity=1, stop=99.0, target=102.0,
                     probability_up=0.7, exit_time=t + pd.Timedelta(minutes=minutes), exit_price=100.0 + pnl)

    streak = _loss_streak([trade(-1, 10), trade(2, 20), trade(-1, 30), trade(-1, 40)], t.tz_convert(
        "America/New_York").date(), timedelta(minutes=5))
    assert streak.count == 2 and streak.last_loss_at == (t + pd.Timedelta(minutes=45)).to_pydatetime()
