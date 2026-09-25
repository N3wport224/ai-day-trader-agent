"""Intraday architecture gaps: flat confirmation, ORB30, margin guard, VWAP attribution, 5m defaults."""
from __future__ import annotations

import pandas as pd
import pytest

from core.alpaca_executor import BrokerSnapshot, FlattenReport
from core.execution_telemetry import EventLog
from core.session_clock import SessionClock, SessionConfig
from core.trading_bot import TradingBot
from tests.test_intraday_trading import RecordingWorkflow, _clock


# ---------------------------------------------------------------------------
# EOD: flatten fills and flat confirmation before the close
# ---------------------------------------------------------------------------

class ClosingBroker:
    """Positions close only after ``closes_after`` flatten attempts."""

    def __init__(self, closes_after=1, stuck=False):
        self.attempts, self.closes_after, self.stuck = 0, closes_after, stuck
        self.positions = [{"symbol": "AAPL", "qty": "10"}]

    def get_snapshot(self):
        return BrokerSnapshot(positions=list(self.positions), open_orders=[])

    def flatten_all(self, reason):
        self.attempts += 1
        if not self.stuck and self.attempts >= self.closes_after:
            self.positions = []
        return FlattenReport(reason=reason, closed=[{"symbol": "AAPL", "qty": 10, "order_id": "c1"}])


def _eod_bot(pm, broker, state):
    return TradingBot(
        RecordingWorkflow(pm), ["AAPL"], execute=True, broker=broker,
        market_clock=lambda: _clock(state["time"]),
        reconcile_fn=lambda *a, **k: type("R", (), {"discrepancies": [], "positions": 0, "open_orders": 0})(),
        session_clock=SessionClock(SessionConfig()), telemetry=EventLog(None),
        now_fn=lambda: pd.Timestamp(f"2026-03-02 {state['time']}", tz="America/New_York").to_pydatetime(),
        interval_seconds=900,
    )


def test_flat_confirmed_once_after_flatten(portfolio_manager) -> None:
    state = {"time": "15:50"}
    bot = _eod_bot(portfolio_manager, ClosingBroker(), state)
    report = bot.run_cycle()
    assert report.phase == "FLATTEN" and report.flat is True
    state["time"] = "15:52"
    assert bot.run_cycle().flat is True
    events = [e for e in bot.telemetry.events if e["event"] == "flat_confirmed"]
    assert len(events) == 1 and events[0]["at_et"] == "15:50:00"


def test_not_flat_alert_near_the_close_once(portfolio_manager) -> None:
    from core.alerts import format_alert

    state = {"time": "15:50"}
    bot = _eod_bot(portfolio_manager, ClosingBroker(stuck=True), state)
    assert bot.run_cycle().flat is False
    assert not [e for e in bot.telemetry.events if e["event"] == "not_flat"]   # 10 min left: keep retrying quietly
    state["time"] = "15:56"
    bot.run_cycle()
    state["time"] = "15:58"
    bot.run_cycle()
    alerts = [e for e in bot.telemetry.events if e["event"] == "not_flat"]
    assert len(alerts) == 1 and alerts[0]["positions"] == [{"symbol": "AAPL", "qty": "10"}]
    assert alerts[0]["minutes_to_close"] == 4.0 and "NOT FLAT" in format_alert(alerts[0])
    assert bot.broker.attempts == 3                                              # retried every cycle


def test_flatten_window_rechecks_every_two_minutes(portfolio_manager) -> None:
    state = {"time": "15:50"}
    bot = _eod_bot(portfolio_manager, ClosingBroker(), state)
    bot.run_cycle()
    assert bot._next_sleep() == 120
    state["time"] = "11:00"
    bot.run_cycle()
    assert bot._next_sleep() == 900


def test_flatten_fill_confirmation_polls_until_filled(monkeypatch) -> None:
    from core.alpaca_executor import AlpacaExecutor

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FLATTEN_CONFIRM_SECONDS", "30")
    ex = AlpacaExecutor(telemetry=EventLog(None), price_lookup=lambda s: 0.0)
    ex.sleep = lambda s: None
    polls = {"n": 0}

    def get_order(self, oid):
        polls["n"] += 1
        if polls["n"] < 3:
            return {"id": oid, "status": "new"}
        return {"id": oid, "status": "filled", "filled_avg_price": "187.21", "filled_qty": "10",
                "submitted_at": "2026-03-02T20:50:01Z", "filled_at": "2026-03-02T20:50:03Z"}

    monkeypatch.setattr(AlpacaExecutor, "get_order", get_order)
    report = FlattenReport(reason="eod", closed=[{"symbol": "AAPL", "qty": 10, "order_id": "o1"}])
    ex._confirm_flatten_fills(report)
    assert polls["n"] == 3 and report.closed[0]["fill_price"] == 187.21
    assert report.closed[0]["filled_at"] == "2026-03-02T20:50:03Z"

    monkeypatch.setenv("FLATTEN_CONFIRM_SECONDS", "0")
    monkeypatch.setattr(AlpacaExecutor, "get_order", lambda self, oid: {"id": oid, "status": "new"})
    report = FlattenReport(reason="eod", closed=[{"symbol": "MSFT", "qty": 5, "order_id": "o2"}])
    ex._confirm_flatten_fills(report)
    assert report.closed[0]["status"] == "unconfirmed"


# ---------------------------------------------------------------------------
# Microstructure features: VWAP (+bands, ratio, zone), ORB15/30, zero lookahead
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from core.features import VWAP_ZONES, add_intraday_features, compute_features  # noqa: E402
from core.ml_training import synthetic_intraday_bars  # noqa: E402


def _session(freq="5min", day="2026-03-02"):
    idx = pd.date_range(f"{day} 09:30", f"{day} 15:55" if freq == "5min" else f"{day} 15:59", freq=freq,
                        tz="America/New_York").tz_convert("UTC")
    n = len(idx)
    close = 100 + np.sin(np.arange(n) / 7.0)
    return pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
                         "volume": np.linspace(1000, 3000, n)}, index=idx)


def _feats(bars, minutes):
    return add_intraday_features(compute_features(bars), bar_length=pd.Timedelta(minutes=minutes))


@pytest.mark.parametrize("freq, minutes", [("5min", 5), ("1min", 1)])
def test_orb15_and_orb30_values_and_timing(freq, minutes) -> None:
    bars = _session(freq)
    f = _feats(bars, minutes)
    local = f.index.tz_convert("America/New_York")
    first30 = bars[local < pd.Timestamp("2026-03-02 10:00", tz="America/New_York")]
    first15 = bars[local < pd.Timestamp("2026-03-02 09:45", tz="America/New_York")]
    at_1000 = f.loc[local == pd.Timestamp("2026-03-02 10:00", tz="America/New_York")].iloc[0]
    assert at_1000["orb30_high"] == pytest.approx(first30["high"].max())
    assert at_1000["orb30_low"] == pytest.approx(first30["low"].min())
    assert at_1000["orb_high"] == pytest.approx(first15["high"].max())
    before = f[local < pd.Timestamp("2026-03-02 10:00", tz="America/New_York")]
    assert before["orb30_high"].isna().all()                     # hidden until the window has closed
    assert f.loc[local == pd.Timestamp("2026-03-02 09:45", tz="America/New_York"), "orb_high"].notna().all()


def test_orb30_needs_bars_no_longer_than_the_window() -> None:
    idx = pd.date_range("2026-03-02 14:30", periods=40, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0}, index=idx)
    f = add_intraday_features(compute_features(bars), bar_length=pd.Timedelta(hours=1))
    assert f["orb30_high"].isna().all() and f["orb_high"].isna().all()


def test_vwap_resets_each_session_with_bands_ratio_and_zone() -> None:
    bars = pd.concat([_session(day="2026-03-02"), _session(day="2026-03-03")])
    bars.loc[bars.index[len(bars) // 2]:, ["open", "high", "low", "close"]] += 50  # day 2 trades far higher
    f = _feats(bars, 5)
    day2_open = f.index[len(bars) // 2]
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    assert f.loc[day2_open, "vwap"] == pytest.approx(typical.loc[day2_open])       # reset at 9:30, not blended
    row = f.iloc[40]
    assert row["vwap_upper_1"] - row["vwap"] == pytest.approx(row["vwap_sigma"])
    assert row["vwap_upper_2"] - row["vwap"] == pytest.approx(2 * row["vwap_sigma"])
    assert row["vwap_lower_2"] == pytest.approx(row["vwap"] - 2 * row["vwap_sigma"])
    assert row["vwap_dist"] == pytest.approx((row["close"] - row["vwap"]) / row["vwap"])   # the VWAP ratio
    zone_index = int(np.searchsorted([-2, -1, 0, 1, 2], row["vwap_z"], side="right"))
    assert row["vwap_zone"] == VWAP_ZONES[zone_index]
    assert set(f["vwap_zone"].dropna()) <= set(VWAP_ZONES)


@pytest.mark.parametrize("freq, minutes", [("5min", 5), ("1min", 1)])
def test_new_intraday_features_have_zero_lookahead(freq, minutes) -> None:
    bars = synthetic_intraday_bars(days=4, freq=freq, seed=11)
    cols = ["vwap", "vwap_upper_1", "vwap_lower_2", "vwap_dist", "vwap_z", "rvol", "orb_high", "orb30_high",
            "orb30_low", "close_vs_orb30_high", "orb30_range_pct"]
    full = _feats(bars, minutes)
    for cut in (bars.index[len(bars) // 3], bars.index[len(bars) // 2 + 7]):
        prefix = _feats(bars[bars.index <= cut], minutes)
        pd.testing.assert_frame_equal(full.loc[:cut, cols], prefix[cols])
        shocked = bars.copy()
        shocked.loc[shocked.index > cut, ["open", "high", "low", "close"]] *= 3
        shocked.loc[shocked.index > cut, "volume"] *= 10
        pd.testing.assert_frame_equal(full.loc[:cut, cols], _feats(shocked, minutes).loc[:cut, cols])


# ---------------------------------------------------------------------------
# Intraday margin & regulatory guard
# ---------------------------------------------------------------------------

from core.risk_manager import RiskLimits, RiskManager  # noqa: E402

MARGIN_ACCOUNT = {"equity": "30000", "last_equity": "30000", "buying_power": "60000", "status": "ACTIVE"}


def _margin_buy(account, quantity=100, price=100.0, **limits):
    manager = RiskManager(RiskLimits(max_position_pct=1.0, max_portfolio_heat_pct=0, **limits))
    return manager.check_order(side="BUY", symbol="AAPL", quantity=quantity, price=price, account=account,
                               position=None, orders_today=[], stop_loss=price * 0.98, take_profit=price * 1.04)


@pytest.mark.parametrize("change, fragment", [
    ({"status": "ACCOUNT_CLOSED"}, "ACCOUNT_CLOSED"),
    ({"trade_suspended_by_user": True}, "suspended"),
    ({"intraday_margin_deficit": "150"}, "deficit"),
])
def test_margin_guard_hard_blocks_restricted_accounts(change, fragment):
    decision = _margin_buy({**MARGIN_ACCOUNT, **change})
    assert not decision.approved and fragment in decision.reason


def test_margin_cushion_is_a_fixed_share_of_equity():
    # $30k equity, 5% cushion = $1,500 kept back from $10k buying power.
    decision = _margin_buy({**MARGIN_ACCOUNT, "buying_power": "10000"})
    assert decision.approved and decision.quantity == 85
    # A second order can't eat into the cushion either.
    second = _margin_buy({**MARGIN_ACCOUNT, "buying_power": "1600"})
    assert second.approved and second.quantity == 1


def test_day_trading_buying_power_caps_intraday_entries():
    account = {**MARGIN_ACCOUNT, "daytrading_buying_power": "5000"}
    assert _margin_buy(account, day_trading=True).quantity == 35      # ($5,000 - $1,500) / $100
    assert _margin_buy(account, day_trading=False).quantity == 100    # DTBP only binds in day-trading mode


def test_maintenance_margin_headroom_caps_exposure():
    # $30k equity - $28k maintenance - $1.5k cushion = $500 excess; / 30% = $1,666 new exposure.
    decision = _margin_buy({**MARGIN_ACCOUNT, "maintenance_margin": "28000"})
    assert decision.approved and decision.quantity == 16


def test_optional_intraday_margin_fields_are_honoured_when_reported():
    account = {**MARGIN_ACCOUNT, "intraday_buying_power": "4000", "intraday_margin_excess": "3000"}
    assert _margin_buy(account).quantity == 25                          # min(4000-1500, (3000-1500)/0.3)
    assert _margin_buy({**account, "intraday_margin_excess": "1600"}).quantity == 3


def test_order_that_would_breach_margin_is_blocked_not_shrunk_to_zero():
    decision = _margin_buy({**MARGIN_ACCOUNT, "buying_power": "1550"})
    assert not decision.approved
    assert decision.reason.startswith("Intraday margin") and "one AAPL share" in decision.reason


def test_exhausted_buying_power_still_reports_no_room():
    decision = _margin_buy({**MARGIN_ACCOUNT, "buying_power": "50"})
    assert not decision.approved and decision.reason.startswith("No room")


def test_pdt_guard_blocks_fourth_day_trade_under_25k():
    small = {"equity": "20000", "last_equity": "20000", "buying_power": "20000"}
    assert _margin_buy({**small, "daytrade_count": 2}, quantity=10, day_trading=True).approved
    blocked = _margin_buy({**small, "daytrade_count": 3}, quantity=10, day_trading=True)
    assert not blocked.approved and "PDT" in blocked.reason


def test_margin_limits_read_from_environment(monkeypatch):
    monkeypatch.setenv("MARGIN_BUFFER_PCT", "10")
    monkeypatch.setenv("MAINTENANCE_MARGIN_RATE", "0.5")
    limits = RiskLimits.from_env()
    assert limits.margin_buffer_pct == 10 and limits.maintenance_rate == 0.5


# ---------------------------------------------------------------------------
# VWAP-location attribution and 5-minute CLI defaults
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from core.backtester import Trade, attribution, format_report  # noqa: E402
from core.features import VWAP_ZONES, vwap_zone  # noqa: E402
from tests.test_intraday_trading import _day_bars, _et, _intraday_backtest  # noqa: E402


@pytest.mark.parametrize("z, zone", [(-2.5, "below -2σ"), (-1.5, "-2σ to -1σ"), (-0.1, "-1σ to VWAP"),
                                     (0.0, "VWAP to +1σ"), (1.2, "+1σ to +2σ"), (2.0, "above +2σ"),
                                     (float("nan"), None), (None, None)])
def test_vwap_zone_labels(z, zone):
    assert vwap_zone(z) == zone


def test_backtest_records_vwap_location_at_entry():
    warmup = _day_bars("2026-02-27")
    day = _day_bars("2026-03-02")
    drift = np.linspace(0, 3, len(day)) + np.tile([0.0, 0.3, -0.2], len(day) // 3 + 1)[:len(day)]
    for col in ("open", "high", "low", "close"):
        day[col] = day[col] + drift
    bars = pd.concat([warmup, day])
    result = _intraday_backtest(bars, {_et(bars, "14:00"): 0.9})

    trade = result.trades[0]
    assert trade.vwap_zone in VWAP_ZONES and trade.vwap_zone.startswith(("VWAP", "+", "above"))  # uptrend
    assert result.trades_frame()["vwap_zone"].tolist() == [trade.vwap_zone]
    assert "By VWAP location at entry" in format_report(result)


def test_vwap_attribution_orders_bands_low_to_high():
    t0 = pd.Timestamp("2026-03-02 15:00", tz="UTC")

    def trade(zone, pnl):
        return Trade(symbol="A", entry_time=t0, entry_price=100.0, quantity=1, stop=99.0, target=102.0,
                     probability_up=0.7, exit_time=t0, exit_price=100.0 + pnl, vwap_zone=zone)

    table = attribution([trade("above +2σ", -1), trade("-2σ to -1σ", 2), trade("VWAP to +1σ", 1)], "vwap_zone")
    assert table["vwap_zone"].tolist() == ["-2σ to -1σ", "VWAP to +1σ", "above +2σ"]
    assert table["total_pnl"].tolist() == [2.0, 1.0, -1.0]


class _Parsed(Exception):
    def __init__(self, parser):
        self.defaults = {a.dest: a.default for a in parser._actions}


def _cli_defaults(module, monkeypatch):
    def stop(parser, argv=None):
        raise _Parsed(parser)

    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", stop)
    with pytest.raises(_Parsed) as parsed:
        module.main([])
    return parsed.value.defaults


def test_cli_defaults_to_5_minute_bars(monkeypatch):
    from core.market_history import default_history_days
    from scripts import backtest, train_model

    monkeypatch.delenv("ML_TIMEFRAME", raising=False)
    for module in (backtest, train_model):
        defaults = _cli_defaults(module, monkeypatch)
        assert defaults["timeframe"] == "5Min" and defaults["days"] is None
    assert default_history_days("5m", 730) == 120 and default_history_days("1m", 730) == 30
    assert default_history_days("1h", 730) == 730 and default_history_days("1d", 365) == 365
