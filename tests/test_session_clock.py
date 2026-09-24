from __future__ import annotations

import pandas as pd
import pytest

from core.market_history import bar_length, is_intraday, normalize_timeframe, timeframe_from_length
from core.session_clock import SessionClock, SessionConfig, SessionPhase


def et(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="America/New_York")


CLOCK = SessionClock(SessionConfig(opening_lockout_minutes=15, entry_cutoff_minutes=15, flatten_minutes=10))


@pytest.mark.parametrize(
    "when, phase",
    [
        ("2026-03-02 09:29", SessionPhase.CLOSED),
        ("2026-03-02 09:30", SessionPhase.OPENING_LOCKOUT),
        ("2026-03-02 09:44:59", SessionPhase.OPENING_LOCKOUT),
        ("2026-03-02 09:45", SessionPhase.OPEN),
        ("2026-03-02 15:44:59", SessionPhase.OPEN),
        ("2026-03-02 15:45", SessionPhase.ENTRY_CUTOFF),
        ("2026-03-02 15:50", SessionPhase.FLATTEN),
        ("2026-03-02 15:59", SessionPhase.FLATTEN),
        ("2026-03-02 16:00", SessionPhase.CLOSED),
        ("2026-03-07 11:00", SessionPhase.CLOSED),   # Saturday
    ],
)
def test_phases_on_a_regular_day(when, phase) -> None:
    assert CLOCK.phase(et(when)) is phase


def test_accepts_utc_and_naive_utc_timestamps() -> None:
    assert CLOCK.phase(pd.Timestamp("2026-03-02 15:00", tz="UTC")) is SessionPhase.OPEN        # 10:00 ET
    assert CLOCK.phase("2026-03-02T20:52:00Z") is SessionPhase.FLATTEN                           # 15:52 ET
    assert CLOCK.phase(pd.Timestamp("2026-03-02 20:52")) is SessionPhase.FLATTEN                 # naive = UTC


def test_early_close_moves_cutoff_and_flatten() -> None:
    close = et("2026-11-27 13:00")  # day after Thanksgiving
    assert CLOCK.phase(et("2026-11-27 12:44"), close) is SessionPhase.OPEN
    assert CLOCK.phase(et("2026-11-27 12:45"), close) is SessionPhase.ENTRY_CUTOFF
    assert CLOCK.phase(et("2026-11-27 12:50"), close) is SessionPhase.FLATTEN
    assert CLOCK.phase(et("2026-11-27 13:05"), close) is SessionPhase.CLOSED


def test_overnight_mode_never_flattens() -> None:
    clock = SessionClock(SessionConfig(no_overnight=False))
    assert clock.phase(et("2026-03-02 15:55")) is SessionPhase.ENTRY_CUTOFF
    assert clock.seconds_until_flatten(et("2026-03-02 15:00")) is None


def test_alpaca_clock_payload() -> None:
    open_clock = {"is_open": True, "timestamp": "2026-03-02T15:52:00-05:00", "next_close": "2026-03-02T16:00:00-05:00"}
    assert CLOCK.phase_from_alpaca_clock(open_clock) is SessionPhase.FLATTEN
    assert CLOCK.phase_from_alpaca_clock({"is_open": False}) is SessionPhase.CLOSED


def test_seconds_until_flatten() -> None:
    assert CLOCK.seconds_until_flatten(et("2026-03-02 15:40")) == 600
    assert CLOCK.seconds_until_flatten(et("2026-03-02 15:51")) is None
    assert CLOCK.can_enter(et("2026-03-02 10:00")) and not CLOCK.can_enter(et("2026-03-02 09:35"))
    assert CLOCK.must_flatten(et("2026-03-02 15:55"))


def test_config_validation_and_env(monkeypatch) -> None:
    with pytest.raises(ValueError):
        SessionConfig(entry_cutoff_minutes=5, flatten_minutes=10)
    monkeypatch.setenv("OPENING_LOCKOUT_MINUTES", "30")
    monkeypatch.setenv("NO_OVERNIGHT", "false")
    cfg = SessionConfig.from_env()
    assert cfg.opening_lockout_minutes == 30 and cfg.no_overnight is False


def test_timeframe_aliases_and_intraday_detection() -> None:
    assert normalize_timeframe("5m") == "5Min"
    assert normalize_timeframe("1m") == "1Min"
    assert normalize_timeframe("1h") == "1Hour"
    assert normalize_timeframe("1Day") == "1Day"
    assert is_intraday("5m") and is_intraday("1Min") and not is_intraday("1h")
    assert bar_length("5m") == pd.Timedelta(minutes=5)
    assert timeframe_from_length(pd.Timedelta(minutes=1)) == "1Min"
    with pytest.raises(ValueError):
        normalize_timeframe("7m")
