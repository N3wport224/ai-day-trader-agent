from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.feature_pipeline import build_feature_frame
from core.features import INTRADAY_FEATURES, add_intraday_features, compute_features
from core.ml_training import LabelParams, build_dataset, regular_session_ids, synthetic_intraday_bars, triple_barrier_labels

FIVE = pd.Timedelta(minutes=5)


def _session(day: str, closes, volumes, start="09:30", freq="5min", spread=0.5):
    idx = pd.date_range(pd.Timestamp(f"{day} {start}", tz="America/New_York"), periods=len(closes), freq=freq)
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close,
         "volume": np.array(volumes, dtype=float)},
        index=idx.tz_convert("UTC"),
    )


def _intraday(bars, **kwargs):
    return add_intraday_features(compute_features(bars), bar_length=FIVE, **kwargs)


def test_vwap_matches_hand_calculation_and_resets_each_session() -> None:
    day1 = _session("2026-03-02", [10, 12, 11], [100, 300, 200])
    day2 = _session("2026-03-03", [20, 22], [100, 100])
    f = _intraday(pd.concat([day1, day2]))

    typical = np.array([10, 12, 11])  # (h + l + c) / 3 == close with symmetric spread
    vols = np.array([100, 300, 200])
    expected = np.cumsum(typical * vols) / np.cumsum(vols)
    assert f["vwap"].iloc[:3].tolist() == pytest.approx(expected.tolist())
    assert f["vwap"].iloc[3] == pytest.approx(20.0)            # new session: reset
    assert f["vwap"].iloc[4] == pytest.approx(21.0)

    # Volume-weighted sigma and bands on the 3rd bar of day 1.
    vwap3 = expected[-1]
    sigma = np.sqrt(np.sum(vols * (typical - vwap3) ** 2) / vols.sum())
    assert f["vwap_sigma"].iloc[2] == pytest.approx(sigma)
    assert f["vwap_upper_2"].iloc[2] == pytest.approx(vwap3 + 2 * sigma)
    assert f["vwap_lower_1"].iloc[2] == pytest.approx(vwap3 - sigma)
    assert f["vwap_dist"].iloc[2] == pytest.approx((11 - vwap3) / vwap3)


def test_extended_hours_bars_are_excluded_from_vwap() -> None:
    premarket = _session("2026-03-02", [50, 50], [1e6, 1e6], start="09:00")
    regular = _session("2026-03-02", [10, 12], [100, 100])
    f = _intraday(pd.concat([premarket, regular]))

    assert f["vwap"].iloc[:2].isna().all()
    assert f["vwap"].iloc[2] == pytest.approx(10.0)
    assert f["minutes_from_open"].iloc[2:].tolist() == [0.0, 5.0]


def test_opening_range_is_hidden_until_the_window_closes() -> None:
    closes = [10, 11, 9, 10.5, 10.8, 12.0]  # three 5-min bars form the 15-min range
    f = _intraday(_session("2026-03-02", closes, [100] * 6), orb_minutes=15)

    assert f["orb_high"].iloc[:3].isna().all()                  # 9:30-9:45: still forming
    assert f["orb_high"].iloc[3:].tolist() == [11.5] * 3        # max high of first 3 bars
    assert f["orb_low"].iloc[3:].tolist() == [8.5] * 3
    assert f["close_vs_orb_high"].iloc[5] == pytest.approx(12.0 / 11.5 - 1)
    assert f["orb_range_pct"].iloc[3] == pytest.approx((11.5 - 8.5) / 8.5)


def test_orb_undefined_for_bars_longer_than_the_window() -> None:
    bars = _session("2026-03-02", [10, 11, 12], [100] * 3, freq="1h")
    f = add_intraday_features(compute_features(bars), bar_length=pd.Timedelta(hours=1), orb_minutes=15)
    assert f["orb_high"].isna().all()


def test_rvol_uses_same_time_bucket_on_prior_sessions_only() -> None:
    days = pd.bdate_range("2026-03-02", periods=6)
    sessions = [_session(str(d.date()), [10, 10], [100 * (i + 1), 50]) for i, d in enumerate(days)]
    f = _intraday(pd.concat(sessions), rvol_sessions=4)
    opening = f[f["minutes_from_open"] == 0]

    assert opening["rvol"].iloc[:3].isna().all()                 # needs >= 3 prior sessions
    # Day 4 opening volume 400 vs mean(100, 200, 300) = 200 -> 2.0
    assert opening["rvol"].iloc[3] == pytest.approx(2.0)
    # Day 6 opening 600 vs mean of the previous 4 sessions (200..500) = 350
    assert opening["rvol"].iloc[5] == pytest.approx(600 / 350)


@pytest.mark.parametrize("freq", ["5min", "1min"])
def test_intraday_pipeline_has_no_lookahead(freq) -> None:
    bars = synthetic_intraday_bars(days=12, freq=freq, seed=3)
    timeframe = "5Min" if freq == "5min" else "1Min"
    full = build_feature_frame(bars, timeframe)
    cols = INTRADAY_FEATURES + ["vwap", "vwap_sigma", "orb_high", "orb_low", "regime"]
    for t in (len(bars) // 3, len(bars) // 2, len(bars) - 1):
        prefix = build_feature_frame(bars.iloc[: t + 1], timeframe)
        pd.testing.assert_series_equal(prefix.iloc[-1][cols], full.iloc[t][cols], check_names=False)


def test_non_intraday_timeframes_get_empty_intraday_columns() -> None:
    idx = pd.date_range("2025-01-02", periods=60, freq="D", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e5}, index=idx)
    f = build_feature_frame(bars, "1Day")
    assert set(INTRADAY_FEATURES) <= set(f.columns) and f[INTRADAY_FEATURES].isna().all().all()


def test_labels_stop_at_the_session_close_in_day_trading_mode() -> None:
    day1 = _session("2026-03-02", [100, 100, 100], [1, 1, 1])
    day2 = _session("2026-03-03", [110, 110, 110], [1, 1, 1])   # gap up overnight
    frame = pd.concat([day1, day2])
    frame["atr"] = 1.0
    params = LabelParams(horizon=3, stop_atr_mult=1.0, target_atr_mult=2.0)

    swing = triple_barrier_labels(frame, params)
    intraday = triple_barrier_labels(frame, params, regular_session_ids(frame.index))

    assert swing.iloc[2] == 1.0       # overnight gap "hits" the target
    assert intraday.iloc[2] == 0.0    # but a day trader is flat at the close


def test_intraday_training_dataset_includes_intraday_features() -> None:
    bars = synthetic_intraday_bars(days=25, seed=2)
    data = build_dataset(bars, LabelParams(), FIVE)
    assert set(INTRADAY_FEATURES) <= set(data.columns)
    assert data["rvol"].notna().any() and data["vwap_dist"].notna().any()
