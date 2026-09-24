from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import TECHNICAL_FEATURES, atr, bars_from_candles, compute_features, rsi
from core.ml_training import synthetic_bars


def _bars(rows):
    index = pd.date_range("2026-01-05 14:30", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)


def test_candlestick_anatomy_and_gap() -> None:
    bars = _bars(
        [
            [10.0, 11.0, 9.0, 10.5, 100],   # range 2, body +0.5
            [11.0, 12.0, 10.0, 10.0, 100],  # gap up 0.5 from 10.5; bearish body
            [10.0, 10.0, 10.0, 10.0, 100],  # zero-range bar
        ]
    )

    f = compute_features(bars)

    assert f["body_ratio"].tolist() == pytest.approx([0.25, -0.5, 0.0])
    assert f["upper_wick_ratio"].tolist() == pytest.approx([0.25, 0.5, 0.0])
    assert f["lower_wick_ratio"].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert np.isnan(f["gap_pct"].iloc[0])
    assert f["gap_pct"].iloc[1] == pytest.approx((11.0 - 10.5) / 10.5 * 100)


def test_features_have_no_lookahead_bias() -> None:
    """Row t must be identical whether or not later bars exist."""
    bars = synthetic_bars(n=400, seed=3)
    full = compute_features(bars)

    for t in (30, 199, 250, 399):
        prefix = compute_features(bars.iloc[: t + 1])
        pd.testing.assert_series_equal(prefix.iloc[-1], full.iloc[t], check_names=False)


def test_future_bars_do_not_change_past_rows() -> None:
    bars = synthetic_bars(n=300, seed=5)
    shocked = bars.copy()
    shocked.iloc[250:, :4] *= 3  # huge move after t=249

    base = compute_features(bars).iloc[:250]
    after = compute_features(shocked).iloc[:250]

    pd.testing.assert_frame_equal(base, after)


def test_indicator_warmup_is_nan_not_misleading() -> None:
    f = compute_features(synthetic_bars(n=250, seed=1))

    assert f["ema200"].iloc[:199].isna().all()
    assert f["ema200"].iloc[199:].notna().all()
    assert f["rsi"].iloc[:14].isna().all()
    assert f["rsi"].dropna().between(0, 100).all()
    assert set(TECHNICAL_FEATURES) <= set(f.columns)


def test_rsi_extremes_and_atr_matches_wilder() -> None:
    rising = pd.Series(np.arange(1.0, 40.0))
    assert rsi(rising).iloc[-1] == 100.0

    high = pd.Series([11.0, 12.0, 13.0, 12.5])
    low = pd.Series([9.0, 10.0, 11.5, 11.0])
    close = pd.Series([10.0, 11.5, 12.0, 11.5])
    # True ranges: 2.0, 2.0, 1.5, 1.5 -> Wilder(3): 2.0, 2.0, 1.8333, 1.7222 (seeded at bar 0)
    result = atr(high, low, close, period=3)
    assert result.iloc[:2].isna().all()
    expected = [2.0]
    for tr in (2.0, 1.5, 1.5):
        expected.append(expected[-1] + (tr - expected[-1]) / 3)
    assert result.iloc[2:].tolist() == pytest.approx(expected[2:4])


def test_rejects_unsorted_or_incomplete_frames() -> None:
    bars = synthetic_bars(n=30)
    with pytest.raises(ValueError, match="oldest-first"):
        compute_features(bars.iloc[::-1])
    with pytest.raises(ValueError, match="missing columns"):
        compute_features(bars.drop(columns=["volume"]))


def test_bars_from_candles_sorts_fetcher_output() -> None:
    candles = [
        {"datetime": "2026-01-05T15:30:00Z", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "10"},
        {"datetime": "2026-01-05T14:30:00Z", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "5"},
    ]

    frame = bars_from_candles(candles)

    assert frame["close"].tolist() == [1.5, 2.5]
    assert frame.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Multi-timeframe (daily anchor) alignment
# ---------------------------------------------------------------------------

from core.features import MACRO_FEATURES, add_macro_features, resample_bars  # noqa: E402


def _hourly(n=24 * 90, seed=4):
    return synthetic_bars(n=n, seed=seed, start="2025-01-01 00:00")


def test_macro_features_only_use_completed_daily_candles() -> None:
    hourly = _hourly()
    daily = resample_bars(hourly, "1D")
    feats = add_macro_features(
        compute_features(hourly), daily, primary_bar_length=pd.Timedelta(hours=1)
    )
    daily_ema50 = daily["close"].ewm(span=50, adjust=False, min_periods=50).mean()

    # An hourly bar on day D (closing before midnight) must see day D-1's EMA.
    ts = pd.Timestamp("2025-03-10 15:00", tz="UTC")
    assert feats.loc[ts, "macro_ema50"] == pytest.approx(daily_ema50.loc[pd.Timestamp("2025-03-09", tz="UTC")])
    # The 23:00 bar closes exactly at midnight, when day D's candle has closed.
    ts_last = pd.Timestamp("2025-03-10 23:00", tz="UTC")
    assert feats.loc[ts_last, "macro_ema50"] == pytest.approx(daily_ema50.loc[pd.Timestamp("2025-03-10", tz="UTC")])


def test_macro_features_have_no_lookahead() -> None:
    hourly = _hourly()
    cutoff = pd.Timestamp("2025-03-01 12:00", tz="UTC")
    shocked = hourly.copy()
    shocked.loc[shocked.index > cutoff, ["open", "high", "low", "close"]] *= 2

    def build(bars):
        return add_macro_features(
            compute_features(bars), resample_bars(bars, "1D"), primary_bar_length=pd.Timedelta(hours=1)
        )

    base, alt = build(hourly), build(shocked)
    cols = MACRO_FEATURES + ["macro_ema50"]
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], alt.loc[:cutoff, cols])


def test_macro_alignment_flag_and_warmup() -> None:
    hourly = _hourly()
    feats = add_macro_features(
        compute_features(hourly), resample_bars(hourly, "1D"), primary_bar_length=pd.Timedelta(hours=1)
    )
    early = feats.loc[: "2025-02-15"]
    late = feats.loc["2025-03-15":].dropna(subset=["macro_ema50"])

    assert early["macro_trend_aligned"].isna().all()  # < 50 daily candles
    assert set(late["macro_trend_aligned"].unique()) <= {0.0, 1.0}
    expected = ((late["close"] > late["macro_ema50"]) & (late["macro_ema20_vs_ema50"] > 0)).astype(float)
    pd.testing.assert_series_equal(late["macro_trend_aligned"], expected, check_names=False)


def test_resample_bars_aggregates_ohlcv() -> None:
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    bars = pd.DataFrame({"open": range(48), "high": range(1, 49), "low": range(48),
                         "close": range(48), "volume": [1.0] * 48}, index=idx, dtype=float)

    daily = resample_bars(bars, "1D")

    assert daily["open"].tolist() == [0.0, 24.0]
    assert daily["high"].tolist() == [24.0, 48.0]
    assert daily["close"].tolist() == [23.0, 47.0]
    assert daily["volume"].tolist() == [24.0, 24.0]
