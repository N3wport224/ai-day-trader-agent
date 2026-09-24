#!/usr/bin/env python3
"""
Technical feature engineering for the ML strategy.

``compute_features`` turns an OHLCV frame (oldest bar first) into one feature
row per bar. Every feature is causal: the value at row ``t`` depends only on
bars ``0..t``, i.e. what was known at that candle's close. Nothing uses
``shift(-n)``, centered windows, or full-sample statistics, so the same code
serves both training and live inference without lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Scale-free columns fed to the model (raw prices like EMA levels are kept in
# the frame for stops/diagnostics but are not model inputs).
TECHNICAL_FEATURES = [
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "gap_pct",
    "close_vs_ema20",
    "close_vs_ema50",
    "close_vs_ema200",
    "ema20_vs_ema50",
    "ema50_vs_ema200",
    "ema20_slope",
    "rsi",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "atr_pct",
    "return_1",
    "return_5",
    "return_20",
    "volume_z20",
]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")
    frame = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    if isinstance(frame.index, pd.DatetimeIndex) and not frame.index.is_monotonic_increasing:
        raise ValueError("Bars must be sorted oldest-first")
    return frame


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (recursive, so strictly causal)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in 0..100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses in the window: RSI is 100 (or 50 when flat), not NaN.
    out = out.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    true_range.iloc[0] = high.iloc[0] - low.iloc[0] if len(true_range) else np.nan
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the input OHLCV columns plus all technical features."""
    frame = _validate(df)
    o, h, l, c, v = (frame[col] for col in OHLCV_COLUMNS)
    out = frame.copy()

    # Candlestick anatomy. A zero-range bar (h == l) has no body or wicks.
    rng = (h - l).replace(0.0, np.nan)
    out["body_ratio"] = ((c - o) / rng).fillna(0.0)
    out["upper_wick_ratio"] = ((h - np.maximum(o, c)) / rng).fillna(0.0)
    out["lower_wick_ratio"] = ((np.minimum(o, c) - l) / rng).fillna(0.0)
    prev_close = c.shift(1)
    out["gap_pct"] = (o - prev_close) / prev_close * 100

    # Trend.
    for span in (20, 50, 200):
        out[f"ema{span}"] = ema(c, span)
        out[f"close_vs_ema{span}"] = c / out[f"ema{span}"] - 1
    out["ema20_vs_ema50"] = out["ema20"] / out["ema50"] - 1
    out["ema50_vs_ema200"] = out["ema50"] / out["ema200"] - 1
    out["ema20_slope"] = out["ema20"].pct_change(5, fill_method=None)

    # Momentum.
    out["rsi"] = rsi(c)
    line, signal_line, hist = macd(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = line, signal_line, hist
    out["macd_pct"] = line / c
    out["macd_signal_pct"] = signal_line / c
    out["macd_hist_pct"] = hist / c

    # Volatility.
    out["atr"] = atr(h, l, c)
    out["atr_pct"] = out["atr"] / c

    # Returns and volume (trailing windows only).
    for n in (1, 5, 20):
        out[f"return_{n}"] = c.pct_change(n, fill_method=None)
    vol_mean = v.rolling(20, min_periods=20).mean()
    vol_std = v.rolling(20, min_periods=20).std(ddof=0)
    out["volume_z20"] = ((v - vol_mean) / vol_std.replace(0.0, np.nan)).fillna(0.0).where(vol_mean.notna())

    return out.replace([np.inf, -np.inf], np.nan)


def bars_from_candles(candles: list[dict]) -> pd.DataFrame:
    """Build an oldest-first OHLCV frame from the fetcher's candle dicts."""
    frame = pd.DataFrame(candles)
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    frame.index = pd.to_datetime(frame.pop("datetime"), utc=True)
    return frame[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").sort_index()
