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

from typing import Any, Optional

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


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    if len(tr):
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


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


# ---------------------------------------------------------------------------
# Multi-timeframe (macro anchor) features
# ---------------------------------------------------------------------------

MACRO_FEATURES = ["macro_close_vs_ema50", "macro_ema20_vs_ema50", "macro_trend_aligned"]


def resample_bars(bars: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    """Aggregate OHLCV bars into a higher timeframe (labels = period start)."""
    agg = bars.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["close"])


def add_macro_features(
    features: pd.DataFrame,
    macro_bars: pd.DataFrame,
    *,
    primary_bar_length: pd.Timedelta,
    macro_bar_length: pd.Timedelta = pd.Timedelta(days=1),
) -> pd.DataFrame:
    """Attach higher-timeframe trend context to each primary bar without lookahead.

    A macro (e.g. daily) candle is only usable once it has closed, i.e. at
    ``macro_index + macro_bar_length``. Each primary bar is matched, as of its
    own close time (``index + primary_bar_length``), to the latest macro candle
    that had closed by then (``merge_asof`` backward). An hourly bar during
    day D therefore sees day D-1's EMA values, never day D's unfinished close.
    """
    out = features.copy()
    macro = _validate(macro_bars).sort_index()
    ema20, ema50 = ema(macro["close"], 20), ema(macro["close"], 50)
    def utc_ns(index) -> pd.DatetimeIndex:
        index = pd.DatetimeIndex(index)
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        return index.as_unit("ns")

    anchor = pd.DataFrame(
        {
            "macro_available_at": utc_ns(macro.index + macro_bar_length),
            "macro_ema20": ema20.to_numpy(),
            "macro_ema50": ema50.to_numpy(),
        }
    )
    primary = pd.DataFrame(
        {"primary_close_at": utc_ns(out.index + primary_bar_length), "_row": np.arange(len(out))}
    )
    merged = pd.merge_asof(
        primary.sort_values("primary_close_at"),
        anchor.sort_values("macro_available_at"),
        left_on="primary_close_at",
        right_on="macro_available_at",
        direction="backward",
    ).sort_values("_row")

    macro_ema20 = merged["macro_ema20"].to_numpy()
    macro_ema50 = merged["macro_ema50"].to_numpy()
    close = out["close"].to_numpy()
    out["macro_ema50"] = macro_ema50
    out["macro_close_vs_ema50"] = close / macro_ema50 - 1
    out["macro_ema20_vs_ema50"] = macro_ema20 / macro_ema50 - 1
    aligned = (close > macro_ema50) & (macro_ema20 > macro_ema50)
    out["macro_trend_aligned"] = np.where(np.isnan(macro_ema50) | np.isnan(macro_ema20), np.nan, aligned.astype(float))
    return out


# ---------------------------------------------------------------------------
# Intraday microstructure features (VWAP, relative volume, opening range)
# ---------------------------------------------------------------------------

INTRADAY_FEATURES = [
    "vwap_dist",
    "vwap_z",
    "rvol",
    "close_vs_orb_high",
    "close_vs_orb_low",
    "orb_range_pct",
    "close_vs_orb30_high",
    "close_vs_orb30_low",
    "orb30_range_pct",
    "minutes_from_open",
]
# Where the close sits against session VWAP (a label for attribution, not a model input).
VWAP_ZONES = ["below -2σ", "-2σ to -1σ", "-1σ to VWAP", "VWAP to +1σ", "+1σ to +2σ", "above +2σ"]


def vwap_zone(z: Any) -> Optional[str]:
    """The VWAP band a price sits in, from its distance to VWAP in sigma."""
    try:
        z = float(z)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(z):
        return None
    for edge, label in zip((-2, -1, 0, 1, 2), VWAP_ZONES):
        if z < edge:
            return label
    return VWAP_ZONES[-1]
_SESSION_OPEN_MIN = 9 * 60 + 30
_SESSION_CLOSE_MIN = 16 * 60


def add_intraday_features(
    features: pd.DataFrame,
    *,
    bar_length: pd.Timedelta,
    orb_minutes: int = 15,
    rvol_sessions: int = 20,
) -> pd.DataFrame:
    """Session-anchored VWAP (+1/2 sigma bands), RVOL and opening-range levels.

    All values at a bar use only that session's bars up to and including it
    (plus earlier sessions for RVOL), so they are known at the bar's close:

    - VWAP resets at 9:30 ET: cumulative sum(typical price x volume) /
      sum(volume) over regular-session bars; sigma is the volume-weighted
      standard deviation of typical price around it. Extended-hours bars get
      NaN (they aren't part of the anchored session).
    - RVOL: bar volume / mean volume of the same time-of-day bucket over the
      previous ``rvol_sessions`` sessions (today excluded).
    - Opening ranges: high/low of the first ``orb_minutes`` (default 15) and of
      the first 30 minutes; NaN until each window has closed. Not defined when
      bars are longer than the window.
    - vwap_dist is the VWAP ratio (close - VWAP) / VWAP; vwap_z the distance in
      sigma; vwap_zone labels the band (for attribution).
    """
    out = features.copy()
    local = out.index.tz_convert("America/New_York")
    minutes = local.hour * 60 + local.minute
    bar_minutes = bar_length / pd.Timedelta(minutes=1)
    regular = (minutes >= _SESSION_OPEN_MIN) & (minutes < _SESSION_CLOSE_MIN) & (local.weekday < 5)
    session = pd.Series(np.where(regular, local.date, None), index=out.index)

    typical = (out["high"] + out["low"] + out["close"]) / 3
    vol = out["volume"].where(regular)
    pv = (typical * vol).groupby(session).cumsum()
    pv2 = (typical ** 2 * vol).groupby(session).cumsum()
    cum_vol = vol.groupby(session).cumsum().replace(0.0, np.nan)
    vwap = pv / cum_vol
    sigma = np.sqrt((pv2 / cum_vol - vwap ** 2).clip(lower=0.0))
    out["vwap"] = vwap.where(regular)
    out["vwap_sigma"] = sigma.where(regular)
    for k in (1, 2):
        out[f"vwap_upper_{k}"] = out["vwap"] + k * out["vwap_sigma"]
        out[f"vwap_lower_{k}"] = out["vwap"] - k * out["vwap_sigma"]
    out["vwap_dist"] = (out["close"] - out["vwap"]) / out["vwap"]
    out["vwap_z"] = ((out["close"] - out["vwap"]) / out["vwap_sigma"].replace(0.0, np.nan)).where(regular)

    # Relative volume vs the same time-of-day bucket on previous sessions.
    bucket = pd.Series(np.where(regular, minutes, -1), index=out.index)
    prior_mean = (
        out["volume"].where(regular)
        .groupby(bucket)
        .transform(lambda s: s.shift(1).rolling(rvol_sessions, min_periods=max(3, rvol_sessions // 4)).mean())
    )
    out["rvol"] = (out["volume"] / prior_mean.replace(0.0, np.nan)).where(regular)

    # Opening ranges (15 min by default, and 30 min): only once the window has fully closed.
    out["orb_high"], out["orb_low"] = _opening_range(out, minutes, regular, session, bar_minutes, orb_minutes)
    out["close_vs_orb_high"] = out["close"] / out["orb_high"] - 1
    out["close_vs_orb_low"] = out["close"] / out["orb_low"] - 1
    out["orb_range_pct"] = (out["orb_high"] - out["orb_low"]) / out["orb_low"]
    out["orb30_high"], out["orb30_low"] = _opening_range(out, minutes, regular, session, bar_minutes, 30)
    out["close_vs_orb30_high"] = out["close"] / out["orb30_high"] - 1
    out["close_vs_orb30_low"] = out["close"] / out["orb30_low"] - 1
    out["orb30_range_pct"] = (out["orb30_high"] - out["orb30_low"]) / out["orb30_low"]

    out["minutes_from_open"] = pd.Series(minutes - _SESSION_OPEN_MIN, index=out.index, dtype=float).where(regular)
    out = out.replace([np.inf, -np.inf], np.nan)
    z = out["vwap_z"]  # label added after the numeric clean-up (text column)
    out["vwap_zone"] = pd.Series(
        np.select([z < -2, z < -1, z < 0, z < 1, z < 2, z >= 2], VWAP_ZONES, default=""), index=out.index
    ).where(z.notna())
    return out


def _opening_range(out, minutes, regular, session, bar_minutes: float, window: int):
    """(high, low) of the first ``window`` minutes of each session, exposed only
    on bars that start after the window has closed (no lookahead). NaN when bars
    are longer than the window."""
    if not 0 < bar_minutes <= window:
        return pd.Series(np.nan, index=out.index), pd.Series(np.nan, index=out.index)
    in_window = regular & (minutes + bar_minutes <= _SESSION_OPEN_MIN + window)
    after_window = regular & (minutes >= _SESSION_OPEN_MIN + window)
    high = out["high"].where(in_window).groupby(session).transform("max")
    low = out["low"].where(in_window).groupby(session).transform("min")
    return high.where(after_window), low.where(after_window)


def bars_from_candles(candles: list[dict]) -> pd.DataFrame:
    """Build an oldest-first OHLCV frame from the fetcher's candle dicts."""
    frame = pd.DataFrame(candles)
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    frame.index = pd.to_datetime(frame.pop("datetime"), utc=True)
    return frame[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").sort_index()
