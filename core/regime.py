#!/usr/bin/env python3
"""
Market regime and volatility classification.

Each bar is labelled from information available at its close:

    TRENDING_BULL   trend present and +DI > -DI
    TRENDING_BEAR   trend present and -DI > +DI
    CHOPPY          no trend (sideways / mean-reverting)
    UNKNOWN         not enough history for ADX yet

"Trend present" combines ADX with a normalised volatility read:
    ADX >= ADX_TREND_THRESHOLD                                  (established trend)
    or ADX >= threshold - 5 and ATR% percentile >= 0.5         (emerging trend with
                                                                 expanding volatility)

ATR% (ATR / close) is ranked against its own trailing ``window`` bars, so
"high volatility" means high for this symbol recently, not in absolute terms.
All indicators are recursive/trailing, so the labels are causal.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from core.features import true_range

TRENDING_BULL = "TRENDING_BULL"
TRENDING_BEAR = "TRENDING_BEAR"
CHOPPY = "CHOPPY"
UNKNOWN = "UNKNOWN"
REGIMES = (TRENDING_BULL, TRENDING_BEAR, CHOPPY, UNKNOWN)

REGIME_COLUMNS = ["adx", "plus_di", "minus_di", "atr_pctile", "atr_pct_median", "high_volatility", "regime"]


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX with +DI / -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_w = _wilder(true_range(high, low, close), period).replace(0.0, np.nan)
    plus_di = 100 * _wilder(plus_dm, period) / atr_w
    minus_di = 100 * _wilder(minus_dm, period) / atr_w
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame({"adx": _wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di})


def adx_threshold_from_env() -> float:
    try:
        return float(os.getenv("ADX_TREND_THRESHOLD", "25"))
    except ValueError:
        return 25.0


def classify(
    adx_value: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    atr_pctile: pd.Series,
    adx_threshold: float,
) -> pd.Series:
    strong = adx_value >= adx_threshold
    emerging = (adx_value >= adx_threshold - 5) & (atr_pctile >= 0.5)
    trending = strong | emerging
    bullish = plus_di > minus_di
    labels = np.where(trending & bullish, TRENDING_BULL, np.where(trending, TRENDING_BEAR, CHOPPY))
    return pd.Series(np.where(adx_value.isna(), UNKNOWN, labels), index=adx_value.index)


def add_regime_columns(
    features: pd.DataFrame,
    *,
    adx_threshold: float | None = None,
    period: int = 14,
    window: int = 100,
) -> pd.DataFrame:
    """Return ``features`` plus REGIME_COLUMNS (needs high/low/close and atr_pct)."""
    out = features.copy()
    adx_threshold = adx_threshold if adx_threshold is not None else adx_threshold_from_env()
    ind = adx(out["high"], out["low"], out["close"], period)
    out["adx"], out["plus_di"], out["minus_di"] = ind["adx"], ind["plus_di"], ind["minus_di"]

    atr_pct = out["atr_pct"] if "atr_pct" in out else None
    if atr_pct is None:
        raise ValueError("add_regime_columns needs compute_features() output (atr_pct)")
    min_periods = max(20, window // 2)
    out["atr_pctile"] = atr_pct.rolling(window, min_periods=min_periods).rank(pct=True)
    out["atr_pct_median"] = atr_pct.rolling(window, min_periods=min_periods).median()
    out["high_volatility"] = (out["atr_pctile"] >= 0.9).astype(float).where(out["atr_pctile"].notna())
    out["regime"] = classify(out["adx"], out["plus_di"], out["minus_di"], out["atr_pctile"], adx_threshold)
    return out
