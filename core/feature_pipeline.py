#!/usr/bin/env python3
"""
One place that turns raw bars into the full feature frame, so live inference,
training and backtesting can't drift apart:

    compute_features -> add_regime_columns -> add_intraday_features (1m/5m/15m)
                     -> add_macro_features (optional) -> add_market_features (optional)

Macro (higher-timeframe) anchor: daily bars for intraday timeframes, weekly
bars (resampled from the dailies) when the primary timeframe is already 1Day.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from core.features import (
    INTRADAY_FEATURES,
    MACRO_FEATURES,
    add_intraday_features,
    add_macro_features,
    compute_features,
    resample_bars,
)
from core.market_context import add_market_features
from core.market_history import bar_length, is_intraday, normalize_timeframe
from core.regime import add_regime_columns

MACRO_TIMEFRAME = {"1Min": "1Day", "5Min": "1Day", "15Min": "1Day", "1Hour": "1Day", "1Day": "1Week"}
MACRO_LENGTH = {"1Day": pd.Timedelta(days=1), "1Week": pd.Timedelta(days=7)}


def macro_timeframe(timeframe: str) -> str:
    return MACRO_TIMEFRAME[normalize_timeframe(timeframe)]


def macro_from_primary(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Derive the macro anchor by resampling the primary bars (daily -> weekly,
    or intraday -> daily when no separate daily history is available)."""
    rule = "7D" if macro_timeframe(timeframe) == "1Week" else "1D"
    return resample_bars(bars, rule)


def build_feature_frame(
    bars: pd.DataFrame,
    timeframe: str,
    *,
    macro_bars: Optional[pd.DataFrame] = None,
    adx_threshold: Optional[float] = None,
    market_bars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """``market_bars``: the index (SPY) on the same timeframe, for relative
    strength / market-context features (NaN without it)."""
    timeframe = normalize_timeframe(timeframe)
    features = add_regime_columns(compute_features(bars), adx_threshold=adx_threshold)
    if is_intraday(timeframe):
        features = add_intraday_features(
            features,
            bar_length=pd.Timedelta(bar_length(timeframe)),
            orb_minutes=int(os.getenv("ORB_MINUTES", "15")),
            rvol_sessions=int(os.getenv("RVOL_LOOKBACK_SESSIONS", "20")),
        )
    else:
        for column in INTRADAY_FEATURES:
            features[column] = np.nan
    if macro_bars is not None and len(macro_bars):
        features = add_macro_features(
            features,
            macro_bars,
            primary_bar_length=pd.Timedelta(bar_length(timeframe)),
            macro_bar_length=MACRO_LENGTH[macro_timeframe(timeframe)],
        )
    else:
        for column in MACRO_FEATURES:
            features[column] = np.nan
    return add_market_features(
        features, market_bars, bar_length=pd.Timedelta(bar_length(timeframe)), intraday=is_intraday(timeframe)
    )
