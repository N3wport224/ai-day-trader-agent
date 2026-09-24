#!/usr/bin/env python3
"""
One place that turns raw bars into the full feature frame, so live inference,
training and backtesting can't drift apart:

    compute_features  -> add_regime_columns -> add_macro_features (optional)

Macro (higher-timeframe) anchor: daily bars for intraday timeframes, weekly
bars (resampled from the dailies) when the primary timeframe is already 1Day.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.features import MACRO_FEATURES, add_macro_features, compute_features, resample_bars
from core.market_history import bar_length
from core.regime import add_regime_columns

MACRO_TIMEFRAME = {"15Min": "1Day", "1Hour": "1Day", "1Day": "1Week"}
MACRO_LENGTH = {"1Day": pd.Timedelta(days=1), "1Week": pd.Timedelta(days=7)}


def macro_timeframe(timeframe: str) -> str:
    return MACRO_TIMEFRAME[timeframe]


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
) -> pd.DataFrame:
    features = add_regime_columns(compute_features(bars), adx_threshold=adx_threshold)
    if macro_bars is not None and len(macro_bars):
        return add_macro_features(
            features,
            macro_bars,
            primary_bar_length=pd.Timedelta(bar_length(timeframe)),
            macro_bar_length=MACRO_LENGTH[macro_timeframe(timeframe)],
        )
    for column in MACRO_FEATURES:
        features[column] = np.nan
    return features
