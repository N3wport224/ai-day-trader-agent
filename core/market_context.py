#!/usr/bin/env python3
"""
Market context: how the broad market (SPY by default) is doing, and how the
symbol is doing relative to it.

Most single stocks move with the index, so a long setup in a falling tape
starts with a headwind, and stocks outperforming the index while it rises
("relative strength") are the classic intraday long candidates. Features
(all causal, aligned on the symbol's own bar timestamps):

  rs_12, rs_48     symbol log return minus market log return over 12 / 48 bars
  mkt_ret_12       market log return over 12 bars
  mkt_trend        market close / market EMA50 - 1
  mkt_vwap_dist    market distance from its session VWAP in ATRs (intraday only)
  market_ok        1 if the tape is not weak, 0 if it is, NaN if unknown
                   (weak = below EMA50 AND below session VWAP; on daily bars,
                   below EMA50 AND a negative 12-bar return). Used by the
                   optional market filter gate, not by the model.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from core.features import add_intraday_features, compute_features

MARKET_FEATURES = ["rs_12", "rs_48", "mkt_ret_12", "mkt_trend", "mkt_vwap_dist"]
MARKET_COLUMNS = MARKET_FEATURES + ["market_ok"]


def market_symbol() -> str:
    return os.getenv("MARKET_SYMBOL", "SPY").strip().upper() or "SPY"


def add_market_features(
    features: pd.DataFrame,
    market_bars: Optional[pd.DataFrame],
    *,
    bar_length: Optional[pd.Timedelta] = None,
    intraday: bool = False,
) -> pd.DataFrame:
    """Add MARKET_COLUMNS to ``features`` (NaN when no market bars)."""
    out = features.copy()
    if market_bars is None or len(market_bars) < 60:
        for column in MARKET_COLUMNS:
            out[column] = np.nan
        return out

    market = compute_features(market_bars)
    if intraday and bar_length is not None:
        market = add_intraday_features(market, bar_length=bar_length)
    mkt = pd.DataFrame(index=market.index)
    log_close = np.log(market["close"])
    mkt["mkt_ret_12"] = log_close - log_close.shift(12)
    mkt["mkt_ret_48"] = log_close - log_close.shift(48)
    mkt["mkt_trend"] = market["close"] / market["ema50"] - 1
    mkt["mkt_vwap_dist"] = market["vwap_dist"] if "vwap_dist" in market else np.nan

    # Same bar grid as the symbol; a missing market bar uses the previous one
    # (at most 2 bars back), which is causal.
    aligned = mkt.reindex(mkt.index.union(out.index)).ffill(limit=2).reindex(out.index)
    sym_log = np.log(out["close"])
    out["rs_12"] = (sym_log - sym_log.shift(12)) - aligned["mkt_ret_12"]
    out["rs_48"] = (sym_log - sym_log.shift(48)) - aligned["mkt_ret_48"]
    out["mkt_ret_12"] = aligned["mkt_ret_12"]
    out["mkt_trend"] = aligned["mkt_trend"]
    out["mkt_vwap_dist"] = aligned["mkt_vwap_dist"]

    below_trend = out["mkt_trend"] < 0
    second = out["mkt_vwap_dist"] < 0 if intraday else out["mkt_ret_12"] < 0
    known = out["mkt_trend"].notna() & (out["mkt_vwap_dist"].notna() if intraday else out["mkt_ret_12"].notna())
    out["market_ok"] = np.where(known, (~(below_trend & second)).astype(float), np.nan)
    return out
