#!/usr/bin/env python3
"""
Offline dataset building, labeling and training for core/ml_strategy.py.

Labels use a triple-barrier rule on each completed bar t:
  entry = close[t], stop = entry - k_stop*ATR[t], target = entry + k_target*ATR[t]
  label = 1 if a later bar (t+1 .. t+horizon) reaches the target before the
  stop, else 0. If one bar touches both, it counts as a loss (conservative).
Labels look forward by design; features never do. The chronological split
purges the last `horizon` bars before the test period so no training label
depends on test-period prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.features import compute_features
from core.ml_strategy import ARTIFACT_VERSION, FEATURE_COLUMNS
from core.news_sentiment import SENTIMENT_FEATURES, ScoredArticle, rolling_sentiment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelParams:
    horizon: int = 12
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 3.0


def triple_barrier_labels(features: pd.DataFrame, params: LabelParams) -> pd.Series:
    """1.0 if target hit before stop within the horizon, 0.0 otherwise, NaN if unknowable."""
    close = features["close"].to_numpy()
    high = features["high"].to_numpy()
    low = features["low"].to_numpy()
    atr = features["atr"].to_numpy()
    n = len(features)
    labels = np.full(n, np.nan)

    for t in range(n - params.horizon):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        stop = close[t] - params.stop_atr_mult * atr[t]
        target = close[t] + params.target_atr_mult * atr[t]
        outcome = 0.0
        for j in range(t + 1, t + params.horizon + 1):
            if low[j] <= stop:
                break
            if high[j] >= target:
                outcome = 1.0
                break
        labels[t] = outcome
    return pd.Series(labels, index=features.index, name="label")


def build_dataset(
    bars: pd.DataFrame,
    params: LabelParams,
    bar_length: timedelta,
    scored_news: Optional[Sequence[ScoredArticle]] = None,
    half_life: timedelta = timedelta(hours=6),
) -> pd.DataFrame:
    """Features + sentiment (as of each bar's close) + label for one symbol."""
    features = compute_features(bars)
    if scored_news is None:
        sentiment = pd.DataFrame(np.nan, index=features.index, columns=SENTIMENT_FEATURES)
        sentiment["sentiment_available"] = 0.0
    else:
        close_times = [ts.to_pydatetime() + bar_length for ts in features.index]
        sentiment = rolling_sentiment(close_times, scored_news, half_life=half_life)
        sentiment.index = features.index
    data = features.join(sentiment)
    data["label"] = triple_barrier_labels(features, params)
    return data.dropna(subset=["label"])


def purged_time_split(
    data: pd.DataFrame, test_fraction: float, purge: timedelta
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on bar time with a purge gap before the test period."""
    times = data.index.sort_values()
    cutoff = times[int(len(times) * (1 - test_fraction))]
    train = data[data.index < cutoff - purge]
    test = data[data.index >= cutoff]
    return train, test


def make_pipeline(random_state: int = 42):
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=40,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=random_state,
                    verbose=-1,
                ),
            )
        ]
    )


def evaluate(pipeline, test: pd.DataFrame, threshold: float, params: LabelParams) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y = test["label"].to_numpy()
    proba = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    taken = proba >= threshold
    win_rate = float(y[taken].mean()) if taken.any() else float("nan")
    # Expected R per trade if every barrier outcome were hit exactly: win = +k_t/k_s R, loss <= -1 R.
    reward_ratio = params.target_atr_mult / params.stop_atr_mult
    expectancy = win_rate * reward_ratio - (1 - win_rate) if taken.any() else float("nan")
    return {
        "test_rows": int(len(test)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        "signals_at_threshold": int(taken.sum()),
        "win_rate_at_threshold": win_rate,
        "approx_expectancy_r": float(expectancy),
    }


def train(
    datasets: Dict[str, pd.DataFrame],
    params: LabelParams,
    bar_length: timedelta,
    *,
    threshold: float = 0.6,
    test_fraction: float = 0.2,
    timeframe: str = "1Hour",
) -> Dict:
    """Train on all symbols' rows and return a ready-to-save artifact dict."""
    data = pd.concat(datasets.values()).sort_index()
    if len(data) < 200:
        raise ValueError(f"Only {len(data)} labeled rows; fetch more history")
    if data["label"].nunique() < 2:
        raise ValueError("Labels contain a single class; widen the horizon or history")

    train_rows, test_rows = purged_time_split(data, test_fraction, purge=params.horizon * bar_length)
    pipeline = make_pipeline()
    pipeline.fit(train_rows[FEATURE_COLUMNS], train_rows["label"])
    metrics = evaluate(pipeline, test_rows, threshold, params)
    metrics["train_rows"] = int(len(train_rows))

    # Refit on everything for deployment; metrics above are out-of-sample.
    final = make_pipeline()
    final.fit(data[FEATURE_COLUMNS], data["label"])

    return {
        "version": ARTIFACT_VERSION,
        "pipeline": final,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_params": {
            "horizon": params.horizon,
            "stop_atr_mult": params.stop_atr_mult,
            "target_atr_mult": params.target_atr_mult,
        },
        "timeframe": timeframe,
        "symbols": sorted(datasets),
        "uses_sentiment": bool(data["sentiment_available"].max() > 0),
        "train_base_rate": float(data["label"].mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }


def synthetic_bars(n: int = 1500, seed: int = 7, start: str = "2025-01-02 14:30") -> pd.DataFrame:
    """Random-walk OHLCV bars with mild trend regimes, for demos and tests."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0015, n // 100 + 1), 100)[:n]
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(10_000, 100_000, n).astype(float)
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
