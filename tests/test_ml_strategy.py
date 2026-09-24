from __future__ import annotations

from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from core.features import compute_features
from core.ml_strategy import FEATURE_COLUMNS, MLStrategy, feature_vector, load_artifact
from core.ml_training import (
    LabelParams,
    build_dataset,
    purged_time_split,
    synthetic_bars,
    train,
    triple_barrier_labels,
)
from core.news_sentiment import UNAVAILABLE, SentimentSnapshot

HOUR = timedelta(hours=1)


@pytest.fixture(scope="module")
def artifact():
    params = LabelParams(horizon=8, stop_atr_mult=1.0, target_atr_mult=1.5)
    datasets = {f"S{i}": build_dataset(synthetic_bars(900, seed=i), params, HOUR) for i in range(3)}
    return train(datasets, params, HOUR, threshold=0.55)


class FixedModel:
    """Stands in for a trained pipeline with a known probability."""

    def __init__(self, p: float):
        self.p = p
        self.seen = None

    def predict_proba(self, X):
        self.seen = X
        return np.array([[1 - self.p, self.p]])


def _fixed_artifact(p: float) -> dict:
    return {"pipeline": FixedModel(p), "label_params": {"stop_atr_mult": 2.0, "target_atr_mult": 4.0}}


@pytest.fixture(scope="module")
def features():
    return compute_features(synthetic_bars(300, seed=11))


def test_triple_barrier_labels() -> None:
    idx = pd.date_range("2026-01-05", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 103.5, 100.0, 100.0],
            "low": [100.0, 99.0, 99.5, 97.0, 100.0],
            "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )
    labels = triple_barrier_labels(frame, LabelParams(horizon=2, stop_atr_mult=2.0, target_atr_mult=3.0))

    # t0: bar2 high 103.5 >= 103 before any low <= 98 -> win
    # t1: bar2 hits target (103) -> win; t2: bar3 low 97 <= 98 -> loss
    assert labels.tolist()[:3] == [1.0, 1.0, 0.0]
    assert labels.iloc[3:].isna().all()  # not enough future bars to know


def test_same_bar_touching_both_barriers_counts_as_loss() -> None:
    idx = pd.date_range("2026-01-05", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"close": [100.0] * 3, "high": [100.0, 110.0, 100.0], "low": [100.0, 90.0, 100.0], "atr": [1.0] * 3},
        index=idx,
    )
    assert triple_barrier_labels(frame, LabelParams(horizon=1)).iloc[0] == 0.0


def test_purged_split_leaves_gap_before_test() -> None:
    idx = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    data = pd.DataFrame({"label": 0.0}, index=idx)

    train_rows, test_rows = purged_time_split(data, 0.2, purge=5 * HOUR)

    assert test_rows.index.min() - train_rows.index.max() > 5 * HOUR
    assert len(test_rows) == 20


def test_trained_artifact_roundtrip_and_model_mode(tmp_path, artifact, features) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)

    loaded = load_artifact(path)
    signal = MLStrategy(loaded, confidence_threshold=0.55).predict(features, UNAVAILABLE)

    assert loaded["feature_columns"] == FEATURE_COLUMNS
    assert {"auc", "base_rate", "signals_at_threshold"} <= set(loaded["metrics"])
    assert 0 < loaded["train_base_rate"] < 1
    assert signal.mode == "model"
    assert 0 <= signal.probability_up <= 1
    assert signal.signal in {"BUY", "SELL", "HOLD"}


def test_missing_or_mismatched_artifact_falls_back_to_heuristic(tmp_path, features) -> None:
    assert load_artifact(tmp_path / "absent.joblib") is None

    stale = tmp_path / "stale.joblib"
    joblib.dump({"version": 1, "feature_columns": ["rsi"], "pipeline": None}, stale)
    assert load_artifact(stale) is None

    strategy = MLStrategy(model_path=str(tmp_path / "absent.joblib"))
    signal = strategy.predict(features, SentimentSnapshot(0.5, 3, True))
    assert strategy.mode == "heuristic" and signal.mode == "heuristic"
    assert any(r.startswith("sentiment=") for r in signal.reasons)


@pytest.mark.parametrize(
    "p, expected_signal, expected_confidence",
    [(0.8, "BUY", 0.8), (0.1, "SELL", 0.9), (0.55, "HOLD", 0.55), (0.3, "HOLD", 0.3)],
)
def test_threshold_maps_probability_to_signal(features, p, expected_signal, expected_confidence) -> None:
    signal = MLStrategy(_fixed_artifact(p), confidence_threshold=0.6, exit_threshold=0.2).predict(
        features, UNAVAILABLE
    )

    assert signal.signal == expected_signal
    assert signal.confidence == pytest.approx(expected_confidence)


def test_exit_threshold_defaults_to_half_the_training_base_rate(monkeypatch, features) -> None:
    monkeypatch.delenv("ML_EXIT_THRESHOLD", raising=False)
    artifact = {**_fixed_artifact(0.1), "train_base_rate": 0.16}

    strategy = MLStrategy(artifact, confidence_threshold=0.6)

    # P=0.10 is below the 16% base rate but not below half of it: no churn exit.
    assert strategy.exit_threshold == pytest.approx(0.08)
    assert strategy.predict(features, UNAVAILABLE).signal == "HOLD"
    # Without a base rate (heuristic) the symmetric 1 - threshold applies.
    assert MLStrategy(model_path="/nonexistent", confidence_threshold=0.6).exit_threshold == pytest.approx(0.4)


def test_stops_use_training_atr_multiples(features) -> None:
    signal = MLStrategy(_fixed_artifact(0.9)).predict(features, UNAVAILABLE)
    latest = features.iloc[-1]

    assert signal.stop_distance == pytest.approx(2.0 * latest["atr"], abs=1e-4)
    assert signal.target_distance == pytest.approx(4.0 * latest["atr"], abs=1e-4)
    assert signal.stop_loss < latest["close"] < signal.take_profit


def test_missing_sentiment_is_passed_as_nan_to_model(features) -> None:
    artifact = _fixed_artifact(0.5)
    MLStrategy(artifact).predict(features, UNAVAILABLE)
    seen = artifact["pipeline"].seen

    assert list(seen.columns) == FEATURE_COLUMNS
    assert np.isnan(seen["sentiment_score"].iloc[0])
    assert seen["sentiment_available"].iloc[0] == 0.0


def test_no_buy_without_atr_history() -> None:
    short = compute_features(synthetic_bars(10, seed=2))  # ATR needs 14 bars

    signal = MLStrategy(_fixed_artifact(0.95)).predict(short, UNAVAILABLE)

    assert signal.signal == "HOLD"
    assert "ATR unavailable" in " ".join(signal.reasons)


def test_heuristic_follows_trend_and_sentiment() -> None:
    idx = pd.date_range("2025-06-02", periods=260, freq="h", tz="UTC")
    rising = 100 * np.exp(np.linspace(0, 0.3, 260)) * (1 + 0.002 * np.sin(np.arange(260)))
    bars = pd.DataFrame(
        {"open": rising * 0.999, "high": rising * 1.003, "low": rising * 0.996, "close": rising, "volume": 1e5},
        index=idx,
    )
    feats = compute_features(bars)
    strategy = MLStrategy(model_path="/nonexistent", confidence_threshold=0.6)

    bullish = strategy.predict(feats, SentimentSnapshot(0.8, 5, True))
    bearish_news = strategy.predict(feats, SentimentSnapshot(-0.9, 5, True))

    assert bullish.probability_up > 0.5
    assert bearish_news.probability_up < bullish.probability_up


def test_feature_vector_uses_last_completed_bar(features) -> None:
    row = feature_vector(features, SentimentSnapshot(0.25, 4, True))

    assert row.index[0] == features.index[-1]
    assert row["sentiment_score"].iloc[0] == 0.25
    assert row["rsi"].iloc[0] == features["rsi"].iloc[-1]


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        MLStrategy({"pipeline": FixedModel(0.5)}, confidence_threshold=0.4)
