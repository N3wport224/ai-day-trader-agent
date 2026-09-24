from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import compute_features
from core.ml_strategy import MLStrategy
from core.ml_training import synthetic_bars
from core.news_sentiment import UNAVAILABLE
from core.regime import (
    CHOPPY,
    TRENDING_BEAR,
    TRENDING_BULL,
    UNKNOWN,
    add_regime_columns,
    adx,
    classify,
)


def _trend_bars(direction: float, n: int = 300, noise: float = 0.0015, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(direction + rng.normal(0, noise, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.002, n)) * close
    idx = pd.date_range("2025-01-02", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
         "close": close, "volume": 1e5},
        index=idx,
    )


def _choppy_bars(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 0.5, n)  # i.i.d. noise around a flat level: no persistence
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.3, n))
    idx = pd.date_range("2025-01-02", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
         "close": close, "volume": 1e5},
        index=idx,
    )


def _regimes(bars: pd.DataFrame) -> pd.Series:
    return add_regime_columns(compute_features(bars), adx_threshold=25)["regime"]


def test_classifies_bull_bear_and_choppy_markets() -> None:
    assert _regimes(_trend_bars(+0.006)).iloc[-100:].value_counts().idxmax() == TRENDING_BULL
    assert _regimes(_trend_bars(-0.006)).iloc[-100:].value_counts().idxmax() == TRENDING_BEAR
    assert _regimes(_choppy_bars()).iloc[-100:].value_counts().idxmax() == CHOPPY


def test_warmup_is_unknown_not_a_guess() -> None:
    regimes = _regimes(_trend_bars(+0.006))
    assert (regimes.iloc[:14] == UNKNOWN).all()


def test_adx_is_bounded_and_dis_follow_direction() -> None:
    bars = _trend_bars(+0.006)
    ind = adx(bars["high"], bars["low"], bars["close"])
    tail = ind.iloc[-100:]

    assert tail["adx"].between(0, 100).all()
    assert (tail["plus_di"] > tail["minus_di"]).mean() > 0.8


def test_emerging_trend_needs_volatility_expansion() -> None:
    adx_v = pd.Series([22.0, 22.0, 30.0, 10.0])
    plus = pd.Series([30.0, 30.0, 10.0, 30.0])
    minus = pd.Series([10.0, 10.0, 30.0, 10.0])
    pctile = pd.Series([0.8, 0.2, 0.5, 0.9])

    labels = classify(adx_v, plus, minus, pctile, adx_threshold=25).tolist()

    assert labels == [TRENDING_BULL, CHOPPY, TRENDING_BEAR, CHOPPY]


def test_regime_columns_are_causal() -> None:
    bars = synthetic_bars(n=400, seed=3)
    full = add_regime_columns(compute_features(bars))
    for t in (120, 250, 399):
        prefix = add_regime_columns(compute_features(bars.iloc[: t + 1]))
        pd.testing.assert_series_equal(prefix.iloc[-1], full.iloc[t], check_names=False)


# ---------------------------------------------------------------------------
# Entry gating in MLStrategy.decide
# ---------------------------------------------------------------------------

@pytest.fixture
def latest_row():
    feats = add_regime_columns(compute_features(synthetic_bars(300, seed=2)))
    return feats.iloc[-1].copy()


def _decide(row, regime, p=0.75, **kwargs):
    row = row.copy()
    row["regime"] = regime
    strategy = MLStrategy(artifact=None, model_path="/none", confidence_threshold=0.6, **kwargs)
    return strategy.decide(p, row, UNAVAILABLE)


@pytest.mark.parametrize("regime", [CHOPPY, TRENDING_BEAR, UNKNOWN])
def test_suppress_policy_blocks_longs_outside_bull_regime(latest_row, regime) -> None:
    signal = _decide(latest_row, regime, regime_policy="suppress")

    assert signal.signal == "HOLD"
    assert signal.gated_by == "regime"
    assert signal.regime == regime


def test_bull_regime_allows_entries(latest_row) -> None:
    signal = _decide(latest_row, TRENDING_BULL, regime_policy="suppress")

    assert signal.signal == "BUY" and signal.gated_by is None


def test_penalty_policy_raises_threshold_in_adverse_regimes(latest_row) -> None:
    blocked = _decide(latest_row, CHOPPY, p=0.65, regime_policy="penalty", regime_bump=0.10)
    cleared = _decide(latest_row, CHOPPY, p=0.72, regime_policy="penalty", regime_bump=0.10)

    assert blocked.signal == "HOLD" and blocked.gated_by == "regime"
    assert "requires P >= 0.70" in " ".join(blocked.reasons)
    assert cleared.signal == "BUY"


def test_gates_never_block_exits(latest_row) -> None:
    signal = _decide(latest_row, TRENDING_BEAR, p=0.05, regime_policy="suppress", mtf_confirmation=True,
                     exit_threshold=0.2)

    assert signal.signal == "SELL" and signal.gated_by is None


def test_mtf_gate_requires_daily_alignment(latest_row) -> None:
    aligned, misaligned, missing = latest_row.copy(), latest_row.copy(), latest_row.copy()
    aligned["macro_trend_aligned"], misaligned["macro_trend_aligned"] = 1.0, 0.0
    missing["macro_trend_aligned"] = np.nan

    assert _decide(aligned, TRENDING_BULL, mtf_confirmation=True).signal == "BUY"
    for row in (misaligned, missing):
        signal = _decide(row, TRENDING_BULL, mtf_confirmation=True)
        assert signal.signal == "HOLD" and signal.gated_by == "mtf"


def test_regime_policy_off_ignores_regime(latest_row) -> None:
    assert _decide(latest_row, TRENDING_BEAR, regime_policy="off").signal == "BUY"


def test_invalid_regime_policy_rejected() -> None:
    with pytest.raises(ValueError):
        MLStrategy(artifact=None, model_path="/none", regime_policy="yolo")
