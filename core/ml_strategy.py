#!/usr/bin/env python3
"""
ML inference: technical + sentiment features -> directional signal.

The model is a binary classifier estimating P(price reaches an ATR-based
take-profit before an ATR-based stop within N bars) — see
core/ml_training.py. It is long-only:

- P >= threshold          -> BUY  (confidence = P)
- P <= exit threshold     -> SELL (exit an existing long; confidence = 1 - P)
- otherwise               -> HOLD

P is not "probability the price goes up": target hits are rare by design
(a 2:1 reward/risk target might be hit first on only ~15% of bars), so a
low P usually just means "no edge". The exit threshold is therefore set
relative to the model's training base rate (half of it by default), not
at 1 - threshold. Brackets handle normal exits; SELL is for setups that
look clearly worse than average. Override with ML_EXIT_THRESHOLD.

When no trained artifact is available (or it doesn't match the current
feature set) a transparent trend/momentum/sentiment heuristic produces the
same output shape, flagged ``mode="heuristic"``. Missing sentiment is passed
to the model as NaN (LightGBM routes missing values natively) and simply
dropped from the heuristic.

Signals are advisory: the executor's RiskManager and bracket orders remain
the final gatekeepers.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.features import INTRADAY_FEATURES, MACRO_FEATURES, TECHNICAL_FEATURES
from core.news_sentiment import SENTIMENT_FEATURES, UNAVAILABLE, SentimentSnapshot
from core.regime import TRENDING_BULL

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = TECHNICAL_FEATURES + SENTIMENT_FEATURES
# Optional inputs a model may be trained with (--mtf; intraday timeframes).
KNOWN_FEATURES = FEATURE_COLUMNS + MACRO_FEATURES + INTRADAY_FEATURES
REGIME_POLICIES = ("off", "suppress", "penalty")
ARTIFACT_VERSION = 1
DEFAULT_MODEL_PATH = "models/ml_signal.joblib"


@dataclass(frozen=True)
class MLSignal:
    signal: str                   # BUY / SELL / HOLD
    confidence: float             # 0..1, confidence in `signal`
    probability_up: float         # P(take-profit before stop)
    mode: str                     # "model" or "heuristic"
    price: float
    atr: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    stop_distance: Optional[float]
    target_distance: Optional[float]
    sentiment: SentimentSnapshot
    reasons: List[str] = field(default_factory=list)
    regime: Optional[str] = None          # TRENDING_BULL / TRENDING_BEAR / CHOPPY / UNKNOWN
    macro_aligned: Optional[float] = None  # 1.0 if aligned with the daily trend, 0.0 if not
    gated_by: Optional[str] = None        # "regime" or "mtf" when a BUY was vetoed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "probability_up": round(self.probability_up, 4),
            "mode": self.mode,
            "price": self.price,
            "atr": self.atr,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "sentiment_score": self.sentiment.score if self.sentiment.available else None,
            "sentiment_articles": self.sentiment.article_count,
            "sentiment_available": self.sentiment.available,
            "reasons": list(self.reasons),
            "regime": self.regime,
            "macro_aligned": self.macro_aligned,
            "gated_by": self.gated_by,
        }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def feature_vector(
    features: pd.DataFrame,
    sentiment: SentimentSnapshot,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One-row model input from the latest completed bar plus sentiment."""
    if features.empty:
        raise ValueError("No completed bars to build features from")
    columns = columns or FEATURE_COLUMNS
    bar_columns = [c for c in columns if c not in SENTIMENT_FEATURES]
    missing = [c for c in bar_columns if c not in features.columns]
    if missing:
        raise ValueError(f"Features missing columns the model needs: {missing}")
    row = features.iloc[[-1]][bar_columns].copy()
    for name, value in sentiment.as_features().items():
        row[name] = value
    return row[columns].astype(float)


def load_artifact(path: str | os.PathLike) -> Optional[Dict[str, Any]]:
    """Load a model artifact written by scripts/train_model.py.

    joblib uses pickle, which can execute code: only load artifacts you
    trained yourself.
    """
    path = Path(path)
    if not path.exists():
        return None
    import joblib

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        logger.error(f"Could not load model artifact {path}: {exc}")
        return None
    if not isinstance(artifact, dict) or artifact.get("version") != ARTIFACT_VERSION:
        logger.error(f"Model artifact {path} has an unsupported format; retrain it")
        return None
    columns = list(artifact.get("feature_columns", []))
    if not columns or not set(columns) <= set(KNOWN_FEATURES) or not set(FEATURE_COLUMNS) <= set(columns):
        logger.error(f"Model artifact {path} was trained on different features; retrain it")
        return None
    return artifact


def _snapshot_from_row(row: pd.Series) -> SentimentSnapshot:
    """Inverse of SentimentSnapshot.as_features() for one feature row."""
    if not row.get("sentiment_available"):
        return UNAVAILABLE
    return SentimentSnapshot(
        score=float(row["sentiment_score"]),
        article_count=int(row["sentiment_article_count"]),
        available=True,
    )


class MLStrategy:
    def __init__(
        self,
        artifact: Optional[Dict[str, Any]] = None,
        *,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        stop_atr_mult: Optional[float] = None,
        target_atr_mult: Optional[float] = None,
        exit_threshold: Optional[float] = None,
        regime_policy: Optional[str] = None,
        regime_bump: Optional[float] = None,
        mtf_confirmation: Optional[bool] = None,
    ) -> None:
        if artifact is None:
            artifact = load_artifact(model_path or os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.artifact = artifact
        labels = (artifact or {}).get("label_params", {})
        # Use the same ATR multiples the model was trained to predict.
        self.stop_atr_mult = stop_atr_mult or labels.get("stop_atr_mult") or _env_float("ATR_STOP_MULT", 1.5)
        self.target_atr_mult = target_atr_mult or labels.get("target_atr_mult") or _env_float("ATR_TARGET_MULT", 3.0)
        threshold = confidence_threshold or _env_float("ML_CONFIDENCE_THRESHOLD", 0.6)
        if not 0.5 < threshold < 1:
            raise ValueError("ML_CONFIDENCE_THRESHOLD must be between 0.5 and 1")
        self.threshold = threshold

        exit_threshold = exit_threshold or _env_float("ML_EXIT_THRESHOLD", 0.0)
        if not exit_threshold:
            base_rate = (artifact or {}).get("train_base_rate")
            exit_threshold = 0.5 * base_rate if base_rate else 1 - threshold
        if not 0 < exit_threshold < threshold:
            raise ValueError("ML_EXIT_THRESHOLD must be between 0 and ML_CONFIDENCE_THRESHOLD")
        self.exit_threshold = exit_threshold

        # Entry gates (longs only; exits are never blocked).
        policy = (regime_policy or os.getenv("REGIME_FILTER", "penalty")).strip().lower()
        if policy not in REGIME_POLICIES:
            raise ValueError(f"REGIME_FILTER must be one of {REGIME_POLICIES}")
        self.regime_policy = policy
        self.regime_bump = regime_bump if regime_bump is not None else _env_float("REGIME_THRESHOLD_BUMP", 0.10)
        if mtf_confirmation is None:
            mtf_confirmation = os.getenv("MTF_CONFIRMATION", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.mtf_confirmation = mtf_confirmation
        self.feature_columns = list((artifact or {}).get("feature_columns") or FEATURE_COLUMNS)

    def without_model(self) -> "MLStrategy":
        """Same thresholds and gates, heuristic probabilities."""
        clone = copy.copy(self)
        clone.artifact = None
        clone.feature_columns = list(FEATURE_COLUMNS)
        return clone

    @property
    def uses_macro(self) -> bool:
        """True when the model or the MTF gate needs daily-timeframe features."""
        return self.mtf_confirmation or any(c in MACRO_FEATURES for c in self.feature_columns)

    @property
    def mode(self) -> str:
        return "model" if self.artifact else "heuristic"

    def predict(self, features: pd.DataFrame, sentiment: SentimentSnapshot) -> MLSignal:
        """``features`` is compute_features() output; the last row must be a closed bar."""
        latest = features.iloc[-1]
        reasons: List[str] = []
        if self.artifact:
            probability_up = float(
                self.artifact["pipeline"].predict_proba(
                    feature_vector(features, sentiment, self.feature_columns)
                )[0, 1]
            )
            reasons.append(f"model P(target before stop)={probability_up:.2f}")
        else:
            probability_up = self._heuristic_probability(latest, sentiment, reasons)
        return self.decide(probability_up, latest, sentiment, reasons)

    def probabilities(self, features: pd.DataFrame, sentiment: pd.DataFrame) -> np.ndarray:
        """P(target before stop) for every row at once (backtesting).

        ``sentiment`` has SENTIMENT_FEATURES columns aligned to ``features``.
        Equivalent to calling predict() on each prefix because every feature
        is causal.
        """
        if self.artifact:
            bar_columns = [c for c in self.feature_columns if c not in SENTIMENT_FEATURES]
            vectors = features[bar_columns].join(sentiment[SENTIMENT_FEATURES])[self.feature_columns]
            return self.artifact["pipeline"].predict_proba(vectors.astype(float))[:, 1]
        return np.array(
            [
                self._heuristic_probability(row, _snapshot_from_row(sent), [])
                for (_, row), (_, sent) in zip(features.iterrows(), sentiment.iterrows())
            ]
        )

    def decide(
        self,
        probability_up: float,
        latest: pd.Series,
        sentiment: SentimentSnapshot,
        reasons: Optional[List[str]] = None,
    ) -> MLSignal:
        """Turn a probability for one closed bar into a signal with ATR levels."""
        reasons = list(reasons or [])
        price = float(latest["close"])
        atr = float(latest.get("atr", float("nan")))

        if not sentiment.available:
            reasons.append("sentiment unavailable; technical features only")

        if probability_up >= self.threshold:
            signal, confidence = "BUY", probability_up
        elif probability_up <= self.exit_threshold:
            signal, confidence = "SELL", 1 - probability_up
            reasons.append(f"at or below exit threshold {self.exit_threshold:.2f}")
        else:
            signal, confidence = "HOLD", probability_up
            reasons.append(f"below confidence threshold {self.threshold:.2f}")

        regime = latest.get("regime") if isinstance(latest.get("regime"), str) else None
        macro_aligned = latest.get("macro_trend_aligned")
        macro_aligned = float(macro_aligned) if macro_aligned is not None and pd.notna(macro_aligned) else None
        gated_by = None
        if signal == "BUY":
            gated_by = self._entry_gate(probability_up, regime, macro_aligned, reasons)
            if gated_by:
                signal, confidence = "HOLD", probability_up

        stop_distance = target_distance = stop_loss = take_profit = None
        if math.isfinite(atr) and atr > 0:
            stop_distance = round(self.stop_atr_mult * atr, 4)
            target_distance = round(self.target_atr_mult * atr, 4)
            stop_loss = round(price - stop_distance, 2)
            take_profit = round(price + target_distance, 2)
        elif signal == "BUY":
            signal = "HOLD"
            reasons.append("ATR unavailable (not enough history); no entry")

        return MLSignal(
            signal=signal,
            confidence=float(confidence),
            probability_up=float(probability_up),
            mode=self.mode,
            price=price,
            atr=atr,
            stop_loss=stop_loss,
            take_profit=take_profit,
            stop_distance=stop_distance,
            target_distance=target_distance,
            sentiment=sentiment,
            reasons=reasons,
            regime=regime,
            macro_aligned=macro_aligned,
            gated_by=gated_by,
        )

    def _entry_gate(
        self,
        probability_up: float,
        regime: Optional[str],
        macro_aligned: Optional[float],
        reasons: List[str],
    ) -> Optional[str]:
        """Return "regime"/"mtf" if a long entry should be vetoed, else None."""
        if regime is not None and self.regime_policy != "off" and regime != TRENDING_BULL:
            if self.regime_policy == "suppress":
                reasons.append(f"{regime} regime: long entries suppressed")
                return "regime"
            required = min(0.99, self.threshold + self.regime_bump)
            if probability_up < required:
                reasons.append(f"{regime} regime requires P >= {required:.2f}")
                return "regime"
            reasons.append(f"{regime} regime: cleared raised threshold {required:.2f}")
        if self.mtf_confirmation and macro_aligned != 1.0:
            reasons.append(
                "not aligned with the daily trend (needs close > daily EMA50 and daily EMA20 > EMA50)"
                if macro_aligned == 0.0 else "daily trend unavailable; no MTF confirmation"
            )
            return "mtf"
        return None

    @staticmethod
    def _heuristic_probability(latest: pd.Series, sentiment: SentimentSnapshot, reasons: List[str]) -> float:
        """Map trend/momentum/sentiment votes to a pseudo-probability in (0, 1)."""
        votes: List[float] = []

        def vote(name: str, value: float, weight: float = 1.0) -> None:
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return
            votes.append(weight * float(np.clip(value, -1, 1)))
            reasons.append(f"{name}={value:+.2f}")

        close, ema50, ema200 = latest.get("close"), latest.get("ema50"), latest.get("ema200")
        if pd.notna(ema50) and pd.notna(ema200):
            trend = 1.0 if close > ema50 > ema200 else -1.0 if close < ema50 < ema200 else 0.0
            vote("trend", trend, 1.5)
        if pd.notna(latest.get("macd_hist_pct")) and pd.notna(latest.get("atr_pct")) and latest["atr_pct"] > 0:
            vote("macd", latest["macd_hist_pct"] / latest["atr_pct"])
        if pd.notna(latest.get("rsi")):
            # Reward momentum (50-70), fade extremes.
            r = latest["rsi"]
            vote("rsi", (r - 50) / 20 if r <= 70 else (80 - r) / 10)
        if sentiment.available and sentiment.article_count > 0:
            vote("sentiment", sentiment.score, 1.0)

        if not votes:
            reasons.append("no usable indicators yet")
            return 0.5
        score = sum(votes) / (len(votes) + 0.5)  # shrink toward neutral
        return float(0.5 + 0.5 * np.clip(score, -1, 1))
