#!/usr/bin/env python3
"""
Adapter that runs the ML pipeline and returns the analysis dict that
TradingWorkflow already understands, so the ML strategy plugs into the
existing path unchanged:

    MLSignalEngine -> TradingWorkflow -> AlpacaExecutor.submit
                                          -> RiskManager.check_order
                                          -> bracket order

Sizing here is only a request: position caps, buying power, the daily loss
limit and the daily entry limit are still enforced by the RiskManager.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import pandas as pd

from core.features import compute_features
from core.market_history import get_history
from core.ml_strategy import MLSignal, MLStrategy
from core.news_sentiment import SentimentSnapshot, live_sentiment
from core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

MIN_BARS = 60


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class MLSignalEngine:
    """Callable with the TradingWorkflow ``AnalysisRunner`` signature."""

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        *,
        strategy: Optional[MLStrategy] = None,
        history_loader: Optional[Callable[[str], pd.DataFrame]] = None,
        sentiment_loader: Optional[Callable[[str], SentimentSnapshot]] = None,
        timeframe: Optional[str] = None,
        lookback_days: Optional[int] = None,
        risk_per_trade_pct: Optional[float] = None,
    ) -> None:
        self.portfolio_manager = portfolio_manager
        self.strategy = strategy or MLStrategy()
        self.timeframe = timeframe or (self.strategy.artifact or {}).get("timeframe") or os.getenv("ML_TIMEFRAME", "1Hour")
        self.lookback_days = lookback_days or int(os.getenv("ML_LOOKBACK_DAYS", "60"))
        self.history_loader = history_loader or (
            lambda symbol: get_history(symbol, self.timeframe, self.lookback_days)
        )
        self.sentiment_loader = sentiment_loader or live_sentiment
        self.risk_per_trade_pct = risk_per_trade_pct or _env_float("RISK_PER_TRADE_PCT", 1.0)
        logger.info(f"ML signal engine ready: mode={self.strategy.mode}, timeframe={self.timeframe}")

    def __call__(
        self,
        symbol: str,
        api_keys: Dict[str, Optional[str]],
        portfolio_name: str = "default",
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        bars = self.history_loader(symbol)
        if bars is None or len(bars) < MIN_BARS:
            return self._error(symbol, "no_data", f"Need at least {MIN_BARS} completed bars for {symbol}")

        features = compute_features(bars)
        sentiment = self.sentiment_loader(symbol)
        ml = self.strategy.predict(features, sentiment)
        quantity = self._size(ml, symbol, portfolio_name, user_id)
        return self._analysis(symbol, ml, quantity, features, portfolio_name)

    # ------------------------------------------------------------------

    def _size(self, ml: MLSignal, symbol: str, portfolio_name: str, user_id: Optional[int]) -> int:
        if ml.signal == "BUY":
            if not ml.stop_distance or ml.stop_distance <= 0:
                return 0
            capital = self._capital(portfolio_name, user_id)
            # Risk a fixed % of capital between entry and stop.
            return max(0, int(capital * self.risk_per_trade_pct / 100 // ml.stop_distance))
        if ml.signal == "SELL":
            holdings = self.portfolio_manager.get_holdings(portfolio_name, user_id=user_id) \
                if self.portfolio_manager.get_portfolio(portfolio_name, user_id=user_id) else []
            return next((int(h["quantity"]) for h in holdings if h["symbol"] == symbol), 0)
        return 0

    def _capital(self, portfolio_name: str, user_id: Optional[int]) -> float:
        if self.portfolio_manager.get_portfolio(portfolio_name, user_id=user_id):
            value = self.portfolio_manager.get_portfolio_value(portfolio_name, user_id=user_id)
            return float(value["total_value"])
        return _env_float("TRADING_CAPITAL", 5000.0)

    def _analysis(
        self, symbol: str, ml: MLSignal, quantity: int, features: pd.DataFrame, portfolio_name: str
    ) -> Dict[str, Any]:
        action = ml.signal if quantity > 0 else "HOLD"
        reasons = list(ml.reasons)
        if ml.signal != "HOLD" and quantity == 0:
            reasons.append("no shares to trade (no position to exit or stop distance too wide for risk budget)")
        latest = features.iloc[-1]
        indicators = {
            key: (None if pd.isna(latest.get(key)) else float(latest.get(key)))
            for key in ("rsi", "macd", "macd_signal", "ema20", "ema50", "ema200", "atr")
        }
        indicators["sma_20"] = indicators["ema20"]  # keep the dashboard's existing field populated
        indicators["ema_20"] = indicators["ema20"]
        risk_parameters: Dict[str, Any] = {
            "stop_loss": ml.stop_loss,
            "take_profit": ml.take_profit,
            "stop_distance": ml.stop_distance,
            "target_distance": ml.target_distance,
            "position_value": round(quantity * ml.price, 2),
        }
        if ml.stop_distance and ml.target_distance:
            risk_parameters["risk_reward_ratio"] = round(ml.target_distance / ml.stop_distance, 2)

        return {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bar_time": str(features.index[-1]),
            "recommendation": action,
            "signal": action,
            "quantity": quantity,
            "confidence": round(ml.confidence, 4),
            "primary_strategy": f"ml_{ml.mode}",
            "reason": "; ".join(reasons),
            "primary_reason": "; ".join(reasons),
            "risk_parameters": risk_parameters,
            "technical_indicators": indicators,
            "all_signals": {
                "technical": {"signal": action, "current_price": ml.price, "indicators": indicators,
                              "strength": ml.confidence, "priority": 2, "reason": "ML technical features"},
                "sentiment": {
                    "signal": "HOLD",
                    "sentiment_score": ml.sentiment.score if ml.sentiment.available else 0.0,
                    "strength": abs(ml.sentiment.score) if ml.sentiment.available else 0.0,
                    "priority": 1,
                    "reason": (
                        f"24h weighted sentiment {ml.sentiment.score:+.2f} over {ml.sentiment.article_count} articles"
                        if ml.sentiment.available else "sentiment unavailable"
                    ),
                },
                "ml": ml.as_dict(),
            },
            "portfolio_context": {"portfolio_name": portfolio_name},
        }

    @staticmethod
    def _error(symbol: str, error_type: str, message: str) -> Dict[str, Any]:
        return {
            "error": True,
            "error_type": error_type,
            "message": message,
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
