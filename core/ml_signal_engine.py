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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

from core.feature_pipeline import build_feature_frame, macro_from_primary, macro_timeframe
from core.market_history import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TIMEFRAME,
    MARKET_TZ,
    BarCache,
    bar_length,
    get_history,
    is_intraday,
    normalize_timeframe,
)
from core.market_context import market_symbol
from core.ml_strategy import MLSignal, MLStrategy
from core.news_sentiment import SentimentSnapshot, live_sentiment
from core.portfolio_manager import PortfolioManager
from core.risk_manager import SizingConfig, position_size, risk_pct_for_trade

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
        sizing: Optional[SizingConfig] = None,
        macro_loader: Optional[Callable[[str], pd.DataFrame]] = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        market_loader: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
    ) -> None:
        self.portfolio_manager = portfolio_manager
        self.strategy = strategy or MLStrategy()
        model_timeframe = (self.strategy.artifact or {}).get("timeframe")
        self.timeframe = normalize_timeframe(timeframe or model_timeframe or os.getenv("ML_TIMEFRAME", DEFAULT_TIMEFRAME))
        if model_timeframe and normalize_timeframe(model_timeframe) != self.timeframe:
            # A model trained on hourly bars says nothing about 5-minute bars.
            logger.error(
                f"Model was trained on {model_timeframe} bars but the bot runs on {self.timeframe}; "
                f"ignoring the model (heuristic mode). Retrain with --timeframe {self.timeframe}."
            )
            self.strategy = self.strategy.without_model()
        env_lookback = os.getenv("ML_LOOKBACK_DAYS")
        self.lookback_days = lookback_days or (int(env_lookback) if env_lookback else DEFAULT_LOOKBACK_DAYS[self.timeframe])
        # The live loop re-reads history every cycle; the cache fetches only
        # the newest bars after the first load (BAR_CACHE=false disables it).
        self.use_cache = os.getenv("BAR_CACHE", "true").strip().lower() not in {"0", "false", "no", "off"}
        self._bar_cache = BarCache(self.timeframe, self.lookback_days) if self.use_cache else None
        self._macro_cache: Optional[BarCache] = None
        self.history_loader = history_loader or (
            self._bar_cache.get if self._bar_cache is not None
            else (lambda symbol: get_history(symbol, self.timeframe, self.lookback_days))
        )
        self.sentiment_loader = sentiment_loader or live_sentiment
        sizing = sizing or SizingConfig.from_env()
        if risk_per_trade_pct:
            sizing = SizingConfig(**{**sizing.__dict__, "base_risk_pct": risk_per_trade_pct})
        self.sizing = sizing
        self.macro_loader = macro_loader or self._default_macro_loader
        self.now_fn = now_fn
        self.market_loader = market_loader or (lambda: self.history_loader(market_symbol()))
        # Intraday only: refuse to signal on bars this many bar-lengths old
        # while the regular session is running (feed outage, delayed fallback).
        self.max_bar_age_bars = _env_float("MAX_BAR_AGE_BARS", 3.0)
        logger.info(
            f"ML signal engine ready: mode={self.strategy.mode}, timeframe={self.timeframe}, "
            f"regime filter={self.strategy.regime_policy}, MTF gate={self.strategy.mtf_confirmation}, "
            f"market filter={self.strategy.market_filter}, "
            f"sizing={self.sizing.method}"
        )

    def _default_macro_loader(self, symbol: str) -> pd.DataFrame:
        weekly = macro_timeframe(self.timeframe) == "1Week"
        # 50 daily EMA bars need ~75 trading days; fetch extra for warm-up.
        days = 500 if weekly else int(os.getenv("ML_MACRO_LOOKBACK_DAYS", "200"))
        if self.use_cache:
            if self._macro_cache is None:
                self._macro_cache = BarCache("1Day", days)
            daily = self._macro_cache.get(symbol)
        else:
            daily = get_history(symbol, "1Day", days)
        return macro_from_primary(daily, "1Day") if weekly else daily

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
        stale = self._stale_reason(bars)
        if stale:
            logger.warning(f"{symbol}: {stale}")
            return self._error(symbol, "stale_data", stale)

        macro = None
        if self.strategy.uses_macro:
            try:
                macro = self.macro_loader(symbol)
            except Exception as exc:  # missing macro data blocks MTF-gated entries, never crashes
                logger.warning(f"Macro ({macro_timeframe(self.timeframe)}) bars unavailable for {symbol}: {exc}")
        market = None
        if self.strategy.uses_market:
            try:
                market = self.market_loader()
            except Exception as exc:  # missing index data blocks market-filtered entries, never crashes
                logger.warning(f"Market ({market_symbol()}) bars unavailable: {exc}")
        features = build_feature_frame(bars, self.timeframe, macro_bars=macro, market_bars=market)
        sentiment = self.sentiment_loader(symbol)
        ml = self.strategy.predict(features, sentiment)
        quantity, risk_pct = self._size(ml, symbol, portfolio_name, user_id, features.iloc[-1])
        return self._analysis(symbol, ml, quantity, features, portfolio_name, risk_pct)

    # ------------------------------------------------------------------

    def _stale_reason(self, bars: pd.DataFrame) -> Optional[str]:
        if not is_intraday(self.timeframe) or self.max_bar_age_bars <= 0:
            return None
        now = pd.Timestamp(self.now_fn())
        length = bar_length(self.timeframe)
        max_age = length * self.max_bar_age_bars
        local = now.tz_convert(MARKET_TZ)
        session_open = local.normalize() + pd.Timedelta(hours=9, minutes=30)
        session_close = local.normalize() + pd.Timedelta(hours=16)
        if local.weekday() >= 5 or not (session_open + length + max_age <= local <= session_close):
            return None  # outside regular hours the last bar is legitimately old
        age = now - (bars.index[-1] + length)
        if age > max_age:
            return (f"latest completed {self.timeframe} bar closed {age.total_seconds() / 60:.0f} min ago "
                    f"(limit {max_age.total_seconds() / 60:.0f} min); no signal on stale data")
        return None

    def _size(
        self, ml: MLSignal, symbol: str, portfolio_name: str, user_id: Optional[int], latest: pd.Series
    ) -> tuple[int, Optional[float]]:
        if ml.signal == "BUY":
            risk_pct = risk_pct_for_trade(
                self.sizing,
                probability=ml.probability_up,
                reward_risk=(ml.target_distance / ml.stop_distance) if ml.stop_distance else None,
                atr_pct=latest.get("atr_pct"),
                atr_pct_median=latest.get("atr_pct_median"),
                calibrated=self.strategy.mode == "model",
            )
            capital = self._capital(portfolio_name, user_id)
            return position_size(capital, risk_pct, ml.stop_distance), risk_pct
        if ml.signal == "SELL":
            holdings = self.portfolio_manager.get_holdings(portfolio_name, user_id=user_id) \
                if self.portfolio_manager.get_portfolio(portfolio_name, user_id=user_id) else []
            return next((int(h["quantity"]) for h in holdings if h["symbol"] == symbol), 0), None
        return 0, None

    def _capital(self, portfolio_name: str, user_id: Optional[int]) -> float:
        if self.portfolio_manager.get_portfolio(portfolio_name, user_id=user_id):
            value = self.portfolio_manager.get_portfolio_value(portfolio_name, user_id=user_id)
            return float(value["total_value"])
        return _env_float("TRADING_CAPITAL", 5000.0)

    def _analysis(
        self,
        symbol: str,
        ml: MLSignal,
        quantity: int,
        features: pd.DataFrame,
        portfolio_name: str,
        risk_pct: Optional[float] = None,
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
            "sizing_method": self.sizing.method,
            "risk_pct": round(risk_pct, 4) if risk_pct is not None else None,
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
