from __future__ import annotations

import numpy as np
import pytest

from core.alpaca_executor import AlpacaExecutor, ExecutionResult
from core.ml_signal_engine import MLSignalEngine
from core.ml_strategy import MLStrategy
from core.ml_training import synthetic_bars
from core.news_sentiment import UNAVAILABLE, SentimentSnapshot
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskLimits, RiskManager
from core.trading_bot import TradingBot, create_workflow
from core.trading_workflow import TradingWorkflow


class FixedModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


def _engine(pm, p, *, sentiment=UNAVAILABLE, bars=None, risk_pct=1.0):
    strategy = MLStrategy(
        {"pipeline": FixedModel(p), "label_params": {"stop_atr_mult": 1.5, "target_atr_mult": 3.0}},
        confidence_threshold=0.6,
    )
    history = bars if bars is not None else synthetic_bars(300, seed=4)
    return MLSignalEngine(
        pm,
        strategy=strategy,
        history_loader=lambda symbol: history,
        sentiment_loader=lambda symbol: sentiment,
        risk_per_trade_pct=risk_pct,
    )


def test_buy_signal_sizes_by_risk_budget_and_passes_atr_levels(portfolio_manager: PortfolioManager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    engine = _engine(portfolio_manager, 0.8)

    analysis = engine("AAPL", {}, "paper")
    risk = analysis["risk_parameters"]
    ml = analysis["all_signals"]["ml"]

    assert analysis["recommendation"] == "BUY"
    assert analysis["primary_strategy"] == "ml_model"
    # 1% of $10,000 at risk between entry and a 1.5 ATR stop
    assert analysis["quantity"] == int(100 // risk["stop_distance"])
    assert risk["target_distance"] == pytest.approx(2 * risk["stop_distance"], rel=1e-3)
    assert risk["stop_loss"] < ml["price"] < risk["take_profit"]
    assert risk["position_value"] == pytest.approx(analysis["quantity"] * ml["price"], abs=0.01)


def test_low_confidence_is_hold_with_zero_quantity(portfolio_manager: PortfolioManager) -> None:
    analysis = _engine(portfolio_manager, 0.55)("AAPL", {}, "missing-portfolio")

    assert analysis["recommendation"] == "HOLD"
    assert analysis["quantity"] == 0
    assert "below confidence threshold" in analysis["reason"]


def test_sell_signal_exits_only_held_shares(portfolio_manager: PortfolioManager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    engine = _engine(portfolio_manager, 0.1)

    flat = engine("AAPL", {}, "paper")
    portfolio_manager.update_holding("paper", "AAPL", 7, 100.0)
    held = engine("AAPL", {}, "paper")

    assert flat["recommendation"] == "HOLD" and flat["quantity"] == 0
    assert held["recommendation"] == "SELL" and held["quantity"] == 7


def test_not_enough_history_is_reported_as_error(portfolio_manager: PortfolioManager) -> None:
    analysis = _engine(portfolio_manager, 0.9, bars=synthetic_bars(20))("AAPL", {}, "paper")

    assert analysis["error"] is True
    assert analysis["error_type"] == "no_data"


def test_sentiment_details_are_reported(portfolio_manager: PortfolioManager) -> None:
    analysis = _engine(portfolio_manager, 0.5, sentiment=SentimentSnapshot(0.4, 6, True))("AAPL", {}, "x")

    ml = analysis["all_signals"]["ml"]
    assert ml["sentiment_available"] is True
    assert ml["sentiment_score"] == 0.4
    assert "over 6 articles" in analysis["all_signals"]["sentiment"]["reason"]


def test_ml_signal_still_goes_through_risk_manager_and_brackets(
    monkeypatch, portfolio_manager: PortfolioManager
) -> None:
    """End to end: ML BUY -> workflow -> executor.submit -> RiskManager -> bracket order."""
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(
        AlpacaExecutor, "get_account",
        lambda self: {"equity": "10000", "last_equity": "10000", "buying_power": "10000"},
    )
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, symbol: None)
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: [])
    placed = {}

    def fake_bracket(self, symbol, qty, stop, target):
        placed.update(symbol=symbol, qty=qty, stop=stop, target=target)
        return {"id": "bracket-1", "status": "accepted", "qty": str(qty), "symbol": symbol}

    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order", fake_bracket)

    portfolio_manager.create_portfolio("paper", 100_000)  # big local budget...
    live_price = 100.0
    executor = AlpacaExecutor(
        risk_manager=RiskManager(RiskLimits(max_position_pct=0.25, min_price=1.0)),
        price_lookup=lambda symbol: live_price,
    )
    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=_engine(portfolio_manager, 0.9, risk_pct=5.0),
        api_key_loader=lambda: {},
        paper_order_executor_factory=lambda: executor,
    )

    result = workflow.run("AAPL", "paper", record_paper_trade=True, submit_alpaca_paper_order=True)
    risk = result.analysis["risk_parameters"]

    assert result.alpaca_order["id"] == "bracket-1"
    # ...but the broker account's 25% position cap ($2,500 at $100) still wins.
    assert placed["qty"] == 25
    # ATR distances are re-centred on the live price.
    assert placed["stop"] == pytest.approx(round(live_price - risk["stop_distance"], 2))
    assert placed["target"] == pytest.approx(round(live_price + risk["target_distance"], 2))
    assert portfolio_manager.get_trade_history("paper", 1)[0]["quantity"] == 25


def test_create_workflow_selects_strategy(portfolio_manager: PortfolioManager) -> None:
    ml_workflow = create_workflow(portfolio_manager, "ml")
    classic = create_workflow(portfolio_manager, "classic")

    assert isinstance(ml_workflow.analysis_runner, MLSignalEngine)
    assert not isinstance(classic.analysis_runner, MLSignalEngine)
    with pytest.raises(ValueError):
        create_workflow(portfolio_manager, "magic")


def test_bot_dry_run_with_ml_engine_never_orders(portfolio_manager: PortfolioManager) -> None:
    class ExplodingExecutor:
        def submit(self, signal):
            raise AssertionError("dry run must not submit")

    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=_engine(portfolio_manager, 0.9),
        api_key_loader=lambda: {},
        paper_order_executor_factory=ExplodingExecutor,
    )

    report = TradingBot(workflow, ["AAPL"]).run_cycle()

    assert report.results[0].analysis["recommendation"] == "BUY"
    assert report.results[0].skipped_reason == "Paper trading disabled"
    assert report.orders == []


def test_engine_reports_regime_applies_mtf_gate_and_dynamic_sizing(portfolio_manager: PortfolioManager) -> None:
    from core.features import resample_bars
    from core.risk_manager import SizingConfig

    portfolio_manager.create_portfolio("paper", 10_000)
    hourly = synthetic_bars(24 * 90, seed=6, start="2025-01-01 00:00")
    macro_calls = []

    def daily_loader(symbol):
        macro_calls.append(symbol)
        return resample_bars(hourly, "1D")

    strategy = MLStrategy(
        {"pipeline": FixedModel(0.9), "label_params": {"stop_atr_mult": 1.5, "target_atr_mult": 3.0}},
        confidence_threshold=0.6, regime_policy="off", mtf_confirmation=True,
    )
    engine = MLSignalEngine(
        portfolio_manager, strategy=strategy, history_loader=lambda s: hourly,
        sentiment_loader=lambda s: UNAVAILABLE, macro_loader=daily_loader,
        sizing=SizingConfig(method="volatility", base_risk_pct=1.0),
    )

    analysis = engine("AAPL", {}, "paper")
    ml = analysis["all_signals"]["ml"]

    assert macro_calls == ["AAPL"]
    assert ml["regime"] in {"TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"}
    assert ml["macro_aligned"] in {0.0, 1.0}
    if ml["macro_aligned"] == 1.0:
        assert analysis["recommendation"] == "BUY"
        assert analysis["risk_parameters"]["sizing_method"] == "volatility"
        assert 0.5 <= analysis["risk_parameters"]["risk_pct"] <= 1.5
    else:
        assert analysis["recommendation"] == "HOLD" and ml["gated_by"] == "mtf"


def test_engine_blocks_mtf_entries_when_daily_bars_unavailable(portfolio_manager: PortfolioManager) -> None:
    def broken(symbol):
        raise ConnectionError("daily feed down")

    strategy = MLStrategy(
        {"pipeline": FixedModel(0.9), "label_params": {}}, confidence_threshold=0.6,
        regime_policy="off", mtf_confirmation=True,
    )
    engine = MLSignalEngine(portfolio_manager, strategy=strategy, history_loader=lambda s: synthetic_bars(300, seed=4),
                            sentiment_loader=lambda s: UNAVAILABLE, macro_loader=broken)

    analysis = engine("AAPL", {}, "paper")

    assert analysis["recommendation"] == "HOLD"
    assert analysis["all_signals"]["ml"]["gated_by"] == "mtf"


def test_engine_ignores_model_trained_on_a_different_timeframe(portfolio_manager: PortfolioManager, caplog) -> None:
    strategy = MLStrategy(
        {"pipeline": FixedModel(0.9), "label_params": {}, "timeframe": "1Hour"},
        confidence_threshold=0.6, regime_policy="off",
    )

    engine = MLSignalEngine(portfolio_manager, strategy=strategy, timeframe="5m",
                            history_loader=lambda s: synthetic_bars(300, seed=4),
                            sentiment_loader=lambda s: UNAVAILABLE)

    assert engine.timeframe == "5Min"
    assert engine.strategy.mode == "heuristic"
    assert strategy.mode == "model"  # the caller's strategy object is untouched
    assert "trained on 1Hour bars" in caplog.text
    assert engine.lookback_days == 45  # 5-minute default history
