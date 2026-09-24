from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.alpaca_executor import AlpacaExecutor, BrokerSnapshot, ExecutionConfig, spread_bps
from core.backtester import Backtester, BacktestConfig
from core.execution_telemetry import EventLog
from core.risk_manager import PortfolioRisk, RiskLimits, RiskManager, portfolio_risk
from tests.test_backtester import DAY, _flat_bars, _strategy

ACCOUNT = {"equity": "100000", "last_equity": "100000", "buying_power": "100000"}


# ---------------------------------------------------------------------------
# Portfolio heat and position count
# ---------------------------------------------------------------------------

def test_portfolio_risk_uses_highest_stop_and_counts_unprotected() -> None:
    positions = [
        {"symbol": "AAPL", "qty": "10", "current_price": "100"},
        {"symbol": "MSFT", "qty": "5", "current_price": "200"},
        {"symbol": "NVDA", "qty": "4", "current_price": "50"},   # no stop: 3% of value
        {"symbol": "SHRT", "qty": "-3", "current_price": "10"},  # ignored (long-only accounting)
    ]
    orders = [
        {"symbol": "AAPL", "side": "sell", "type": "stop", "stop_price": "95"},
        {"symbol": "AAPL", "side": "sell", "type": "stop", "stop_price": "97"},   # trailed: highest wins
        {"symbol": "MSFT", "side": "sell", "type": "stop", "stop_price": "205"},  # above price: locked profit
        {"symbol": "NVDA", "side": "sell", "type": "limit", "limit_price": "60"},  # target, not a stop
    ]
    risk = portfolio_risk(positions, orders, default_stop_pct=3.0)
    assert risk.open_positions == 3
    assert risk.open_risk == pytest.approx(10 * 3 + 0 + 4 * 50 * 0.03)


def _check(limits, portfolio, quantity=500, position=None):
    return RiskManager(limits).check_order(
        side="BUY", symbol="AAPL", quantity=quantity, price=100.0, account=ACCOUNT,
        stop_loss=98.0, take_profit=104.0, portfolio=portfolio, position=position,
    )


def test_heat_cap_shrinks_then_blocks_entries() -> None:
    limits = RiskLimits(min_price=1.0, max_position_pct=1.0, max_portfolio_heat_pct=2.0, max_open_positions=5)
    # $2,000 heat budget, $1,500 already at risk, $2/share risk -> 250 shares.
    shrunk = _check(limits, PortfolioRisk(open_positions=2, open_risk=1500))
    assert shrunk.approved and shrunk.quantity == 250 and "heat cap" in shrunk.reason
    assert _check(limits, PortfolioRisk(2, 400)).quantity == 500           # fits
    blocked = _check(limits, PortfolioRisk(2, 2000))
    assert not blocked.approved and "Portfolio heat limit" in blocked.reason
    assert _check(replace_limits(limits, max_portfolio_heat_pct=0), PortfolioRisk(2, 1e9)).approved
    assert _check(limits, None).quantity == 500                             # no portfolio info: cap not applied


def test_max_open_positions_blocks_new_symbols_only() -> None:
    limits = RiskLimits(min_price=1.0, max_position_pct=1.0, max_portfolio_heat_pct=0, max_open_positions=2)
    blocked = _check(limits, PortfolioRisk(open_positions=2))
    assert not blocked.approved and "Max open positions reached (2/2)" in blocked.reason
    adding = _check(limits, PortfolioRisk(open_positions=2), position={"qty": "10", "market_value": "1000"})
    assert adding.approved  # adding to an existing holding doesn't open a new position


def test_heat_and_position_env(monkeypatch) -> None:
    monkeypatch.setenv("MAX_PORTFOLIO_HEAT_PCT", "3.5")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "7")
    limits = RiskLimits.from_env()
    assert limits.max_portfolio_heat_pct == 3.5 and limits.max_open_positions == 7


def replace_limits(limits, **kw):
    from dataclasses import replace

    return replace(limits, **kw)


def test_backtester_limits_concurrent_positions() -> None:
    bars = {f"S{i}": _flat_bars() for i in range(4)}
    day = bars["S0"].index[30]
    limits = RiskLimits(min_price=1.0, max_position_pct=0.2, max_daily_trades=10, max_open_positions=2,
                        max_portfolio_heat_pct=0)
    config = BacktestConfig(initial_capital=10_000, bar_length=DAY, slippage_bps=0)
    result = Backtester(_strategy({day: 0.9}), RiskManager(limits), config).run(bars)
    assert len(result.trades) == 2
    assert result.blocked == {"Max open positions reached": 2}


# ---------------------------------------------------------------------------
# Execution cost: spread filter, marketable limits, stale entries
# ---------------------------------------------------------------------------

def test_spread_bps() -> None:
    assert spread_bps(99.95, 100.05) == pytest.approx(10.0)
    assert spread_bps(0, 100) is None and spread_bps(101, 100) is None


def test_execution_config_env(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_ORDER_TYPE", "market")
    monkeypatch.setenv("MAX_SPREAD_BPS", "8")
    config = ExecutionConfig.from_env()
    assert config.entry_order_type == "market" and config.max_spread_bps == 8
    monkeypatch.setenv("ENTRY_ORDER_TYPE", "bogus")
    assert ExecutionConfig.from_env().entry_order_type == "limit"


@pytest.fixture
def make_executor(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)
    monkeypatch.setattr(AlpacaExecutor, "get_position", lambda self, symbol: None)
    monkeypatch.setattr(AlpacaExecutor, "get_orders_today", lambda self: [])
    monkeypatch.setattr(AlpacaExecutor, "get_account", lambda self: ACCOUNT)

    def build(quote=None, execution=None, **limit_kw):
        ex = AlpacaExecutor(
            risk_manager=RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0, **limit_kw)),
            price_lookup=lambda symbol: 100.0,
            quote_lookup=lambda symbol: quote,
            telemetry=EventLog(None),
            execution=execution or ExecutionConfig(),
        )
        ex.placed = []

        def bracket(self, symbol, qty, stop, target, **kw):
            self.placed.append({"qty": qty, "stop": stop, "target": target, **kw})
            return {"id": f"o{len(self.placed)}", "qty": str(qty), "symbol": symbol}

        monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order", bracket)
        return ex

    return build


SIGNAL = {"symbol": "AAPL", "recommendation": "BUY", "quantity": 10,
          "risk_parameters": {"stop_distance": 2.0, "target_distance": 4.0}}


def test_wide_spread_skips_entry(make_executor) -> None:
    ex = make_executor(quote={"bid": 99.80, "ask": 100.20})  # 40 bps
    result = ex.submit(SIGNAL)
    assert result.order is None and "Spread too wide for AAPL: 40.0 bps" in result.skipped_reason
    assert ex.placed == [] and ex.telemetry.events[-1]["event"] == "entry_skipped_spread"


def test_entry_is_marketable_limit_off_the_ask(make_executor) -> None:
    ex = make_executor(quote={"bid": 100.00, "ask": 100.04})  # 4 bps: fine
    result = ex.submit(SIGNAL)
    placed = ex.placed[0]
    assert result.order["id"] == "o1"
    assert placed["limit_price"] == round(100.04 * 1.001, 2)  # 10 bps through the ask
    assert placed["stop"] == pytest.approx(98.04) and placed["target"] == pytest.approx(104.04)
    assert ex.expected_prices["o1"] == pytest.approx(100.04)


def test_market_entry_and_missing_quote(make_executor) -> None:
    ex = make_executor(quote=None, execution=ExecutionConfig(entry_order_type="market"))
    ex.submit(SIGNAL)
    assert "limit_price" not in ex.placed[0]  # plain market bracket; no quote does not block


def test_executor_applies_heat_cap_from_broker_snapshot(monkeypatch, make_executor) -> None:
    ex = make_executor(quote=None, max_portfolio_heat_pct=1.0)  # $1,000 budget
    monkeypatch.setattr(AlpacaExecutor, "get_snapshot", lambda self: BrokerSnapshot(
        positions=[{"symbol": "MSFT", "qty": "100", "current_price": "100"}],
        open_orders=[{"symbol": "MSFT", "side": "sell", "type": "stop", "stop_price": "92"}],  # $800 at risk
    ))
    ex.submit({**SIGNAL, "quantity": 500})
    assert ex.placed[0]["qty"] == 100  # $200 left / $2 per share


def test_cancel_stale_entries_only_unfilled_parents(make_executor, monkeypatch) -> None:
    ex = make_executor(execution=ExecutionConfig(entry_ttl_seconds=120))
    cancelled = []
    monkeypatch.setattr(AlpacaExecutor, "cancel_order", lambda self, oid: cancelled.append(oid) or True)
    now = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)
    old = (now - timedelta(minutes=5)).isoformat()
    fresh = (now - timedelta(seconds=30)).isoformat()
    orders = [
        {"id": "stale", "side": "buy", "status": "new", "submitted_at": old, "symbol": "AAPL"},
        {"id": "young", "side": "buy", "status": "new", "submitted_at": fresh, "symbol": "AAPL"},
        {"id": "partial", "side": "buy", "status": "partially_filled", "submitted_at": old, "symbol": "AAPL"},
        {"id": "leg", "side": "sell", "status": "held", "submitted_at": old, "parent_id": "x", "symbol": "AAPL"},
    ]
    assert ex.cancel_stale_entries(orders, now=now) == ["stale"] and cancelled == ["stale"]
    assert ex.telemetry.events[-1]["event"] == "entry_expired"


def test_backtester_limit_entries_miss_gaps() -> None:
    def run(next_open, next_low, limit_bps):
        bars = _flat_bars()
        bars.loc[bars.index[31], ["open", "high", "low", "close"]] = [next_open, next_open + 0.2, next_low, next_open]
        config = BacktestConfig(initial_capital=10_000, bar_length=DAY, slippage_bps=0, entry_limit_bps=limit_bps)
        limits = RiskLimits(min_price=1.0, max_position_pct=1.0)
        return Backtester(_strategy({bars.index[30]: 0.9}), RiskManager(limits), config).run({"AAA": bars})

    gap = run(101.0, 100.8, 10)                 # opens 1% above; limit 100.10 never trades
    assert gap.trades == [] and gap.session_blocked == {"entry_limit_missed": 1}
    back = run(101.0, 99.9, 10)                 # trades back through the limit: filled at it
    assert back.trades[0].entry_price == pytest.approx(100.10)
    assert run(101.0, 100.8, None).trades[0].entry_price == pytest.approx(101.0)   # market: always fills
    assert run(100.05, 99.9, 10).trades[0].entry_price == pytest.approx(100.05)     # inside the limit: open


# ---------------------------------------------------------------------------
# Market context (relative strength vs SPY, market filter)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.feature_pipeline import build_feature_frame  # noqa: E402
from core.market_context import MARKET_COLUMNS, MARKET_FEATURES, add_market_features  # noqa: E402
from core.ml_strategy import MLStrategy  # noqa: E402
from core.ml_training import synthetic_intraday_bars  # noqa: E402
from core.news_sentiment import UNAVAILABLE  # noqa: E402


def _intraday_pair():
    return synthetic_intraday_bars(days=12, seed=3), synthetic_intraday_bars(days=12, seed=0)


def test_market_features_are_causal() -> None:
    bars, spy = _intraday_pair()
    full = build_feature_frame(bars, "5Min", market_bars=spy)
    cut = bars.index[len(bars) // 2]
    prefix = build_feature_frame(bars[bars.index <= cut], "5Min", market_bars=spy[spy.index <= cut])
    pd.testing.assert_frame_equal(full.loc[:cut, MARKET_COLUMNS], prefix[MARKET_COLUMNS])
    # A market shock after `cut` changes nothing at or before it.
    shocked = spy.copy()
    shocked.loc[shocked.index > cut, ["open", "high", "low", "close"]] *= 0.5
    after = build_feature_frame(bars, "5Min", market_bars=shocked)
    pd.testing.assert_frame_equal(full.loc[:cut, MARKET_COLUMNS], after.loc[:cut, MARKET_COLUMNS])


def test_relative_strength_definition_and_missing_market() -> None:
    bars, spy = _intraday_pair()
    feats = build_feature_frame(bars, "5Min", market_bars=spy)
    t = feats.index[300]
    i, j = bars.index.get_loc(t), spy.index.get_loc(t)
    expected = np.log(bars["close"].iloc[i] / bars["close"].iloc[i - 12]) - np.log(
        spy["close"].iloc[j] / spy["close"].iloc[j - 12])
    assert feats.loc[t, "rs_12"] == pytest.approx(expected)
    same = build_feature_frame(spy, "5Min", market_bars=spy)
    assert same["rs_12"].dropna().abs().max() == pytest.approx(0.0)  # the index vs itself
    assert build_feature_frame(bars, "5Min")[MARKET_COLUMNS].isna().all().all()


def test_market_ok_flags_weak_tape() -> None:
    idx = pd.date_range("2026-03-02 14:30", periods=78, freq="5min", tz="UTC")  # one full session
    falling = pd.DataFrame({"open": np.linspace(110, 90, 78), "close": np.linspace(110, 90, 78),
                            "volume": 1e6}, index=idx)
    falling["high"], falling["low"] = falling["close"] + 0.1, falling["close"] - 0.1
    features = build_feature_frame(falling, "5Min")
    out = add_market_features(features, falling, bar_length=pd.Timedelta(minutes=5), intraday=True)
    assert out["market_ok"].iloc[-1] == 0.0
    rising = falling.iloc[::-1].copy()
    rising.index = idx
    out = add_market_features(features, rising, bar_length=pd.Timedelta(minutes=5), intraday=True)
    assert out["market_ok"].iloc[-1] == 1.0


def test_market_filter_gates_buys() -> None:
    strategy = MLStrategy(artifact=None, model_path="/nonexistent", confidence_threshold=0.6,
                          regime_policy="off", market_filter=True)
    row = pd.Series({"close": 100.0, "atr": 1.0, "market_ok": 0.0})
    weak = strategy.decide(0.9, row, UNAVAILABLE)
    assert weak.signal == "HOLD" and weak.gated_by == "market" and weak.market_ok == 0.0
    unknown = strategy.decide(0.9, row.drop("market_ok"), UNAVAILABLE)
    assert unknown.gated_by == "market" and "unavailable" in unknown.reasons[-1]
    assert strategy.decide(0.9, pd.Series({**row, "market_ok": 1.0}), UNAVAILABLE).signal == "BUY"
    assert strategy.decide(0.05, row, UNAVAILABLE).signal == "SELL"  # exits never blocked
    assert strategy.uses_market


def test_signal_engine_loads_index_bars_when_needed(portfolio_manager, monkeypatch) -> None:
    from core.ml_signal_engine import MLSignalEngine

    bars, spy = _intraday_pair()
    loaded = []

    def history(symbol):
        loaded.append(symbol)
        return spy if symbol == "SPY" else bars

    strategy = MLStrategy(artifact=None, model_path="/nonexistent", regime_policy="off", market_filter=True)
    engine = MLSignalEngine(portfolio_manager, strategy=strategy, timeframe="5m", history_loader=history,
                            sentiment_loader=lambda s: UNAVAILABLE)
    analysis = engine("AAPL", {}, "default")
    assert loaded == ["AAPL", "SPY"]
    assert analysis["all_signals"]["ml"]["market_ok"] in (0.0, 1.0)

    plain = MLSignalEngine(portfolio_manager, strategy=MLStrategy(artifact=None, model_path="/nonexistent"),
                           timeframe="5m", history_loader=history, sentiment_loader=lambda s: UNAVAILABLE)
    loaded.clear()
    plain("AAPL", {}, "default")
    assert loaded == ["AAPL"]  # no index fetch unless a model feature or the filter needs it


def test_training_with_market_features(tmp_path) -> None:
    from core.ml_training import LabelParams, build_dataset, train

    bars, spy = _intraday_pair()
    params = LabelParams(horizon=12, stop_atr_mult=1.0, target_atr_mult=2.0)
    data = build_dataset(bars, params, pd.Timedelta(minutes=5).to_pytimedelta(), market_bars=spy)
    artifact = train({"AAA": data}, params, pd.Timedelta(minutes=5).to_pytimedelta(), timeframe="5Min",
                     use_market=True)
    assert set(MARKET_FEATURES) <= set(artifact["feature_columns"])
    strategy = MLStrategy(artifact, regime_policy="off")
    assert strategy.uses_market and not strategy.market_filter


# ---------------------------------------------------------------------------
# Edge gate
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402

from core.edge_gate import (  # noqa: E402
    EdgeCriteria,
    build_verdict,
    check_live_setup,
    evaluate_edge,
    strategy_fingerprint,
    write_report,
)

GOOD = {"trades": 250, "folds": 4, "profit_factor": 1.45, "avg_r": 0.12, "positive_folds": 3,
        "max_drawdown_pct": -6.5, "win_rate_pct": 48.0}


def test_evaluate_edge_criteria() -> None:
    c = EdgeCriteria()
    assert evaluate_edge(GOOD, c) == []
    bad = evaluate_edge({**GOOD, "trades": 40, "folds": 1, "profit_factor": 0.9, "avg_r": -0.02,
                         "positive_folds": 0, "max_drawdown_pct": -22.0}, c)
    assert len(bad) == 6
    assert evaluate_edge({**GOOD, "profit_factor": float("nan")}, c) == ["profit factor nan < 1.2"]
    assert evaluate_edge({**GOOD, "profit_factor": float("nan"), "win_rate_pct": 100}, c) == []  # no losers


def _live_strategy(**kw):
    return MLStrategy(artifact=None, model_path="/nonexistent", confidence_threshold=0.6, regime_policy="penalty",
                      mtf_confirmation=False, market_filter=False, **kw)


def test_check_live_setup(tmp_path) -> None:
    path = tmp_path / "edge.json"
    setup = strategy_fingerprint(_live_strategy(), "5Min")

    reason, _ = check_live_setup(setup, ["AAPL"], path=path)
    assert "no edge report" in reason

    write_report(build_verdict({**GOOD, "trades": 10}, setup, ["AAPL"]), path)
    reason, _ = check_live_setup(setup, ["AAPL"], path=path)
    assert "did not pass" in reason and "10 trades" in reason

    write_report(build_verdict(GOOD, setup, ["AAPL", "MSFT"]), path)
    assert check_live_setup(setup, ["AAPL"], path=path) == (None, [])
    reason, warnings = check_live_setup(setup, ["AAPL", "TSLA"], path=path)
    assert reason is None and warnings == ["symbols not in the validated set: TSLA"]

    other = strategy_fingerprint(_live_strategy(), "1Min")
    reason, _ = check_live_setup(other, ["AAPL"], path=path)
    assert "differs" in reason and "timeframe" in reason
    gated = strategy_fingerprint(MLStrategy(artifact=None, model_path="/nonexistent", confidence_threshold=0.6,
                                            regime_policy="penalty", market_filter=True), "5Min")
    assert "market_filter" in check_live_setup(gated, ["AAPL"], path=path)[0]

    later = datetime.now(timezone.utc) + timedelta(days=45)
    reason, _ = check_live_setup(setup, ["AAPL"], path=path, now=later)
    assert "days old" in reason


def test_bot_standing_block_vetoes_entries_but_keeps_running(portfolio_manager) -> None:
    from tests.test_operations import Broker, _bot

    seen = []

    class Workflow:
        def __init__(self, pm):
            self.portfolio_manager = pm

        def run(self, symbol, portfolio_name, *, record_paper_trade, submit_alpaca_paper_order,
                entry_block_reason=None):
            from core.trading_workflow import WorkflowResult

            seen.append(entry_block_reason)
            return WorkflowResult(symbol=symbol, portfolio_name=portfolio_name, analysis={"recommendation": "HOLD"})

    bot = _bot(portfolio_manager, Broker(), {"time": "11:00", "open": True}, workflow=Workflow(portfolio_manager),
               standing_entry_block="edge gate: no edge report")
    report = bot.run_cycle()
    assert report.entry_block == "edge gate: no edge report" and seen == ["edge gate: no edge report"]


def test_bot_edge_gate_helper(monkeypatch, tmp_path, portfolio_manager) -> None:
    import logging

    from bot import _edge_gate
    from core.ml_signal_engine import MLSignalEngine
    from core.trading_workflow import TradingWorkflow

    monkeypatch.setenv("EDGE_REPORT_PATH", str(tmp_path / "edge.json"))
    engine = MLSignalEngine(portfolio_manager, strategy=_live_strategy(), timeframe="5m")
    bot = type("B", (), {"workflow": TradingWorkflow(portfolio_manager, analysis_runner=engine),
                         "symbols": ["AAPL"]})()
    log = logging.getLogger("test")

    assert "no edge report" in _edge_gate(bot, "ml", "5Min", log)[0]
    write_report(build_verdict(GOOD, strategy_fingerprint(engine.strategy, "5Min"), ["AAPL"]))
    assert _edge_gate(bot, "ml", "5Min", log) == (None, [])
    assert "classic" in _edge_gate(bot, "classic", "5Min", log)[0]
    monkeypatch.setenv("EDGE_GATE", "false")
    assert _edge_gate(bot, "classic", "5Min", log) == (None, [])


def test_backtest_writes_edge_report_but_never_promotes_synthetic(tmp_path, monkeypatch) -> None:
    import scripts.backtest as backtest

    monkeypatch.setenv("EDGE_REPORT_PATH", str(tmp_path / "live_edge.json"))
    code = backtest.main(["--synthetic", "--symbols", "AAA,BBB", "--mode", "heuristic",
                          "--out", str(tmp_path / "out"), "--promote"])
    assert code == 0
    report = _json.loads((tmp_path / "out" / "edge_report.json").read_text())
    assert set(report) >= {"passed", "failures", "metrics", "setup", "symbols", "created_at"}
    assert report["setup"]["mode"] == "heuristic"
    assert not (tmp_path / "live_edge.json").exists()  # synthetic data can't validate live trading


# ---------------------------------------------------------------------------
# Live edge-decay monitor
# ---------------------------------------------------------------------------

from core.edge_monitor import EdgeMonitor  # noqa: E402
from core.fill_quality import FillRecord  # noqa: E402

_T0 = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


def _fill(symbol, side, qty, price, minute, stop=None):
    return FillRecord(order_id=f"{symbol}-{side}-{minute}", symbol=symbol, side=side, order_type="market",
                      qty=qty, fill_price=price, reference_price=None, reference="none", slippage_bps=None,
                      filled_at=(_T0 + timedelta(minutes=minute)).isoformat(), stop_price=stop)


def _round_trips(monitor, results, start=0):
    """results: list of exit prices for entries at 100 with a 98 stop (R = (exit-100)/2)."""
    fills = []
    for k, exit_price in enumerate(results):
        m = start + 2 * k
        fills += [_fill("AAPL", "buy", 10, 100.0, m, stop=98.0), _fill("AAPL", "sell", 10, exit_price, m + 1)]
    return monitor.update(fills)


def test_round_trips_fifo_with_partial_exits(tmp_path) -> None:
    monitor = EdgeMonitor(state_path=str(tmp_path / "m.json"), telemetry=EventLog(None))
    closed = monitor.update([
        _fill("AAPL", "buy", 10, 100.0, 0, stop=98.0),
        _fill("AAPL", "buy", 5, 101.0, 1, stop=99.0),
        _fill("AAPL", "sell", 12, 104.0, 2),
        _fill("MSFT", "sell", 3, 50.0, 3),  # no tracked entry: ignored
    ])
    assert [(t.qty, t.entry_price, t.r_multiple) for t in closed] == [(10, 100.0, 2.0), (2, 101.0, 1.5)]
    assert monitor.open_lots["AAPL"][0]["qty"] == 3
    restarted = EdgeMonitor(state_path=str(tmp_path / "m.json"), telemetry=EventLog(None))
    assert restarted.update([_fill("AAPL", "sell", 3, 99.0, 4)])[0].r_multiple == pytest.approx(-1.0)


def test_pauses_on_profit_factor_collapse_and_latches(tmp_path) -> None:
    telemetry = EventLog(None)
    monitor = EdgeMonitor(state_path=str(tmp_path / "m.json"), telemetry=telemetry, min_trades=20, window=30)
    _round_trips(monitor, [104.0] * 5 + [98.0] * 14)  # 19 trades: not evaluated yet
    assert monitor.paused is None
    _round_trips(monitor, [98.0], start=100)          # 20th trade: PF = 200 / 300
    assert monitor.paused and "profit factor" in monitor.paused
    assert telemetry.events[-1]["event"] == "edge_decay"
    _round_trips(monitor, [110.0] * 20, start=200)    # later wins don't silently un-pause
    assert monitor.paused
    assert EdgeMonitor(state_path=str(tmp_path / "m.json"), telemetry=EventLog(None)).paused  # persisted
    monitor.reset()
    assert monitor.paused is None and monitor.trades == []


def test_pauses_when_live_r_significantly_below_backtest(tmp_path) -> None:
    # PF (0.63) stays above the 0.5 floor, but mean R (-0.22) is far below the backtest's 0.8.
    results = [104.0, 98.0, 98.0, 99.0, 100.4, 98.0] * 4
    monitor = EdgeMonitor(state_path=str(tmp_path / "m.json"), telemetry=EventLog(None), backtest_avg_r=0.8,
                          min_trades=20, min_profit_factor=0.5)
    _round_trips(monitor, results)
    assert monitor.paused and "significantly below" in monitor.paused
    # Against a modest backtest expectation (0.2) the gap is within noise (t ~ -1.8): no pause.
    ok = EdgeMonitor(state_path=str(tmp_path / "ok.json"), telemetry=EventLog(None), backtest_avg_r=0.2,
                     min_trades=20, min_profit_factor=0.5)
    _round_trips(ok, results)
    assert ok.paused is None


def test_monitor_reads_backtest_avg_r_from_edge_report(tmp_path) -> None:
    write_report(build_verdict(GOOD, {"timeframe": "5Min"}, ["AAPL"]))
    assert EdgeMonitor.from_edge_report(telemetry=EventLog(None)).backtest_avg_r == 0.12


def test_bot_blocks_entries_while_edge_monitor_paused(portfolio_manager, tmp_path) -> None:
    from tests.test_operations import Broker, _bot

    monitor = EdgeMonitor(state_path=str(tmp_path / "m.json"), telemetry=EventLog(None))
    bot = _bot(portfolio_manager, Broker(), {"time": "11:00", "open": True}, edge_monitor=monitor)
    assert bot.run_cycle().entry_block is None
    monitor.paused = "live profit factor 0.5"
    assert bot.run_cycle().entry_block == "edge decay: live profit factor 0.5"


def test_fill_tracker_records_initial_stop_and_cancelled_partials(tmp_path) -> None:
    from core.fill_quality import FillTracker

    tracker = FillTracker({}, state_path=str(tmp_path / "f.json"), telemetry=EventLog(None))
    parent = {"id": "p", "symbol": "AAPL", "side": "buy", "type": "limit", "limit_price": "100.1",
              "status": "canceled", "filled_qty": "4", "filled_avg_price": "100.05",
              "updated_at": _T0.isoformat(),
              "legs": [{"id": "s", "symbol": "AAPL", "side": "sell", "type": "stop", "stop_price": "98.00",
                        "status": "canceled", "filled_qty": "0"}]}
    (record,) = tracker.update([parent], _T0.date())
    assert record.qty == 4 and record.stop_price == 98.0 and record.reference == "limit"


def test_market_ok_needs_both_conditions() -> None:
    # Gap-down open, rally, late pullback: index below its (fast-reacting) EMA50
    # but still above the session VWAP -> not a weak tape.
    idx = pd.date_range("2026-03-02 14:30", periods=78, freq="5min", tz="UTC")
    close = np.array([80.0] * 40 + [100.0] * 30 + [90.0] * 8)
    spy = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1e6},
                       index=idx)
    out = add_market_features(build_feature_frame(spy, "5Min"), spy, bar_length=pd.Timedelta(minutes=5),
                              intraday=True)
    last = out.iloc[-1]
    assert last["mkt_trend"] < 0 and last["mkt_vwap_dist"] > 0
    assert last["market_ok"] == 1.0


def test_missing_market_bar_uses_the_previous_one_not_the_next() -> None:
    bars, spy = _intraday_pair()
    t = bars.index[400]
    gappy = spy.drop(index=t)
    full = build_feature_frame(bars, "5Min", market_bars=gappy)
    prefix = build_feature_frame(bars[bars.index <= t], "5Min", market_bars=gappy[gappy.index <= t])
    pd.testing.assert_series_equal(full.loc[t, MARKET_COLUMNS], prefix.loc[t, MARKET_COLUMNS])
    assert full.loc[t, "mkt_trend"] == pytest.approx(full.loc[bars.index[399], "mkt_trend"])  # carried forward
