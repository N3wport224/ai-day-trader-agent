"""Trade journal: why the bot bought (decision snapshot) and how each trade ended."""
from __future__ import annotations

import json

import pandas as pd

from core.alpaca_executor import AlpacaExecutor, ExecutionConfig
from core.edge_monitor import EdgeMonitor
from core.execution_router import ChaseConfig
from core.execution_telemetry import EventLog
from core.fill_quality import FillRecord
from core.ml_signal_engine import _decision_context
from core.ml_strategy import MLSignal
from core.news_sentiment import SentimentSnapshot
from core.performance import attach_explanations, exit_reason, explain_entry, load_live_trades, trades_csv
from core.risk_manager import RiskLimits, RiskManager

CONTEXT = {"probability_up": 0.64, "threshold": 0.6, "regime": "TRENDING_BULL", "vwap_zone": "VWAP to +1σ",
           "vwap_dist_pct": 0.42, "rvol": 1.8, "minutes_from_open": 45, "above_orb_high": True,
           "market_ok": 1.0, "reasons": ["p=0.64"]}


def _trade(exit_type, entry=100.0, exit_=102.0, exit_at="2026-09-28T14:10:00-04:00"):
    return {"symbol": "AAPL", "entry_price": entry, "exit_price": exit_, "exit_type": exit_type,
            "exit_at": exit_at, "entry_order_id": "o1"}


def test_entry_explained_in_plain_words():
    points = explain_entry(CONTEXT)
    assert points[0] == "Model: 64% chance of reaching the target before the stop (needs 60%)"
    assert "Market regime: uptrend" in points
    assert "Price 0.42% above VWAP (VWAP to +1σ)" in points
    assert "Volume 1.8x normal for this time of day" in points
    assert "Trading above the opening range (breakout)" in points
    assert explain_entry({"reasons": ["heuristic: EMA trend up"]}) == ["heuristic: EMA trend up"]


def test_exit_reasons():
    assert exit_reason(_trade("stop", exit_=98.0)) == "Stop-loss hit"
    assert exit_reason(_trade("trailing_stop", exit_=101.0)) == "Trailing stop: locked in profit"
    assert exit_reason(_trade("limit")) == "Take-profit hit"
    assert exit_reason(_trade("market", exit_at="2026-09-28T15:50:30-04:00")).startswith("End-of-day close")
    assert exit_reason(_trade("limit", exit_at="2026-09-28T19:51:00Z")).startswith("End-of-day close")  # UTC input
    assert exit_reason(_trade("market")).startswith("Sold at market")
    assert exit_reason(_trade(None)) is None


def test_explanations_join_by_any_order_id_including_repriced_orders():
    trades = [{**_trade("limit"), "entry_order_id": "r2"}, {**_trade("stop", exit_=98), "entry_order_id": "zz"}]
    attach_explanations(trades, [{"event": "entry_context", "order_ids": ["o1", "r2"], "context": CONTEXT}])
    assert trades[0]["why"][0].startswith("Model: 64%") and trades[0]["exit_reason"] == "Take-profit hit"
    assert trades[1]["why"] == [] and trades[1]["exit_reason"] == "Stop-loss hit"
    csv = trades_csv(trades)
    assert "exit_reason,why" in csv.splitlines()[0] and "Market regime: uptrend" in csv


def _fill(order_id, side, price, order_type, at):
    return FillRecord(order_id, "AAPL", side, order_type, 10, price, None, "none", None, at, stop_price=99.0)


def test_edge_monitor_records_order_ids_and_exit_type(tmp_path):
    state = tmp_path / "edge.json"
    monitor = EdgeMonitor(state_path=str(state), telemetry=EventLog(None), min_trades=100)
    monitor.update([_fill("o1", "buy", 100.0, "limit", "2026-09-28T14:00:00Z"),
                    _fill("s1", "sell", 99.0, "stop", "2026-09-28T14:30:00Z")])
    trade = load_live_trades(state)[0]
    assert trade["entry_order_id"] == "o1" and trade["exit_type"] == "stop"
    # State written by older versions (no ids) still loads.
    old = json.loads(state.read_text())
    for t in old["trades"]:
        for key in ("entry_order_id", "exit_order_id", "exit_type"):
            t.pop(key)
    old["trades"][0]["some_future_field"] = 1
    state.write_text(json.dumps(old))
    assert len(EdgeMonitor(state_path=str(state), telemetry=EventLog(None)).trades) == 1


def test_engine_snapshot_from_the_feature_row():
    ml = MLSignal("BUY", 0.64, 0.64, "model", 100.0, 1.0, 98.0, 104.0, 2.0, 4.0,
                  SentimentSnapshot(0.2, 3, True), ["p=0.64"], regime="TRENDING_BULL", market_ok=1.0)
    row = pd.Series({"vwap_z": 0.5, "vwap_dist": 0.0042, "rvol": 1.83, "minutes_from_open": 45.0,
                     "close_vs_orb_high": 0.003})
    ctx = _decision_context(ml, row, 0.6)
    assert ctx["vwap_zone"] == "VWAP to +1σ" and ctx["vwap_dist_pct"] == 0.42 and ctx["rvol"] == 1.83
    assert ctx["above_orb_high"] is True and ctx["threshold"] == 0.6 and ctx["sentiment"] == 0.2
    empty = _decision_context(ml, pd.Series({"vwap_z": float("nan")}), None)
    assert empty["vwap_zone"] is None and empty["rvol"] is None


def test_executor_logs_the_entry_context(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    for name, value in {"is_market_open": lambda self: True, "get_position": lambda self, s: None,
                        "get_orders_today": lambda self: [],
                        "get_account": lambda self: {"equity": "100000", "last_equity": "100000",
                                                     "buying_power": "100000"},
                        "_place_bracket_order": lambda self, s, q, stop, target, **k: {"id": "o7", "qty": str(q)}
                        }.items():
        monkeypatch.setattr(AlpacaExecutor, name, value)
    ex = AlpacaExecutor(risk_manager=RiskManager(RiskLimits(min_price=1.0)), price_lookup=lambda s: 100.0,
                        telemetry=EventLog(None), execution=ExecutionConfig(), chase=ChaseConfig(enabled=False))
    ex.submit({"symbol": "AAPL", "recommendation": "BUY", "quantity": 5, "confidence": 0.64,
               "decision_context": CONTEXT, "risk_parameters": {"stop_distance": 2.0, "target_distance": 4.0}})
    event = [e for e in ex.telemetry.events if e["event"] == "entry_context"][0]
    assert event["order_ids"] == ["o7"] and event["context"]["regime"] == "TRENDING_BULL"
    assert event["stop"] == 98.0 and event["target"] == 104.0


def test_performance_api_returns_explained_trades(client, manager):
    paths = {"state": manager.root / "data" / "paper" / "edge_monitor.json",
             "log": manager.root / "logs" / "paper" / "execution_events.jsonl"}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    paths["state"].write_text(json.dumps({"trades": [{
        "symbol": "AAPL", "qty": 10, "entry_price": 100, "exit_price": 104, "stop_price": 98,
        "entry_at": "2026-09-28T14:00:00Z", "exit_at": "2026-09-28T15:00:00Z", "entry_order_id": "o1",
        "exit_order_id": "t1", "exit_type": "limit"}]}))
    paths["log"].write_text(json.dumps({"event": "entry_context", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                                        "order_ids": ["o1"], "context": CONTEXT}) + "\n")
    trade = client.get("/api/control/paper/performance?period=1M").json()["recent_trades"][0]
    assert trade["exit_reason"] == "Take-profit hit" and trade["why"][0].startswith("Model: 64%")
    csv = client.get("/api/control/paper/trades.csv").text
    assert "Take-profit hit" in csv
