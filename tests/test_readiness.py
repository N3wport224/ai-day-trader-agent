"""Go-live scorecard: paper record -> ready / not ready, and the live start gate."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from config.api import control
from core.performance import compare_with_backtest, summarize
from core.readiness import ReadinessCriteria, assess, read_events

NOW = datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)
EDGE = {"passed": True, "metrics": {"avg_r": 0.25, "profit_factor": 1.4, "cost_bps_per_side": 6.0}}
SELFTEST = {"overall": "ok", "checked_at": (NOW - timedelta(days=2)).isoformat(),
            "steps": [{"id": "order_cancel", "label": "Cancel order", "status": "ok"}]}
CRIT = ReadinessCriteria(min_trades=20, min_days=10)


def _trades(n=24, days=12, win_every=2):
    trades = []
    for i in range(n):
        exit_at = NOW - timedelta(days=days - 1 - (i % days), hours=2)
        win = i % win_every == 0
        exit_price = 102.0 if win else 99.2
        trades.append({"symbol": "AAPL", "qty": 10, "exit_at": exit_at.isoformat(), "entry_price": 100.0,
                       "exit_price": exit_price, "stop_price": 99.0, "pnl": round((exit_price - 100) * 10, 2),
                       "r_multiple": round(exit_price - 100.0, 3)})
    return trades


def _fills(n=12, bps=4.0):
    return [{"event": "order_filled", "ts": (NOW - timedelta(days=1)).isoformat(), "slippage_bps": bps}] * n


def _assess(trades=None, edge=EDGE, selftest=SELFTEST, events=None):
    trades = _trades() if trades is None else trades
    summary = summarize(trades)
    return assess(trades=trades, summary=summary,
                  comparison=compare_with_backtest(summary, edge.get("metrics") if edge.get("passed") else None),
                  edge=edge, selftest=selftest, events=_fills() if events is None else events,
                  now=NOW, criteria=CRIT)


def _by_id(result):
    return {c["id"]: c for c in result["checks"]}


def test_a_solid_paper_record_is_ready():
    result = _assess()
    checks = _by_id(result)
    assert result["ready"] and result["overall"] == "ok", checks
    assert checks["trades"]["detail"].startswith("24 closed trades")
    assert checks["days"]["status"] == "ok" and checks["costs"]["status"] == "ok"


def test_too_little_paper_trading_is_not_ready():
    result = _assess(trades=_trades(n=6, days=3))
    checks = _by_id(result)
    assert not result["ready"]
    assert checks["trades"]["status"] == "fail" and "need 20" in checks["trades"]["detail"]
    assert checks["days"]["status"] == "fail" and checks["vs_test"]["status"] == "fail"   # too early to judge
    nothing = _by_id(_assess(trades=[], events=[]))
    assert nothing["incidents"]["status"] == "info"          # no trading is not a clean safety record


def test_results_behind_the_test_block_going_live():
    losing = _trades(win_every=10)          # mostly small losses
    result = _assess(trades=losing)
    checks = _by_id(result)
    assert not result["ready"] and checks["vs_test"]["status"] == "fail"
    assert checks["pnl"]["status"] == "warn"


def test_validation_and_selftest_are_required():
    checks = _by_id(_assess(edge={"passed": False, "failures": ["profit factor 0.9 < 1.2"]}))
    assert checks["validation"]["status"] == "fail" and "0.9" in checks["validation"]["detail"]
    checks = _by_id(_assess(selftest={}))
    assert checks["selftest"]["status"] == "fail" and "Test my setup" in checks["selftest"]["fix"]


def test_execution_costs_compared_with_the_test_assumption():
    assert _by_id(_assess(events=_fills(bps=15.0)))["costs"]["status"] == "warn"      # 15 > 6 + 5
    assert _by_id(_assess(events=_fills(n=3)))["costs"]["status"] == "info"          # too few fills
    old_report = {"passed": True, "metrics": {"avg_r": 0.25, "profit_factor": 1.4}}  # no cost recorded
    assert "6 bps assumed" in _by_id(_assess(edge=old_report))["costs"]["detail"]


def test_safety_incidents():
    unprotected = {"event": "unprotected_position", "ts": (NOW - timedelta(days=3)).isoformat()}
    result = _assess(events=_fills() + [unprotected])
    assert not result["ready"] and _by_id(result)["incidents"]["status"] == "fail"
    minor = {"event": "bot_error", "ts": (NOW - timedelta(days=3)).isoformat()}
    result = _assess(events=_fills() + [minor])
    assert result["ready"] and _by_id(result)["incidents"]["status"] == "warn"
    old = {"event": "not_flat", "ts": (NOW - timedelta(days=45)).isoformat()}
    assert _by_id(_assess(events=_fills() + [old]))["incidents"]["status"] == "ok"   # outside the 30-day window


def test_read_events_filters_kind_and_age(tmp_path):
    log = tmp_path / "events.jsonl"
    lines = [json.dumps({"event": "order_filled", "ts": NOW.isoformat(), "slippage_bps": 2}),
             "not json",
             json.dumps({"event": "order_submitted", "ts": NOW.isoformat()}),
             json.dumps({"event": "not_flat", "ts": (NOW - timedelta(days=60)).isoformat()})]
    log.write_text("\n".join(lines) + "\n")
    events = read_events(log, NOW - timedelta(days=30), {"order_filled", "not_flat"})
    assert [e["event"] for e in events] == ["order_filled"]
    assert read_events(tmp_path / "missing.jsonl", NOW, {"x"}) == []


# ---------------------------------------------------------------------------
# API and the live start gate
# ---------------------------------------------------------------------------

def _scorecard(ready):
    return {"ready": ready, "required": True, "passed": 5, "total": 8, "overall": "ok" if ready else "fail",
            "checks": [{"id": "trades", "label": "Paper track record", "status": "ok" if ready else "fail"}]}


def test_readiness_endpoint(client, monkeypatch):
    monkeypatch.setattr(control, "live_readiness", lambda root=None: _scorecard(False))
    body = client.get("/api/control/live/readiness").json()
    assert body["ready"] is False and body["checks"][0]["label"] == "Paper track record"


def test_live_start_needs_acknowledgment_when_not_ready(client, monkeypatch, manager):
    monkeypatch.setattr(control, "start_block_reason", lambda mode, execute: None)
    monkeypatch.setattr(control, "live_readiness", lambda root=None: _scorecard(False))
    start = {"symbols": ["AAPL"], "execute": True, "confirm_live": True}

    blocked = client.post("/api/control/live/start", json=start)
    assert blocked.status_code == 409 and "Paper track record" in blocked.json()["detail"]
    assert not manager.bots["live"].running()

    accepted = client.post("/api/control/live/start", json={**start, "accept_unready": True})
    assert accepted.status_code == 200 and manager.bots["live"].running()


def test_ready_or_disabled_scorecard_does_not_block(client, monkeypatch, manager):
    monkeypatch.setattr(control, "start_block_reason", lambda mode, execute: None)
    monkeypatch.setattr(control, "live_readiness", lambda root=None: _scorecard(True))
    assert client.post("/api/control/live/start",
                       json={"symbols": ["AAPL"], "execute": True, "confirm_live": True}).status_code == 200
    manager.stop_bot("live")

    monkeypatch.setattr(control, "live_readiness", lambda root=None: _scorecard(False))
    monkeypatch.setenv("LIVE_REQUIRE_READINESS", "false")
    assert client.post("/api/control/live/start",
                       json={"symbols": ["MSFT"], "execute": True, "confirm_live": True}).status_code == 200


def test_watch_only_and_paper_starts_skip_the_scorecard(client, monkeypatch):
    monkeypatch.setattr(control, "start_block_reason", lambda mode, execute: None)
    calls = []
    monkeypatch.setattr(control, "live_readiness", lambda root=None: calls.append(1) or _scorecard(False))
    assert client.post("/api/control/live/start",
                       json={"symbols": ["AAPL"], "execute": False}).status_code == 200
    assert client.post("/api/control/paper/start", json={"symbols": ["AAPL"]}).status_code == 200
    assert calls == []
