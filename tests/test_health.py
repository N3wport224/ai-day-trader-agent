from __future__ import annotations

from datetime import datetime, timedelta, timezone

import config.api.control as control
import config.api.settings as settings
from core import health
from core.edge_gate import build_verdict, write_report

NOW = datetime(2026, 3, 3, 16, 0, tzinfo=timezone.utc)
GOOD = {"trades": 250, "folds": 4, "profit_factor": 1.45, "avg_r": 0.12, "positive_folds": 3,
        "max_drawdown_pct": -6.5, "win_rate_pct": 48.0}


def test_clock_drift_thresholds() -> None:
    assert health.check_clock(NOW, (NOW - timedelta(seconds=2)).isoformat()).status == "ok"
    warn = health.check_clock(NOW, (NOW + timedelta(seconds=12)).isoformat())
    assert warn.status == "warn" and "behind" in warn.detail and "Set time automatically" in warn.fix
    assert health.check_clock(NOW, (NOW - timedelta(seconds=45)).isoformat()).status == "fail"
    assert health.check_clock(NOW, None).status == "info"
    assert health.check_clock(NOW, "2026-03-03T11:00:00-05:00").status == "ok"  # Alpaca sends ET offsets


def test_validation_states() -> None:
    assert health.check_validation({}, NOW).status == "fail"
    failed = {"passed": False, "created_at": NOW.isoformat()}
    assert "no edge" in health.check_validation(failed, NOW).detail
    ok = {"passed": True, "created_at": (NOW - timedelta(days=3)).isoformat()}
    assert health.check_validation(ok, NOW).status == "ok"
    soon = {"passed": True, "created_at": (NOW - timedelta(days=26)).isoformat()}
    assert health.check_validation(soon, NOW).status == "warn"
    old = {"passed": True, "created_at": (NOW - timedelta(days=31)).isoformat()}
    assert "expired" in health.check_validation(old, NOW).detail


def test_model_and_keys_and_data() -> None:
    assert health.check_model(None, {}).status == "warn"
    mismatch = health.check_model({"timeframe": "1Hour"}, {"setup": {"timeframe": "5Min"}})
    assert mismatch.status == "warn" and "5Min" in mismatch.detail
    assert health.check_model({"timeframe": "5Min", "symbols": ["AAPL"]}, {"setup": {"timeframe": "5Min"}}).status == "ok"

    assert health.check_keys("paper", {"ok": False, "message": "No paper keys saved yet."}).status == "fail"
    assert health.check_keys("live", {"ok": False, "message": "No live keys saved yet."}).status == "info"
    assert health.check_keys("paper", {"ok": False, "message": "Alpaca rejected these keys (401)."}).status == "fail"
    assert health.check_keys("paper", {"ok": True, "equity": 1000, "trading_blocked": True}).status == "fail"
    assert health.check_keys("paper", {"ok": True, "equity": 1000}).status == "ok"

    assert health.check_market_data(None, "none", NOW, True).status == "fail"
    assert health.check_market_data(NOW - timedelta(minutes=40), "Alpaca", NOW, True).status == "warn"
    assert health.check_market_data(NOW - timedelta(hours=40), "Alpaca", NOW, False).status == "ok"  # weekend
    assert health.check_market_data(NOW, "Yahoo (delayed backup)", NOW, True).status == "warn"


def test_unattended_and_alerts(monkeypatch) -> None:
    assert health.check_unattended(False).status == "info"
    assert health.check_unattended(True).status == "ok"
    monkeypatch.setenv("KEEP_AWAKE", "false")
    assert health.check_unattended(True).status == "warn"
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    assert health.check_alerts().status == "info"


def test_bots_check(manager) -> None:
    manager.start_bot("paper", ["AAPL"], "5m", execute=False)
    manager.bots["paper"].tracked.proc.kill()
    manager.bots["paper"].tracked.proc.wait()
    checks = {c.id: c for c in health.check_bots(manager, market_open=True)}
    assert checks["bot_paper"].status == "fail" and checks["bot_live"].status == "info"


def test_run_checks_sorts_problems_first_and_survives_errors(manager, tmp_path) -> None:
    def boom():
        raise RuntimeError("network down")

    result = health.run_checks(
        manager=manager,
        connection=lambda mode: {"ok": True, "equity": 5000} if mode == "paper" else {"ok": False, "message": "No live keys saved yet."},
        broker_clock=boom,
        latest_bar=boom,
        load_model=boom,
        report_file=tmp_path / "missing.json",
        autostart_enabled=False,
        now_fn=lambda: NOW,
    )
    statuses = [c["status"] for c in result["checks"]]
    assert result["overall"] == "fail" and statuses == sorted(statuses, key=health.ORDER.get)
    by_id = {c["id"]: c for c in result["checks"]}
    assert by_id["data"]["status"] == "fail" and by_id["validation"]["status"] == "fail"
    assert by_id["keys_paper"]["status"] == "ok" and by_id["clock"]["status"] == "info"


def test_health_endpoint(client, manager, monkeypatch) -> None:
    write_report(build_verdict(GOOD, {"timeframe": "5Min"}, ["AAPL"]))
    monkeypatch.setattr(settings, "check_connection", lambda mode: {"ok": mode == "paper", "equity": 1000,
                                                                    "message": "No live keys saved yet."})
    monkeypatch.setattr(control, "_latest_spy_bar", lambda: (datetime.now(timezone.utc), "Alpaca"))
    body = client.get("/api/control/health").json()
    by_id = {c["id"]: c for c in body["checks"]}
    assert by_id["validation"]["status"] == "ok" and by_id["data"]["status"] == "ok"
    assert by_id["keys_paper"]["status"] == "ok" and by_id["keys_live"]["status"] == "info"
    assert body["overall"] in {"ok", "warn", "fail"} and body["checked_at"]
