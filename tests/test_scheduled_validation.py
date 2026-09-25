from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from bot import EdgeGateWatch
from core.execution_telemetry import EventLog
from core.edge_gate import build_verdict, check_live_setup, write_report
from tests.test_operations import Broker, _bot

GOOD = {"trades": 250, "folds": 4, "profit_factor": 1.45, "avg_r": 0.12, "positive_folds": 3,
        "max_drawdown_pct": -6.5, "win_rate_pct": 48.0}


# ---------------------------------------------------------------------------
# Edge gate re-checked while running
# ---------------------------------------------------------------------------

def test_gate_watch_caches_and_logs_transitions_once() -> None:
    results = iter([(None, []), ("report expired", []), ("report expired", []), (None, [])])
    clock = {"t": 0.0}
    calls = []

    def check():
        calls.append(clock["t"])
        return next(results)

    telemetry = EventLog(None)
    watch = EdgeGateWatch(check, telemetry, logging.getLogger("t"), interval=300, clock=lambda: clock["t"])
    assert watch() is None
    clock["t"] = 100
    assert watch() is None and len(calls) == 1          # cached within the interval
    clock["t"] = 400
    assert watch() == "edge gate: report expired"      # expired mid-run: entries blocked
    clock["t"] = 800
    assert watch() == "edge gate: report expired"
    clock["t"] = 1200
    assert watch() is None                             # fresh passing report: lifted without a restart
    events = [(e["event"], e["passed"]) for e in telemetry.events]
    assert events == [("edge_gate", False), ("edge_gate", True)]  # each transition once, not every check


def test_real_report_expiry_blocks_a_running_bot(tmp_path, monkeypatch) -> None:
    path = tmp_path / "edge.json"
    setup = {"timeframe": "5Min"}
    write_report(build_verdict(GOOD, setup, ["AAPL"]), path)
    now = {"t": datetime.now(timezone.utc)}
    watch = EdgeGateWatch(lambda: check_live_setup(setup, ["AAPL"], path=path, now=now["t"]),
                          EventLog(None), logging.getLogger("t"), interval=0)
    assert watch() is None
    now["t"] += timedelta(days=31)
    assert "days old" in watch()


def test_trading_bot_applies_gate_every_cycle_and_fails_closed(portfolio_manager) -> None:
    state = {"block": None}
    bot = _bot(portfolio_manager, Broker(), {"time": "11:00", "open": True}, entry_gate=lambda: state["block"])
    assert bot.run_cycle().entry_block is None
    state["block"] = "edge gate: report expired"
    assert bot.run_cycle().entry_block == "edge gate: report expired"

    def broken():
        raise RuntimeError("disk error")

    bot.entry_gate = broken
    assert "entry gate check failed" in bot.run_cycle().entry_block


# ---------------------------------------------------------------------------
# Scheduled re-validation
# ---------------------------------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pytest  # noqa: E402

from core.bot_manager import BotManager  # noqa: E402
from core.edge_gate import report_path  # noqa: E402
from core.revalidation import Revalidator, off_hours  # noqa: E402

ET = ZoneInfo("America/New_York")
SAT_NOON = datetime(2026, 3, 7, 12, 0, tzinfo=ET)
TUE_11AM = datetime(2026, 3, 3, 11, 0, tzinfo=ET)


def test_off_hours() -> None:
    assert off_hours(SAT_NOON) and off_hours(datetime(2026, 3, 3, 19, 0, tzinfo=ET))
    assert off_hours(datetime(2026, 3, 3, 7, 30, tzinfo=ET))
    assert not off_hours(TUE_11AM) and not off_hours(datetime(2026, 3, 3, 17, 59, tzinfo=ET))


@pytest.fixture
def world(tmp_path):
    procs = []

    def popen(command, **kwargs):
        proc = subprocess.Popen(["sleep", "30"], **kwargs)
        proc.command = command
        procs.append(proc)
        return proc

    mgr = BotManager(root=tmp_path, popen=popen)
    now = {"t": SAT_NOON}
    reval = Revalidator(mgr, now_fn=lambda: now["t"])
    yield mgr, reval, now, procs
    for p in procs:
        p.kill()
        p.wait()


def _report(created: datetime) -> None:
    write_report(build_verdict(GOOD, {"timeframe": "5Min"}, ["AAPL"]))
    data = json.loads(report_path().read_text())
    data["created_at"] = created.isoformat()
    report_path().write_text(json.dumps(data))


def _finish(mgr, procs, result="validated", finished=None):
    procs[-1].kill(); procs[-1].wait()
    folder = mgr.root / "reports" / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "last_run.json").write_text(json.dumps({"result": result, "trigger": "scheduled",
                                                       "finished_at": (finished or SAT_NOON).isoformat()}))


def test_due_rules(world) -> None:
    mgr, reval, now, procs = world
    _report(SAT_NOON - timedelta(days=8))
    assert reval.due() is None                                   # never validated from the dashboard
    mgr.start_validation(["AAPL", "MSFT"], "5m", 120, True)      # a manual run remembers its settings
    assert reval.due() is None                                   # ...and is still running
    _finish(mgr, procs)
    assert reval.due() == "validation is 8 days old"
    now["t"] = TUE_11AM + timedelta(days=7)
    assert reval.due() is None                                   # market hours: wait
    now["t"] = SAT_NOON
    _report(SAT_NOON - timedelta(days=3))
    assert reval.due() is None                                   # still fresh


def test_tick_runs_scheduled_validation_with_same_settings_once(world, monkeypatch) -> None:
    mgr, reval, now, procs = world
    _report(SAT_NOON - timedelta(days=9))
    mgr.start_validation(["AAPL", "MSFT"], "15m", 90, False)
    _finish(mgr, procs, finished=SAT_NOON - timedelta(days=9))
    reval._save(announced=(SAT_NOON - timedelta(days=9)).isoformat())
    actions = reval.tick()
    assert actions[0]["action"] == "revalidation_started"
    command = procs[-1].command
    assert command[command.index("--symbols") + 1] == "AAPL,MSFT" and "--market" not in command
    assert command[command.index("--timeframe") + 1] == "15m" and command[-2:] == ["--trigger", "scheduled"]
    procs[-1].kill(); procs[-1].wait()                           # run crashed without writing a result
    now["t"] = SAT_NOON + timedelta(hours=2)
    assert reval.tick() == []                                    # retry only after 12 hours
    now["t"] = SAT_NOON + timedelta(hours=13)
    assert reval.tick()[0]["action"] == "revalidation_started"
    monkeypatch.setenv("AUTO_REVALIDATE", "false")
    procs[-1].kill(); procs[-1].wait()
    now["t"] = SAT_NOON + timedelta(days=2)
    assert reval.due() is None


def test_result_announced_once_with_alert_text(world) -> None:
    from core.alerts import format_alert

    mgr, reval, now, procs = world
    mgr.start_validation(["AAPL"], "5m", 120, True)
    _finish(mgr, procs, result="no_edge")
    assert reval.tick() == [{"action": "revalidation_finished", "result": "no_edge"}]
    assert reval.tick() == []
    event = reval.telemetry.events[-1]
    assert event["event"] == "revalidation" and "NO edge" in format_alert(event)


def test_refresh_restarts_older_bots_off_hours_only(world) -> None:
    mgr, reval, now, procs = world
    mgr.start_bot("paper", ["AAPL"], "5m", execute=True)
    bot = procs[-1]
    later = datetime.now(timezone.utc) + timedelta(minutes=5)
    folder = mgr.root / "reports" / "validation"
    folder.mkdir(parents=True)
    (folder / "last_run.json").write_text(json.dumps({"result": "validated", "finished_at": later.isoformat()}))
    reval._save(announced=later.isoformat())

    now["t"] = TUE_11AM
    assert reval.tick() == []                                    # never during market hours
    now["t"] = SAT_NOON
    assert reval.tick() == [{"action": "bot_refresh", "mode": "paper"}]
    assert mgr.desired("paper")["running"] is True               # so the supervisor brings it back
    bot.wait(timeout=5)
    assert mgr.supervise(lambda mode, want: None) == [{"mode": "paper", "action": "restarted"}]
    (folder / "last_run.json").write_text(json.dumps({"result": "validated",
                                                       "finished_at": (later - timedelta(hours=1)).isoformat()}))
    reval._save(announced=(later - timedelta(hours=1)).isoformat())
    assert reval.tick() == []                                    # the restarted bot is newer: no loop


def test_schedule_status_endpoint(client, manager) -> None:
    _report(datetime.now(timezone.utc) - timedelta(days=2))
    body = client.get("/api/control/validate").json()
    sc = body["schedule"]
    assert sc["enabled"] and sc["every_days"] == 7 and 1.9 < sc["report_age_days"] < 2.1
    assert sc["has_settings"] is False
    resp = client.put("/api/settings/auto-revalidate", json={"enabled": False})
    assert resp.json() == {"enabled": False}
    assert client.get("/api/control/validate").json()["schedule"]["enabled"] is False
