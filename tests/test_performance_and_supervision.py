from __future__ import annotations

import csv
import io
import json
import plistlib
import subprocess

import pytest

import config.api.control as control
from core import autostart
from core.bot_manager import BotManager
from core.edge_gate import build_verdict, write_report
from core.env_store import update_env
from core.performance import compare_with_backtest, equity_series, load_live_trades, summarize, trades_csv
from tests.dashboard_helpers import remote


def _trade(symbol, entry, exit_, stop=98.0, qty=10, day=1):
    return {"symbol": symbol, "qty": qty, "entry_price": entry, "exit_price": exit_, "stop_price": stop,
            "entry_at": f"2026-03-0{day}T15:00:00+00:00", "exit_at": f"2026-03-0{day}T16:00:00+00:00"}


def _write_monitor(root, mode, trades):
    path = root / "data" / mode / "edge_monitor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trades": trades, "open_lots": {}, "paused": None}))
    return path


# ---------------------------------------------------------------------------
# core/performance.py
# ---------------------------------------------------------------------------

def test_live_trades_and_summary(tmp_path) -> None:
    path = _write_monitor(tmp_path, "paper", [_trade("AAPL", 100, 104), _trade("MSFT", 100, 98), {"bad": 1}])
    trades = load_live_trades(path)
    assert [t["r_multiple"] for t in trades] == [2.0, -1.0]
    assert trades[0]["pnl"] == 40 and trades[1]["return_pct"] == -2.0
    s = summarize(trades)
    assert s == {**s, "trades": 2, "win_rate_pct": 50.0, "profit_factor": 2.0, "avg_r": 0.5, "total_pnl": 20.0,
                 "losing_trades": 1}
    assert summarize([trades[0]])["profit_factor"] is None  # no losses yet: never infinity (not valid JSON)
    assert load_live_trades(tmp_path / "missing.json") == []


def test_verdicts() -> None:
    bt = {"avg_r": 0.2, "profit_factor": 1.4}
    assert compare_with_backtest(summarize([]), None)["status"] == "no_backtest"
    assert compare_with_backtest({"trades": 5}, bt)["status"] == "too_early"
    good = {"trades": 40, "avg_r": 0.18, "sd_r": 1.2, "profit_factor": 1.3}
    assert compare_with_backtest(good, bt)["status"] == "on_track"
    bad = {"trades": 60, "avg_r": -0.4, "sd_r": 1.0, "profit_factor": 0.7}
    verdict = compare_with_backtest(bad, bt)
    assert verdict["status"] == "behind" and "Don't move to real money" in verdict["message"]
    meh = {"trades": 25, "avg_r": 0.02, "sd_r": 1.5, "profit_factor": 1.02}
    assert compare_with_backtest(meh, bt)["status"] == "watch"


def test_equity_series_skips_padding() -> None:
    history = {"timestamp": [1700000000, 1700086400, 1700172800], "equity": [0, 10000, 10150.5],
               "profit_loss": [0, 0, 150.5], "profit_loss_pct": [0, 0, 0.01505]}
    points = equity_series(history)
    assert [p["equity"] for p in points] == [10000, 10150.5]
    assert points[-1]["pnl_pct"] == pytest.approx(1.505)
    assert points[0]["t"].startswith("2023-11-15")


def test_trades_csv(tmp_path) -> None:
    trades = load_live_trades(_write_monitor(tmp_path, "paper", [_trade("AAPL", 100, 104)]))
    rows = list(csv.DictReader(io.StringIO(trades_csv(trades))))
    assert rows[0]["symbol"] == "AAPL" and rows[0]["pnl"] == "40.0" and rows[0]["r_multiple"] == "2.0"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_performance_endpoint(client, manager, monkeypatch) -> None:
    _write_monitor(manager.root, "paper", [_trade("AAPL", 100, 104), _trade("MSFT", 100, 98)])
    body = client.get("/api/control/paper/performance").json()
    assert body["live"]["trades"] == 2 and body["comparison"]["status"] == "no_backtest"
    assert body["equity"] == [] and "API keys" in body["equity_message"]
    assert body["recent_trades"][0]["symbol"] == "MSFT"  # newest first

    update_env({"ALPACA_API_KEY": "PK1", "ALPACA_SECRET_KEY": "s"})
    good = {"trades": 300, "folds": 4, "profit_factor": 1.5, "avg_r": 0.1, "positive_folds": 3,
            "max_drawdown_pct": -5, "win_rate_pct": 45}
    write_report(build_verdict(good, {}, ["AAPL"]))

    class FakeExec:
        def get_portfolio_history(self, period, timeframe):
            assert (period, timeframe) == ("1W", "1H")
            return {"timestamp": [1700000000, 1700003600], "equity": [10000, 10020],
                    "profit_loss": [0, 20], "profit_loss_pct": [0, 0.002]}

    monkeypatch.setattr(control, "_read_only_executor", lambda mode: FakeExec())
    body = client.get("/api/control/paper/performance?period=1W").json()
    assert [p["equity"] for p in body["equity"]] == [10000, 10020]
    assert body["backtest"]["avg_r"] == 0.1 and body["comparison"]["status"] == "too_early"
    assert client.get("/api/control/paper/performance?period=5Y").status_code == 400


def test_trades_csv_endpoint(client, manager) -> None:
    _write_monitor(manager.root, "live", [_trade("NVDA", 50, 51, stop=49)])
    resp = client.get("/api/control/live/trades.csv")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.text.splitlines()[1].startswith("NVDA,10")


def test_validation_report_endpoint(client, manager) -> None:
    folder = manager.root / "reports" / "validation"
    folder.mkdir(parents=True)
    (folder / "folds.csv").write_text("fold,trades,avg_r\nfold 1/2,40,0.12\nfold 2/2,35,0.08\n")
    (folder / "by_entry_hour.csv").write_text("entry_hour_et,trades,avg_r\n10,20,0.3\n")
    body = client.get("/api/control/validate/report").json()
    assert [f["fold"] for f in body["folds"]] == ["fold 1/2", "fold 2/2"]
    assert body["by_entry_hour"][0]["entry_hour_et"] == "10" and body["by_regime"] == []


# ---------------------------------------------------------------------------
# Supervision / auto-resume
# ---------------------------------------------------------------------------

@pytest.fixture
def sup(tmp_path):
    procs = []

    def popen(command, **kwargs):
        proc = subprocess.Popen(["sleep", "30"], **kwargs)
        procs.append(proc)
        return proc

    clock = {"t": 0.0}
    mgr = BotManager(root=tmp_path, popen=popen, clock=lambda: clock["t"])
    yield mgr, clock, procs
    for p in procs:
        p.kill()
        p.wait()


def test_supervisor_restarts_crashed_bot_with_same_settings(sup) -> None:
    mgr, clock, procs = sup
    mgr.start_bot("paper", ["AAPL", "MSFT"], "15m", execute=True)
    assert mgr.desired("paper")["running"] is True
    procs[0].kill(); procs[0].wait()  # crash
    actions = mgr.supervise(lambda mode, want: None)
    assert actions == [{"mode": "paper", "action": "restarted"}]
    assert mgr.bots["paper"].running()
    assert mgr.bots["paper"].tracked.settings == {"symbols": ["AAPL", "MSFT"], "timeframe": "15m", "execute": True}
    assert mgr.supervise(lambda mode, want: None) == []  # healthy: nothing to do


def test_supervisor_respects_user_stop_budget_and_safety(sup) -> None:
    mgr, clock, procs = sup
    mgr.start_bot("paper", ["AAPL"], "5m", execute=True)
    mgr.stop_bot("paper")
    procs[-1].wait(timeout=5)
    assert mgr.supervise(lambda mode, want: None) == []  # you stopped it: stays stopped

    mgr.start_bot("live", ["AAPL"], "5m", execute=True)
    procs[-1].kill(); procs[-1].wait()
    assert mgr.supervise(lambda mode, want: "Live trading is not armed.") == [
        {"mode": "live", "action": "skipped", "reason": "Live trading is not armed."}]
    for _ in range(3):  # crash loop
        assert mgr.supervise(lambda mode, want: None)[0]["action"] == "restarted"
        procs[-1].kill(); procs[-1].wait()
        clock["t"] += 60
    assert mgr.supervise(lambda mode, want: None)[0]["action"] == "gave_up"
    clock["t"] += 3600  # an hour later the budget resets
    assert mgr.supervise(lambda mode, want: None)[0]["action"] == "restarted"


def test_supervise_once_applies_live_safety_and_logs_once(env_file, sup, monkeypatch) -> None:
    mgr, clock, procs = sup
    update_env({"ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "s", "LIVE_TRADING_ENABLED": "true"})
    mgr.start_bot("live", ["AAPL"], "5m", execute=True)
    procs[-1].kill(); procs[-1].wait()
    update_env({"LIVE_TRADING_ENABLED": "false"})  # disarmed while the bot was down
    control._last_supervisor_note.clear()
    first = control.supervise_once(mgr)
    assert first[0]["action"] == "skipped" and "not armed" in first[0]["reason"]
    control.supervise_once(mgr)
    events = [json.loads(line) for line in (mgr.root / "logs" / "live" / "execution_events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["bot_restart_blocked"]  # not repeated every pass
    monkeypatch.setenv("AUTO_RESUME_BOTS", "false")
    assert control.supervise_once(mgr) == []


# ---------------------------------------------------------------------------
# Start at login
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_autostart_entries(tmp_path, monkeypatch, platform) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    kwargs = dict(platform=platform, home=tmp_path)
    assert autostart.status(**kwargs)["enabled"] is False
    result = autostart.enable(python="/py/bin/python", root=tmp_path / "bot", **kwargs)
    path = autostart.entry_path(**kwargs)
    assert result["enabled"] and path.exists()
    raw = path.read_bytes()
    if platform == "win32":
        assert path.name == "AI Day Trader.cmd" and "Startup" in str(path)
        assert b"\r\n" in raw and b"set NO_BROWSER=1" in raw and b"start_dashboard.py" in raw
    elif platform == "darwin":
        plist = plistlib.loads(raw)
        assert plist["RunAtLoad"] is True and plist["ProgramArguments"][0] == "/py/bin/python"
        assert plist["EnvironmentVariables"] == {"NO_BROWSER": "1"}
    else:
        assert b"Exec=env NO_BROWSER=1" in raw
    assert autostart.disable(**kwargs)["enabled"] is False and not path.exists()


def test_autostart_endpoint_is_local_only(client, monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(autostart, "enable", lambda: calls.append("on") or {"enabled": True, "path": "x", "platform": "linux"})
    assert remote(client).put("/api/settings/autostart", json={"enabled": True}).status_code == 403
    body = client.put("/api/settings/autostart", json={"enabled": True}).json()
    assert body["enabled"] and calls == ["on"] and body["auto_resume"] is True


# ---------------------------------------------------------------------------
# Audit fixes: no double launches, single supervisor, escaping
# ---------------------------------------------------------------------------

def test_concurrent_starts_launch_exactly_one_bot(tmp_path, monkeypatch) -> None:
    import threading
    import time

    import core.bot_manager as bm

    monkeypatch.setattr(bm, "_pid_is_ours", lambda pid, script: True)  # the stand-in isn't literally bot.py

    launched = []

    def slow_popen(command, **kwargs):
        time.sleep(0.3)  # widen the check-then-launch window
        proc = subprocess.Popen(["sleep", "30"], **kwargs)
        launched.append(proc)
        return proc

    # Two managers = the supervisor and a manual Start (or two server processes).
    a, b = BotManager(root=tmp_path, popen=slow_popen), BotManager(root=tmp_path, popen=slow_popen)
    results = []

    def start(mgr):
        try:
            mgr.start_bot("live", ["AAPL"], "5m", execute=True)
            results.append("started")
        except RuntimeError as exc:
            results.append(str(exc))

    threads = [threading.Thread(target=start, args=(m,)) for m in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert len(launched) == 1
        assert sorted(results) == ["live is already running", "started"]
    finally:
        for p in launched:
            p.kill()
            p.wait()


def test_running_check_sees_a_bot_started_elsewhere(sup, tmp_path) -> None:
    mgr, clock, procs = sup
    mgr.start_bot("paper", ["AAPL"], "5m", execute=False)
    procs[0].kill(); procs[0].wait()                 # our bot died...
    other = subprocess.Popen(["sleep", "30"])        # ...and another process started a new one
    procs.append(other)
    mgr.bots["paper"].pid_path.write_text(str(other.pid))
    import core.bot_manager as bm
    original = bm._pid_is_ours
    bm._pid_is_ours = lambda pid, script: True       # the stand-in isn't literally bot.py
    try:
        assert mgr.bots["paper"].running()
        assert mgr.supervise(lambda mode, want: None) == []  # so no duplicate restart
    finally:
        bm._pid_is_ours = original


def test_only_one_supervisor_lock(tmp_path) -> None:
    from core.bot_manager import try_singleton_lock

    first = try_singleton_lock(tmp_path / "supervisor.lock")
    assert first is not None
    assert try_singleton_lock(tmp_path / "supervisor.lock") is None
    first.close()
    again = try_singleton_lock(tmp_path / "supervisor.lock")
    assert again is not None
    again.close()


def test_autostart_escapes_percent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    root = tmp_path / "100% bots"
    autostart.enable(platform="win32", home=tmp_path, python="C:/py/python.exe", root=root)
    text = autostart.entry_path(platform="win32", home=tmp_path).read_text(encoding="utf-8")
    assert "100%% bots" in text and "100% bots" not in text.replace("%%", "")
    autostart.enable(platform="linux", home=tmp_path, python="/usr/bin/python3", root=root)
    desktop = autostart.entry_path(platform="linux", home=tmp_path).read_text(encoding="utf-8")
    assert 'Exec=env NO_BROWSER=1 "/usr/bin/python3" "' in desktop and "100%% bots/start_dashboard.py" in desktop
