from __future__ import annotations

import os
import subprocess
import time

import pytest

import core.bot_manager as bot_manager
from core.bot_manager import BotManager
from core.env_store import update_env
from core.keep_awake import keep_awake


def _sleep_popen(captured):
    def popen(command, **kwargs):
        captured.update(kwargs)
        kwargs.pop("creationflags", None)  # POSIX test host can't use Windows flags
        return subprocess.Popen(["sleep", "30"], **kwargs)
    return popen


def test_env_store_works_without_fchmod(tmp_path, monkeypatch) -> None:
    monkeypatch.delattr(os, "fchmod", raising=False)  # Windows has no os.fchmod
    path = tmp_path / ".env"
    update_env({"XPLAT_TEST_KEY": "value"}, path)
    assert path.read_text(encoding="utf-8") == "XPLAT_TEST_KEY=value\n"
    os.environ.pop("XPLAT_TEST_KEY", None)


def test_bot_gets_stop_file_and_utf8_env_and_stale_request_is_cleared(tmp_path) -> None:
    captured = {}
    manager = BotManager(root=tmp_path, popen=_sleep_popen(captured))
    slot = manager.bots["paper"]
    slot.dir.mkdir(parents=True)
    slot.stop_path.write_text("stop")  # left over from an earlier run
    manager.start_bot("paper", ["AAPL"], "5m", execute=False)
    try:
        assert not slot.stop_path.exists()
        env = captured["env"]
        assert env["BOT_STOP_FILE"] == str(slot.stop_path)
        assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
        assert captured["start_new_session"] is True
    finally:
        slot.stop()


def test_windows_stop_uses_stop_file_not_kill(tmp_path, monkeypatch) -> None:
    captured = {}
    manager = BotManager(root=tmp_path, popen=_sleep_popen(captured))
    manager.start_bot("paper", ["AAPL"], "5m", execute=False)
    slot = manager.bots["paper"]
    monkeypatch.setattr(bot_manager, "IS_WINDOWS", True)
    killed = []
    monkeypatch.setattr(bot_manager.os, "kill", lambda pid, sig: killed.append(sig))
    try:
        assert slot.stop() is True
        assert slot.stop_path.read_text() == "stop" and killed == []  # graceful: no TerminateProcess
    finally:
        monkeypatch.undo()
        slot.tracked.proc.kill()
        slot.tracked.proc.wait()


def test_windows_start_flags(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bot_manager, "IS_WINDOWS", True)
    captured = {}
    manager = BotManager(root=tmp_path, popen=_sleep_popen(captured))
    manager.start_bot("paper", ["AAPL"], "5m", execute=False)
    try:
        assert "start_new_session" not in captured and "creationflags" in captured
    finally:
        manager.bots["paper"].tracked.proc.kill()
        manager.bots["paper"].tracked.proc.wait()


def test_windows_pid_check_never_sends_signals(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bot_manager, "IS_WINDOWS", True)
    monkeypatch.setattr(bot_manager, "_win_pid_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(bot_manager.os, "kill", lambda *a: pytest.fail("os.kill(pid, 0) is Ctrl+C on Windows"))
    assert bot_manager._pid_alive(4242) and not bot_manager._pid_alive(1)


def test_validation_job_is_stopped_directly(tmp_path) -> None:
    manager = BotManager(root=tmp_path, popen=_sleep_popen({}))
    assert manager.validation.graceful_stop_file is False
    manager.start_validation(["AAPL"], "5m", 120, True)
    assert manager.validation.stop() is True
    manager.validation.tracked.proc.wait(timeout=5)


def test_bot_stops_when_stop_file_appears(tmp_path, monkeypatch) -> None:
    from bot import _watch_stop_file

    class FakeBot:
        stopped = False

        def stop(self, reason):
            self.reason, self.stopped = reason, True

    stop_file = tmp_path / "stop.request"
    monkeypatch.setenv("BOT_STOP_FILE", str(stop_file))
    bot = FakeBot()
    _watch_stop_file(bot, __import__("logging").getLogger("t"), interval=0.05)
    stop_file.write_text("stop")
    for _ in range(100):
        if bot.stopped:
            break
        time.sleep(0.02)
    assert bot.stopped and bot.reason == "dashboard" and not stop_file.exists()


def test_keep_awake_opt_out(monkeypatch) -> None:
    monkeypatch.setenv("KEEP_AWAKE", "false")
    assert keep_awake() is False
