#!/usr/bin/env python3
"""
Start, stop and watch the trading bot (bot.py) and the validation job from
the dashboard.

Each mode (paper / live) runs as its own `bot.py` process with its own log,
heartbeat and state files (logs/<mode>/, data/<mode>/), so the two never
share state. A pid file lets the dashboard find a bot it didn't start (or
one still running after the dashboard restarted). Stopping sends SIGTERM,
which bot.py handles gracefully: it finishes the current cycle and exits;
broker-side bracket orders keep protecting any open positions.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("paper", "live")
TIMEFRAMES = ("1m", "5m", "15m", "1h", "1d")
_SYMBOL = re.compile(r"^[A-Z][A-Z.]{0,9}$")
IS_WINDOWS = os.name == "nt"


def mode_env(mode: str, root: Path = PROJECT_ROOT) -> Dict[str, str]:
    """Per-mode runtime file locations (matches bot.apply_mode_paths)."""
    return {
        "BOT_MODE": mode,
        "HEARTBEAT_PATH": str(root / "logs" / mode / "heartbeat.json"),
        "EXECUTION_LOG_PATH": str(root / "logs" / mode / "execution_events.jsonl"),
        "TRAILING_STATE_PATH": str(root / "data" / mode / "trailing_state.json"),
        "FILL_STATE_PATH": str(root / "data" / mode / "fill_state.json"),
        "EDGE_MONITOR_STATE_PATH": str(root / "data" / mode / "edge_monitor.json"),
    }


def clean_symbols(symbols: Any) -> List[str]:
    if isinstance(symbols, str):
        symbols = symbols.replace(" ", ",").split(",")
    cleaned = [s.strip().upper() for s in symbols or [] if s and s.strip()]
    bad = [s for s in cleaned if not _SYMBOL.match(s)]
    if bad:
        raise ValueError(f"Not valid ticker symbols: {', '.join(bad)}")
    if not cleaned:
        raise ValueError("Add at least one symbol (e.g. AAPL, MSFT)")
    if len(cleaned) > 30:
        raise ValueError("Use at most 30 symbols")
    return list(dict.fromkeys(cleaned))


def tail(path: Path, lines: int = 80) -> List[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=lines)]
    except OSError:
        return []


@contextmanager
def file_lock(path: Path, timeout: float = 15.0):
    """Exclusive lock across threads and processes (flock / msvcrt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _lock_fd(fh.fileno(), blocking=False)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Timed out waiting for {path.name}; try again")
                time.sleep(0.05)
        try:
            yield
        finally:
            _unlock_fd(fh.fileno())
    finally:
        fh.close()


def _lock_fd(fd: int, blocking: bool) -> None:
    if os.name == "nt":  # the real platform (IS_WINDOWS can be simulated in tests)
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def _unlock_fd(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


def try_singleton_lock(path: Path):
    """Hold an exclusive lock for the life of this process, or return None if
    another process already holds it (released automatically when we exit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    try:
        _lock_fd(fh.fileno(), blocking=False)
    except OSError:
        fh.close()
        return None
    return fh


def _win_pid_alive(pid: int) -> bool:
    """Windows: os.kill(pid, 0) would send a Ctrl+C event (signal 0 is
    CTRL_C_EVENT there), so query the process handle instead."""
    import ctypes

    process_query_limited_information, still_active = 0x1000, 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if IS_WINDOWS:
        return _win_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_is_ours(pid: int, script: str) -> bool:
    """Best effort: the pid still runs our script (guards against pid reuse)."""
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            return script in cmdline.read_bytes().decode(errors="replace")
        except OSError:
            return False
    return True  # no /proc (macOS/Windows): trust the pid file


@dataclass
class _Tracked:
    proc: Any
    started_at: str
    command: List[str]
    settings: Dict[str, Any]


class ProcessSlot:
    """One managed background process (a bot mode, or the validation job)."""

    def __init__(self, name: str, script: str, root: Path, popen: Callable = subprocess.Popen,
                 graceful_stop_file: bool = True) -> None:
        self.name = name
        self.script = script
        self.root = root
        self.popen = popen
        self.dir = root / "logs" / name
        self.log_path = self.dir / "process.log"
        self.pid_path = self.dir / "process.pid"
        self.meta_path = self.dir / "process.json"
        # bot.py polls this file and stops gracefully (works on every OS;
        # Windows has no SIGTERM, only a hard TerminateProcess).
        self.stop_path = self.dir / "stop.request"
        self.graceful_stop_file = graceful_stop_file
        self.tracked: Optional[_Tracked] = None

    def pid(self) -> Optional[int]:
        if self.tracked is not None and self.tracked.proc.poll() is None:
            return self.tracked.proc.pid
        # Not ours (or ours exited): another dashboard process may have started one.
        try:
            pid = int(self.pid_path.read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if _pid_alive(pid) and _pid_is_ours(pid, self.script) else None

    def running(self) -> bool:
        return self.pid() is not None

    def exit_code(self) -> Optional[int]:
        if self.tracked is not None:
            return self.tracked.proc.poll()
        return None

    def start(self, args: List[str], env: Dict[str, str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Start the process unless it is already running. The check and the
        launch happen under an OS file lock, so a manual Start and an automatic
        restart (other thread or other server process) can never both launch a
        bot; two live bots would double every order."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.dir / "start.lock"):
            return self._start_locked(args, env, settings)

    def _start_locked(self, args: List[str], env: Dict[str, str], settings: Dict[str, Any]) -> Dict[str, Any]:
        if self.running():
            raise RuntimeError(f"{self.name} is already running")
        command = [sys.executable, str(self.root / self.script), *args]
        self.stop_path.unlink(missing_ok=True)  # a stale request must not stop the new run
        log = self.log_path.open("a", encoding="utf-8")
        stamp = datetime.now(timezone.utc).isoformat()
        log.write(f"\n===== {stamp} starting: {' '.join(command[1:])} =====\n")
        log.flush()
        child_env = {**os.environ, **env, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
                     "PYTHONIOENCODING": "utf-8"}
        if self.graceful_stop_file:
            child_env["BOT_STOP_FILE"] = str(self.stop_path)
        kwargs = dict(cwd=str(self.root), env=child_env,
                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        # Ctrl-C in the dashboard's terminal must not kill the bot mid-order.
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                       | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        proc = self.popen(command, **kwargs)
        log.close()
        self.tracked = _Tracked(proc, stamp, command, settings)
        self.pid_path.write_text(str(proc.pid))
        self.meta_path.write_text(json.dumps({"started_at": stamp, "settings": settings}))
        return self.status()

    def stop(self) -> bool:
        """Ask the process to stop. Bots finish their current cycle first."""
        pid = self.pid()
        if pid is None:
            return False
        if self.graceful_stop_file:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.stop_path.write_text("stop")
            if IS_WINDOWS:
                return True  # the bot sees the file within a couple of seconds
        try:
            os.kill(pid, signal.SIGTERM)  # POSIX: graceful; Windows (jobs only): immediate
        except (ProcessLookupError, OSError):
            return False
        return True

    def meta(self) -> Dict[str, Any]:
        if self.tracked is not None:
            return {"started_at": self.tracked.started_at, "settings": self.tracked.settings}
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return {}

    def status(self, log_lines: int = 60) -> Dict[str, Any]:
        pid = self.pid()
        meta = self.meta()
        return {
            "running": pid is not None,
            "pid": pid,
            "started_at": meta.get("started_at"),
            "settings": meta.get("settings") or {},
            "exit_code": self.exit_code(),
            "log": tail(self.log_path, log_lines),
        }


class BotManager:
    """Runs the bots and, via ``supervise``, keeps them running.

    The *desired* state of each bot (running or not, and with which settings)
    is saved in logs/<mode>/desired.json whenever you start or stop it. If a bot
    that should be running isn't (it crashed, or the computer restarted), the
    supervisor restarts it with the same settings, at most MAX_RESTARTS_PER_HOUR
    times an hour, and only if the same safety checks as a manual start pass.
    """

    MAX_RESTARTS_PER_HOUR = 3

    def __init__(self, root: Path = PROJECT_ROOT, popen: Callable = subprocess.Popen,
                 clock: Callable[[], float] = None) -> None:
        self.root = root
        self.clock = clock or time.monotonic
        self._restarts: Dict[str, List[float]] = {mode: [] for mode in MODES}
        self.bots = {mode: ProcessSlot(mode, "bot.py", root, popen) for mode in MODES}
        # The validation job places no orders, so a hard stop is fine everywhere.
        self.validation = ProcessSlot("validate", "scripts/validate_and_train.py", root, popen,
                                      graceful_stop_file=False)

    # -- bots -----------------------------------------------------------

    def start_bot(self, mode: str, symbols: Any, timeframe: str, execute: bool) -> Dict[str, Any]:
        if mode not in MODES:
            raise ValueError("mode must be 'paper' or 'live'")
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {', '.join(TIMEFRAMES)}")
        cleaned = clean_symbols(symbols)
        args = ["--mode", mode, "--symbols", ",".join(cleaned), "--timeframe", timeframe, "--strategy", "ml"]
        if execute:
            args.append("--execute")
        settings = {"symbols": cleaned, "timeframe": timeframe, "execute": execute}
        self.bots[mode].start(args, mode_env(mode, self.root), settings)
        self._save_desired(mode, {"running": True, **settings})
        return self.bot_status(mode)

    def stop_bot(self, mode: str) -> bool:
        """Stop and remember that you want it stopped (no auto-restart)."""
        self._save_desired(mode, {**self.desired(mode), "running": False})
        return self.bots[mode].stop()

    # -- desired state / supervision --------------------------------------

    def _desired_path(self, mode: str) -> Path:
        return self.root / "logs" / mode / "desired.json"

    def desired(self, mode: str) -> Dict[str, Any]:
        try:
            return json.loads(self._desired_path(mode).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"running": False}

    def _save_desired(self, mode: str, state: Dict[str, Any]) -> None:
        path = self._desired_path(mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {**state, "updated_at": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(state), encoding="utf-8")

    def supervise(self, block_reason: Callable[[str, Dict[str, Any]], Optional[str]]) -> List[Dict[str, Any]]:
        """Restart bots that should be running but aren't. ``block_reason(mode,
        settings)`` returns why a restart isn't allowed right now (or None)."""
        actions = []
        now = self.clock()
        for mode in MODES:
            want = self.desired(mode)
            if not want.get("running") or self.bots[mode].running():
                continue
            recent = [t for t in self._restarts[mode] if now - t < 3600]
            self._restarts[mode] = recent
            if len(recent) >= self.MAX_RESTARTS_PER_HOUR:
                actions.append({"mode": mode, "action": "gave_up",
                                "reason": f"restarted {len(recent)} times in the last hour; check the log"})
                continue
            reason = block_reason(mode, want)
            if reason:
                actions.append({"mode": mode, "action": "skipped", "reason": reason})
                continue
            try:
                self.start_bot(mode, want.get("symbols") or [], want.get("timeframe") or "5m",
                               bool(want.get("execute")))
            except (ValueError, RuntimeError, OSError) as exc:
                actions.append({"mode": mode, "action": "failed", "reason": str(exc)})
                continue
            self._restarts[mode].append(now)
            actions.append({"mode": mode, "action": "restarted"})
        return actions

    def heartbeat(self, mode: str) -> Optional[Dict[str, Any]]:
        path = Path(mode_env(mode, self.root)["HEARTBEAT_PATH"])
        try:
            beat = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat["ts"])).total_seconds()
        except (KeyError, ValueError):
            age = None
        beat["age_seconds"] = round(age) if age is not None else None
        return beat

    def events(self, mode: str, limit: int = 40) -> List[Dict[str, Any]]:
        path = Path(mode_env(mode, self.root)["EXECUTION_LOG_PATH"])
        out = []
        for line in tail(path, limit):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return list(reversed(out))

    def bot_status(self, mode: str) -> Dict[str, Any]:
        return {**self.bots[mode].status(), "mode": mode, "heartbeat": self.heartbeat(mode)}

    def stop_all(self) -> None:
        for slot in self.bots.values():
            slot.stop()

    # -- validation job -------------------------------------------------

    def start_validation(self, symbols: Any, timeframe: str, days: int, market: bool) -> Dict[str, Any]:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {', '.join(TIMEFRAMES)}")
        if not 20 <= int(days) <= 3650:
            raise ValueError("days must be between 20 and 3650")
        cleaned = clean_symbols(symbols)
        args = ["--symbols", ",".join(cleaned), "--timeframe", timeframe, "--days", str(int(days))]
        if market:
            args.append("--market")
        settings = {"symbols": cleaned, "timeframe": timeframe, "days": int(days), "market": market}
        return self.validation.start(args, {}, settings)
