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
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("paper", "live")
TIMEFRAMES = ("1m", "5m", "15m", "1h", "1d")
_SYMBOL = re.compile(r"^[A-Z][A-Z.]{0,9}$")


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
        with path.open(errors="replace") as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=lines)]
    except OSError:
        return []


def _pid_alive(pid: int) -> bool:
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

    def __init__(self, name: str, script: str, root: Path, popen: Callable = subprocess.Popen) -> None:
        self.name = name
        self.script = script
        self.root = root
        self.popen = popen
        self.dir = root / "logs" / name
        self.log_path = self.dir / "process.log"
        self.pid_path = self.dir / "process.pid"
        self.meta_path = self.dir / "process.json"
        self.tracked: Optional[_Tracked] = None

    def pid(self) -> Optional[int]:
        if self.tracked is not None:
            return self.tracked.proc.pid if self.tracked.proc.poll() is None else None
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
        if self.running():
            raise RuntimeError(f"{self.name} is already running")
        self.dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(self.root / self.script), *args]
        log = self.log_path.open("a")
        stamp = datetime.now(timezone.utc).isoformat()
        log.write(f"\n===== {stamp} starting: {' '.join(command[1:])} =====\n")
        log.flush()
        kwargs = dict(cwd=str(self.root), env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if os.name == "posix":
            kwargs["start_new_session"] = True  # Ctrl-C in the server terminal doesn't kill the bot mid-order
        proc = self.popen(command, **kwargs)
        log.close()
        self.tracked = _Tracked(proc, stamp, command, settings)
        self.pid_path.write_text(str(proc.pid))
        self.meta_path.write_text(json.dumps({"started_at": stamp, "settings": settings}))
        return self.status()

    def stop(self) -> bool:
        pid = self.pid()
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
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
    def __init__(self, root: Path = PROJECT_ROOT, popen: Callable = subprocess.Popen) -> None:
        self.root = root
        self.bots = {mode: ProcessSlot(mode, "bot.py", root, popen) for mode in MODES}
        self.validation = ProcessSlot("validate", "scripts/validate_and_train.py", root, popen)

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
        return self.bot_status(mode)

    def stop_bot(self, mode: str) -> bool:
        return self.bots[mode].stop()

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
