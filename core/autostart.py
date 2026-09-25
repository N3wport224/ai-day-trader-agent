#!/usr/bin/env python3
"""
Start the dashboard automatically when you log in to your computer, so bots
that were running come back after a restart (see BotManager.supervise).

Windows  a .cmd in your Startup folder (%APPDATA%\\...\\Start Menu\\Programs\\Startup)
macOS    a LaunchAgent (~/Library/LaunchAgents/com.aidaytrader.dashboard.plist)
Linux    an XDG autostart entry (~/.config/autostart/ai-day-trader.desktop)

It starts without opening a browser tab (NO_BROWSER=1); open the dashboard
yourself when you want to look at it. Nothing needs admin rights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.aidaytrader.dashboard"


def _platform(platform: Optional[str]) -> str:
    platform = platform or sys.platform
    if platform.startswith("win"):
        return "windows"
    if platform == "darwin":
        return "mac"
    return "linux"


def entry_path(platform: Optional[str] = None, home: Optional[Path] = None) -> Path:
    kind = _platform(platform)
    home = Path(home or Path.home())
    if kind == "windows":
        appdata = Path(os.getenv("APPDATA") or home / "AppData" / "Roaming")
        return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "AI Day Trader.cmd"
    if kind == "mac":
        return home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    return home / ".config" / "autostart" / "ai-day-trader.desktop"


def _content(kind: str, python: str, root: Path) -> str:
    script = root / "start_dashboard.py"
    if kind == "windows":
        # In a .cmd, "%" starts a variable even inside quotes: double it.
        q = lambda value: str(value).replace("%", "%%")  # noqa: E731
        return (
            "@echo off\r\n"
            f'cd /d "{q(root)}"\r\n'
            "set NO_BROWSER=1\r\n"
            f'start "AI Day Trader" /min "{q(python)}" "{q(script)}"\r\n'
        )
    if kind == "mac":
        log = root / "logs" / "dashboard.log"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{escape(python)}</string><string>{escape(str(script))}</string></array>
  <key>WorkingDirectory</key><string>{escape(str(root))}</string>
  <key>EnvironmentVariables</key><dict><key>NO_BROWSER</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{escape(str(log))}</string>
  <key>StandardErrorPath</key><string>{escape(str(log))}</string>
</dict>
</plist>
"""
    # Desktop-entry Exec: "%" is a field code and must be written "%%".
    exec_python, exec_script = (str(v).replace("%", "%%") for v in (python, script))
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=AI Day Trader dashboard\n"
        f'Exec=env NO_BROWSER=1 "{exec_python}" "{exec_script}"\n'
        f"Path={root}\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def status(platform: Optional[str] = None, home: Optional[Path] = None) -> Dict[str, object]:
    path = entry_path(platform, home)
    return {"enabled": path.exists(), "platform": _platform(platform), "path": str(path)}


def enable(platform: Optional[str] = None, home: Optional[Path] = None,
           python: Optional[str] = None, root: Path = PROJECT_ROOT) -> Dict[str, object]:
    kind = _platform(platform)
    path = entry_path(platform, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    newline = "" if kind == "windows" else None  # the .cmd already carries CRLF
    with path.open("w", encoding="utf-8", newline=newline) as fh:
        fh.write(_content(kind, python or sys.executable, Path(root)))
    return status(platform, home)


def disable(platform: Optional[str] = None, home: Optional[Path] = None) -> Dict[str, object]:
    entry_path(platform, home).unlink(missing_ok=True)
    return status(platform, home)
