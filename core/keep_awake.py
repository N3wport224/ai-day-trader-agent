#!/usr/bin/env python3
"""
Stop the computer from going to sleep while the bot runs.

A sleeping computer can't run the 3:50 pm ET flatten, so day-trade positions
would be carried overnight (broker-side stops still protect them). This only
prevents *idle* sleep: closing a laptop lid or shutting down still stops it.

Windows: SetThreadExecutionState (released automatically when the process exits).
macOS:   `caffeinate -i -w <pid>` (exits with the process).
Linux:   nothing (servers don't idle-sleep; use systemd-inhibit if needed).
Disable with KEEP_AWAKE=false.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def keep_awake() -> bool:
    if os.getenv("KEEP_AWAKE", "true").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            es_continuous, es_system_required = 0x80000000, 0x00000001
            ok = ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
            return bool(ok)
        if sys.platform == "darwin":
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception as exc:  # never block trading over this
        logger.warning(f"Could not prevent sleep: {exc}")
    return False
