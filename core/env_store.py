#!/usr/bin/env python3
"""
Read and update the project's .env file (used by the dashboard's API Keys tab).

Secrets stay on this machine: the file is written atomically with owner-only
permissions (0600), values are never logged, and callers only ever get masked
hints back (see ``mask``).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_\-./:+=@%?&,]*$")
_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def env_path() -> Path:
    return Path(os.getenv("DOTENV_PATH") or PROJECT_ROOT / ".env")


def _format(value: str) -> str:
    if _SAFE_VALUE.match(value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def update_env(values: Mapping[str, Optional[str]], path: Optional[Path] = None) -> Path:
    """Set (or, with None, remove) variables in .env and in this process.

    Other lines, comments and ordering are preserved.
    """
    path = Path(path or env_path())
    for name, value in values.items():
        if not _NAME.match(name):
            raise ValueError(f"Invalid variable name {name!r}")
        if value is not None and ("\n" in value or "\r" in value or "\x00" in value):
            raise ValueError(f"{name} must be a single line")

    lines = path.read_text().splitlines() if path.exists() else []
    pending: Dict[str, Optional[str]] = dict(values)
    out = []
    for line in lines:
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        name = match.group(1) if match else None
        if name in pending:
            value = pending.pop(name)
            if value is not None:
                out.append(f"{name}={_format(value)}")
            continue  # replaced or removed
        out.append(line)
    for name, value in pending.items():
        if value is not None:
            out.append(f"{name}={_format(value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".env.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(out) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    os.chmod(path, 0o600)

    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return path


def mask(value: Optional[str], show: int = 4) -> Optional[str]:
    """'PKABCD1234WXYZ' -> 'PK…WXYZ' (prefix hints paper/live, suffix identifies it)."""
    if not value:
        return None
    if len(value) <= show + 2:
        return "•" * len(value)
    return f"{value[:2]}…{value[-show:]}"
