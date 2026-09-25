#!/usr/bin/env python3
"""
One-click updates from GitHub, without touching your keys or data.

check()   compares the installed version with the latest commit on
          UPDATE_REPO / UPDATE_BRANCH (default N3wport224/ai-day-trader-agent,
          main) and lists what changed.
apply()   installs it:
            - git checkout: `git pull --ff-only` (refused if you changed files)
            - downloaded ZIP: fetches that exact commit's archive and copies it
              over the install, skipping PROTECTED paths
          Every file it would overwrite is first copied to backups/<time>/, and
          packages are reinstalled only when requirements.txt changed.
rollback() puts the files from the latest backup back (ZIP installs).

Never runs while a bot or the validation job is running: stop them first.
Code from GitHub runs with your user's permissions, exactly as when you
download the ZIP yourself; only the configured repository is used, over HTTPS.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
VERSION_FILE = ".installed_version"
# Never overwritten or backed up by an update: your secrets, data and environment.
PROTECTED = {".env", ".venv", "venv", "data", "logs", "models", "reports", "backups", ".git", VERSION_FILE}
PROTECTED_SUFFIXES = (".db", ".sqlite", ".sqlite3")


class UpdateError(RuntimeError):
    pass


def repo() -> str:
    return os.getenv("UPDATE_REPO", "N3wport224/ai-day-trader-agent").strip()


def branch() -> str:
    return os.getenv("UPDATE_BRANCH", "main").strip()


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ai-day-trader-updater"}
    token = os.getenv("UPDATE_GITHUB_TOKEN")  # only needed for a private fork
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def is_protected(rel: str) -> bool:
    parts = PurePosixPath(rel).parts
    return bool(parts) and (parts[0] in PROTECTED or rel.endswith(PROTECTED_SUFFIXES) or "__pycache__" in parts)


# ---------------------------------------------------------------------------
# Installed version
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=120)
    if check and result.returncode != 0:
        raise UpdateError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def install_kind(root: Path = PROJECT_ROOT) -> str:
    return "git" if (root / ".git").exists() and shutil.which("git") else "zip"


def installed_version(root: Path = PROJECT_ROOT) -> Optional[str]:
    if install_kind(root) == "git":
        try:
            return _git(root, "rev-parse", "HEAD")
        except (UpdateError, OSError, subprocess.SubprocessError):
            return None
    try:
        return json.loads((root / VERSION_FILE).read_text(encoding="utf-8")).get("sha")
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def check(root: Path = PROJECT_ROOT, http_get: Callable = requests.get) -> Dict[str, Any]:
    current = installed_version(root)
    try:
        resp = http_get(f"{API}/repos/{repo()}/commits/{branch()}", headers=_headers(), timeout=15)
    except requests.RequestException as exc:
        return {"ok": False, "message": f"Couldn't reach GitHub ({exc.__class__.__name__}).", "current": current}
    if resp.status_code == 404:
        return {"ok": False, "current": current,
                "message": f"GitHub repository {repo()} not found (private repos need UPDATE_GITHUB_TOKEN)."}
    if not resp.ok:
        return {"ok": False, "current": current, "message": f"GitHub returned HTTP {resp.status_code}."}
    latest = resp.json()
    sha = latest.get("sha")
    info = {
        "ok": True,
        "repo": repo(),
        "branch": branch(),
        "install": install_kind(root),
        "current": current,
        "latest": sha,
        "latest_date": ((latest.get("commit") or {}).get("committer") or {}).get("date"),
        "update_available": bool(sha) and sha != current,
        "changes": [],
    }
    if info["update_available"] and current:
        try:
            cmp = http_get(f"{API}/repos/{repo()}/compare/{current}...{sha}", headers=_headers(), timeout=15)
            if cmp.ok:
                commits = cmp.json().get("commits") or []
                info["changes"] = [c["commit"]["message"].splitlines()[0] for c in commits][-20:][::-1]
                info["behind_by"] = cmp.json().get("ahead_by")
        except (requests.RequestException, KeyError, ValueError):
            pass
    elif info["update_available"]:
        info["changes"] = [((latest.get("commit") or {}).get("message") or "").splitlines()[0]]
        info["message"] = "Installed version unknown (downloaded ZIP); updating records it from now on."
    return info


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _safe_members(archive: zipfile.ZipFile) -> List[tuple]:
    """(member, relative path) for archive files, rejecting any path that would
    escape the install folder (zip-slip)."""
    out = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        if len(parts) < 2:
            continue
        rel = PurePosixPath(*parts[1:])  # drop GitHub's "<repo>-<sha>/" folder
        if rel.is_absolute() or ".." in rel.parts or any(":" in p or "\\" in p for p in rel.parts):
            raise UpdateError(f"Refusing unsafe path in update archive: {info.filename}")
        out.append((info, str(rel)))
    return out


def _reinstall_requirements(root: Path, run: Callable) -> Dict[str, Any]:
    result = run([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")],
                 cwd=root, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        return {"ok": False, "message": "Installing updated packages failed: " + " | ".join(tail)}
    return {"ok": True, "message": "Packages updated"}


def apply(
    root: Path = PROJECT_ROOT,
    *,
    busy: Optional[str] = None,
    http_get: Callable = requests.get,
    run: Callable = subprocess.run,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Install the latest version. ``busy`` is why updating isn't safe right now
    (a bot or validation running); it is refused then."""
    if busy:
        raise UpdateError(busy)
    info = check(root, http_get)
    if not info.get("ok"):
        raise UpdateError(info.get("message") or "Couldn't check for updates")
    if not info["update_available"]:
        return {"updated": False, "message": "Already up to date.", "version": info["current"]}
    sha = info["latest"]
    requirements_before = _file_hash(root / "requirements.txt")

    if info["install"] == "git":
        if _git(root, "status", "--porcelain", "--untracked-files=no"):
            raise UpdateError("You have changed files in this folder; update with git yourself (git pull).")
        _git(root, "pull", "--ff-only", "origin", branch())
        backup = None
        changed = None
    else:
        resp = http_get(f"{API}/repos/{repo()}/zipball/{sha}", headers=_headers(), timeout=120)
        if not resp.ok:
            raise UpdateError(f"Download failed (HTTP {resp.status_code})")
        archive = zipfile.ZipFile(io.BytesIO(resp.content))
        members = _safe_members(archive)
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
        backup = root / "backups" / stamp
        changed, added = [], []
        for info_, rel in members:
            if is_protected(rel):
                continue
            target = root / rel
            data = archive.read(info_)
            if target.exists() and target.read_bytes() == data:
                continue
            if target.exists():
                (backup / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup / rel)
            else:
                added.append(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".updating")
            tmp.write_bytes(data)
            mode = (info_.external_attr >> 16) & 0o777
            if mode & 0o111:
                os.chmod(tmp, mode)  # keep launchers executable
            os.replace(tmp, target)
            changed.append(rel)
        backup.mkdir(parents=True, exist_ok=True)
        (backup / "backup.json").write_text(json.dumps(
            {"from": info["current"], "to": sha, "changed": changed, "added": added}, indent=2), encoding="utf-8")
        (root / VERSION_FILE).write_text(json.dumps(
            {"sha": sha, "installed_at": (now or datetime.now(timezone.utc)).isoformat()}), encoding="utf-8")

    packages = None
    if _file_hash(root / "requirements.txt") != requirements_before:
        packages = _reinstall_requirements(root, run)
    return {
        "updated": True,
        "version": sha,
        "from": info["current"],
        "files_changed": len(changed) if changed is not None else None,
        "backup": str(backup) if backup else None,
        "packages": packages,
        "restart_required": True,
        "message": "Updated. Restart the dashboard to use the new version.",
    }


def rollback(root: Path = PROJECT_ROOT, busy: Optional[str] = None) -> Dict[str, Any]:
    """Undo the latest ZIP update: restore the files it overwrote."""
    if busy:
        raise UpdateError(busy)
    backups = sorted(p for p in (root / "backups").glob("*") if (p / "backup.json").exists())
    if not backups:
        raise UpdateError("No update backup to undo.")
    latest = backups[-1]
    meta = json.loads((latest / "backup.json").read_text(encoding="utf-8"))
    restored, removed = [], []
    for rel in meta.get("changed") or []:
        source = latest / rel
        if source.exists() and not is_protected(rel):
            shutil.copy2(source, root / rel)
            restored.append(rel)
    for rel in meta.get("added") or []:  # files the update introduced
        target = root / rel
        if target.is_file() and not is_protected(rel) and ".." not in PurePosixPath(rel).parts:
            target.unlink()
            removed.append(rel)
    if meta.get("from"):
        (root / VERSION_FILE).write_text(json.dumps({"sha": meta["from"]}), encoding="utf-8")
    (latest / "backup.json").rename(latest / "backup.undone.json")
    return {"restored": len(restored), "removed": len(removed), "version": meta.get("from"), "restart_required": True,
            "message": "Previous version restored. Restart the dashboard."}


def restart_dashboard(root: Path = PROJECT_ROOT, popen: Callable = subprocess.Popen,
                      exit_fn: Callable[[int], None] = os._exit, delay: float = 1.5) -> None:
    """Start a fresh dashboard (it waits for this one to free the port), then
    exit this process shortly after the HTTP response has gone out. Running
    bots are separate processes and keep going."""
    import threading
    import time

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "dashboard.log", "a", encoding="utf-8")
    kwargs: Dict[str, Any] = dict(cwd=str(root), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  env={**os.environ, "NO_BROWSER": "1", "WAIT_FOR_PORT": "1"})
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    popen([sys.executable, str(root / "start_dashboard.py")], **kwargs)
    log.close()

    def _exit_soon() -> None:
        time.sleep(delay)
        exit_fn(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
