from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from core import updater

OLD, NEW = "a" * 40, "b" * 40


class Resp:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self.ok = status, 200 <= status < 300
        self._payload, self.content = payload, content

    def json(self):
        return self._payload


def _zip(files: dict, prefix="N3wport224-ai-day-trader-agent-bbbbbbb/") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            info = zipfile.ZipInfo(prefix + name)
            info.external_attr = (0o755 if name.endswith(".command") else 0o644) << 16
            z.writestr(info, data)
    return buf.getvalue()


def _github(archive: bytes, latest=NEW, compare_msgs=("Add feature X", "Fix bug Y")):
    def get(url, headers, timeout):
        if url.endswith(f"/commits/{updater.branch()}"):
            return Resp(payload={"sha": latest, "commit": {"committer": {"date": "2026-09-25T10:00:00Z"},
                                                          "message": "Merge pull request #8\n\nbody"}})
        if "/compare/" in url:
            return Resp(payload={"ahead_by": 2, "commits": [{"commit": {"message": m}} for m in compare_msgs]})
        if "/zipball/" in url:
            assert url.endswith(latest)
            return Resp(content=archive)
        raise AssertionError(url)
    return get


@pytest.fixture
def install(tmp_path):
    root = tmp_path / "bot"
    (root / "core").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "core" / "a.py").write_text("old a")
    (root / "core" / "same.py").write_text("same")
    (root / "requirements.txt").write_text("pandas==1\n")
    (root / ".env").write_text("ALPACA_API_KEY=mine\n")
    (root / "data" / "portfolios.db").write_text("my trades")
    (root / updater.VERSION_FILE).write_text(json.dumps({"sha": OLD}))
    return root


REMOTE = {
    "core/a.py": "new a",
    "core/same.py": "same",
    "core/new.py": "brand new",
    "start_dashboard.command": "#!/bin/bash\n",
    "requirements.txt": "pandas==1\n",
    ".env": "ALPACA_API_KEY=REPO-DEFAULT",       # must never overwrite yours
    "data/portfolios.db": "empty",               # nor your data
    "models/ml_signal.joblib": "x",
}


def test_check_reports_update_and_changes(install) -> None:
    info = updater.check(install, _github(_zip(REMOTE)))
    assert info["update_available"] and info["current"] == OLD and info["latest"] == NEW
    assert info["install"] == "zip" and info["changes"] == ["Fix bug Y", "Add feature X"]  # newest first
    assert updater.check(install, _github(_zip(REMOTE), latest=OLD))["update_available"] is False
    missing = updater.check(install, lambda url, headers, timeout: Resp(404))
    assert missing["ok"] is False and "not found" in missing["message"]


def test_zip_update_keeps_secrets_and_data_backs_up_and_can_undo(install) -> None:
    runs = []
    result = updater.apply(install, http_get=_github(_zip(REMOTE)), run=lambda *a, **k: runs.append(a))
    assert result["updated"] and result["version"] == NEW and result["restart_required"]
    assert (install / "core" / "a.py").read_text() == "new a"
    assert (install / "core" / "new.py").read_text() == "brand new"
    assert (install / ".env").read_text() == "ALPACA_API_KEY=mine\n"            # untouched
    assert (install / "data" / "portfolios.db").read_text() == "my trades"      # untouched
    assert not (install / "models").exists()                                    # protected folder
    assert os.stat(install / "start_dashboard.command").st_mode & 0o111           # still executable
    assert result["files_changed"] == 3 and runs == []                          # same requirements: no pip
    backup = Path(result["backup"])
    assert (backup / "core" / "a.py").read_text() == "old a"
    assert json.loads((install / updater.VERSION_FILE).read_text())["sha"] == NEW

    undo = updater.rollback(install)
    assert undo["restored"] == 1 and undo["removed"] == 2
    assert (install / "core" / "a.py").read_text() == "old a" and not (install / "core" / "new.py").exists()
    assert json.loads((install / updater.VERSION_FILE).read_text())["sha"] == OLD
    with pytest.raises(updater.UpdateError, match="No update backup"):
        updater.rollback(install)


def test_requirements_change_reinstalls_packages(install) -> None:
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    remote = {**REMOTE, "requirements.txt": "pandas==2\n"}
    result = updater.apply(install, http_get=_github(_zip(remote)), run=lambda cmd, **k: calls.append(cmd) or Done())
    assert calls and calls[0][-2:] == ["-r", str(install / "requirements.txt")]
    assert result["packages"]["ok"] is True


def test_refuses_unsafe_archives_and_busy_state(install) -> None:
    evil = _zip({"../../outside.txt": "pwned"})
    with pytest.raises(updater.UpdateError, match="unsafe path"):
        updater.apply(install, http_get=_github(evil))
    assert not (install.parent.parent / "outside.txt").exists()
    with pytest.raises(updater.UpdateError, match="Stop the paper bot"):
        updater.apply(install, busy="Stop the paper bot before updating", http_get=_github(_zip(REMOTE)))
    assert (install / "core" / "a.py").read_text() == "old a"


def test_already_up_to_date(install) -> None:
    result = updater.apply(install, http_get=_github(_zip(REMOTE), latest=OLD))
    assert result == {"updated": False, "message": "Already up to date.", "version": OLD}


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_git_install_pulls_fast_forward(tmp_path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.com")
    _git(origin, "config", "user.name", "t")
    (origin / "a.py").write_text("v1")
    _git(origin, "add", ".")
    _git(origin, "commit", "-qm", "v1")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    (origin / "a.py").write_text("v2")
    _git(origin, "commit", "-qam", "v2")
    latest = _git(origin, "rev-parse", "HEAD")

    get = lambda url, headers, timeout: (Resp(payload={"sha": latest, "commit": {}}) if "/commits/" in url  # noqa: E731
                                         else Resp(payload={"commits": [{"commit": {"message": "v2"}}]}))
    assert updater.install_kind(clone) == "git"
    result = updater.apply(clone, http_get=get)
    assert result["updated"] and (clone / "a.py").read_text() == "v2"
    assert updater.installed_version(clone) == latest

    (clone / "a.py").write_text("my local edit")                      # user changed a tracked file
    (origin / "a.py").write_text("v3")
    _git(origin, "commit", "-qam", "v3")
    latest3 = _git(origin, "rev-parse", "HEAD")
    get3 = lambda url, headers, timeout: Resp(payload={"sha": latest3, "commit": {}, "commits": []})  # noqa: E731
    with pytest.raises(updater.UpdateError, match="changed files"):
        updater.apply(clone, http_get=get3)
    assert (clone / "a.py").read_text() == "my local edit"


def test_restart_starts_new_dashboard_then_exits(tmp_path) -> None:
    import threading

    launched, exited = [], threading.Event()
    updater.restart_dashboard(tmp_path, popen=lambda cmd, **kw: launched.append((cmd, kw)),
                              exit_fn=lambda code: exited.set(), delay=0.05)
    assert exited.wait(2)
    cmd, kw = launched[0]
    assert cmd[-1].endswith("start_dashboard.py") and kw["env"]["WAIT_FOR_PORT"] == "1"
    assert kw["env"]["NO_BROWSER"] == "1"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_update_api_guards(client, manager, monkeypatch) -> None:
    import config.api.updates as updates
    from tests.dashboard_helpers import remote

    client.app.include_router(updates.router, prefix="/api/updates")

    def fake_apply(root, busy=None):
        if busy:
            raise updater.UpdateError(busy)
        return {"updated": True}

    monkeypatch.setattr(updates.updater, "apply", fake_apply)
    assert client.post("/api/updates/apply", json={}).status_code == 400                  # needs confirm
    assert remote(client).post("/api/updates/apply", json={"confirm": True}).status_code == 403
    assert client.post("/api/updates/apply", json={"confirm": True}).json() == {"updated": True}

    manager.start_bot("paper", ["AAPL"], "5m", execute=False)
    resp = client.post("/api/updates/apply", json={"confirm": True})
    assert resp.status_code == 409 and "Stop the paper bot" in resp.json()["detail"]

    restarted = []
    monkeypatch.setattr(updates.updater, "restart_dashboard", lambda root: restarted.append(root))
    assert client.post("/api/updates/restart").json()["restarting"] and restarted == [manager.root]
    assert remote(client).post("/api/updates/restart").status_code == 403

    monkeypatch.setattr(updates.updater, "check", lambda root: {"ok": True, "update_available": True})
    assert "Stop the paper bot" in client.get("/api/updates/check").json()["busy"]


def test_protection_is_case_insensitive(install) -> None:
    for rel in (".ENV", "Data/portfolios.db", "LOGS/x.log", "Models/m.joblib", "core/cache.DB", ".Env"):
        assert updater.is_protected(rel), rel
    assert not updater.is_protected(".env.example") and not updater.is_protected("core/data_utils.py")
    remote = {**REMOTE, ".ENV": "ALPACA_API_KEY=REPO", "Data/portfolios.db": "wiped"}
    updater.apply(install, http_get=_github(_zip(remote)))
    assert (install / ".env").read_text() == "ALPACA_API_KEY=mine\n"
    assert (install / "data" / "portfolios.db").read_text() == "my trades"
    assert not (install / ".ENV").exists() and not (install / "Data").exists()
