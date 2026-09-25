from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone

import pytest

import config.api.control as control
import config.api.settings as settings
from config.api.auth import get_password_hash
from core.alpaca_executor import AlpacaExecutor
from core.bot_manager import clean_symbols
from core.edge_gate import build_verdict, write_report
from core.env_store import mask, update_env
from tests.dashboard_helpers import remote



# ---------------------------------------------------------------------------
# .env store
# ---------------------------------------------------------------------------

def test_update_env_preserves_file_and_is_private(env_file) -> None:
    update_env({"ALPACA_API_KEY": "PKNEW", "NEW_ONE": "a b#c", "WATCHLIST": None})
    text = env_file.read_text()
    assert text.splitlines() == ["# my settings", "ALPACA_API_KEY=PKNEW", 'NEW_ONE="a b#c"']
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert os.environ["ALPACA_API_KEY"] == "PKNEW" and "WATCHLIST" not in os.environ
    from dotenv import dotenv_values

    assert dotenv_values(env_file)["NEW_ONE"] == "a b#c"
    with pytest.raises(ValueError):
        update_env({"X": "line1\nline2"})
    with pytest.raises(ValueError):
        update_env({"bad name": "x"})


def test_mask() -> None:
    assert mask("PKABCDEFGH1234") == "PK…1234"
    assert mask(None) is None and mask("abc") == "•••"


# ---------------------------------------------------------------------------
# First-run setup and keys
# ---------------------------------------------------------------------------

def test_first_run_admin_creation(client, portfolio_manager, monkeypatch) -> None:
    monkeypatch.setattr(portfolio_manager, "list_users", lambda active_only=True: [])
    assert client.get("/api/settings/setup/status").json() == {"needs_admin": True, "local_request": True}
    body = {"username": "owner2", "email": "o@example.com", "password": "correct horse"}
    assert remote(client).post("/api/settings/setup/admin", json=body).status_code == 403
    resp = client.post("/api/settings/setup/admin", json=body)
    assert resp.status_code == 201 and resp.json()["is_admin"] is True
    assert os.environ.get("JWT_SECRET_KEY")  # persisted so logins survive restarts
    monkeypatch.setattr(portfolio_manager, "list_users", lambda active_only=True: [{"id": 1}])
    assert client.post("/api/settings/setup/admin", json=body).status_code == 409


def test_save_keys_masks_and_never_returns_secrets(client, env_file) -> None:
    resp = client.put("/api/settings/keys", json={
        "paper_key_id": "PKPAPERKEY0001", "paper_secret_key": "paper-secret-value",
        "live_key_id": "PKOOPSWRONG999", "live_secret_key": "live-secret-value",
        "alert_webhook_url": "https://discord.example/hook",
    })
    body = resp.json()
    assert resp.status_code == 200
    assert body["paper"] == {"configured": True, "key_id_hint": "PK…0001", "secret_set": True}
    assert body["live"]["configured"] and body["alerts"]["configured"]
    assert any("PAPER keys start" in w for w in body["warnings"])
    text = json.dumps(body) + json.dumps(client.get("/api/settings/keys").json())
    for secret in ("paper-secret-value", "live-secret-value", "PKPAPERKEY0001", "discord.example"):
        assert secret not in text
    saved = env_file.read_text()
    assert "ALPACA_SECRET_KEY=paper-secret-value" in saved and "ALPACA_TRADING_BASE_URL=" in saved
    # Blank fields leave existing values alone.
    client.put("/api/settings/keys", json={"paper_key_id": "", "live_secret_key": "  "})
    assert os.environ["ALPACA_SECRET_KEY"] == "paper-secret-value"


def test_key_changes_refused_from_other_machines(client, monkeypatch) -> None:
    resp = remote(client).put("/api/settings/keys", json={"paper_key_id": "PKX", "paper_secret_key": "s"})
    assert resp.status_code == 403 and "computer running the bot" in resp.json()["detail"]
    assert remote(client).get("/api/settings/keys").status_code == 200  # masked status is fine
    monkeypatch.setenv("SETTINGS_ALLOW_REMOTE", "true")
    assert remote(client).put("/api/settings/keys", json={"paper_key_id": "PKX"}).status_code == 200


def test_invalid_webhook_and_delete_live_disarms(client) -> None:
    assert client.put("/api/settings/keys", json={"alert_webhook_url": "http://x"}).status_code == 400
    update_env({"ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "s", "LIVE_TRADING_ENABLED": "true"})
    body = client.delete("/api/settings/keys/live").json()
    assert body["live"]["configured"] is False and body["live_armed"] is False


def test_connection_check(client, monkeypatch) -> None:
    assert client.post("/api/settings/test/live").json()["ok"] is False  # no keys yet
    update_env({"ALPACA_LIVE_API_KEY": "AKLIVE", "ALPACA_LIVE_SECRET_KEY": "s"})
    seen = {}

    class Resp:
        status_code, ok = 200, True

        def json(self):
            return {"status": "ACTIVE", "account_number": "123456789", "equity": "2500", "buying_power": "5000"}

    def fake_get(url, headers, timeout):
        seen["url"] = url
        return Resp()

    monkeypatch.setattr(settings.requests, "get", fake_get)
    body = client.post("/api/settings/test/live").json()
    assert seen["url"] == "https://api.alpaca.markets/v2/account"  # live endpoint, read-only, no arming needed
    assert body["ok"] and body["equity"] == 2500 and body["account_number_hint"] == "12…6789"

    class Denied(Resp):
        status_code, ok = 403, False

    monkeypatch.setattr(settings.requests, "get", lambda url, headers, timeout: Denied())
    assert "rejected" in client.post("/api/settings/test/live").json()["message"]


# ---------------------------------------------------------------------------
# Executor modes
# ---------------------------------------------------------------------------

def test_live_executor_interlocks(env_file, monkeypatch) -> None:
    with pytest.raises(ValueError, match="ALPACA_LIVE_API_KEY"):
        AlpacaExecutor(mode="live")
    update_env({"ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "s"})
    with pytest.raises(ValueError, match="not armed"):
        AlpacaExecutor(mode="live")
    assert AlpacaExecutor(mode="live", require_armed=False).base_url == "https://api.alpaca.markets/v2"
    update_env({"LIVE_TRADING_ENABLED": "true"})
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://evil.example/v2")  # config can't redirect live
    live = AlpacaExecutor(mode="live")
    assert live.base_url == "https://api.alpaca.markets/v2" and live.headers["APCA-API-KEY-ID"] == "AK1"
    with pytest.raises(ValueError, match="Unknown trading mode"):
        AlpacaExecutor(mode="real")


def test_paper_dependency_cannot_be_switched_to_live() -> None:
    import inspect

    from core.alpaca_executor_provider import get_alpaca_executor

    assert inspect.signature(get_alpaca_executor).parameters == {}


# ---------------------------------------------------------------------------
# Bot control
# ---------------------------------------------------------------------------

def test_clean_symbols() -> None:
    assert clean_symbols("aapl, msft MSFT brk.b") == ["AAPL", "MSFT", "BRK.B"]
    with pytest.raises(ValueError):
        clean_symbols("AAPL;rm -rf")
    with pytest.raises(ValueError):
        clean_symbols("")


def test_paper_bot_start_status_stop(client, manager, monkeypatch) -> None:
    assert client.post("/api/control/paper/start", json={"symbols": ["AAPL"]}).status_code == 400  # no keys
    update_env({"ALPACA_API_KEY": "PK1", "ALPACA_SECRET_KEY": "s"})
    monkeypatch.setattr(control, "_ensure_portfolio", lambda db, mode: None)
    resp = client.post("/api/control/paper/start", json={"symbols": ["aapl", "msft"], "timeframe": "5m"})
    assert resp.status_code == 200 and resp.json()["running"]
    assert resp.json()["settings"] == {"symbols": ["AAPL", "MSFT"], "timeframe": "5m", "execute": True}
    assert "--execute" in manager.bots["paper"].tracked.command
    assert client.post("/api/control/paper/start", json={"symbols": ["AAPL"]}).status_code == 400  # already running

    status = client.get("/api/control/paper/status").json()
    assert status["running"] and status["keys_configured"] and "starting" in status["log"][-1]
    assert client.get("/api/control/overview").json()["bots"]["paper"]["running"]

    assert client.post("/api/control/paper/stop").json()["stopping"] is True
    manager.bots["paper"].tracked.proc.wait(timeout=5)
    assert not client.get("/api/control/paper/status").json()["running"]


def test_bad_timeframe_and_symbols_rejected(client, monkeypatch) -> None:
    update_env({"ALPACA_API_KEY": "PK1", "ALPACA_SECRET_KEY": "s"})
    monkeypatch.setattr(control, "_ensure_portfolio", lambda db, mode: None)
    assert client.post("/api/control/paper/start", json={"symbols": ["AAPL"], "timeframe": "2h"}).status_code == 400
    assert client.post("/api/control/paper/start", json={"symbols": ["$(x)"]}).status_code == 400
    assert client.post("/api/control/crypto/start", json={"symbols": ["AAPL"]}).status_code == 404


def test_live_requires_arming_edge_and_confirmation(client, manager, portfolio_manager, monkeypatch) -> None:
    update_env({"ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "s"})
    monkeypatch.setattr(control, "_ensure_portfolio", lambda db, mode: None)
    start = {"symbols": ["AAPL"], "execute": True, "confirm_live": True}
    assert "not armed" in client.post("/api/control/live/start", json=start).json()["detail"]

    # Arming: exact phrase + correct password, from this machine.
    user = {"hashed_password": get_password_hash("hunter2hunter2")}
    monkeypatch.setattr(portfolio_manager, "get_user_by_username", lambda name: user)
    arm = {"phrase": control.ARM_PHRASE, "password": "hunter2hunter2"}
    assert client.post("/api/control/live/arm", json={**arm, "phrase": "yes"}).status_code == 400
    assert client.post("/api/control/live/arm", json={**arm, "password": "wrong"}).status_code == 403
    assert remote(client).post("/api/control/live/arm", json=arm).status_code == 403
    assert client.post("/api/control/live/arm", json=arm).json() == {"live_armed": True}

    # Armed, but the strategy hasn't been validated.
    detail = client.post("/api/control/live/start", json=start).json()["detail"]
    assert "not passed validation" in detail
    write_report(build_verdict({"trades": 5, "folds": 1}, {}, ["AAPL"]))
    assert "not passed validation" in client.post("/api/control/live/start", json=start).json()["detail"]

    good = {"trades": 300, "folds": 4, "profit_factor": 1.5, "avg_r": 0.1, "positive_folds": 3,
            "max_drawdown_pct": -5}
    write_report(build_verdict(good, {}, ["AAPL"]))
    assert "Confirm" in client.post("/api/control/live/start",
                                    json={**start, "confirm_live": False}).json()["detail"]
    assert remote(client).post("/api/control/live/start", json=start).status_code == 403
    resp = client.post("/api/control/live/start", json=start)
    assert resp.status_code == 200 and "--mode" in manager.bots["live"].tracked.command
    assert manager.bots["live"].tracked.command[manager.bots["live"].tracked.command.index("--mode") + 1] == "live"

    # Disarm stops the live bot and turns the switch off.
    assert client.post("/api/control/live/disarm").json() == {"live_armed": False}
    manager.bots["live"].tracked.proc.wait(timeout=5)
    assert not manager.bots["live"].running()
    assert os.environ["LIVE_TRADING_ENABLED"] == "false"


def test_live_watch_only_needs_arming_but_not_edge(client, monkeypatch) -> None:
    update_env({"ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "s", "LIVE_TRADING_ENABLED": "true"})
    resp = client.post("/api/control/live/start", json={"symbols": ["AAPL"], "execute": False})
    assert resp.status_code == 200 and resp.json()["settings"]["execute"] is False


def test_flatten_stops_bot_and_closes_everything(client, manager, monkeypatch) -> None:
    update_env({"ALPACA_API_KEY": "PK1", "ALPACA_SECRET_KEY": "s"})
    monkeypatch.setattr(control, "_ensure_portfolio", lambda db, mode: None)
    client.post("/api/control/paper/start", json={"symbols": ["AAPL"]})
    assert client.post("/api/control/paper/flatten", json={}).status_code == 400  # needs confirm
    monkeypatch.setattr(control, "_flatten", lambda mode: {"cancelled_orders": 2, "closed": [{"symbol": "AAPL"}],
                                                           "failures": []})
    resp = client.post("/api/control/paper/flatten", json={"confirm": True})
    assert resp.json()["cancelled_orders"] == 2
    manager.bots["paper"].tracked.proc.wait(timeout=5)
    assert not manager.bots["paper"].running()


def test_account_endpoint_without_keys_and_with_error(client, monkeypatch) -> None:
    assert client.get("/api/control/paper/account").json()["connected"] is False
    update_env({"ALPACA_API_KEY": "PK1", "ALPACA_SECRET_KEY": "s"})

    def boom(mode):
        raise RuntimeError("network down")

    monkeypatch.setattr(control, "_account_snapshot", boom)
    body = client.get("/api/control/paper/account").json()
    assert body["connected"] is False and "network down" in body["message"]


def test_status_includes_heartbeat_and_events(client, manager, tmp_path) -> None:
    logs = tmp_path / "logs" / "paper"
    logs.mkdir(parents=True)
    (logs / "heartbeat.json").write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "phase": "OPEN"}))
    (logs / "execution_events.jsonl").write_text(
        json.dumps({"event": "order_submitted", "symbol": "AAPL"}) + "\nnot json\n"
        + json.dumps({"event": "order_filled", "symbol": "AAPL"}) + "\n")
    status = client.get("/api/control/paper/status").json()
    assert status["heartbeat"]["phase"] == "OPEN" and status["heartbeat"]["age_seconds"] <= 5
    assert [e["event"] for e in status["events"]] == ["order_filled", "order_submitted"]  # newest first


def test_validation_job(client, manager) -> None:
    body = {"symbols": ["AAPL"], "timeframe": "5m", "days": 120, "market": True}
    assert client.post("/api/control/validate", json=body).status_code == 400  # needs data keys
    update_env({"ALPACA_API_KEY": "PK1", "ALPACA_SECRET_KEY": "s"})
    assert client.post("/api/control/validate", json={**body, "days": 5}).status_code == 400
    resp = client.post("/api/control/validate", json=body)
    assert resp.status_code == 200 and resp.json()["running"]
    assert "--market" in manager.validation.tracked.command
    status = client.get("/api/control/validate").json()
    assert status["running"] and status["edge"]["exists"] is False
    assert client.post("/api/control/validate/stop").json()["stopping"]


def test_proxied_requests_count_as_remote(client) -> None:
    body = {"paper_key_id": "PKX", "paper_secret_key": "s"}
    assert client.put("/api/settings/keys", json=body, headers={"X-Forwarded-For": "198.51.100.7"}).status_code == 403
    assert client.put("/api/settings/keys", json=body, headers={"Forwarded": "for=198.51.100.7"}).status_code == 403
    assert client.put("/api/settings/keys", json=body, headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 200
