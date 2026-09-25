#!/usr/bin/env python3
"""
First-run setup and API-key settings for the dashboard.

Security model:
- Everything here that reads or changes secrets is admin-only, and by
  default only accepted from this machine (127.0.0.1 / ::1), because keys
  typed into a browser would otherwise cross the network in plain HTTP.
  Set SETTINGS_ALLOW_REMOTE=true only behind HTTPS.
- Keys are written to .env with owner-only permissions and are never sent
  back to the browser: responses carry masked hints ("PK…WXYZ") only.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from config.api.auth import User, get_admin_user, get_password_hash
from config.api.dependencies import get_portfolio_manager
from core.alpaca_executor import LIVE_BASE_URL, PAPER_BASE_URL, live_trading_armed
from core.alpaca_executor_provider import clear_alpaca_executor_cache
from core.env_store import env_path, mask, update_env
from core.portfolio_manager import PortfolioManager

router = APIRouter()

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

KEY_GROUPS = {
    "paper": ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
    "live": ("ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY"),
}


def remote_settings_allowed() -> bool:
    return os.getenv("SETTINGS_ALLOW_REMOTE", "false").strip().lower() in {"1", "true", "yes", "on"}


def is_local_request(request: Request) -> bool:
    """True only for a browser on this machine. Behind a reverse proxy every
    request arrives from 127.0.0.1, so any forwarding header naming a
    non-local client makes the request count as remote."""
    host = request.client.host if request.client else ""
    if host not in LOCAL_HOSTS:
        return False
    forwarded = [request.headers.get("x-forwarded-for", ""), request.headers.get("x-real-ip", "")]
    hops = [h.strip() for value in forwarded for h in value.split(",") if h.strip()]
    if request.headers.get("forwarded"):
        return False  # RFC 7239 header: a proxy is involved; don't guess
    return all(h in LOCAL_HOSTS for h in hops)


def require_local(request: Request) -> None:
    if not (is_local_request(request) or remote_settings_allowed()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="For safety, keys and live-trading settings can only be changed from the computer running "
                   "the bot (open http://127.0.0.1:8000/dashboard there).",
        )


# ---------------------------------------------------------------------------
# First-run setup
# ---------------------------------------------------------------------------

class SetupStatus(BaseModel):
    needs_admin: bool
    local_request: bool


class AdminCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_-]+$")
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8, max_length=128)


def _has_users(db: PortfolioManager) -> bool:
    return bool(db.list_users(active_only=False))


@router.get("/setup/status", response_model=SetupStatus)
async def setup_status(request: Request, db: PortfolioManager = Depends(get_portfolio_manager)):
    """Public: tells the login screen whether to offer first-run admin creation."""
    return {"needs_admin": not await run_in_threadpool(_has_users, db), "local_request": is_local_request(request)}


@router.post("/setup/admin", status_code=status.HTTP_201_CREATED)
async def create_first_admin(
    body: AdminCreate, request: Request, db: PortfolioManager = Depends(get_portfolio_manager)
):
    """Create the first admin account. Only works while no account exists, from this machine."""
    require_local(request)
    if await run_in_threadpool(_has_users, db):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account already exists; sign in.")
    hashed = await run_in_threadpool(get_password_hash, body.password)
    try:
        user = await run_in_threadpool(
            db.create_user, username=body.username, email=body.email, hashed_password=hashed, is_admin=True
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not os.getenv("JWT_SECRET_KEY"):
        # Persist a signing key so logins survive restarts (takes effect next start).
        import secrets

        await run_in_threadpool(update_env, {"JWT_SECRET_KEY": secrets.token_hex(32)})
    return {"username": user["username"], "is_admin": True}


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

class KeyUpdate(BaseModel):
    paper_key_id: Optional[str] = Field(None, max_length=128)
    paper_secret_key: Optional[str] = Field(None, max_length=256)
    live_key_id: Optional[str] = Field(None, max_length=128)
    live_secret_key: Optional[str] = Field(None, max_length=256)
    alert_webhook_url: Optional[str] = Field(None, max_length=512)


def key_status() -> Dict[str, Any]:
    groups = {}
    for mode, (key_name, secret_name) in KEY_GROUPS.items():
        key, secret = os.getenv(key_name), os.getenv(secret_name)
        groups[mode] = {
            "configured": bool(key and secret),
            "key_id_hint": mask(key),
            "secret_set": bool(secret),
        }
    webhook = os.getenv("ALERT_WEBHOOK_URL")
    return {
        **groups,
        "alerts": {"configured": bool(webhook)},
        "live_armed": live_trading_armed(),
        "env_file": str(env_path()),
        "remote_changes_allowed": remote_settings_allowed(),
    }


@router.get("/keys")
async def get_keys(current_user: User = Depends(get_admin_user)):
    """Which keys are configured (masked hints only; secrets never leave the server)."""
    return key_status()


@router.put("/keys")
async def put_keys(body: KeyUpdate, request: Request, current_user: User = Depends(get_admin_user)):
    """Save keys to .env. Blank fields are left unchanged."""
    require_local(request)
    fields = {
        "paper_key_id": "ALPACA_API_KEY",
        "paper_secret_key": "ALPACA_SECRET_KEY",
        "live_key_id": "ALPACA_LIVE_API_KEY",
        "live_secret_key": "ALPACA_LIVE_SECRET_KEY",
        "alert_webhook_url": "ALERT_WEBHOOK_URL",
    }
    updates = {}
    for field, env_name in fields.items():
        value = getattr(body, field)
        if value is not None and value.strip():
            if any(ch.isspace() for ch in value.strip()):
                raise HTTPException(status_code=400, detail=f"{field} must not contain spaces")
            updates[env_name] = value.strip()
    if "ALERT_WEBHOOK_URL" in updates and not updates["ALERT_WEBHOOK_URL"].startswith("https://"):
        raise HTTPException(status_code=400, detail="The alert webhook must be an https:// URL")

    warnings = []
    if updates.get("ALPACA_LIVE_API_KEY", "").startswith("PK"):
        warnings.append("The live key ID starts with 'PK', which is how Alpaca PAPER keys start. "
                        "Live keys usually start with 'AK'.")
    if updates.get("ALPACA_API_KEY", "").startswith("AK"):
        warnings.append("The paper key ID starts with 'AK', which is how Alpaca LIVE keys start. "
                        "Paper keys usually start with 'PK'.")
    if "ALPACA_API_KEY" in updates:
        updates.setdefault("ALPACA_TRADING_BASE_URL", PAPER_BASE_URL)

    if updates:
        await run_in_threadpool(update_env, updates)
        clear_alpaca_executor_cache()
    return {**key_status(), "saved": sorted(k for k in updates if k != "ALPACA_TRADING_BASE_URL"),
            "warnings": warnings}


@router.delete("/keys/{group}")
async def delete_keys(group: str, request: Request, current_user: User = Depends(get_admin_user)):
    """Remove a key group ('paper', 'live' or 'alerts') from .env."""
    require_local(request)
    if group == "alerts":
        names = ("ALERT_WEBHOOK_URL",)
    elif group in KEY_GROUPS:
        names = KEY_GROUPS[group]
    else:
        raise HTTPException(status_code=404, detail="Unknown key group")
    changes: Dict[str, Optional[str]] = {name: None for name in names}
    if group == "live":
        changes["LIVE_TRADING_ENABLED"] = "false"  # no live keys -> disarmed
    await run_in_threadpool(update_env, changes)
    clear_alpaca_executor_cache()
    return key_status()


def check_connection(mode: str) -> Dict[str, Any]:
    """Read-only account check with the stored keys (works for live without arming)."""
    key_name, secret_name = KEY_GROUPS[mode]
    key, secret = os.getenv(key_name), os.getenv(secret_name)
    if not key or not secret:
        return {"ok": False, "mode": mode, "message": f"No {mode} keys saved yet."}
    base = LIVE_BASE_URL if mode == "live" else PAPER_BASE_URL
    try:
        resp = requests.get(
            f"{base}/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10,
        )
    except requests.RequestException as exc:
        return {"ok": False, "mode": mode, "message": f"Could not reach Alpaca: {exc.__class__.__name__}"}
    if resp.status_code in (401, 403):
        other = "live" if mode == "paper" else "paper"
        return {"ok": False, "mode": mode,
                "message": f"Alpaca rejected these keys ({resp.status_code}). Check you copied both parts, and "
                           f"that they are {mode} keys, not {other} keys."}
    if not resp.ok:
        return {"ok": False, "mode": mode, "message": f"Alpaca returned HTTP {resp.status_code}"}
    account = resp.json()
    return {
        "ok": True,
        "mode": mode,
        "message": f"Connected to your Alpaca {mode} account.",
        "account_status": account.get("status"),
        "account_number_hint": mask(str(account.get("account_number") or "")),
        "equity": float(account.get("equity") or 0),
        "buying_power": float(account.get("buying_power") or 0),
        "pattern_day_trader": bool(account.get("pattern_day_trader")),
        "trading_blocked": bool(account.get("trading_blocked")),
    }


@router.post("/test/{mode}")
async def post_test_connection(mode: str, request: Request, current_user: User = Depends(get_admin_user)):
    if mode not in KEY_GROUPS:
        raise HTTPException(status_code=404, detail="Unknown mode")
    require_local(request)
    return await run_in_threadpool(check_connection, mode)


# ---------------------------------------------------------------------------
# Start with the computer
# ---------------------------------------------------------------------------

class AutostartRequest(BaseModel):
    enabled: bool


@router.get("/autostart")
async def get_autostart(current_user: User = Depends(get_admin_user)):
    from core import autostart

    return {**autostart.status(), "auto_resume": _auto_resume()}


@router.put("/autostart")
async def put_autostart(body: AutostartRequest, request: Request, current_user: User = Depends(get_admin_user)):
    """Start the dashboard (and so any bots that should be running) when you log in."""
    require_local(request)
    from core import autostart

    try:
        result = await run_in_threadpool(autostart.enable if body.enabled else autostart.disable)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not change the login item: {exc}") from exc
    return {**result, "auto_resume": _auto_resume()}


def _auto_resume() -> bool:
    return os.getenv("AUTO_RESUME_BOTS", "true").strip().lower() not in {"0", "false", "no", "off"}


class ToggleRequest(BaseModel):
    enabled: bool


@router.put("/auto-revalidate")
async def put_auto_revalidate(body: ToggleRequest, request: Request, current_user: User = Depends(get_admin_user)):
    """Turn the weekly automatic re-validation on or off (saved to .env)."""
    require_local(request)
    await run_in_threadpool(update_env, {"AUTO_REVALIDATE": "true" if body.enabled else "false"})
    return {"enabled": body.enabled}
