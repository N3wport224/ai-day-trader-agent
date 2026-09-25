#!/usr/bin/env python3
"""Dashboard updates: check GitHub, install (with backup), undo, restart.

Installing, undoing and restarting are admin-only and only accepted from
this computer, and never while a bot or the validation job is running.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from config.api.auth import User, get_admin_user
from config.api.control import get_bot_manager
from config.api.settings import require_local
from core import updater
from core.bot_manager import BotManager

router = APIRouter()


def busy_reason(manager: BotManager) -> Optional[str]:
    running = [mode for mode, slot in manager.bots.items() if slot.running()]
    if running:
        return (f"Stop the {' and '.join(running)} bot{'s' if len(running) > 1 else ''} before updating "
                "(open positions keep their broker-side stops).")
    if manager.validation.running():
        return "Wait for the validation run to finish before updating."
    return None


class ConfirmRequest(BaseModel):
    confirm: bool = False


@router.get("/check")
async def check_updates(current_user: User = Depends(get_admin_user),
                        manager: BotManager = Depends(get_bot_manager)):
    result = await run_in_threadpool(updater.check, manager.root)
    return {**result, "busy": busy_reason(manager)}


@router.post("/apply")
async def apply_update(body: ConfirmRequest, request: Request, current_user: User = Depends(get_admin_user),
                       manager: BotManager = Depends(get_bot_manager)):
    require_local(request)
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirm the update.")
    try:
        return await run_in_threadpool(lambda: updater.apply(manager.root, busy=busy_reason(manager)))
    except updater.UpdateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/rollback")
async def rollback_update(body: ConfirmRequest, request: Request, current_user: User = Depends(get_admin_user),
                          manager: BotManager = Depends(get_bot_manager)):
    require_local(request)
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirm the undo.")
    try:
        return await run_in_threadpool(lambda: updater.rollback(manager.root, busy=busy_reason(manager)))
    except updater.UpdateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/restart")
async def restart(request: Request, current_user: User = Depends(get_admin_user),
                  manager: BotManager = Depends(get_bot_manager)):
    """Restart the dashboard so updated code is loaded (bots keep running)."""
    require_local(request)
    updater.restart_dashboard(manager.root)
    return {"restarting": True, "message": "Restarting; the page will reconnect in a few seconds."}
