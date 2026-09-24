#!/usr/bin/env python3
"""Shared FastAPI dependency providers."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from core.portfolio_manager import PortfolioManager
from core.portfolio_manager_provider import (
    clear_portfolio_manager_cache,
    get_portfolio_manager,
)

__all__ = [
    "claim_legacy_portfolios",
    "clear_portfolio_manager_cache",
    "get_portfolio_manager",
]


async def claim_legacy_portfolios(db: PortfolioManager, user) -> None:
    """Hand CLI-created (unowned) portfolios to an admin, never to regular users."""
    if user.is_admin:
        await run_in_threadpool(db.claim_unowned_portfolios, user.id)
