#!/usr/bin/env python3
"""Process-scoped AlpacaExecutor provider."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from typing import Tuple

from core.alpaca_executor import AlpacaExecutor


def _secret_fingerprint(value: str | None) -> str:
    """Return a stable non-secret cache key fragment."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configured_executor_cache_key(mode: str = "paper") -> Tuple[str, ...]:
    """Return a cache key that changes when Alpaca credentials/config changes."""
    if mode == "live":
        api_key, secret_key = os.getenv("ALPACA_LIVE_API_KEY"), os.getenv("ALPACA_LIVE_SECRET_KEY")
        return ("live", _secret_fingerprint(api_key), _secret_fingerprint(secret_key),
                os.getenv("LIVE_TRADING_ENABLED", ""))
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
    base_url = (
        os.getenv("ALPACA_TRADING_BASE_URL")
        or os.getenv("ALPACA_BASE_URL")
        or "https://paper-api.alpaca.markets/v2"
    )
    return (
        "paper",
        _secret_fingerprint(api_key),
        _secret_fingerprint(secret_key),
        base_url.rstrip("/"),
    )


@lru_cache(maxsize=8)
def _get_cached_alpaca_executor(_cache_key: Tuple[str, ...]) -> AlpacaExecutor:
    """Create one Alpaca executor per mode/credentials/base-url configuration."""
    return AlpacaExecutor(mode=_cache_key[0])


def get_alpaca_executor() -> AlpacaExecutor:
    """Return the process-scoped Alpaca PAPER executor.

    Takes no arguments on purpose: it is a FastAPI dependency, and a parameter
    would become a query parameter a request could use to reach live trading.
    """
    return _get_cached_alpaca_executor(_configured_executor_cache_key("paper"))


def get_executor_for_mode(mode: str) -> AlpacaExecutor:
    """Paper or live executor (live raises unless keys are set and live is armed)."""
    if mode not in ("paper", "live"):
        raise ValueError(f"Unknown trading mode {mode!r}")
    return _get_cached_alpaca_executor(_configured_executor_cache_key(mode))


def clear_alpaca_executor_cache() -> None:
    """Clear cached Alpaca executors after test or configuration changes."""
    _get_cached_alpaca_executor.cache_clear()
