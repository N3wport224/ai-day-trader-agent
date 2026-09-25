"""Shared helpers for the dashboard API tests (fixtures live in conftest.py)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from config.api.auth import User

ADMIN = User(id=1, username="owner", email="owner@example.com", is_active=True, is_admin=True,
             created_at="2026-01-01T00:00:00")


def remote(client: TestClient) -> TestClient:
    """The same app, as seen from another machine on the network."""
    return TestClient(client.app, client=("203.0.113.9", 50000))
