from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_manager import PortfolioManager


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "portfolios.db"


@pytest.fixture
def portfolio_manager(test_db_path: Path) -> PortfolioManager:
    manager = PortfolioManager(str(test_db_path))
    manager.create_user(
        username="api-user",
        email="api-user@example.com",
        hashed_password="test-password-hash",
    )
    manager.create_user(
        username="other-user",
        email="other-user@example.com",
        hashed_password="test-password-hash",
    )
    return manager


@pytest.fixture(autouse=True)
def _isolate_runtime_files(tmp_path: Path, monkeypatch) -> None:
    """Keep telemetry and trailing-stop state out of the repo during tests."""
    monkeypatch.setenv("EXECUTION_LOG_PATH", str(tmp_path / "execution_events.jsonl"))
    monkeypatch.setenv("TRAILING_STATE_PATH", str(tmp_path / "trailing_state.json"))
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "heartbeat.json"))
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
