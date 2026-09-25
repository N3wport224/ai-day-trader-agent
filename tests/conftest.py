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
    monkeypatch.setenv("FILL_STATE_PATH", str(tmp_path / "fill_state.json"))
    monkeypatch.setenv("EDGE_MONITOR_STATE_PATH", str(tmp_path / "edge_monitor.json"))
    monkeypatch.setenv("EDGE_REPORT_PATH", str(tmp_path / "edge_report.json"))
    # Tests feed historical synthetic bars; the live stale-bar guard is tested explicitly.
    monkeypatch.setenv("MAX_BAR_AGE_BARS", "0")
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    # Entries read open positions/stops for the portfolio heat cap; default to
    # an empty account so executor tests never touch the network.
    from core.alpaca_executor import AlpacaExecutor, BrokerSnapshot

    monkeypatch.setattr(AlpacaExecutor, "get_snapshot", lambda self: BrokerSnapshot(positions=[], open_orders=[]))
    monkeypatch.setattr(AlpacaExecutor, "get_quote", lambda self, symbol: None)  # no quote: spread filter skipped

    def _no_network_get_order(self, order_id):
        import requests

        raise requests.exceptions.ConnectionError("tests never reach Alpaca")

    monkeypatch.setattr(AlpacaExecutor, "get_order", _no_network_get_order)
    monkeypatch.setenv("FLATTEN_CONFIRM_SECONDS", "0")


# ---------------------------------------------------------------------------
# Dashboard API fixtures (tests/test_dashboard_control.py and friends)
# ---------------------------------------------------------------------------

import os  # noqa: E402
import subprocess  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import config.api.control as control  # noqa: E402
import config.api.settings as settings  # noqa: E402
from config.api.auth import get_admin_user  # noqa: E402
from config.api.dependencies import get_portfolio_manager  # noqa: E402
from core.bot_manager import BotManager  # noqa: E402
from tests.dashboard_helpers import ADMIN  # noqa: E402

@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# my settings\nWATCHLIST=AAPL,MSFT\nALPACA_API_KEY=old\n")
    monkeypatch.setenv("DOTENV_PATH", str(path))
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY",
                 "LIVE_TRADING_ENABLED", "ALERT_WEBHOOK_URL", "SETTINGS_ALLOW_REMOTE", "JWT_SECRET_KEY",
                 "AUTO_REVALIDATE"):
        monkeypatch.delenv(name, raising=False)
    yield path
    # update_env writes os.environ directly; undo so other tests are unaffected
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY",
                 "LIVE_TRADING_ENABLED", "ALERT_WEBHOOK_URL", "ALPACA_TRADING_BASE_URL", "JWT_SECRET_KEY",
                 "AUTO_REVALIDATE"):
        os.environ.pop(name, None)


@pytest.fixture
def manager(tmp_path):
    def popen(command, **kwargs):  # a harmless stand-in for bot.py
        return subprocess.Popen(["sleep", "30"], **kwargs)

    mgr = BotManager(root=tmp_path, popen=popen)
    yield mgr
    for slot in [*mgr.bots.values(), mgr.validation]:
        slot.stop()


@pytest.fixture
def client(env_file, manager, portfolio_manager):
    app = FastAPI()
    app.include_router(settings.router, prefix="/api/settings")
    app.include_router(control.router, prefix="/api/control")
    app.dependency_overrides[get_admin_user] = lambda: ADMIN
    app.dependency_overrides[get_portfolio_manager] = lambda: portfolio_manager
    app.dependency_overrides[control.get_bot_manager] = lambda: manager
    return TestClient(app, client=("127.0.0.1", 50000))
