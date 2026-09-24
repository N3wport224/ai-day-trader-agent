from __future__ import annotations

from pathlib import Path

from core.portfolio_manager import PortfolioManager
from scripts import create_admin


def test_create_admin_from_environment(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_USERNAME", "rootuser")
    monkeypatch.setenv("ADMIN_EMAIL", "rootuser@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me-12345")

    assert create_admin.main() == 0

    db = PortfolioManager(str(db_path))
    user = db.get_user_by_username("rootuser")
    assert user is not None
    assert user["email"] == "rootuser@example.com"
    assert user["is_admin"] == 1


def test_create_admin_rejects_duplicates(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_USERNAME", "rootuser")
    monkeypatch.setenv("ADMIN_EMAIL", "rootuser@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me-12345")

    assert create_admin.main() == 0
    assert create_admin.main() == 1


def test_create_admin_reports_short_username_accurately(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "admin.db"))
    monkeypatch.setenv("ADMIN_USERNAME", "me")
    monkeypatch.setenv("ADMIN_EMAIL", "me@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me-12345")

    assert create_admin.main() == 1

    err = capsys.readouterr().err
    assert "3-50 characters" in err
    assert "already exists" not in err


def test_create_user_reports_duplicate_email(tmp_path: Path) -> None:
    db = PortfolioManager(str(tmp_path / "users.db"))
    db.create_user("firstuser", "same@example.com", "hash")

    try:
        db.create_user("seconduser", "same@example.com", "hash")
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate email should be rejected")
