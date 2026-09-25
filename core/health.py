#!/usr/bin/env python3
"""
System check: everything that has to be right for the bot to trade, each with
a plain-language fix. Used by the dashboard's System check tab.

Every check returns {id, label, status, detail, fix} where status is
"ok", "warn", "fail" or "info". Network checks use short timeouts and never
raise: a failed check is a result, not an error.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ORDER = {"fail": 0, "warn": 1, "info": 2, "ok": 3}


@dataclass
class Check:
    id: str
    label: str
    status: str
    detail: str
    fix: str = ""


def _parse(ts: Any) -> Optional[datetime]:
    try:
        value = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Individual checks (pure where possible; callers inject network access)
# ---------------------------------------------------------------------------

def check_python() -> Check:
    v = sys.version_info
    if v < (3, 10):
        return Check("python", "Python version", "fail", f"Python {v.major}.{v.minor}",
                     "Install Python 3.12 from python.org and delete the .venv folder so it is rebuilt.")
    return Check("python", "Python version", "ok", f"Python {v.major}.{v.minor}.{v.micro}")


def check_ml_library() -> Check:
    try:
        importlib.import_module("lightgbm")
    except Exception as exc:
        fix = ("On a Mac run `brew install libomp` in Terminal, then restart the dashboard."
               if sys.platform == "darwin" else "Reinstall the packages: delete the .venv folder and start again.")
        return Check("ml", "Machine-learning library", "fail", f"LightGBM can't load ({exc.__class__.__name__})", fix)
    return Check("ml", "Machine-learning library", "ok", "LightGBM loads")


def check_keys(mode: str, result: Dict[str, Any]) -> Check:
    label = f"{mode.capitalize()} account"
    if not result.get("ok"):
        message = result.get("message") or "not connected"
        if "No " in message and "keys saved" in message:
            status = "fail" if mode == "paper" else "info"
            fix = "Add your keys on the API Keys tab." if mode == "paper" else "Only needed for real-money trading."
            return Check(f"keys_{mode}", label, status, message, fix)
        return Check(f"keys_{mode}", label, "fail", message,
                     "Check the keys on the API Keys tab (Test connection), and your internet connection.")
    if result.get("trading_blocked"):
        return Check(f"keys_{mode}", label, "fail", "Alpaca has blocked trading on this account",
                     "Log in to alpaca.markets to see why.")
    return Check(f"keys_{mode}", label, "ok",
                 f"Connected (equity ${result.get('equity', 0):,.2f}"
                 + (", flagged pattern day trader" if result.get("pattern_day_trader") else "") + ")")


def check_clock(local_now: datetime, broker_timestamp: Optional[str]) -> Check:
    broker = _parse(broker_timestamp)
    if broker is None:
        return Check("clock", "Computer clock", "info", "Couldn't compare with Alpaca's clock (not connected)")
    drift = (local_now - broker).total_seconds()
    text = f"{abs(drift):.1f}s {'ahead of' if drift > 0 else 'behind'} Alpaca"
    fix = ("Turn on automatic time: Windows Settings → Time & language → Date & time → 'Set time automatically'; "
           "Mac System Settings → General → Date & Time → 'Set time and date automatically'.")
    if abs(drift) > 30:
        return Check("clock", "Computer clock", "fail", text, fix)
    if abs(drift) > 5:
        return Check("clock", "Computer clock", "warn", text, fix)
    return Check("clock", "Computer clock", "ok", text)


def check_market_data(latest_bar: Optional[datetime], source: str, now: datetime, market_open: Optional[bool]) -> Check:
    if latest_bar is None:
        return Check("data", "Market data", "fail", "No price data from Alpaca or Yahoo",
                     "Check your internet connection and paper keys (market data uses them).")
    age = now - latest_bar
    detail = f"{source}: latest SPY 5-minute bar {latest_bar.astimezone().strftime('%b %d %H:%M')}"
    if market_open and age > timedelta(minutes=20):
        return Check("data", "Market data", "warn", detail + " (stale while the market is open)",
                     "The data feed may be delayed; the bot won't trade on stale data.")
    if source != "Alpaca":
        return Check("data", "Market data", "warn", detail,
                     "Alpaca data isn't working, so a delayed backup is used. Check your paper keys.")
    return Check("data", "Market data", "ok", detail)


def check_validation(report: Dict[str, Any], now: datetime, max_age_days: float = 30) -> Check:
    if not report:
        return Check("validation", "Strategy validation", "fail", "Not validated yet: bots won't open trades",
                     "Run 'Validate strategy' on the Get Started tab.")
    created = _parse(report.get("created_at"))
    age = (now - created).days if created else None
    if not report.get("passed"):
        return Check("validation", "Strategy validation", "fail",
                     "Last validation found no edge: bots won't open trades",
                     "See the test details on Get Started; try other liquid stocks or more history.")
    if age is None or age > max_age_days:
        return Check("validation", "Strategy validation", "fail", "Validation expired: bots won't open trades",
                     "Run 'Validate strategy' again (or leave weekly re-validation on).")
    if age > max_age_days - 7:
        return Check("validation", "Strategy validation", "warn", f"Passed {age} days ago; expires soon",
                     "Weekly re-validation should renew it; otherwise run it on Get Started.")
    when = "today" if age == 0 else f"{age} day{'s' if age != 1 else ''} ago"
    return Check("validation", "Strategy validation", "ok", f"Passed {when}")


def check_model(artifact: Optional[Dict[str, Any]], report: Dict[str, Any]) -> Check:
    if artifact is None:
        return Check("model", "Trained model", "warn", "No trained model: the bot uses its simple fallback rules",
                     "Validation trains it automatically when the strategy passes.")
    tested = (report.get("setup") or {}).get("timeframe")
    if tested and artifact.get("timeframe") and artifact["timeframe"] != tested:
        return Check("model", "Trained model", "warn",
                     f"Model is for {artifact['timeframe']} bars but the validation was on {tested}",
                     "Validate again with the bar size you trade; it retrains the model.")
    trained = _parse(artifact.get("data_end"))
    detail = f"{artifact.get('timeframe')} model on {', '.join(artifact.get('symbols') or [])}"
    if trained:
        detail += f", data up to {trained.strftime('%b %d')}"
    return Check("model", "Trained model", "ok", detail)


def check_disk(root: Path) -> Check:
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < 0.2:
        return Check("disk", "Disk space", "fail", f"{free_gb:.2f} GB free", "Free up disk space; logs and data can't be saved.")
    if free_gb < 1:
        return Check("disk", "Disk space", "warn", f"{free_gb:.1f} GB free", "Free up some disk space.")
    return Check("disk", "Disk space", "ok", f"{free_gb:.0f} GB free")


def check_bots(manager: Any, market_open: Optional[bool]) -> List[Check]:
    checks = []
    for mode in ("paper", "live"):
        desired = manager.desired(mode).get("running")
        running = manager.bots[mode].running()
        label = f"{mode.capitalize()} bot"
        if desired and not running:
            checks.append(Check(f"bot_{mode}", label, "fail", "Should be running but isn't",
                                "Check its log on the trading tab; the dashboard retries up to 3 times an hour."))
        elif running:
            beat = manager.heartbeat(mode) or {}
            age = beat.get("age_seconds")
            if market_open and age is not None and age > 900:
                checks.append(Check(f"bot_{mode}", label, "warn", f"Running, but no check-in for {age // 60} min",
                                    "It may be stuck; stop and start it on its tab."))
            else:
                checks.append(Check(f"bot_{mode}", label, "ok", "Running" + (f", last check {age}s ago" if age is not None else "")))
        else:
            checks.append(Check(f"bot_{mode}", label, "info", "Not running"))
    return checks


def check_unattended(autostart_enabled: bool) -> Check:
    if os.getenv("KEEP_AWAKE", "true").strip().lower() in {"0", "false", "no", "off"}:
        return Check("unattended", "Running unattended", "warn", "KEEP_AWAKE is off: the computer may sleep mid-day",
                     "Remove KEEP_AWAKE=false from .env.")
    if not autostart_enabled:
        return Check("unattended", "Running unattended", "info",
                     "Keeps the computer awake while bots run; won't start by itself after a reboot",
                     "Turn on 'Start the dashboard when I log in' on the API Keys tab.")
    return Check("unattended", "Running unattended", "ok", "Keeps the computer awake; starts at login")


def check_alerts() -> Check:
    if os.getenv("ALERT_WEBHOOK_URL"):
        return Check("alerts", "Phone alerts", "ok", "Webhook configured")
    return Check("alerts", "Phone alerts", "info", "Off: you won't be notified of trades or problems",
                 "Add a Discord or Slack webhook on the API Keys tab.")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_checks(
    *,
    manager: Any,
    connection: Callable[[str], Dict[str, Any]],
    broker_clock: Callable[[], Optional[Dict[str, Any]]],
    latest_bar: Callable[[], tuple],
    load_model: Callable[[], Optional[Dict[str, Any]]],
    report_file: Path,
    autostart_enabled: bool,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Dict[str, Any]:
    now = now_fn()
    checks: List[Check] = [check_python(), check_ml_library()]

    paper = _safe(lambda: connection("paper"), {"ok": False, "message": "check failed"})
    live = _safe(lambda: connection("live"), {"ok": False, "message": "check failed"})
    checks += [check_keys("paper", paper), check_keys("live", live)]

    clock = _safe(broker_clock, None) or {}
    market_open = clock.get("is_open") if clock else None
    checks.append(check_clock(now_fn(), clock.get("timestamp")))

    bar, source = _safe(latest_bar, (None, "none"))
    checks.append(check_market_data(bar, source, now, market_open))

    try:
        report = json.loads(Path(report_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    max_age = float(os.getenv("EDGE_REPORT_MAX_AGE_DAYS", "30"))
    checks += [check_validation(report, now, max_age), check_model(_safe(load_model, None), report)]
    checks += [check_disk(manager.root)]
    checks += check_bots(manager, market_open)
    checks += [check_unattended(autostart_enabled), check_alerts()]

    worst = min((ORDER[c.status] for c in checks), default=3)
    overall = {0: "fail", 1: "warn"}.get(worst, "ok")
    return {"overall": overall, "checked_at": now.isoformat(), "market_open": market_open,
            "checks": [asdict(c) for c in sorted(checks, key=lambda c: ORDER[c.status])]}


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default
