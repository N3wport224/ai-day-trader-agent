#!/usr/bin/env python3
"""
Dashboard control of the trading bot: paper (fake money) and live (real money).

Live-money interlocks, all enforced server-side:
- Separate live API keys, saved from the API Keys tab.
- Arming: typed confirmation phrase + your password, from this machine only.
  Arming only allows live trading; it starts nothing by itself.
- Starting the live bot with orders needs a passing, current edge report
  (the strategy validated out of sample) and an explicit confirmation. The
  bot itself re-checks the edge gate and cannot bypass it in live mode.
- Disarming stops the live bot. Emergency flatten stops the bot, cancels all
  orders and closes all positions (it works even when disarmed).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from config.api.auth import User, get_admin_user, verify_password
from config.api.dependencies import get_portfolio_manager
from config.api.settings import KEY_GROUPS, require_local
from core.alpaca_executor import AlpacaExecutor, live_trading_armed
from core.alpaca_executor_provider import clear_alpaca_executor_cache
from core.bot_manager import MODES, PROJECT_ROOT, BotManager, mode_env
from core.performance import PERIODS, compare_with_backtest, equity_series, load_live_trades, summarize, trades_csv
from core.edge_gate import report_path
from core.env_store import update_env
from core.execution_telemetry import EventLog, default_alert_sink
from core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)
router = APIRouter()

ARM_PHRASE = "I UNDERSTAND THIS USES REAL MONEY"


@lru_cache(maxsize=1)
def get_bot_manager() -> BotManager:
    return BotManager()


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise HTTPException(status_code=404, detail="Unknown mode; use 'paper' or 'live'")
    return mode


def _keys_configured(mode: str) -> bool:
    key_name, secret_name = KEY_GROUPS[mode]
    return bool(os.getenv(key_name) and os.getenv(secret_name))


def _read_only_executor(mode: str) -> AlpacaExecutor:
    telemetry = EventLog(mode_env(mode)["EXECUTION_LOG_PATH"])
    return AlpacaExecutor(mode=mode, require_armed=False, telemetry=telemetry, price_lookup=lambda s: 0.0)


def edge_report_summary() -> Dict[str, Any]:
    try:
        report = json.loads(report_path().read_text())
    except (OSError, ValueError):
        return {"exists": False, "passed": False,
                "message": "Strategy not validated yet. Run 'Validate strategy' on the Get Started tab."}
    metrics = report.get("metrics") or {}
    return {
        "exists": True,
        "passed": bool(report.get("passed")),
        "failures": report.get("failures") or [],
        "created_at": report.get("created_at"),
        "symbols": report.get("symbols") or [],
        "timeframe": (report.get("setup") or {}).get("timeframe"),
        "metrics": {k: metrics.get(k) for k in ("trades", "profit_factor", "avg_r", "win_rate_pct",
                                               "max_drawdown_pct", "positive_folds", "folds")},
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/overview")
async def overview(current_user: User = Depends(get_admin_user),
                   manager: BotManager = Depends(get_bot_manager)):
    """Everything the Get Started checklist and the sidebar badges need."""
    return {
        "keys": {mode: _keys_configured(mode) for mode in MODES},
        "live_armed": live_trading_armed(),
        "edge": edge_report_summary(),
        "bots": {mode: {"running": manager.bots[mode].running()} for mode in MODES},
        "validation": {"running": manager.validation.running()},
    }


# ---------------------------------------------------------------------------
# Validation job
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=30)
    timeframe: str = "5m"
    days: int = 120
    market: bool = True


@router.post("/validate")
async def start_validation(body: ValidateRequest, current_user: User = Depends(get_admin_user),
                           manager: BotManager = Depends(get_bot_manager)):
    if not (_keys_configured("paper") or _keys_configured("live")):
        raise HTTPException(status_code=400, detail="Add your paper API keys first (they are also used for "
                                                    "market data).")
    try:
        return manager.start_validation(body.symbols, body.timeframe, body.days, body.market)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/validate")
async def validation_status(current_user: User = Depends(get_admin_user),
                            manager: BotManager = Depends(get_bot_manager)):
    return {**manager.validation.status(log_lines=120), "edge": edge_report_summary()}


@router.post("/validate/stop")
async def stop_validation(current_user: User = Depends(get_admin_user),
                          manager: BotManager = Depends(get_bot_manager)):
    return {"stopping": manager.validation.stop()}


VALIDATION_REPORT_DIR = Path("reports") / "validation"  # relative to the bot manager's root


def _read_csv_rows(path: Path, limit: int = 60) -> List[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return [row for _, row in zip(range(limit), csv.DictReader(fh))]
    except OSError:
        return []


@router.get("/validate/report")
async def validation_report(current_user: User = Depends(get_admin_user),
                            manager: BotManager = Depends(get_bot_manager)):
    """Detail tables from the last validation run (per test period, per entry hour, per regime)."""
    folder = manager.root / VALIDATION_REPORT_DIR
    return {
        "folds": _read_csv_rows(folder / "folds.csv"),
        "by_entry_hour": _read_csv_rows(folder / "by_entry_hour.csv"),
        "by_regime": _read_csv_rows(folder / "by_regime.csv"),
    }


# Per-mode routes come after the fixed paths above so /validate/... isn't taken as a mode.
@router.get("/{mode}/status")
async def bot_status(mode: str, current_user: User = Depends(get_admin_user),
                     manager: BotManager = Depends(get_bot_manager)):
    _check_mode(mode)
    return {
        **manager.bot_status(mode),
        "keys_configured": _keys_configured(mode),
        "live_armed": live_trading_armed() if mode == "live" else None,
        "events": manager.events(mode),
    }


def _account_snapshot(mode: str) -> Dict[str, Any]:
    executor = _read_only_executor(mode)
    account = executor.get_account()
    positions = executor.get_positions()
    orders = executor.get_open_orders()
    clock = executor.get_clock()
    equity, last = float(account.get("equity") or 0), float(account.get("last_equity") or 0)
    return {
        "mode": mode,
        "status": account.get("status"),
        "equity": equity,
        "cash": float(account.get("cash") or 0),
        "buying_power": float(account.get("buying_power") or 0),
        "day_pnl": round(equity - last, 2) if last else None,
        "day_pnl_pct": round((equity - last) / last * 100, 2) if last else None,
        "pattern_day_trader": bool(account.get("pattern_day_trader")),
        "daytrade_count": account.get("daytrade_count"),
        "trading_blocked": bool(account.get("trading_blocked")),
        "market_open": bool(clock.get("is_open")),
        "next_open": clock.get("next_open"),
        "next_close": clock.get("next_close"),
        "positions": [
            {
                "symbol": p.get("symbol"),
                "qty": float(p.get("qty") or 0),
                "avg_entry_price": float(p.get("avg_entry_price") or 0),
                "current_price": float(p.get("current_price") or 0),
                "market_value": float(p.get("market_value") or 0),
                "unrealized_pl": float(p.get("unrealized_pl") or 0),
                "unrealized_plpc": round(float(p.get("unrealized_plpc") or 0) * 100, 2),
            }
            for p in positions
        ],
        "open_orders": len(orders),
    }


@router.get("/{mode}/account")
async def account(mode: str, current_user: User = Depends(get_admin_user)):
    _check_mode(mode)
    if not _keys_configured(mode):
        return {"mode": mode, "connected": False, "message": f"Add your {mode} API keys on the API Keys tab."}
    try:
        return {"connected": True, **await run_in_threadpool(_account_snapshot, mode)}
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        hint = " Check that the keys are correct." if code in (401, 403) else ""
        return {"mode": mode, "connected": False,
                "message": f"Alpaca refused the request (HTTP {code}).{hint}"}
    except requests.RequestException as exc:
        return {"mode": mode, "connected": False,
                "message": f"Could not reach Alpaca ({exc.__class__.__name__}). Check your internet connection."}
    except Exception as exc:
        logger.exception(f"{mode} account snapshot failed")
        return {"mode": mode, "connected": False, "message": f"Could not load the {mode} account: {exc}"}


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=30)
    timeframe: str = "5m"
    execute: bool = True            # False = watch only (analyse, never order)
    confirm_live: bool = False      # the live tab's "Yes, trade real money" confirmation


def _ensure_portfolio(db: PortfolioManager, mode: str) -> None:
    name = "live" if mode == "live" else os.getenv("BOT_PORTFOLIO", "default")
    if not db.get_portfolio(name):
        try:
            equity = float(_read_only_executor(mode).get_account().get("equity") or 0)
        except Exception:
            equity = 0.0
        db.create_portfolio(name, equity if equity > 0 else 10_000.0)


def start_block_reason(mode: str, execute: bool) -> Optional[str]:
    """Safety conditions shared by manual starts and automatic restarts."""
    if not _keys_configured(mode):
        return f"Add your {mode} API keys on the API Keys tab first."
    if mode == "live":
        if not live_trading_armed():
            return "Live trading is not armed. Arm it on this tab first."
        if execute:
            edge = edge_report_summary()
            if not edge["passed"]:
                return ("The strategy has not passed validation, so it can't trade real money. "
                        + (edge.get("message") or "; ".join(edge.get("failures") or [])))
    return None


def auto_resume_enabled() -> bool:
    return os.getenv("AUTO_RESUME_BOTS", "true").strip().lower() not in {"0", "false", "no", "off"}


_last_supervisor_note: Dict[str, Any] = {}


def supervise_once(manager: BotManager) -> List[Dict[str, Any]]:
    """One supervisor pass: restart bots that should be running (crash, reboot)."""
    if not auto_resume_enabled():
        return []
    actions = manager.supervise(lambda mode, want: start_block_reason(mode, bool(want.get("execute"))))
    for action in actions:
        mode = action["mode"]
        key = (action["action"], action.get("reason"))
        if action["action"] != "restarted" and _last_supervisor_note.get(mode) == key:
            continue  # don't repeat the same "skipped"/"gave up" note every pass
        _last_supervisor_note[mode] = key
        level = logging.WARNING if action["action"] == "restarted" else logging.ERROR
        EventLog(mode_env(mode, manager.root)["EXECUTION_LOG_PATH"], alerts=default_alert_sink()).record(
            "bot_restarted" if action["action"] == "restarted" else "bot_restart_blocked",
            level, reason=action.get("reason") or "bot was not running (crash or computer restart)",
        )
        logger.log(level, f"Supervisor: {mode} bot {action['action']} ({action.get('reason') or 'ok'})")
    return actions


@router.post("/{mode}/start")
async def start_bot(mode: str, body: StartRequest, request: Request,
                    current_user: User = Depends(get_admin_user),
                    manager: BotManager = Depends(get_bot_manager),
                    db: PortfolioManager = Depends(get_portfolio_manager)):
    _check_mode(mode)
    if mode == "live":
        require_local(request)
    if mode == "live" and body.execute and _keys_configured(mode) and live_trading_armed() and not body.confirm_live:
        raise HTTPException(status_code=400, detail="Confirm that the live bot will trade real money.")
    blocked = start_block_reason(mode, body.execute)
    if blocked:
        raise HTTPException(status_code=400, detail=blocked)
    if body.execute:
        await run_in_threadpool(_ensure_portfolio, db, mode)
    try:
        result = manager.start_bot(mode, body.symbols, body.timeframe, body.execute)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.warning(f"{current_user.username} started the {mode} bot "
                   f"({'orders ON' if body.execute else 'watch only'}) on {', '.join(body.symbols)}")
    return result


@router.post("/{mode}/stop")
async def stop_bot(mode: str, current_user: User = Depends(get_admin_user),
                   manager: BotManager = Depends(get_bot_manager)):
    _check_mode(mode)
    stopped = manager.stop_bot(mode)
    return {"stopping": stopped, "message": "Stopping after the current cycle; open positions keep their "
                                            "broker-side stops." if stopped else "The bot was not running."}


def _performance(mode: str, period: str, root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    trades = load_live_trades(Path(mode_env(mode, root)["EDGE_MONITOR_STATE_PATH"]))
    live = summarize(trades)
    edge = edge_report_summary()
    backtest = edge.get("metrics") if edge.get("passed") else None
    result: Dict[str, Any] = {
        "mode": mode,
        "period": period,
        "live": live,
        "backtest": backtest,
        "comparison": compare_with_backtest(live, backtest),
        "recent_trades": list(reversed(trades[-50:])),
        "equity": [],
        "equity_message": None,
    }
    if not _keys_configured(mode):
        result["equity_message"] = f"Add your {mode} API keys to see the account's equity history."
        return result
    alpaca_period, timeframe = PERIODS[period]
    try:
        history = _read_only_executor(mode).get_portfolio_history(alpaca_period, timeframe)
        result["equity"] = equity_series(history)
        if not result["equity"]:
            result["equity_message"] = "No equity history for this period yet."
    except requests.RequestException as exc:
        result["equity_message"] = f"Could not load equity history from Alpaca ({exc.__class__.__name__})."
    return result


@router.get("/{mode}/performance")
async def performance(mode: str, period: str = "1M", current_user: User = Depends(get_admin_user),
                      manager: BotManager = Depends(get_bot_manager)):
    _check_mode(mode)
    if period not in PERIODS:
        raise HTTPException(status_code=400, detail=f"period must be one of {', '.join(PERIODS)}")
    return await run_in_threadpool(_performance, mode, period, manager.root)


@router.get("/{mode}/trades.csv")
async def trades_csv_download(mode: str, current_user: User = Depends(get_admin_user),
                              manager: BotManager = Depends(get_bot_manager)):
    _check_mode(mode)
    trades = load_live_trades(Path(mode_env(mode, manager.root)["EDGE_MONITOR_STATE_PATH"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Response(
        content=trades_csv(trades),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{mode}_trades_{stamp}.csv"'},
    )


class FlattenRequest(BaseModel):
    confirm: bool = False


def _flatten(mode: str) -> Dict[str, Any]:
    report = _read_only_executor(mode).flatten_all("manual_dashboard")
    return {"cancelled_orders": report.cancelled, "closed": report.closed, "failures": report.failures}


@router.post("/{mode}/flatten")
async def flatten(mode: str, body: FlattenRequest, request: Request,
                  current_user: User = Depends(get_admin_user),
                  manager: BotManager = Depends(get_bot_manager)):
    """Emergency: stop the bot, cancel every order and close every position."""
    _check_mode(mode)
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirm the flatten.")
    if mode == "live":
        require_local(request)
    if not _keys_configured(mode):
        raise HTTPException(status_code=400, detail=f"No {mode} API keys saved.")
    manager.stop_bot(mode)
    try:
        return await run_in_threadpool(_flatten, mode)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Flatten failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Live arming
# ---------------------------------------------------------------------------

class ArmRequest(BaseModel):
    phrase: str = Field(..., max_length=100)
    password: str = Field(..., max_length=128)


@router.post("/live/arm")
async def arm_live(body: ArmRequest, request: Request, current_user: User = Depends(get_admin_user),
                   db: PortfolioManager = Depends(get_portfolio_manager)):
    require_local(request)
    if body.phrase.strip() != ARM_PHRASE:
        raise HTTPException(status_code=400, detail=f'Type exactly: {ARM_PHRASE}')
    user = await run_in_threadpool(db.get_user_by_username, current_user.username)
    if not user or not await run_in_threadpool(verify_password, body.password, user["hashed_password"]):
        raise HTTPException(status_code=403, detail="Password is incorrect.")
    if not _keys_configured("live"):
        raise HTTPException(status_code=400, detail="Save your live API keys on the API Keys tab first.")
    await run_in_threadpool(update_env, {"LIVE_TRADING_ENABLED": "true"})
    clear_alpaca_executor_cache()
    logger.warning(f"LIVE TRADING ARMED by {current_user.username}")
    return {"live_armed": True}


@router.post("/live/disarm")
async def disarm_live(current_user: User = Depends(get_admin_user),
                      manager: BotManager = Depends(get_bot_manager)):
    """Always allowed (from anywhere): turning real-money trading off is never risky."""
    manager.stop_bot("live")
    await run_in_threadpool(update_env, {"LIVE_TRADING_ENABLED": "false"})
    clear_alpaca_executor_cache()
    logger.warning(f"Live trading disarmed by {current_user.username}")
    return {"live_armed": False}
