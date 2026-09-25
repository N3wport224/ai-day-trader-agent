#!/usr/bin/env python3
"""
Keep the validated strategy fresh without anyone remembering to do it.

The edge report expires after EDGE_REPORT_MAX_AGE_DAYS (30), after which the
bot stops opening trades. The dashboard's supervisor therefore:

1. Re-runs the validation (walk-forward test, then retraining) with the
   settings you last validated with, once the report is REVALIDATE_EVERY_DAYS
   (7) old, only outside market hours (weekends, or before 8:00 / after
   18:00 ET on weekdays), retrying at most every 12 hours.
2. Announces the outcome (telemetry + phone alert). If the strategy no longer
   passes, running bots stop opening trades (the per-cycle edge gate).
3. After a successful run, restarts running bots outside market hours so
   they load the newly trained model.

AUTO_REVALIDATE=false turns this off.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from core.bot_manager import MODES, BotManager
from core.edge_gate import report_path
from core.execution_telemetry import EventLog, default_alert_sink

logger = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("America/New_York")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def auto_revalidate_enabled() -> bool:
    return os.getenv("AUTO_REVALIDATE", "true").strip().lower() not in {"0", "false", "no", "off"}


def off_hours(now: datetime) -> bool:
    """True when heavy work (and bot restarts) can't interfere with trading."""
    local = now.astimezone(MARKET_TZ)
    return local.weekday() >= 5 or local.hour < 8 or local.hour >= 18


def _parse(ts: Any) -> Optional[datetime]:
    try:
        value = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class Revalidator:
    def __init__(self, manager: BotManager, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.manager = manager
        self.now_fn = now_fn
        self.every = timedelta(days=_env_float("REVALIDATE_EVERY_DAYS", 7))
        self.retry = timedelta(hours=_env_float("REVALIDATE_RETRY_HOURS", 12))
        self.state_path = manager.root / "logs" / "validate" / "scheduler.json"
        self.telemetry = EventLog(str(manager.root / "logs" / "validate" / "events.jsonl"), alerts=default_alert_sink())

    # -- state ---------------------------------------------------------------

    def _state(self) -> Dict[str, Any]:
        return _read_json(self.state_path)

    def _save(self, **changes: Any) -> None:
        state = {**self._state(), **changes}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def last_run(self) -> Dict[str, Any]:
        return _read_json(self.manager.root / "reports" / "validation" / "last_run.json")

    def report_created(self) -> Optional[datetime]:
        return _parse(_read_json(report_path()).get("created_at"))

    # -- schedule ------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """For the dashboard: when the next automatic re-validation is due."""
        created = self.report_created()
        max_age = timedelta(days=_env_float("EDGE_REPORT_MAX_AGE_DAYS", 30))
        now = self.now_fn()
        return {
            "enabled": auto_revalidate_enabled(),
            "every_days": self.every.days,
            "has_settings": bool(self.manager.last_validation_settings()),
            "report_created_at": created.isoformat() if created else None,
            "report_age_days": round((now - created).total_seconds() / 86400, 1) if created else None,
            "expires_at": (created + max_age).isoformat() if created else None,
            "next_due_at": (created + self.every).isoformat() if created else None,
            "last_run": self.last_run() or None,
        }

    def due(self) -> Optional[str]:
        """Why a scheduled run should start now, or None."""
        if not auto_revalidate_enabled():
            return None
        settings = self.manager.last_validation_settings()
        created = self.report_created()
        if not settings or created is None:
            return None  # never validated from the dashboard: nothing to repeat
        if not (os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_LIVE_API_KEY")):
            return None  # no market-data keys: a run could only fail
        if self.manager.validation.running():
            return None
        now = self.now_fn()
        if not off_hours(now) or now - created < self.every:
            return None
        attempted = _parse(self._state().get("last_attempt_at"))
        if attempted and now - attempted < self.retry:
            return None
        return f"validation is {(now - created).days} days old"

    # -- actions -------------------------------------------------------------

    def tick(self) -> List[Dict[str, Any]]:
        """One supervisor pass: start a due run, announce a finished one, refresh bots."""
        actions: List[Dict[str, Any]] = []
        reason = self.due()
        if reason:
            settings = self.manager.last_validation_settings()
            self._save(last_attempt_at=self.now_fn().isoformat())
            try:
                self.manager.start_validation(settings["symbols"], settings["timeframe"], settings["days"],
                                              settings["market"], trigger="scheduled")
                actions.append({"action": "revalidation_started", "reason": reason})
                self.telemetry.record("revalidation_started", reason=reason)
            except (KeyError, ValueError, RuntimeError, OSError) as exc:
                actions.append({"action": "revalidation_failed_to_start", "reason": str(exc)})
        actions += self._announce()
        actions += self._refresh_bots()
        return actions

    def _announce(self) -> List[Dict[str, Any]]:
        run = self.last_run()
        finished = run.get("finished_at")
        if not finished or self.manager.validation.running() or self._state().get("announced") == finished:
            return []
        self._save(announced=finished)
        level = logging.INFO if run.get("result") == "validated" else logging.WARNING
        self.telemetry.record("revalidation", level, result=run.get("result"), trigger=run.get("trigger"))
        return [{"action": "revalidation_finished", "result": run.get("result")}]

    def _refresh_bots(self) -> List[Dict[str, Any]]:
        """Restart running bots (outside market hours) so they load the model a
        successful validation just trained. The supervisor starts them again."""
        run = self.last_run()
        finished = _parse(run.get("finished_at"))
        if run.get("result") != "validated" or finished is None or not off_hours(self.now_fn()):
            return []
        actions = []
        for mode in MODES:
            slot = self.manager.bots[mode]
            started = _parse(slot.meta().get("started_at"))
            if not slot.running() or not self.manager.desired(mode).get("running"):
                continue
            if started is None or started >= finished:
                continue
            if slot.stop():  # graceful; desired state stays "running", so it is restarted
                actions.append({"action": "bot_refresh", "mode": mode})
                logger.info(f"Restarting the {mode} bot to load the newly validated model")
        return actions
