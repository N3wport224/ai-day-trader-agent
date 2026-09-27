#!/usr/bin/env python3
"""
Ready for real money? A go-live scorecard built from the paper account.

Paper trading exists to answer one question: does the strategy behave live
the way the validation test said it would? This turns the paper record into
a checklist. Each item is ok / warn / fail with a plain-language reason:

  validation        the strategy passes validation and it hasn't expired
  selftest          a self-test with the order path passed in the last 14 days
  track record      at least READINESS_MIN_TRADES (20) closed paper trades...
  trading days      ...spread over at least READINESS_MIN_DAYS (10) days
  vs the test       paper results in line with the validation (not "behind")
  paper P&L         net positive (a warning only: small samples are noisy)
  execution costs   real fill slippage within the backtest's cost assumption
                    plus READINESS_SLIPPAGE_TOLERANCE_BPS (5)
  safety incidents  no unprotected position or "not flat at the close" in the
                    last 30 days (other incidents are warnings)

Any "fail" means not ready. Starting the live bot while not ready needs an
explicit acknowledgment (LIVE_REQUIRE_READINESS=false turns the check off).
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from core.health import ORDER, Check, check_selftest

MARKET_TZ = ZoneInfo("America/New_York")
DEFAULT_COST_BPS = 6.0          # backtest default: 5 bps slippage + half of a 2 bps spread
CRITICAL_INCIDENTS = {"unprotected_position": "position without a stop-loss",
                      "not_flat": "still holding positions near the close"}
WARNING_INCIDENTS = {"protection_unverified": "stop-loss could not be verified after a fill",
                     "entry_chase_error": "entry order handling error",
                     "bot_error": "bot error",
                     "trailing_stop_failed": "trailing stop could not be moved"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def readiness_required() -> bool:
    return os.getenv("LIVE_REQUIRE_READINESS", "true").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ReadinessCriteria:
    min_trades: int = 20
    min_days: int = 10
    slippage_tolerance_bps: float = 5.0
    incident_days: int = 30

    @classmethod
    def from_env(cls) -> "ReadinessCriteria":
        return cls(
            min_trades=int(_env_float("READINESS_MIN_TRADES", cls.min_trades)),
            min_days=int(_env_float("READINESS_MIN_DAYS", cls.min_days)),
            slippage_tolerance_bps=_env_float("READINESS_SLIPPAGE_TOLERANCE_BPS", cls.slippage_tolerance_bps),
        )


def _parse(ts: Any) -> Optional[datetime]:
    try:
        value = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def read_events(path: Path, since: datetime, names: Iterable[str], max_lines: int = 50_000) -> List[Dict[str, Any]]:
    """Events of the given kinds from a telemetry JSONL file, newer than ``since``
    (only the last ``max_lines`` lines are scanned)."""
    wanted = set(names)
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as fh:
            lines = deque(fh, maxlen=max_lines)
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") in wanted and (_parse(event.get("ts")) or since) >= since:
            events.append(event)
    return events


def assess(
    *,
    trades: List[Dict[str, Any]],
    summary: Dict[str, Any],
    comparison: Dict[str, Any],
    edge: Dict[str, Any],
    selftest: Dict[str, Any],
    events: List[Dict[str, Any]],
    now: Optional[datetime] = None,
    criteria: Optional[ReadinessCriteria] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    c = criteria or ReadinessCriteria.from_env()
    checks: List[Check] = []

    # 1. Validation
    if edge.get("passed"):
        checks.append(Check("validation", "Strategy validation", "ok", "Passes on data it had never seen"))
    else:
        checks.append(Check("validation", "Strategy validation", "fail",
                            edge.get("message") or "; ".join(edge.get("failures") or []) or "Not passing",
                            "Run Validate strategy on Get Started. Live trading needs a pass."))

    # 2. Self-test (with the order path, recent)
    st = check_selftest(selftest, now)
    status = "ok" if st.status == "ok" else "fail"
    checks.append(Check("selftest", "Paper self-test", status, st.detail,
                        "" if status == "ok" else "Paper tab > Test my setup, during market hours."))

    # 3-4. Track record
    n = summary.get("trades") or 0
    checks.append(Check("trades", "Paper track record", "ok" if n >= c.min_trades else "fail",
                        f"{n} closed trade{'s' if n != 1 else ''} (need {c.min_trades})",
                        "" if n >= c.min_trades else "Keep the paper bot running; results before this are mostly luck."))
    days = {(_parse(t.get("exit_at")) or now).astimezone(MARKET_TZ).date() for t in trades if t.get("exit_at")}
    checks.append(Check("days", "Days of paper trading", "ok" if len(days) >= c.min_days else "fail",
                        f"Trades on {len(days)} day{'s' if len(days) != 1 else ''} (need {c.min_days})",
                        "" if len(days) >= c.min_days else
                        "Results from a few days can be one lucky or unlucky market; keep going."))

    # 5. Versus the validation test
    verdict = comparison.get("status")
    mapping = {"on_track": "ok", "watch": "warn", "behind": "fail", "too_early": "fail",
               "no_backtest": "fail", "unknown": "warn"}
    checks.append(Check("vs_test", "Paper results vs the test", mapping.get(verdict, "warn"),
                        comparison.get("message") or "No comparison yet"))

    # 6. Paper P&L
    pnl = summary.get("total_pnl")
    if n:
        checks.append(Check("pnl", "Paper profit", "ok" if (pnl or 0) > 0 else "warn",
                            f"${pnl:+,.2f} over {n} trades; profit factor {summary.get('profit_factor') or 'n/a'}",
                            "" if (pnl or 0) > 0 else "Net negative so far. Don't risk real money on it yet."))

    # 7. Execution costs vs the backtest's assumption
    assumed = (edge.get("metrics") or {}).get("cost_bps_per_side")
    assumed = float(assumed) if assumed is not None else DEFAULT_COST_BPS
    slips = [float(e["slippage_bps"]) for e in events
             if e.get("event") == "order_filled" and e.get("slippage_bps") is not None]
    if len(slips) < 10:
        checks.append(Check("costs", "Real trading costs", "info",
                            f"{len(slips)} measured fills; need 10 to compare with the test's {assumed:g} bps"))
    else:
        mean = sum(slips) / len(slips)
        within = mean <= assumed + c.slippage_tolerance_bps
        checks.append(Check("costs", "Real trading costs", "ok" if within else "warn",
                            f"Fills averaged {mean:+.1f} bps vs {assumed:g} bps assumed in the test "
                            f"({len(slips)} fills)",
                            "" if within else "Real costs are higher than the test assumed, so it flatters the "
                                              "strategy. Re-validate with a higher --slippage-bps, or trade more "
                                              "liquid stocks."))

    # 8. Safety incidents
    since = now - timedelta(days=c.incident_days)
    recent = [e for e in events if (_parse(e.get("ts")) or now) >= since]
    critical = [e for e in recent if e.get("event") in CRITICAL_INCIDENTS]
    minor = [e for e in recent if e.get("event") in WARNING_INCIDENTS]
    if critical:
        kinds = sorted({CRITICAL_INCIDENTS[e["event"]] for e in critical})
        checks.append(Check("incidents", "Safety incidents", "fail",
                            f"{len(critical)} in the last {c.incident_days} days: {', '.join(kinds)}",
                            "Find out why in the Paper tab's Activity and bot log before going live."))
    elif minor:
        kinds = sorted({WARNING_INCIDENTS[e["event"]] for e in minor})
        checks.append(Check("incidents", "Safety incidents", "warn",
                            f"{len(minor)} minor in the last {c.incident_days} days: {', '.join(kinds)}",
                            "Review them in the Paper tab's Activity."))
    elif not n:
        checks.append(Check("incidents", "Safety incidents", "info", "No paper trading to judge yet"))
    else:
        checks.append(Check("incidents", "Safety incidents", "ok", f"None in the last {c.incident_days} days"))

    worst = min((ORDER[ch.status] for ch in checks), default=3)
    ready = worst > ORDER["fail"]
    passed = sum(1 for ch in checks if ch.status == "ok")
    return {
        "ready": ready,
        "overall": {0: "fail", 1: "warn"}.get(worst, "ok"),
        "passed": passed,
        "total": len(checks),
        "checked_at": now.isoformat(),
        "required": readiness_required(),
        "checks": [asdict(ch) for ch in checks],
    }


def incident_and_fill_events(log_path: Path, now: Optional[datetime] = None, days: int = 30) -> List[Dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    return read_events(log_path, now - timedelta(days=days),
                       {"order_filled", *CRITICAL_INCIDENTS, *WARNING_INCIDENTS})
