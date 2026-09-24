#!/usr/bin/env python3
"""
Scheduled trading loop: scan a watchlist during market hours and act on it.

Dry-run by default (analyze and log only). With ``execute=True`` actionable
signals go through TradingWorkflow, which applies the executor's risk checks
and bracket orders before anything reaches Alpaca paper trading.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.execution_telemetry import EventLog
from core.portfolio_manager import PortfolioManager
from core.session_clock import SessionClock, SessionPhase, to_market_time
from core.trading_workflow import TradingWorkflow, WorkflowResult

logger = logging.getLogger(__name__)

STRATEGIES = ("ml", "classic")


def create_workflow(
    portfolio_manager: PortfolioManager, strategy: str = "ml", timeframe: Optional[str] = None
) -> TradingWorkflow:
    """Build the workflow for a strategy. Both paths end in the same executor
    and RiskManager, so risk limits and bracket orders apply either way."""
    if strategy == "ml":
        from core.ml_signal_engine import MLSignalEngine

        return TradingWorkflow(
            portfolio_manager, analysis_runner=MLSignalEngine(portfolio_manager, timeframe=timeframe)
        )
    if strategy == "classic":
        return TradingWorkflow(portfolio_manager)
    raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")


@dataclass
class CycleReport:
    started_at: datetime
    market_open: bool
    results: List[WorkflowResult] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    reconciliation: Optional[Any] = None           # core.reconciliation.ReconciliationReport
    stop_adjustments: List[Any] = field(default_factory=list)
    broker_error: Optional[str] = None
    phase: str = "OPEN"                            # core.session_clock.SessionPhase value
    entry_block: Optional[str] = None              # why BUYs were vetoed this cycle
    flatten: Optional[Any] = None                  # core.alpaca_executor.FlattenReport
    session_report: Optional[Dict[str, Any]] = None  # end-of-session summary (first closed cycle)

    @property
    def orders(self) -> List[WorkflowResult]:
        return [r for r in self.results if r.alpaca_order]


def _walk_orders(orders):
    """Orders with their nested bracket legs."""
    for order in orders or []:
        yield order
        yield from _walk_orders(order.get("legs") or [])


class TradingBot:
    def __init__(
        self,
        workflow: TradingWorkflow,
        symbols: Sequence[str],
        portfolio_name: str = "default",
        *,
        execute: bool = False,
        interval_seconds: int = 900,
        market_clock: Optional[Callable[[], Dict[str, Any]]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        broker: Optional[Any] = None,
        trailing_manager: Optional[Any] = None,
        reconcile_fn: Optional[Callable[..., Any]] = None,
        session_clock: Optional[SessionClock] = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        telemetry: Optional[EventLog] = None,
        heartbeat_path: Optional[str] = None,
        timeframe: Optional[str] = None,
        fill_tracker: Optional[Any] = None,
        standing_entry_block: Optional[str] = None,
        edge_monitor: Optional[Any] = None,
    ) -> None:
        """``broker`` (an AlpacaExecutor) enables the start-of-cycle broker
        snapshot and reconciliation; ``trailing_manager`` raises stops on
        winners (execute mode only). ``session_clock`` applies the intraday
        session rules (opening lockout, entry cutoff, EOD flatten) using the
        broker's market clock. ``fill_tracker`` (core.fill_quality.FillTracker)
        measures live slippage from the broker's filled orders each cycle."""
        cleaned = [s.strip().upper() for s in symbols if s and s.strip()]
        if not cleaned:
            raise ValueError("Watchlist is empty; pass --symbols or set WATCHLIST")
        self.workflow = workflow
        self.symbols = list(dict.fromkeys(cleaned))
        self.portfolio_name = portfolio_name
        self.execute = execute
        self.interval_seconds = max(60, interval_seconds)
        self.market_clock = market_clock
        self._stop_event = threading.Event()
        self._stop_reason: Optional[str] = None
        # Default sleep is interruptible, so stop() takes effect between cycles
        # without waiting out the interval (and never interrupts an order).
        self.sleep = sleep or (lambda seconds: self._stop_event.wait(seconds))
        self.telemetry = telemetry or EventLog.from_env()
        self.heartbeat_path = Path(heartbeat_path or os.getenv("HEARTBEAT_PATH", "logs/heartbeat.json"))
        self.timeframe = timeframe
        self._session_seen: Optional[date] = None
        self._session_reported: Optional[date] = None
        self._breaker_alerted: Optional[date] = None
        self.broker = broker
        self.trailing_manager = trailing_manager
        self.fill_tracker = fill_tracker
        # Blocks every new entry for the whole run (e.g. the edge gate failed);
        # exits, brackets, trailing stops and the EOD flatten still run.
        self.standing_entry_block = standing_entry_block
        # core.edge_monitor.EdgeMonitor: pauses entries if live results decay.
        self.edge_monitor = edge_monitor
        if reconcile_fn is None and broker is not None:
            from core.reconciliation import reconcile as reconcile_fn
        self.reconcile_fn = reconcile_fn
        self.session_clock = session_clock
        self.now_fn = now_fn
        self._last_clock: Optional[Dict[str, Any]] = None
        self._last_snapshot: Optional[Any] = None

    def _read_clock(self) -> Optional[Dict[str, Any]]:
        if self.market_clock is None:
            return None
        try:
            return self.market_clock()
        except Exception as exc:  # network hiccup: skip this cycle, don't trade blind
            logger.error(f"Could not read market clock: {exc}")
            return {"is_open": False}

    def _phase(self, clock: Optional[Dict[str, Any]]) -> SessionPhase:
        if clock is None:
            return SessionPhase.OPEN  # no broker clock (offline dry run): always scan
        if self.session_clock is None:
            return SessionPhase.OPEN if clock.get("is_open") else SessionPhase.CLOSED
        return self.session_clock.phase_from_alpaca_clock(clock)

    def _market_date(self) -> date:
        return to_market_time(self.now_fn()).date()

    def run_cycle(self) -> CycleReport:
        report = self._run_cycle()
        self._write_heartbeat(report)
        return report

    def _run_cycle(self) -> CycleReport:
        clock = self._last_clock = self._read_clock()
        phase = self._phase(clock)
        report = CycleReport(
            started_at=datetime.now(timezone.utc),
            market_open=phase is not SessionPhase.CLOSED,
            phase=phase.value,
        )
        if phase is SessionPhase.CLOSED:
            self._maybe_session_report(report)
            logger.info("Market closed; skipping cycle")
            return report
        if clock is not None:
            self._session_seen = self._market_date()

        if self.broker is not None and not self._sync_with_broker(report):
            return report

        if phase is SessionPhase.FLATTEN:
            self._flatten(report)
            return report
        if self.standing_entry_block:
            report.entry_block = report.entry_block or self.standing_entry_block
        if self.edge_monitor is not None and self.edge_monitor.paused:
            report.entry_block = report.entry_block or f"edge decay: {self.edge_monitor.paused}"
        if phase is SessionPhase.OPENING_LOCKOUT:
            report.entry_block = report.entry_block or "opening lockout (opening range still forming)"
        elif phase is SessionPhase.ENTRY_CUTOFF:
            report.entry_block = report.entry_block or "end-of-day entry cutoff"
        if report.entry_block:
            logger.info(f"Session {phase.value}: new entries blocked ({report.entry_block}); exits still managed")

        for symbol in self.symbols:
            try:
                result = self.workflow.run(
                    symbol,
                    self.portfolio_name,
                    record_paper_trade=self.execute,
                    submit_alpaca_paper_order=self.execute,
                    **({"entry_block_reason": report.entry_block} if report.entry_block else {}),
                )
            except Exception as exc:
                logger.error(f"{symbol}: cycle error: {exc}")
                report.errors[symbol] = str(exc)
                continue

            report.results.append(result)
            analysis = result.analysis
            summary = (
                f"{symbol}: {analysis.get('recommendation', analysis.get('signal', '?'))} "
                f"{analysis.get('quantity', 0)} @ confidence {analysis.get('confidence', '?')}"
            )
            ml = (analysis.get("all_signals") or {}).get("ml")
            if ml:
                sentiment = (
                    f"{ml['sentiment_score']:+.2f}/{ml['sentiment_articles']} articles"
                    if ml.get("sentiment_available") else "n/a"
                )
                macro = ml.get("macro_aligned")
                mtf = "n/a" if macro is None else ("aligned" if macro == 1.0 else "not aligned")
                gate = f" vetoed by {ml['gated_by']}" if ml.get("gated_by") else ""
                summary += (
                    f" [{ml['mode']} P(up)={ml['probability_up']:.2f} regime {ml.get('regime') or 'n/a'}"
                    f" daily trend {mtf}{gate} sentiment {sentiment}]"
                )
            if result.alpaca_order:
                logger.info(f"{summary} -> ORDER {result.alpaca_order.get('id')}")
            else:
                logger.info(f"{summary} -> {result.skipped_reason or 'no order'}")
        return report

    def _sync_with_broker(self, report: CycleReport) -> bool:
        """Snapshot broker state, reconcile the local book, trail stops.
        Returns False (skip the cycle) if broker state can't be read."""
        try:
            snapshot = self.broker.get_snapshot()
        except Exception as exc:
            report.broker_error = str(exc)
            logger.error(f"Could not read broker positions/orders ({exc}); skipping cycle rather than trading blind")
            return False

        self._last_snapshot = snapshot
        breaker = getattr(getattr(self.broker, "risk_manager", None), "update_breaker", None)
        if breaker is not None:
            try:
                reason = breaker(self.broker.get_account())
            except Exception as exc:
                logger.error(f"Could not read account for the drawdown breaker ({exc}); blocking entries")
                reason = "account equity unavailable"
            if reason:
                report.entry_block = reason
                logger.warning(reason)
                today = self._market_date()
                if self._breaker_alerted != today:
                    self._breaker_alerted = today
                    self.telemetry.record("breaker_tripped", logging.WARNING, reason=reason)

        report.reconciliation = self.reconcile_fn(
            snapshot,
            self.workflow.portfolio_manager,
            self.portfolio_name,
            mode=None if self.execute else "report",
        )
        recon = report.reconciliation
        if recon.discrepancies:
            logger.warning(f"Reconciliation: {recon.kinds()} (synced: {recon.synced})")
        else:
            logger.info(
                f"Reconciliation clean: {recon.positions} positions, {recon.open_orders} open orders match local book"
            )

        self._track_fills(self._market_date())

        if self.execute and hasattr(self.broker, "cancel_stale_entries"):
            try:
                expired = self.broker.cancel_stale_entries(snapshot.open_orders)
                if expired:
                    logger.info(f"Cancelled {len(expired)} unfilled entry order(s) past their TTL")
            except Exception as exc:
                logger.warning(f"Could not cancel stale entry orders: {exc}")

        if self.execute and self.trailing_manager is not None:
            report.stop_adjustments = self.trailing_manager.update(snapshot)
            for adj in report.stop_adjustments:
                status = "raised" if adj.ok else "FAILED to raise"
                logger.info(f"{adj.symbol}: stop {status} {adj.old_stop} -> {adj.new_stop} ({adj.detail})")
        return True

    def _track_fills(self, session_date: date, orders: Optional[List[Dict[str, Any]]] = None) -> None:
        if self.fill_tracker is None or not hasattr(self.broker, "get_orders_today"):
            return
        try:
            orders = self.broker.get_orders_today() if orders is None else orders
            fills = self.fill_tracker.update(orders, session_date)
            if self.edge_monitor is not None and fills:
                self.edge_monitor.update(fills)
            for fill in fills:
                if fill.slippage_bps is not None:
                    logger.info(f"Fill {fill.side} {fill.qty:g} {fill.symbol} @ {fill.fill_price} vs "
                                f"{fill.reference} {fill.reference_price}: {fill.slippage_bps:+.1f} bps")
        except Exception as exc:  # measurement only; never blocks trading
            logger.warning(f"Fill tracking failed: {exc}")

    def _flatten(self, report: CycleReport) -> None:
        """EOD: cancel working orders and close every position (no overnight risk)."""
        snapshot = self._last_snapshot
        positions = len(snapshot.positions) if snapshot is not None else "?"
        orders = len(snapshot.open_orders) if snapshot is not None else "?"
        if not (self.execute and self.broker is not None):
            logger.info(f"FLATTEN phase (dry run): would cancel {orders} open orders and close {positions} positions")
            return
        if snapshot is not None and not snapshot.positions and not snapshot.open_orders:
            logger.info("FLATTEN phase: already flat")
            return
        report.flatten = self.broker.flatten_all("eod")
        closed = ", ".join(c["symbol"] for c in report.flatten.closed) or "none"
        logger.warning(
            f"EOD flatten: cancelled {report.flatten.cancelled} orders, closed {closed}; "
            f"{len(report.flatten.failures)} failure(s)" + (" (will retry next cycle)" if report.flatten.failures else "")
        )

    def _next_sleep(self) -> float:
        """Normal interval, but wake exactly at the flatten time if it comes sooner."""
        clock = self._last_clock
        if self.session_clock is None or not clock or not clock.get("is_open"):
            return self.interval_seconds
        now = self.now_fn()
        until = self.session_clock.seconds_until_flatten(now, clock.get("next_close"))
        if until is not None and 0 < until < self.interval_seconds:
            return until + 1
        return self.interval_seconds

    def _maybe_session_report(self, report: CycleReport) -> None:
        """After a session the bot traded through, summarise it once."""
        if self.broker is None or self._session_seen is None or self._session_reported == self._session_seen:
            return
        session_date = self._session_seen
        self._session_reported = session_date
        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            orders = self.broker.get_orders_today() if hasattr(self.broker, "get_orders_today") else []
        except Exception as exc:
            logger.error(f"Session report unavailable: {exc}")
            return
        equity = float(account.get("equity") or 0)
        start = float(account.get("last_equity") or 0)
        all_orders = list(_walk_orders(orders))
        fills = [o for o in all_orders if o.get("status") == "filled" or float(o.get("filled_qty") or 0) > 0]
        self._track_fills(session_date, orders)
        no_overnight = bool(self.session_clock and self.session_clock.config.no_overnight)
        report.session_report = {
            "date": str(session_date),
            "equity": round(equity, 2),
            "pnl": round(equity - start, 2) if start else 0.0,
            "pnl_pct": round((equity - start) / start * 100, 3) if start else 0.0,
            "fills": len(fills),
            "open_positions": len(positions),
            "open_symbols": [p.get("symbol") for p in positions],
            "no_overnight": no_overnight,
        }
        if self.fill_tracker is not None:
            report.session_report["fill_quality"] = self.fill_tracker.summary()
        if self.edge_monitor is not None:
            report.session_report["live_edge"] = {**self.edge_monitor.stats(), "paused": self.edge_monitor.paused}
        level = logging.WARNING if (no_overnight and positions) else logging.INFO
        self.telemetry.record("session_report", level, **report.session_report)

    def _write_heartbeat(self, report: CycleReport) -> None:
        """Small JSON file external monitoring can check for freshness."""
        snapshot = self._last_snapshot
        beat = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": report.phase,
            "execute": self.execute,
            "timeframe": self.timeframe,
            "symbols": self.symbols,
            "orders_this_cycle": len(report.orders),
            "errors": report.errors,
            "entry_block": report.entry_block,
            "broker_error": report.broker_error,
            "positions": len(snapshot.positions) if snapshot is not None else None,
            "next_cycle_in_seconds": None if self._stop_event.is_set() else round(self._next_sleep(), 1),
        }
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.heartbeat_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(beat, default=str, indent=2))
            tmp.replace(self.heartbeat_path)  # atomic: monitors never read half a file
        except OSError as exc:
            logger.error(f"Could not write heartbeat: {exc}")

    def stop(self, reason: str = "requested") -> None:
        """Finish the current cycle, then exit ``run`` (safe from signal handlers)."""
        self._stop_reason = reason
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self, max_cycles: Optional[int] = None) -> List[CycleReport]:
        mode = "EXECUTE (paper orders)" if self.execute else "DRY RUN (no orders)"
        logger.info(
            f"Trading bot started: {mode}, {len(self.symbols)} symbols, "
            f"every {self.interval_seconds}s, portfolio '{self.portfolio_name}'"
        )
        self.telemetry.record("bot_started", mode=mode, timeframe=self.timeframe, symbols=",".join(self.symbols))
        reports: List[CycleReport] = []
        attempts = 0  # counts failed cycles too, so --once can't loop forever
        try:
            while not self.stopped and (max_cycles is None or attempts < max_cycles):
                attempts += 1
                try:
                    reports.append(self.run_cycle())
                except Exception as exc:  # one bad cycle must not kill an unattended bot
                    logger.exception(f"Cycle failed: {exc}")
                    self.telemetry.record("bot_error", logging.ERROR, message=str(exc))
                if self.stopped or (max_cycles is not None and attempts >= max_cycles):
                    break
                self.sleep(self._next_sleep())
        finally:
            reason = self._stop_reason or ("completed" if max_cycles is not None else "exited")
            self.telemetry.record("bot_stopped", reason=reason, cycles=attempts)
            logger.info(f"Trading bot stopped ({reason})")
        return reports
