#!/usr/bin/env python3
"""
Scheduled trading loop: scan a watchlist during market hours and act on it.

Dry-run by default (analyze and log only). With ``execute=True`` actionable
signals go through TradingWorkflow, which applies the executor's risk checks
and bracket orders before anything reaches Alpaca paper trading.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.portfolio_manager import PortfolioManager
from core.trading_workflow import TradingWorkflow, WorkflowResult

logger = logging.getLogger(__name__)

STRATEGIES = ("ml", "classic")


def create_workflow(portfolio_manager: PortfolioManager, strategy: str = "ml") -> TradingWorkflow:
    """Build the workflow for a strategy. Both paths end in the same executor
    and RiskManager, so risk limits and bracket orders apply either way."""
    if strategy == "ml":
        from core.ml_signal_engine import MLSignalEngine

        return TradingWorkflow(portfolio_manager, analysis_runner=MLSignalEngine(portfolio_manager))
    if strategy == "classic":
        return TradingWorkflow(portfolio_manager)
    raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")


@dataclass
class CycleReport:
    started_at: datetime
    market_open: bool
    results: List[WorkflowResult] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def orders(self) -> List[WorkflowResult]:
        return [r for r in self.results if r.alpaca_order]


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
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cleaned = [s.strip().upper() for s in symbols if s and s.strip()]
        if not cleaned:
            raise ValueError("Watchlist is empty; pass --symbols or set WATCHLIST")
        self.workflow = workflow
        self.symbols = list(dict.fromkeys(cleaned))
        self.portfolio_name = portfolio_name
        self.execute = execute
        self.interval_seconds = max(60, interval_seconds)
        self.market_clock = market_clock
        self.sleep = sleep

    def _market_open(self) -> bool:
        if self.market_clock is None:
            return True
        try:
            return bool(self.market_clock().get("is_open", False))
        except Exception as exc:  # network hiccup: skip this cycle, don't trade blind
            logger.error(f"Could not read market clock: {exc}")
            return False

    def run_cycle(self) -> CycleReport:
        report = CycleReport(started_at=datetime.now(timezone.utc), market_open=self._market_open())
        if not report.market_open:
            logger.info("Market closed; skipping cycle")
            return report

        for symbol in self.symbols:
            try:
                result = self.workflow.run(
                    symbol,
                    self.portfolio_name,
                    record_paper_trade=self.execute,
                    submit_alpaca_paper_order=self.execute,
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
                summary += (
                    f" [{ml['mode']} P(up)={ml['probability_up']:.2f} sentiment {sentiment}]"
                )
            if result.alpaca_order:
                logger.info(f"{summary} -> ORDER {result.alpaca_order.get('id')}")
            else:
                logger.info(f"{summary} -> {result.skipped_reason or 'no order'}")
        return report

    def run(self, max_cycles: Optional[int] = None) -> List[CycleReport]:
        mode = "EXECUTE (paper orders)" if self.execute else "DRY RUN (no orders)"
        logger.info(
            f"Trading bot started: {mode}, {len(self.symbols)} symbols, "
            f"every {self.interval_seconds}s, portfolio '{self.portfolio_name}'"
        )
        reports: List[CycleReport] = []
        while max_cycles is None or len(reports) < max_cycles:
            reports.append(self.run_cycle())
            if max_cycles is not None and len(reports) >= max_cycles:
                break
            self.sleep(self.interval_seconds)
        return reports
