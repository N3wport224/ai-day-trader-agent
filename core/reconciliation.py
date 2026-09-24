#!/usr/bin/env python3
"""
Broker/local state reconciliation, run at the start of every bot cycle.

Alpaca is the source of truth. Each cycle compares its positions and open
orders with the local portfolio's holdings and reports:

  missing_locally      broker holds a symbol the local book doesn't
  missing_at_broker    local book holds a symbol the broker doesn't
  quantity_mismatch    both hold it, in different sizes
  unprotected          a long position with no working stop order
  short_position       a negative position (the bot is long-only)

In ``sync`` mode the local holdings are overwritten from the broker (the
trade history is left untouched, so cash derived from trades may differ from
the broker's cash). ``report`` mode only logs. Findings are written to the
execution telemetry log.

Assumption: the bot's local portfolio mirrors the whole Alpaca account.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.alpaca_executor import BrokerSnapshot, protective_stop_orders
from core.execution_telemetry import EventLog
from core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class Discrepancy:
    kind: str
    symbol: str
    broker_qty: float = 0.0
    local_qty: float = 0.0
    detail: str = ""


@dataclass
class ReconciliationReport:
    positions: int
    open_orders: int
    discrepancies: List[Discrepancy] = field(default_factory=list)
    synced: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.discrepancies

    def kinds(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.discrepancies:
            counts[d.kind] = counts.get(d.kind, 0) + 1
        return counts


def reconcile(
    snapshot: BrokerSnapshot,
    portfolio_manager: PortfolioManager,
    portfolio_name: str,
    *,
    mode: Optional[str] = None,
    telemetry: Optional[EventLog] = None,
) -> ReconciliationReport:
    mode = (mode or os.getenv("RECONCILE_MODE", "sync")).lower()
    telemetry = telemetry or EventLog.from_env()

    broker = {}
    for p in snapshot.positions:
        broker[p["symbol"]] = (float(p.get("qty") or 0), float(p.get("avg_entry_price") or 0))
    has_portfolio = portfolio_manager.get_portfolio(portfolio_name) is not None
    local = {
        h["symbol"]: float(h["quantity"])
        for h in (portfolio_manager.get_holdings(portfolio_name) if has_portfolio else [])
    }

    report = ReconciliationReport(positions=len(broker), open_orders=len(snapshot.open_orders))
    for symbol, (qty, avg) in broker.items():
        if qty < 0:
            report.discrepancies.append(Discrepancy("short_position", symbol, qty, local.get(symbol, 0.0),
                                                    "negative position; the bot is long-only"))
        elif qty > 0:
            stops = protective_stop_orders(snapshot.open_orders, symbol)
            covered = sum(float(o.get("qty") or 0) for o in stops)
            if covered < qty:
                report.discrepancies.append(
                    Discrepancy("unprotected", symbol, qty, local.get(symbol, 0.0),
                                f"working stop orders cover {covered:g} of {qty:g} shares")
                )
        if symbol not in local:
            report.discrepancies.append(Discrepancy("missing_locally", symbol, qty, 0.0))
        elif local[symbol] != qty:
            report.discrepancies.append(Discrepancy("quantity_mismatch", symbol, qty, local[symbol]))
    for symbol, qty in local.items():
        if symbol not in broker:
            report.discrepancies.append(Discrepancy("missing_at_broker", symbol, 0.0, qty))

    if mode == "sync" and has_portfolio:
        for d in report.discrepancies:
            if d.kind in {"missing_locally", "quantity_mismatch"} and d.broker_qty > 0:
                portfolio_manager.update_holding(portfolio_name, d.symbol, int(d.broker_qty), broker[d.symbol][1])
                report.synced.append(d.symbol)
            elif d.kind == "missing_at_broker":
                portfolio_manager.update_holding(portfolio_name, d.symbol, 0)
                report.synced.append(d.symbol)

    telemetry.record(
        "reconciliation",
        logging.WARNING if report.discrepancies else logging.INFO,
        portfolio=portfolio_name,
        mode=mode,
        positions=report.positions,
        open_orders=report.open_orders,
        discrepancies=[d.__dict__ for d in report.discrepancies],
        synced=report.synced,
    )
    return report
