#!/usr/bin/env python3
"""
Live edge-decay monitor: pause new entries when live results stop matching
the backtest.

A strategy that passed the edge gate can still stop working (regime change,
crowding, a data or execution problem the backtest didn't model). This
monitor pairs live fills (from core/fill_quality.FillTracker) into round-trip
trades, FIFO per symbol, and measures each in R (profit / initial risk to the
bracket stop). Over the last EDGE_MONITOR_WINDOW closed trades (default 30,
evaluated once at least EDGE_MONITOR_MIN_TRADES = 20 exist) it pauses new
entries when either:

  * live profit factor < EDGE_DECAY_MIN_PROFIT_FACTOR (0.8), or
  * live mean R is significantly below the validated backtest's average R
    (one-sided t-statistic < -EDGE_DECAY_T_STAT, default 2.0), using the
    avg R recorded in the edge report.

The pause latches (persisted in EDGE_MONITOR_STATE_PATH) until you review
it and restart with ``bot.py --reset-edge-monitor``. Exits, stops and the
EOD flatten keep running while paused.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.execution_telemetry import EventLog

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class LiveTrade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    stop_price: Optional[float]
    entry_at: str
    exit_at: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def r_multiple(self) -> Optional[float]:
        if not self.stop_price or self.entry_price <= self.stop_price:
            return None
        return (self.exit_price - self.entry_price) / (self.entry_price - self.stop_price)


class EdgeMonitor:
    def __init__(
        self,
        *,
        state_path: Optional[str] = None,
        telemetry: Optional[EventLog] = None,
        backtest_avg_r: Optional[float] = None,
        window: Optional[int] = None,
        min_trades: Optional[int] = None,
        min_profit_factor: Optional[float] = None,
        t_stat: Optional[float] = None,
    ) -> None:
        self.state_path = Path(state_path or os.getenv("EDGE_MONITOR_STATE_PATH", "data/edge_monitor.json"))
        self.telemetry = telemetry or EventLog.from_env()
        self.backtest_avg_r = backtest_avg_r
        self.window = window or int(_env_float("EDGE_MONITOR_WINDOW", 30))
        self.min_trades = min_trades or int(_env_float("EDGE_MONITOR_MIN_TRADES", 20))
        self.min_profit_factor = (min_profit_factor if min_profit_factor is not None
                                  else _env_float("EDGE_DECAY_MIN_PROFIT_FACTOR", 0.8))
        self.t_stat = t_stat if t_stat is not None else _env_float("EDGE_DECAY_T_STAT", 2.0)
        self.open_lots: Dict[str, List[Dict[str, Any]]] = {}
        self.trades: List[LiveTrade] = []
        self.paused: Optional[str] = None
        self._load()

    @classmethod
    def from_edge_report(cls, **kwargs: Any) -> "EdgeMonitor":
        """Use the validated backtest's average R as the live benchmark."""
        from core.edge_gate import report_path

        avg_r = None
        try:
            avg_r = json.loads(report_path().read_text()).get("metrics", {}).get("avg_r")
        except (OSError, ValueError):
            pass
        return cls(backtest_avg_r=float(avg_r) if avg_r is not None else None, **kwargs)

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        self.open_lots = state.get("open_lots") or {}
        self.trades = [LiveTrade(**t) for t in state.get("trades") or []]
        self.paused = state.get("paused")

    def _save(self) -> None:
        state = {
            "open_lots": self.open_lots,
            "trades": [asdict(t) for t in self.trades[-500:]],
            "paused": self.paused,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(self.state_path)
        except OSError as exc:
            logger.error(f"Could not persist edge monitor state: {exc}")

    def reset(self) -> None:
        """Clear a pause (after review) and start a fresh measurement window."""
        self.paused = None
        self.trades = []
        self._save()

    # ------------------------------------------------------------------

    def update(self, fills: Iterable[Any]) -> List[LiveTrade]:
        """Consume new FillRecords; returns round trips closed by them."""
        closed: List[LiveTrade] = []
        for fill in sorted(fills, key=lambda f: f.filled_at):
            if fill.qty <= 0:
                continue
            lots = self.open_lots.setdefault(fill.symbol, [])
            if fill.side == "buy":
                lots.append({"qty": fill.qty, "price": fill.fill_price, "stop": fill.stop_price,
                             "at": fill.filled_at})
                continue
            remaining = fill.qty
            while remaining > 1e-9 and lots:
                lot = lots[0]
                take = min(remaining, lot["qty"])
                closed.append(LiveTrade(fill.symbol, take, lot["price"], fill.fill_price, lot["stop"],
                                        lot["at"], fill.filled_at))
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] <= 1e-9:
                    lots.pop(0)
            if remaining > 1e-9:
                logger.debug(f"{fill.symbol}: sell of {remaining:g} shares with no tracked entry (ignored)")
            if not lots:
                self.open_lots.pop(fill.symbol, None)
        if closed:
            self.trades.extend(closed)
            self._evaluate()
        self._save()
        return closed

    def stats(self) -> Dict[str, Any]:
        recent = self.trades[-self.window:]
        wins = sum(t.pnl for t in recent if t.pnl > 0)
        losses = -sum(t.pnl for t in recent if t.pnl < 0)
        rs = [t.r_multiple for t in recent if t.r_multiple is not None]
        mean_r = sum(rs) / len(rs) if rs else None
        sd = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / (len(rs) - 1)) if len(rs) > 1 else None
        return {
            "trades": len(recent),
            "profit_factor": round(wins / losses, 3) if losses > 0 else (math.inf if wins > 0 else None),
            "mean_r": round(mean_r, 4) if mean_r is not None else None,
            "sd_r": round(sd, 4) if sd is not None else None,
            "r_trades": len(rs),
            "pnl": round(sum(t.pnl for t in recent), 2),
            "backtest_avg_r": self.backtest_avg_r,
        }

    def _evaluate(self) -> None:
        if self.paused:
            return
        s = self.stats()
        if s["trades"] < self.min_trades:
            return
        reason = None
        if s["profit_factor"] is not None and s["profit_factor"] < self.min_profit_factor:
            reason = (f"live profit factor {s['profit_factor']} over the last {s['trades']} trades is below "
                      f"{self.min_profit_factor}")
        elif (self.backtest_avg_r is not None and s["mean_r"] is not None and s["sd_r"]
              and s["r_trades"] >= self.min_trades):
            t = (s["mean_r"] - self.backtest_avg_r) / (s["sd_r"] / math.sqrt(s["r_trades"]))
            if t < -self.t_stat:
                reason = (f"live mean R {s['mean_r']:+.3f} over {s['r_trades']} trades is significantly below the "
                          f"backtest's {self.backtest_avg_r:+.3f} (t = {t:.2f})")
        if reason:
            self.paused = reason
            logger.error(f"EDGE DECAY: pausing new entries: {reason}")
            self.telemetry.record("edge_decay", logging.ERROR, reason=reason, **s)
