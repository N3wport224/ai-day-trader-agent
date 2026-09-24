#!/usr/bin/env python3
"""
Live trailing-stop manager for open bracket positions.

Each cycle, for every long position with a working stop order:
  1. Remember the position's initial stop the first time it is seen (the
     broker stop before we ever moved it) and track the highest price seen
     (Alpaca's position ``current_price`` each cycle).
  2. Compute the new stop with risk_manager.trailing_stop_price — the same
     rule the backtester uses: breakeven once price has moved
     TRAILING_STOP_TRIGGER_R in favour, then trail TRAILING_STOP_DISTANCE_R
     below the high. Stops only move up.
  3. Replace the broker stop order when the new stop is higher.

State (initial stop, high-water mark) is kept in TRAILING_STATE_PATH so a
restart doesn't lose the original R. A position whose entry price or size
changes starts fresh. Note the high-water mark is sampled once per cycle,
so it can lag intrabar highs; the broker stop itself still triggers in real
time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from core.alpaca_executor import BrokerSnapshot, protective_stop_orders
from core.execution_telemetry import EventLog, classify_rejection
from core.risk_manager import TrailingConfig, trailing_stop_price

logger = logging.getLogger(__name__)


@dataclass
class StopAdjustment:
    symbol: str
    order_id: str
    old_stop: float
    new_stop: float
    ok: bool
    detail: str = ""


class TrailingStopManager:
    def __init__(
        self,
        broker,
        *,
        config: Optional[TrailingConfig] = None,
        state_path: Optional[str] = None,
        telemetry: Optional[EventLog] = None,
    ) -> None:
        self.broker = broker
        self.config = config or TrailingConfig.from_env()
        self.state_path = Path(state_path or os.getenv("TRAILING_STATE_PATH", "data/trailing_state.json"))
        self.telemetry = telemetry or EventLog.from_env()
        self.state: Dict[str, Dict[str, float]] = self._load()
        cfg = self.config
        logger.info(
            f"Trailing stops {'enabled' if cfg.enabled else 'disabled'}: trigger {cfg.trigger_r}R, "
            f"lock {cfg.lock_r}R, trail {cfg.distance_r}R; state {self.state_path} "
            f"({len(self.state)} tracked position(s))"
        )

    def _load(self) -> Dict[str, Dict[str, float]]:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, indent=2))
        except OSError as exc:
            logger.error(f"Could not save trailing-stop state: {exc}")

    def update(self, snapshot: BrokerSnapshot) -> List[StopAdjustment]:
        if not self.config.enabled:
            return []
        adjustments: List[StopAdjustment] = []
        held = set()

        for position in snapshot.positions:
            symbol = position["symbol"]
            qty = float(position.get("qty") or 0)
            if qty <= 0:
                continue
            held.add(symbol)
            entry = float(position.get("avg_entry_price") or 0)
            price = float(position.get("current_price") or 0)
            stops = protective_stop_orders(snapshot.open_orders, symbol)
            if not stops or entry <= 0 or price <= 0:
                continue
            stop_order = max(stops, key=lambda o: float(o.get("stop_price") or 0))
            current_stop = float(stop_order.get("stop_price") or 0)

            state = self.state.get(symbol)
            if not state or state.get("entry") != entry or state.get("qty") != qty:
                state = {"entry": entry, "qty": qty, "initial_stop": current_stop, "high_water": price}
            state["high_water"] = max(state["high_water"], price)
            self.state[symbol] = state

            new_stop = trailing_stop_price(
                entry=entry,
                initial_stop=state["initial_stop"],
                current_stop=current_stop,
                high_water=state["high_water"],
                config=self.config,
            )
            if new_stop <= current_stop or new_stop >= price:
                continue
            adjustments.append(self._move(symbol, stop_order["id"], current_stop, new_stop, state))

        for symbol in list(self.state):
            if symbol not in held:
                del self.state[symbol]  # position closed
        self._save()
        return adjustments

    def _move(self, symbol: str, order_id: str, old: float, new: float, state: Dict[str, Any]) -> StopAdjustment:
        risk = state["entry"] - state["initial_stop"]
        locked_r = (new - state["entry"]) / risk if risk > 0 else float("nan")
        try:
            self.broker.replace_stop_price(order_id, new)
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            rejection = classify_rejection(getattr(response, "status_code", None), getattr(response, "text", "") or str(exc))
            self.telemetry.record(
                "trailing_stop_failed", logging.WARNING, symbol=symbol, order_id=order_id,
                old_stop=old, new_stop=new, **rejection.as_dict(),
            )
            return StopAdjustment(symbol, order_id, old, new, False, rejection.message)
        self.telemetry.record(
            "trailing_stop_raised", symbol=symbol, order_id=order_id, old_stop=old, new_stop=new,
            entry=state["entry"], initial_stop=state["initial_stop"], high_water=state["high_water"],
            locked_r=round(locked_r, 2),
        )
        return StopAdjustment(symbol, order_id, old, new, True, f"locks {locked_r:+.2f}R")
