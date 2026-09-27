#!/usr/bin/env python3
"""
Live performance for the dashboard: the account's equity curve, the bot's
closed round-trip trades, and how live results compare with what the
validated backtest promised (the question paper trading has to answer
before real money is considered).
"""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

MARKET_TZ = ZoneInfo("America/New_York")
MIN_TRADES_TO_JUDGE = 20
PERIODS = {  # dashboard range -> (Alpaca period, bar timeframe)
    "1D": ("1D", "5Min"),
    "1W": ("1W", "1H"),
    "1M": ("1M", "1D"),
    "3M": ("3M", "1D"),
    "1Y": ("1A", "1D"),
}


def load_live_trades(state_path: Path) -> List[Dict[str, Any]]:
    """Closed round trips recorded by core/edge_monitor.EdgeMonitor, newest last."""
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    trades = []
    for t in state.get("trades") or []:
        try:
            qty, entry, exit_ = float(t["qty"]), float(t["entry_price"]), float(t["exit_price"])
        except (KeyError, TypeError, ValueError):
            continue
        stop = t.get("stop_price")
        risk = (entry - float(stop)) if stop else 0.0
        trades.append({
            "symbol": t.get("symbol"),
            "qty": qty,
            "entry_at": t.get("entry_at"),
            "exit_at": t.get("exit_at"),
            "entry_price": round(entry, 4),
            "exit_price": round(exit_, 4),
            "stop_price": float(stop) if stop else None,
            "pnl": round((exit_ - entry) * qty, 2),
            "return_pct": round((exit_ / entry - 1) * 100, 3) if entry else None,
            "r_multiple": round((exit_ - entry) / risk, 3) if risk > 0 else None,
            "entry_order_id": t.get("entry_order_id"),
            "exit_type": t.get("exit_type"),
        })
    return trades


STOP_TYPES = {"stop", "stop_limit", "trailing_stop"}
REGIME_WORDS = {"TRENDING_BULL": "uptrend", "TRENDING_BEAR": "downtrend", "CHOPPY": "choppy market"}


def exit_reason(trade: Dict[str, Any]) -> Optional[str]:
    """How a round trip ended, in plain words (None when unknown, e.g. older trades)."""
    kind = str(trade.get("exit_type") or "").lower()
    if not kind:
        return None
    try:
        closed = datetime.fromisoformat(str(trade.get("exit_at")).replace("Z", "+00:00")).astimezone(MARKET_TZ)
    except ValueError:
        closed = None
    if kind in STOP_TYPES:
        if trade["exit_price"] > trade["entry_price"]:
            return "Trailing stop: locked in profit"
        return "Stop-loss hit"
    if closed is not None and (closed.hour, closed.minute) >= (15, 49):
        return "End-of-day close (never held overnight)"
    if kind == "limit":
        return "Take-profit hit"
    return "Sold at market (sell signal or manual close)"


def explain_entry(context: Dict[str, Any]) -> List[str]:
    """What the bot saw when it bought, as short plain-language points."""
    points = []
    p, threshold = context.get("probability_up"), context.get("threshold")
    if p is not None:
        points.append(f"Model: {p * 100:.0f}% chance of reaching the target before the stop"
                      + (f" (needs {threshold * 100:.0f}%)" if threshold else ""))
    if context.get("regime") in REGIME_WORDS:
        points.append(f"Market regime: {REGIME_WORDS[context['regime']]}")
    if context.get("vwap_dist_pct") is not None:
        side = "above" if context["vwap_dist_pct"] >= 0 else "below"
        points.append(f"Price {abs(context['vwap_dist_pct']):.2f}% {side} VWAP"
                      + (f" ({context['vwap_zone']})" if context.get("vwap_zone") else ""))
    if context.get("rvol") is not None:
        points.append(f"Volume {context['rvol']:.1f}x normal for this time of day")
    if context.get("above_orb_high"):
        points.append("Trading above the opening range (breakout)")
    if context.get("minutes_from_open") is not None:
        points.append(f"{context['minutes_from_open']:.0f} min after the open")
    if context.get("market_ok") == 0.0:
        points.append("Note: the overall market (SPY) was weak")
    if not points:
        points = [r for r in context.get("reasons") or [] if r]
    return points


def attach_explanations(trades: List[Dict[str, Any]], context_events: List[Dict[str, Any]]) -> None:
    """Add "why" (entry reasoning) and "exit_reason" to each trade, in place."""
    by_order: Dict[str, Dict[str, Any]] = {}
    for event in context_events:
        for order_id in event.get("order_ids") or []:
            by_order[order_id] = event
    for trade in trades:
        event = by_order.get(trade.get("entry_order_id") or "")
        trade["why"] = explain_entry(event.get("context") or {}) if event else []
        trade["exit_reason"] = exit_reason(trade)


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    rs = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    mean_r = sum(rs) / len(rs) if rs else None
    sd_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / (len(rs) - 1)) if len(rs) > 1 else None
    return {
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        # None when there are no losing trades yet (infinity isn't valid JSON)
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses else None,
        "losing_trades": len(losses),
        "avg_r": round(mean_r, 3) if mean_r is not None else None,
        "sd_r": round(sd_r, 3) if sd_r is not None else None,
        "total_pnl": round(sum(pnls), 2),
        "best_trade": round(max(pnls), 2) if pnls else None,
        "worst_trade": round(min(pnls), 2) if pnls else None,
    }


def compare_with_backtest(live: Dict[str, Any], backtest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Plain-language verdict: is live trading matching the validated test?"""
    if not backtest:
        return {"status": "no_backtest",
                "message": "No validated backtest to compare with yet. Run 'Validate strategy' on Get Started."}
    n = live["trades"]
    if n < MIN_TRADES_TO_JUDGE:
        return {"status": "too_early",
                "message": f"{n} closed trade{'s' if n != 1 else ''} so far. About {MIN_TRADES_TO_JUDGE} are needed "
                           "before live results say anything reliable; day-to-day swings are mostly luck."}
    bt_r = backtest.get("avg_r")
    live_r, sd = live.get("avg_r"), live.get("sd_r")
    if bt_r is None or live_r is None:
        return {"status": "unknown", "message": "Not enough information to compare (no stop prices recorded)."}
    t_stat = (live_r - bt_r) / (sd / math.sqrt(n)) if sd else 0.0
    pf = live.get("profit_factor")
    if t_stat < -2 or (pf is not None and pf < 0.8):
        return {"status": "behind", "t_stat": round(t_stat, 2),
                "message": f"Live results ({live_r:+.2f}R per trade) are clearly worse than the test "
                           f"({bt_r:+.2f}R). Don't move to real money; re-validate on recent data."}
    if live_r >= 0.5 * bt_r and (pf is None or pf >= 1.0):
        return {"status": "on_track", "t_stat": round(t_stat, 2),
                "message": f"Live results ({live_r:+.2f}R per trade) are in line with the test ({bt_r:+.2f}R)."}
    return {"status": "watch", "t_stat": round(t_stat, 2),
            "message": f"Live results ({live_r:+.2f}R per trade) trail the test ({bt_r:+.2f}R) but not by more "
                       "than chance can explain yet. Keep paper trading and check again."}


def equity_series(history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Alpaca /account/portfolio/history -> [{t, equity, pnl, pnl_pct}] without empty points."""
    points = []
    stamps = history.get("timestamp") or []
    equity = history.get("equity") or []
    pnl = history.get("profit_loss") or []
    pnl_pct = history.get("profit_loss_pct") or []
    for i, ts in enumerate(stamps):
        value = equity[i] if i < len(equity) else None
        if value is None or value == 0:
            continue  # Alpaca pads days before the account existed with null/0
        points.append({
            "t": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
            "equity": round(float(value), 2),
            "pnl": round(float(pnl[i]), 2) if i < len(pnl) and pnl[i] is not None else None,
            "pnl_pct": round(float(pnl_pct[i]) * 100, 3) if i < len(pnl_pct) and pnl_pct[i] is not None else None,
        })
    return points


CSV_COLUMNS = ["symbol", "qty", "entry_at", "entry_price", "exit_at", "exit_price", "stop_price",
               "pnl", "return_pct", "r_multiple", "exit_reason", "why"]


def trades_csv(trades: List[Dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for trade in trades:
        row = {**trade, "why": "; ".join(trade.get("why") or [])}
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_COLUMNS})
    return out.getvalue()
