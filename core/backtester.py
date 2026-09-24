#!/usr/bin/env python3
"""
Event-driven backtest of the ML trading path.

The backtest reuses the live decision code rather than re-implementing it:
  features      core/features.compute_features        (causal)
  signals       core/ml_strategy.MLStrategy.decide     (same thresholds / ATR levels)
  sizing        core/ml_signal_engine.risk_based_quantity
  risk checks   core/risk_manager.RiskManager.check_order (same limits)

and simulates what the broker would do:
  - A decision is made at a bar's close; the order fills at the NEXT
    regular-session bar's open, plus adverse slippage.
  - BUYs are brackets: stop and target are the signal's ATR distances
    re-centred on the fill price (as AlpacaExecutor does with the live quote).
  - Stops/targets trigger only in bars overlapping the regular session. A bar that opens
    through a level fills at the open; a bar touching both levels counts as
    the stop (conservative). Stop fills pay slippage; target fills don't.
  - SELL signals exit an open position at the next regular-session open.
  - Intraday decisions happen only on bars that close while the market is
    open (as the live bot only runs while Alpaca's clock says open).
  - One position per symbol; open positions are closed at the final bar.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import time, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.features import compute_features
from core.ml_signal_engine import risk_based_quantity
from core.ml_strategy import MLStrategy, _snapshot_from_row
from core.news_sentiment import SENTIMENT_FEATURES
from core.risk_manager import RiskManager

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN, SESSION_CLOSE = time(9, 30), time(16, 0)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    risk_per_trade_pct: float = 1.0
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    bar_length: timedelta = timedelta(hours=1)
    regular_hours_only: bool = True


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    stop: float
    target: float
    probability_up: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    costs: float = 0.0

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity - self.costs

    @property
    def r_multiple(self) -> float:
        risk = (self.entry_price - self.stop) * self.quantity
        return self.pnl / risk if risk > 0 else float("nan")


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity: pd.Series
    blocked: Dict[str, int]
    skipped_outside_session: int
    signals: Dict[str, int]
    benchmark_return: float
    config: BacktestConfig
    metrics: Dict[str, float] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "entry_price": round(t.entry_price, 4),
                    "quantity": t.quantity,
                    "stop": t.stop,
                    "target": t.target,
                    "probability_up": round(t.probability_up, 4),
                    "exit_time": t.exit_time,
                    "exit_price": round(t.exit_price, 4),
                    "exit_reason": t.exit_reason,
                    "pnl": round(t.pnl, 2),
                    "r_multiple": round(t.r_multiple, 3),
                }
                for t in self.trades
            ]
        )


def _is_regular_bar(ts: pd.Timestamp, config: BacktestConfig) -> bool:
    """Bars overlapping the regular session can fill orders and trigger
    brackets. A 9:00 bar that includes pre-market counts too (conservative:
    its pre-market low can hit a stop)."""
    if not config.regular_hours_only or config.bar_length >= timedelta(days=1):
        return True
    start = ts.tz_convert(MARKET_TZ)
    end = (ts + config.bar_length).tz_convert(MARKET_TZ)
    return start.weekday() < 5 and start.time() < SESSION_CLOSE and end.time() > SESSION_OPEN


def _can_decide(ts: pd.Timestamp, config: BacktestConfig) -> bool:
    """The live bot acts only while the market is open, i.e. on bars that
    close strictly before the 16:00 ET close."""
    if not config.regular_hours_only or config.bar_length >= timedelta(days=1):
        return True
    close_local = (ts + config.bar_length).tz_convert(MARKET_TZ)
    return close_local.weekday() < 5 and SESSION_OPEN < close_local.time() < SESSION_CLOSE


def _reason_category(reason: str, symbol: str) -> str:
    """Group risk-block reasons ("No room for AAPL: cap $2,500 ..." -> "No room")."""
    text = reason.split(":")[0].split("(")[0].replace(f" for {symbol}", "").replace(symbol, "")
    text = re.sub(r"\$?[\d.,]+%?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _day_key(ts: pd.Timestamp):
    return ts.tz_convert(MARKET_TZ).date()


class Backtester:
    def __init__(self, strategy: MLStrategy, risk_manager: RiskManager, config: BacktestConfig) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config

    def run(
        self,
        bars: Dict[str, pd.DataFrame],
        sentiment: Optional[Dict[str, pd.DataFrame]] = None,
        start: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """Simulate from ``start`` (default: first bar). Earlier bars only warm up indicators."""
        cfg = self.config
        slip = cfg.slippage_bps / 10_000

        prepared: Dict[str, pd.DataFrame] = {}
        for symbol, frame in bars.items():
            feats = compute_features(frame)
            sent = (sentiment or {}).get(symbol)
            if sent is None:
                sent = pd.DataFrame(np.nan, index=feats.index, columns=SENTIMENT_FEATURES)
                sent["sentiment_available"] = 0.0
            sent = sent.reindex(feats.index)
            sent["sentiment_available"] = sent["sentiment_available"].fillna(0.0)
            feats = feats.join(sent[SENTIMENT_FEATURES])
            feats["p_up"] = self.strategy.probabilities(feats, feats[SENTIMENT_FEATURES])
            if start is not None:
                feats = feats[feats.index >= start]
            prepared[symbol] = feats

        timeline = sorted(set().union(*(f.index for f in prepared.values())))
        cash = cfg.initial_capital
        positions: Dict[str, Trade] = {}
        pending_entries: Dict[str, dict] = {}
        pending_exits: set = set()
        last_close: Dict[str, float] = {}
        trades: List[Trade] = []
        blocked: Dict[str, int] = {}
        signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        skipped_outside_session = 0
        equity_points: Dict[pd.Timestamp, float] = {}
        entries_by_day: Dict[object, int] = {}
        prev_day_equity: Optional[float] = None
        current_day = None
        day_equity = cfg.initial_capital
        bars_in_market = 0

        def equity() -> float:
            return cash + sum(p.quantity * last_close.get(s, p.entry_price) for s, p in positions.items())

        def close_position(symbol: str, ts, price: float, reason: str, slipped: bool) -> None:
            nonlocal cash
            trade = positions.pop(symbol)
            fill = price * (1 - slip) if slipped else price
            trade.exit_time, trade.exit_price, trade.exit_reason = ts, fill, reason
            trade.costs += cfg.commission_per_share * trade.quantity
            cash += fill * trade.quantity - cfg.commission_per_share * trade.quantity
            trades.append(trade)

        for ts in timeline:
            day = _day_key(ts)
            if day != current_day:
                if current_day is not None:
                    prev_day_equity = day_equity
                current_day = day

            for symbol, feats in prepared.items():
                if ts not in feats.index:
                    continue
                row = feats.loc[ts]
                o, h, l, c = row["open"], row["high"], row["low"], row["close"]
                regular = _is_regular_bar(ts, cfg)

                if regular:
                    # 1. Fills queued at the previous decision.
                    if symbol in pending_exits and symbol in positions:
                        close_position(symbol, ts, o, "signal_exit", slipped=True)
                    pending_exits.discard(symbol)

                    order = pending_entries.pop(symbol, None)
                    if order and symbol not in positions:
                        fill = o * (1 + slip)
                        qty = min(order["quantity"], int(cash // fill))
                        if qty > 0:
                            cash -= fill * qty + cfg.commission_per_share * qty
                            positions[symbol] = Trade(
                                symbol=symbol,
                                entry_time=ts,
                                entry_price=fill,
                                quantity=qty,
                                stop=round(fill - order["stop_distance"], 2),
                                target=round(fill + order["target_distance"], 2),
                                probability_up=order["p_up"],
                                costs=cfg.commission_per_share * qty,
                            )

                    # 2. Bracket legs (the entry bar included).
                    trade = positions.get(symbol)
                    if trade:
                        if o <= trade.stop:
                            close_position(symbol, ts, o, "stop_gap", slipped=True)
                        elif o >= trade.target:
                            close_position(symbol, ts, o, "target_gap", slipped=False)
                        elif l <= trade.stop:
                            close_position(symbol, ts, trade.stop, "stop", slipped=True)
                        elif h >= trade.target:
                            close_position(symbol, ts, trade.target, "target", slipped=False)

                last_close[symbol] = c

                # 3. Decide at this bar's close.
                if not np.isfinite(row["p_up"]):
                    continue
                if not _can_decide(ts, cfg):
                    # The live bot doesn't run while the market is closed.
                    if row["p_up"] >= self.strategy.threshold:
                        skipped_outside_session += 1
                    continue
                signal = self.strategy.decide(float(row["p_up"]), row, _snapshot_from_row(row))
                signals[signal.signal] += 1

                if signal.signal == "SELL" and symbol in positions:
                    pending_exits.add(symbol)
                elif signal.signal == "BUY" and symbol not in positions and symbol not in pending_entries:
                    eq = equity()
                    # Live orders fill at once and consume buying power; queued
                    # backtest orders must reserve it until the next open.
                    reserved = sum(o["quantity"] * o["price"] * (1 + slip) for o in pending_entries.values())
                    qty = risk_based_quantity(eq, cfg.risk_per_trade_pct, signal.stop_distance)
                    decision = self.risk_manager.check_order(
                        side="BUY",
                        symbol=symbol,
                        quantity=qty,
                        price=c,
                        account={
                            "equity": eq,
                            "last_equity": prev_day_equity or cfg.initial_capital,
                            "buying_power": max(0.0, cash - reserved),
                        },
                        position=None,
                        orders_today=[{"side": "buy"}] * entries_by_day.get(day, 0),
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                    )
                    if decision.approved:
                        pending_entries[symbol] = {
                            "quantity": decision.quantity,
                            "stop_distance": signal.stop_distance,
                            "target_distance": signal.target_distance,
                            "p_up": signal.probability_up,
                            "price": c,
                        }
                        entries_by_day[day] = entries_by_day.get(day, 0) + 1
                    else:
                        key = _reason_category(decision.reason, symbol)
                        blocked[key] = blocked.get(key, 0) + 1

            day_equity = equity()
            equity_points[ts] = day_equity
            bars_in_market += bool(positions)

        for symbol in list(positions):
            close_position(symbol, timeline[-1], last_close[symbol], "end_of_test", slipped=True)
        if timeline:
            equity_points[timeline[-1]] = cash

        benchmark = [
            f["close"].iloc[-1] / f["close"].iloc[0] - 1 for f in prepared.values() if len(f) > 1
        ]
        result = BacktestResult(
            trades=trades,
            equity=pd.Series(equity_points, dtype=float).sort_index(),
            blocked=blocked,
            skipped_outside_session=skipped_outside_session,
            signals=signals,
            benchmark_return=float(np.mean(benchmark)) if benchmark else float("nan"),
            config=cfg,
        )
        result.metrics = compute_metrics(result)
        result.metrics["exposure_pct"] = round(bars_in_market / len(timeline) * 100, 1) if timeline else 0.0
        return result


def compute_metrics(result: BacktestResult) -> Dict[str, float]:
    equity = result.equity
    trades = result.trades
    start_equity = result.config.initial_capital
    end_equity = float(equity.iloc[-1]) if len(equity) else start_equity
    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]

    daily = equity.groupby([_day_key(ts) for ts in equity.index]).last() if len(equity) else equity
    daily_returns = daily.pct_change().dropna()
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else float("nan")
    )
    drawdown = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
    return {
        "start_equity": start_equity,
        "end_equity": round(end_equity, 2),
        "total_return_pct": round((end_equity / start_equity - 1) * 100, 2),
        "benchmark_return_pct": round(result.benchmark_return * 100, 2),
        "max_drawdown_pct": round(float(drawdown) * 100, 2),
        "sharpe_daily": round(sharpe, 2) if math.isfinite(sharpe) else float("nan"),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else float("nan"),
        "avg_r": round(float(np.nanmean([t.r_multiple for t in trades])), 3) if trades else float("nan"),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else float("nan"),
    }


def format_report(result: BacktestResult, title: str = "Backtest") -> str:
    m = result.metrics
    lines = [
        f"== {title} ==",
        f"Period:            {result.equity.index[0]} -> {result.equity.index[-1]}" if len(result.equity) else "Period: (no bars)",
        f"Equity:            ${m['start_equity']:,.0f} -> ${m['end_equity']:,.2f}  ({m['total_return_pct']:+.2f}%)",
        f"Buy & hold (avg):  {m['benchmark_return_pct']:+.2f}%",
        f"Max drawdown:      {m['max_drawdown_pct']:.2f}%   time in market {m.get('exposure_pct', 0)}%",
        f"Sharpe (daily):    {m['sharpe_daily']}",
        f"Trades:            {m['trades']}  win rate {m['win_rate_pct']}%  avg R {m['avg_r']}  profit factor {m['profit_factor']}",
        f"Signals:           {result.signals}",
    ]
    if result.blocked:
        lines.append(f"Blocked by risk:   {result.blocked}")
    lines.append(f"BUY-level signals outside market hours (not traded): {result.skipped_outside_session}")
    exits: Dict[str, int] = {}
    for t in result.trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
    if exits:
        lines.append(f"Exit reasons:      {exits}")
    return "\n".join(lines)
