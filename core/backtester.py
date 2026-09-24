#!/usr/bin/env python3
"""
Event-driven backtest of the ML trading path.

The backtest reuses the live decision code rather than re-implementing it:
  features      core/feature_pipeline.build_feature_frame (causal; regime + MTF)
  signals       core/ml_strategy.MLStrategy.decide     (same thresholds, gates, ATR levels)
  sizing        core/risk_manager.risk_pct_for_trade / position_size
  risk checks   core/risk_manager.RiskManager.check_order (same limits)
  trailing      core/risk_manager.trailing_stop_price  (same rule as the live manager)

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
  - Trailing stops (optional) are recomputed from each bar's high after its
    exits are processed, so a raised stop only applies from the next bar.
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

from core.feature_pipeline import build_feature_frame, macro_from_primary
from core.market_history import timeframe_from_length
from core.ml_strategy import MLStrategy, _snapshot_from_row
from core.news_sentiment import SENTIMENT_FEATURES
from core.session_clock import SessionClock, SessionConfig, SessionPhase
from core.risk_manager import (
    RiskManager,
    SizingConfig,
    TrailingConfig,
    position_size,
    risk_pct_for_trade,
    trailing_stop_price,
)

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
    sizing_method: str = "fixed"
    kelly_fraction: float = 0.25
    trailing: TrailingConfig = field(default_factory=lambda: TrailingConfig(enabled=False))
    # Intraday session rules (opening lockout, entry cutoff, EOD flatten); None = off.
    session: Optional[SessionConfig] = None
    spread_bps: float = 0.0   # full bid/ask spread; half is paid on every fill

    @property
    def intraday_rules(self) -> bool:
        return self.session is not None and self.bar_length < timedelta(days=1)

    @property
    def sizing(self) -> SizingConfig:
        return SizingConfig(
            method=self.sizing_method, base_risk_pct=self.risk_per_trade_pct, kelly_fraction=self.kelly_fraction
        )

    @property
    def timeframe(self) -> str:
        return timeframe_from_length(self.bar_length)


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
    initial_stop: Optional[float] = None
    high_water: Optional[float] = None
    regime: Optional[str] = None   # market regime when the entry signal fired

    @property
    def entry_hour_et(self) -> int:
        return int(self.entry_time.tz_convert(MARKET_TZ).hour)

    def __post_init__(self) -> None:
        if self.initial_stop is None:
            self.initial_stop = self.stop
        if self.high_water is None:
            self.high_water = self.entry_price

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity - self.costs

    @property
    def r_multiple(self) -> float:
        risk = (self.entry_price - self.initial_stop) * self.quantity
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
    gated: Dict[str, int] = field(default_factory=dict)
    regime_mix: Dict[str, int] = field(default_factory=dict)
    session_blocked: Dict[str, int] = field(default_factory=dict)

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
                    "initial_stop": t.initial_stop,
                    "regime": t.regime,
                    "entry_hour_et": t.entry_hour_et,
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
        macro_bars: Optional[Dict[str, pd.DataFrame]] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """Simulate bars in [``start``, ``end``) (default: all). Earlier bars only
        warm up indicators; positions still open at ``end`` are closed there.

        ``macro_bars`` are higher-timeframe bars per symbol (daily for intraday
        runs). If the strategy needs them and none are given, they are
        resampled from the primary bars.
        """
        cfg = self.config
        slip = (cfg.slippage_bps + cfg.spread_bps / 2) / 10_000
        sizing = cfg.sizing
        calibrated = self.strategy.mode == "model"

        prepared: Dict[str, pd.DataFrame] = {}
        for symbol, frame in bars.items():
            macro = (macro_bars or {}).get(symbol)
            if macro is None and self.strategy.uses_macro:
                macro = macro_from_primary(frame, cfg.timeframe)
            feats = build_feature_frame(frame, cfg.timeframe, macro_bars=macro)
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
            if end is not None:
                feats = feats[feats.index < end]
            prepared[symbol] = feats

        timeline = sorted(set().union(*(f.index for f in prepared.values())))

        # Intraday session rules. Each day's actual close is taken from its last
        # regular bar, so early-close (13:00) days flatten on time too.
        clock = SessionClock(cfg.session) if cfg.intraday_rules else None
        no_overnight = bool(clock and cfg.session.no_overnight)
        closes_by_day: Dict[object, pd.Timestamp] = {}
        last_bar_of_day: Dict[str, set] = {}
        if clock is not None:
            for symbol, feats in prepared.items():
                regular_idx = [t for t in feats.index if _is_regular_bar(t, cfg)]
                by_day: Dict[object, pd.Timestamp] = {}
                for t in regular_idx:
                    by_day[_day_key(t)] = t
                last_bar_of_day[symbol] = set(by_day.values())
                for d, t in by_day.items():
                    end = min(t + cfg.bar_length, pd.Timestamp(f"{d} 16:00", tz="America/New_York"))
                    closes_by_day[d] = max(closes_by_day.get(d, end), end)

        def phase_at(t: pd.Timestamp) -> SessionPhase:
            return clock.phase(t, closes_by_day.get(_day_key(t)))

        session_blocked: Dict[str, int] = {}
        day_trade_days: List[object] = []
        session_days: List[object] = []
        cash = cfg.initial_capital
        positions: Dict[str, Trade] = {}
        pending_entries: Dict[str, dict] = {}
        pending_exits: set = set()
        last_close: Dict[str, float] = {}
        trades: List[Trade] = []
        blocked: Dict[str, int] = {}
        signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        skipped_outside_session = 0
        gated: Dict[str, int] = {}
        regime_mix: Dict[str, int] = {}
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
            if _day_key(trade.entry_time) == _day_key(ts):
                day_trade_days.append(_day_key(ts))  # same-day round trip (PDT)
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
                session_days.append(day)
            # Drawdown breaker sees every bar, not just entry attempts.
            self.risk_manager.update_breaker(
                {"equity": equity(), "last_equity": prev_day_equity or cfg.initial_capital}, day
            )

            for symbol, feats in prepared.items():
                if ts not in feats.index:
                    continue
                row = feats.loc[ts]
                o, h, l, c = row["open"], row["high"], row["low"], row["close"]
                regular = _is_regular_bar(ts, cfg)

                if regular:
                    flatten_now = no_overnight and phase_at(ts) is SessionPhase.FLATTEN
                    # 0. EOD flatten: the first bar opening in the FLATTEN window
                    #    closes every position at its open.
                    if flatten_now and symbol in positions:
                        close_position(symbol, ts, o, "eod_flatten", slipped=True)

                    # 1. Fills queued at the previous decision.
                    if symbol in pending_exits and symbol in positions:
                        close_position(symbol, ts, o, "signal_exit", slipped=True)
                    pending_exits.discard(symbol)

                    order = pending_entries.pop(symbol, None)
                    if order and clock is not None and (order["day"] != day or phase_at(ts) is not SessionPhase.OPEN
                                                        and phase_at(ts) is not SessionPhase.ENTRY_CUTOFF):
                        # Live orders fill within the session they were sent in.
                        session_blocked["expired_entry"] = session_blocked.get("expired_entry", 0) + 1
                        order = None
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
                                regime=order.get("regime"),
                            )

                    # 2. Bracket legs (the entry bar included).
                    trade = positions.get(symbol)
                    if trade:
                        stop_kind = "trailing_stop" if trade.stop > trade.initial_stop else "stop"
                        if o <= trade.stop:
                            close_position(symbol, ts, o, f"{stop_kind}_gap", slipped=True)
                        elif o >= trade.target:
                            close_position(symbol, ts, o, "target_gap", slipped=False)
                        elif l <= trade.stop:
                            close_position(symbol, ts, trade.stop, stop_kind, slipped=True)
                        elif h >= trade.target:
                            close_position(symbol, ts, trade.target, "target", slipped=False)

                    # 3. Trail the stop for positions still open (effective next bar).
                    trade = positions.get(symbol)
                    if trade and cfg.trailing.enabled:
                        trade.high_water = max(trade.high_water, h)
                        trade.stop = trailing_stop_price(
                            entry=trade.entry_price,
                            initial_stop=trade.initial_stop,
                            current_stop=trade.stop,
                            high_water=trade.high_water,
                            config=cfg.trailing,
                        )

                    # 3b. No bar opens inside the FLATTEN window today (coarse bars
                    #     or an early close): flatten at the last regular bar's close.
                    if no_overnight and symbol in positions and ts in last_bar_of_day.get(symbol, ()):
                        close_position(symbol, ts, c, "eod_flatten", slipped=True)

                last_close[symbol] = c

                # 4. Decide at this bar's close.
                if not np.isfinite(row["p_up"]):
                    continue
                if not _can_decide(ts, cfg):
                    # The live bot doesn't run while the market is closed.
                    if row["p_up"] >= self.strategy.threshold:
                        skipped_outside_session += 1
                    continue
                signal = self.strategy.decide(float(row["p_up"]), row, _snapshot_from_row(row))
                if clock is not None and signal.signal == "BUY":
                    phase = phase_at(ts + cfg.bar_length)
                    if phase is not SessionPhase.OPEN:
                        session_blocked[phase.value] = session_blocked.get(phase.value, 0) + 1
                        continue
                signals[signal.signal] += 1
                regime_mix[signal.regime or "n/a"] = regime_mix.get(signal.regime or "n/a", 0) + 1
                if signal.gated_by:
                    gated[signal.gated_by] = gated.get(signal.gated_by, 0) + 1

                if signal.signal == "SELL" and symbol in positions:
                    pending_exits.add(symbol)
                elif signal.signal == "BUY" and symbol not in positions and symbol not in pending_entries:
                    eq = equity()
                    # Live orders fill at once and consume buying power; queued
                    # backtest orders must reserve it until the next open.
                    reserved = sum(o["quantity"] * o["price"] * (1 + slip) for o in pending_entries.values())
                    risk_pct = risk_pct_for_trade(
                        sizing,
                        probability=signal.probability_up,
                        reward_risk=(signal.target_distance / signal.stop_distance) if signal.stop_distance else None,
                        atr_pct=row.get("atr_pct"),
                        atr_pct_median=row.get("atr_pct_median"),
                        calibrated=calibrated,
                    )
                    if risk_pct <= 0:
                        gated["kelly_no_edge"] = gated.get("kelly_no_edge", 0) + 1
                        continue
                    qty = position_size(eq, risk_pct, signal.stop_distance)
                    decision = self.risk_manager.check_order(
                        side="BUY",
                        symbol=symbol,
                        quantity=qty,
                        price=c,
                        account={
                            "equity": eq,
                            "last_equity": prev_day_equity or cfg.initial_capital,
                            "buying_power": max(0.0, cash - reserved),
                            "pattern_day_trader": False,
                            "daytrade_count": sum(1 for d in day_trade_days if d in set(session_days[-5:])),
                        },
                        position=None,
                        orders_today=[{"side": "buy"}] * entries_by_day.get(day, 0),
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        session_date=day,
                    )
                    if decision.approved:
                        pending_entries[symbol] = {
                            "quantity": decision.quantity,
                            "stop_distance": signal.stop_distance,
                            "target_distance": signal.target_distance,
                            "p_up": signal.probability_up,
                            "price": c,
                            "day": day,
                            "regime": signal.regime,
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
        result.gated = gated
        result.regime_mix = regime_mix
        result.session_blocked = session_blocked
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


def attribution(trades: List[Trade], by: str) -> pd.DataFrame:
    """Per-group trade stats; ``by`` is "regime" or "entry_hour_et".

    Answers "where does the strategy make or lose money?" — e.g. whether
    losses cluster in CHOPPY regimes or in the first hour of the session.
    """
    columns = [by, "trades", "win_rate_pct", "avg_r", "total_pnl", "worst_trade_pnl", "share_of_pnl_pct"]
    if not trades:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(
        {
            by: [getattr(t, by) if getattr(t, by) is not None else "n/a" for t in trades],
            "pnl": [t.pnl for t in trades],
            "r": [t.r_multiple for t in trades],
        }
    )
    total = frame["pnl"].sum()
    grouped = frame.groupby(by, sort=True)
    table = pd.DataFrame(
        {
            "trades": grouped.size(),
            "win_rate_pct": grouped["pnl"].apply(lambda p: (p > 0).mean() * 100).round(1),
            "avg_r": grouped["r"].mean().round(3),
            "total_pnl": grouped["pnl"].sum().round(2),
            "worst_trade_pnl": grouped["pnl"].min().round(2),
            "share_of_pnl_pct": (grouped["pnl"].sum() / total * 100).round(1) if total else np.nan,
        }
    ).reset_index()
    return table[columns]


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
    if result.gated:
        lines.append(f"Entries vetoed by gates: {result.gated}")
    if result.regime_mix:
        total = sum(result.regime_mix.values())
        mix = ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(result.regime_mix.items()))
        lines.append(f"Regime mix (decision bars): {mix}")
    cfg = result.config
    trailing = (
        f"on (trigger {cfg.trailing.trigger_r}R, lock {cfg.trailing.lock_r}R, trail {cfg.trailing.distance_r}R)"
        if cfg.trailing.enabled else "off"
    )
    lines.append(f"Sizing: {cfg.sizing_method} ({cfg.risk_per_trade_pct}% base risk); trailing stops: {trailing}")
    if cfg.intraday_rules:
        sess = cfg.session
        lines.append(
            f"Session: lockout {sess.opening_lockout_minutes} min, entry cutoff {sess.entry_cutoff_minutes} min "
            f"and flatten {sess.flatten_minutes} min before close, "
            f"{'no overnight' if sess.no_overnight else 'overnight allowed'}; "
            f"costs {cfg.slippage_bps} bps slippage + {cfg.spread_bps} bps spread"
        )
        if result.session_blocked:
            lines.append(f"Entries blocked by session rules: {result.session_blocked}")
    exits: Dict[str, int] = {}
    for t in result.trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
    if exits:
        lines.append(f"Exit reasons:      {exits}")
    if result.trades:
        lines.append("\nBy entry regime:")
        lines.append(attribution(result.trades, "regime").to_string(index=False))
        if result.config.bar_length < timedelta(days=1):
            lines.append("\nBy entry hour (ET):")
            lines.append(attribution(result.trades, "entry_hour_et").to_string(index=False))
    return "\n".join(lines)
