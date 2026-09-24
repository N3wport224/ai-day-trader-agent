from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.backtester import Backtester, BacktestConfig, _can_decide, _is_regular_bar
from core.features import compute_features
from core.ml_strategy import MLStrategy
from core.ml_training import synthetic_bars
from core.news_sentiment import SENTIMENT_FEATURES, UNAVAILABLE
from core.risk_manager import RiskLimits, RiskManager

DAY = timedelta(days=1)


class ScheduledModel:
    """predict_proba returns a scripted P(up) per bar timestamp (default 0.5)."""

    def __init__(self, schedule: dict):
        self.schedule = schedule

    def predict_proba(self, X):
        p = np.array([self.schedule.get(ts, 0.5) for ts in X.index])
        return np.column_stack([1 - p, p])


def _strategy(schedule, threshold=0.6, exit_threshold=0.2):
    artifact = {
        "pipeline": ScheduledModel(schedule),
        "label_params": {"stop_atr_mult": 1.0, "target_atr_mult": 2.0},
    }
    return MLStrategy(artifact, confidence_threshold=threshold, exit_threshold=exit_threshold)


def _flat_bars(n=40, price=100.0, rng=1.0):
    """Daily bars with constant range so ATR == rng."""
    idx = pd.date_range("2026-01-05", periods=n, freq="D", tz="UTC")
    close = np.full(n, price)
    return pd.DataFrame(
        {"open": close, "high": close + rng / 2, "low": close - rng / 2, "close": close, "volume": 1e5},
        index=idx,
    )


def _run(bars, schedule, *, limits=None, slippage_bps=0.0, **strategy_kwargs):
    config = BacktestConfig(initial_capital=10_000, risk_per_trade_pct=1.0, slippage_bps=slippage_bps, bar_length=DAY)
    manager = RiskManager(limits or RiskLimits(min_price=1.0, max_position_pct=1.0))
    return Backtester(_strategy(schedule, **strategy_kwargs), manager, config).run({"AAA": bars})


def test_entry_fills_next_open_with_slippage_and_recentred_bracket() -> None:
    bars = _flat_bars()
    decide_at = bars.index[30]
    bars.loc[bars.index[31], "open"] = 101.0  # next bar opens higher
    bars.loc[bars.index[31], ["high", "low", "close"]] = [101.4, 100.6, 101.0]

    result = _run(bars, {decide_at: 0.9}, slippage_bps=10)
    trade = result.trades[0]
    atr = compute_features(bars)["atr"].loc[decide_at]

    assert trade.entry_time == bars.index[31]
    assert trade.entry_price == pytest.approx(101.0 * 1.001)
    assert trade.stop == pytest.approx(round(trade.entry_price - round(atr, 4), 2))
    assert trade.target == pytest.approx(round(trade.entry_price + round(2 * atr, 4), 2))
    # 1% of $10,000 risked over a 1 ATR stop (100 shares), capped by the cash
    # available at the higher fill price.
    assert trade.quantity == min(int(100 // round(atr, 4)), int(10_000 // trade.entry_price))


def test_stop_and_target_fills_including_gaps_and_ambiguous_bars() -> None:
    def scenario(next_bar):
        bars = _flat_bars()
        bars.loc[bars.index[32], ["open", "high", "low", "close"]] = next_bar
        return _run(bars, {bars.index[30]: 0.9}).trades[0]

    # Entry at 100 on bar 31; ATR ~1 -> stop ~99, target ~102.
    assert scenario([100.0, 100.2, 98.5, 99.0]).exit_reason == "stop"
    assert scenario([100.0, 102.5, 99.8, 102.0]).exit_reason == "target"
    both = scenario([100.0, 103.0, 98.0, 101.0])
    assert both.exit_reason == "stop"  # touching both counts as the stop
    gap_down = scenario([97.0, 97.5, 96.0, 97.0])
    assert gap_down.exit_reason == "stop_gap" and gap_down.exit_price == pytest.approx(97.0)
    gap_up = scenario([105.0, 106.0, 104.5, 105.0])
    assert gap_up.exit_reason == "target_gap" and gap_up.exit_price == pytest.approx(105.0)


def test_sell_signal_exits_at_next_open() -> None:
    bars = _flat_bars()
    trade = _run(bars, {bars.index[30]: 0.9, bars.index[33]: 0.05}).trades[0]

    assert trade.exit_reason == "signal_exit"
    assert trade.exit_time == bars.index[34]


def test_risk_manager_limits_apply_in_backtest() -> None:
    bars = {f"S{i}": _flat_bars() for i in range(4)}
    day = bars["S0"].index[30]
    config = BacktestConfig(initial_capital=10_000, bar_length=DAY, slippage_bps=0)
    manager = RiskManager(RiskLimits(max_daily_trades=2, min_price=1.0, max_position_pct=0.2))

    result = Backtester(_strategy({day: 0.9}), manager, config).run(bars)

    assert len(result.trades) == 2
    assert all(t.quantity == 20 for t in result.trades)  # 20% position cap of $10,000
    assert result.blocked == {"Daily entry limit reached": 2}


def test_queued_orders_reserve_buying_power() -> None:
    bars = {f"S{i}": _flat_bars() for i in range(3)}
    day = bars["S0"].index[30]
    config = BacktestConfig(initial_capital=10_000, bar_length=DAY, slippage_bps=0)
    manager = RiskManager(RiskLimits(max_daily_trades=10, min_price=1.0, max_position_pct=0.6))

    result = Backtester(_strategy({day: 0.9}), manager, config).run(bars)

    # S0 takes $6,000 of the cap; S1 is shrunk to the $4,000 left; S2 has no room.
    assert sorted(t.quantity for t in result.trades) == [40, 60]
    assert result.blocked == {"No room": 1}


def test_daily_loss_limit_blocks_new_entries() -> None:
    bars = _flat_bars(n=45)
    # Day 31: enter; day 32: crash through the stop -> ~1% loss; day 33 signal is blocked
    bars.loc[bars.index[32], ["open", "high", "low", "close"]] = [100.0, 100.0, 90.0, 90.0]
    schedule = {bars.index[30]: 0.9, bars.index[32]: 0.9}
    limits = RiskLimits(min_price=1.0, max_position_pct=1.0, max_daily_loss_pct=0.5)

    result = _run(bars, schedule, limits=limits)

    assert len(result.trades) == 1
    assert any(k.startswith("Daily loss limit") for k in result.blocked)


def test_trades_before_t_ignore_prices_after_t() -> None:
    """No lookahead: changing the future can't change already-closed trades."""
    bars = synthetic_bars(n=1200, seed=9)
    strategy = MLStrategy(artifact=None, model_path="/none", confidence_threshold=0.55)
    config = BacktestConfig(initial_capital=50_000, slippage_bps=5)
    limits = RiskLimits(min_price=1.0, max_daily_trades=10)
    cutoff = bars.index[900]
    shocked = bars.copy()
    shocked.loc[shocked.index > cutoff, ["open", "high", "low", "close"]] *= 1.5

    base = Backtester(strategy, RiskManager(limits), config).run({"X": bars})
    alt = Backtester(strategy, RiskManager(limits), config).run({"X": shocked})

    def closed_before(result):
        return [(t.entry_time, t.exit_time, round(t.pnl, 6)) for t in result.trades if t.exit_time < cutoff]

    assert closed_before(base)  # the test is meaningful
    assert closed_before(base) == closed_before(alt)


def test_batch_probabilities_match_per_bar_predict() -> None:
    bars = synthetic_bars(n=260, seed=2)
    feats = compute_features(bars)
    sent = pd.DataFrame(np.nan, index=feats.index, columns=SENTIMENT_FEATURES)
    sent["sentiment_available"] = 0.0
    strategy = MLStrategy(artifact=None, model_path="/none")

    batch = strategy.probabilities(feats, sent)

    for t in (50, 150, 259):
        single = strategy.predict(compute_features(bars.iloc[: t + 1]), UNAVAILABLE).probability_up
        assert batch[t] == pytest.approx(single)


def test_intraday_session_rules() -> None:
    cfg = BacktestConfig(bar_length=timedelta(hours=1))
    ts = lambda s: pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")  # noqa: E731

    assert _is_regular_bar(ts("2026-03-02 09:00"), cfg)       # overlaps the 9:30 open
    assert not _is_regular_bar(ts("2026-03-02 08:00"), cfg)   # pre-market only
    assert not _is_regular_bar(ts("2026-03-02 16:00"), cfg)   # after hours
    assert not _is_regular_bar(ts("2026-03-07 11:00"), cfg)   # Saturday
    assert _can_decide(ts("2026-03-02 09:00"), cfg)            # closes 10:00, market open
    assert not _can_decide(ts("2026-03-02 15:00"), cfg)       # closes 16:00, market closed


def test_pure_random_walk_has_no_edge() -> None:
    """A lookahead bug would show up as reliable profits on unpredictable prices."""
    avg_r = []
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        n = 2000
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        spread = np.abs(rng.normal(0, 0.004, n)) * close
        idx = pd.date_range("2025-01-02 14:30", periods=n, freq="h", tz="UTC")
        bars = pd.DataFrame(
            {"open": open_, "high": np.maximum(open_, close) + spread,
             "low": np.minimum(open_, close) - spread, "close": close, "volume": 1e5},
            index=idx,
        )
        result = Backtester(
            MLStrategy(artifact=None, model_path="/none", confidence_threshold=0.55),
            RiskManager(RiskLimits(min_price=1.0, max_daily_trades=10)),
            BacktestConfig(slippage_bps=0),
        ).run({"RW": bars})
        avg_r.append(result.metrics["avg_r"])

    assert np.mean(avg_r) < 0.15


def test_backtest_script_walkforward_writes_reports(tmp_path: Path) -> None:
    from scripts import backtest

    out = tmp_path / "bt"
    code = backtest.main(
        ["--synthetic", "--symbols", "AAA,BBB", "--days", "200", "--threshold", "0.55", "--out", str(out)]
    )

    assert code == 0
    assert (out / "trades.csv").exists() and (out / "equity.csv").exists()
    assert (out / "threshold_sweep.csv").exists() and (out / "calibration.csv").exists()
    assert "total_return_pct" in (out / "summary.json").read_text()


def test_threshold_and_calibration_tables() -> None:
    from core.ml_training import LabelParams, calibration_table, threshold_table

    proba = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [0, 0, 1, 0, 1]
    table = threshold_table(proba, labels, LabelParams(stop_atr_mult=1.0, target_atr_mult=2.0), thresholds=(0.4, 0.8, 0.95))

    assert table["signals"].tolist() == [3, 1, 0]
    assert table["hit_rate"].iloc[0] == pytest.approx(2 / 3, abs=1e-4)
    assert table["approx_expectancy_r"].iloc[0] == pytest.approx(2 / 3 * 2 - 1 / 3, abs=1e-3)
    assert np.isnan(table["hit_rate"].iloc[2])

    calib = calibration_table(proba, labels, bins=2)
    # Buckets are right-closed: [0, 0.5] holds 0.1, 0.3, 0.5; (0.5, 1] holds 0.7, 0.9.
    assert calib["bars"].tolist() == [3, 2]
    assert calib["hit_rate"].tolist() == pytest.approx([1 / 3, 1 / 2], abs=1e-4)


def test_buy_signals_outside_market_hours_are_counted_not_traded() -> None:
    idx = pd.date_range("2026-03-02 14:00", periods=60, freq="h", tz="UTC")  # includes nights
    close = np.full(60, 100.0)
    bars = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e5}, index=idx
    )
    night = pd.Timestamp("2026-03-03 04:00", tz="UTC")   # 23:00 ET
    session = pd.Timestamp("2026-03-03 15:00", tz="UTC")  # 10:00 ET bar, closes 11:00
    config = BacktestConfig(initial_capital=10_000, slippage_bps=0)

    result = Backtester(
        _strategy({night: 0.9, session: 0.9}),
        RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0)),
        config,
    ).run({"AAA": bars})

    assert result.skipped_outside_session == 1
    assert [t.entry_time for t in result.trades] == [session + timedelta(hours=1)]
