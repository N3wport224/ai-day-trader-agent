#!/usr/bin/env python3
"""
Backtest the ML trading path (features -> model -> RiskManager -> brackets).

Modes:
  walkforward (default)  Train on the first --train-fraction of history, then
                         trade only the unseen remainder. The honest test.
  model                  Use a saved model (ML_MODEL_PATH / --model). Warns if
                         the test period overlaps the model's training data.
  heuristic              The no-model fallback rules.

Risk limits come from .env exactly as in live trading (MAX_DAILY_LOSS_PCT,
MAX_DAILY_TRADES, MAX_PORTFOLIO_ALLOCATION, MIN_PRICE, ...).

Examples:
  python scripts/backtest.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days 730
  python scripts/backtest.py --mode heuristic --timeframe 1Day --days 1500
  python scripts/backtest.py --news --out reports/bt      # with news sentiment, CSVs
  python scripts/backtest.py --synthetic                  # offline demo

  # Before/after benchmark of the regime filter, daily-trend confirmation,
  # volatility-scaled Kelly sizing and trailing stops on identical data:
  python scripts/backtest.py --regime suppress --mtf --mtf-gate --sizing kelly --trailing --compare

  # Intraday day trading on 5-minute bars: 15-min opening lockout, no entries
  # after 15:45 ET, forced flatten at 15:50 ET, 2 bps spread + 5 bps slippage
  python scripts/backtest.py --timeframe 5m --days 60 --no-overnight --opening-lockout-minutes 15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402

from core.backtester import Backtester, BacktestConfig, format_report  # noqa: E402
from core.market_history import bar_length, get_history  # noqa: E402
from core.ml_strategy import DEFAULT_MODEL_PATH, REGIME_POLICIES, MLStrategy, load_artifact  # noqa: E402
from dataclasses import replace  # noqa: E402

from core.market_history import is_intraday, normalize_timeframe  # noqa: E402
from core.session_clock import SessionConfig  # noqa: E402
from core.ml_training import (  # noqa: E402
    LabelParams,
    build_dataset,
    calibration_table,
    synthetic_bars,
    synthetic_intraday_bars,
    threshold_table,
    train,
)
from core.news_sentiment import (  # noqa: E402
    AlpacaNewsClient,
    get_scorer,
    half_life_from_env,
    rolling_sentiment,
    score_articles,
)
from core.risk_manager import SIZING_METHODS, RiskLimits, RiskManager, TrailingConfig  # noqa: E402

logger = logging.getLogger("backtest")


def _load_bars(args, symbols, end):
    bars = {}
    for i, symbol in enumerate(symbols):
        if args.synthetic and is_intraday(args.timeframe):
            frame = synthetic_intraday_bars(days=args.days, freq=str(bar_length(args.timeframe)), seed=i + 1)
        elif args.synthetic:
            frame = synthetic_bars(n=max(args.days * 7, 1000), seed=i + 1)
        else:
            frame = get_history(symbol, args.timeframe, args.days, end=end)
        if len(frame) < 300:
            logger.warning(f"{symbol}: only {len(frame)} bars, skipping")
            continue
        bars[symbol] = frame
    return bars


def _load_sentiment(bars, bar_len, end):
    client = AlpacaNewsClient()
    if not client.configured:
        logger.warning("Alpaca credentials missing; backtesting without news sentiment")
        return None, None
    scorer = get_scorer()
    logger.info(f"Scoring news with the {scorer.name} sentiment model")
    sentiment, scored_by_symbol = {}, {}
    for symbol, frame in bars.items():
        try:
            articles = client.fetch(symbol, frame.index[0].to_pydatetime() - timedelta(days=1), end, max_articles=20000)
        except Exception as exc:
            logger.warning(f"{symbol}: news unavailable ({exc})")
            continue
        scored = score_articles(articles, scorer)
        scored_by_symbol[symbol] = scored
        closes = [ts.to_pydatetime() + bar_len for ts in frame.index]
        sent = rolling_sentiment(closes, scored, half_life=half_life_from_env())
        sent.index = frame.index
        sentiment[symbol] = sent
        logger.info(f"{symbol}: {len(scored)} scored articles")
    return sentiment, scored_by_symbol


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _variant(args, *, baseline: bool) -> dict:
    """Settings for one backtest run. The baseline switches every new layer off."""
    if baseline:
        return {
            "name": "baseline",
            "use_macro": False,
            "strategy": {"regime_policy": "off", "mtf_confirmation": False},
            "sizing": "fixed",
            "trailing": TrailingConfig(enabled=False),
        }
    return {
        "name": "enhanced" if args.compare else "run",
        "use_macro": args.mtf,
        "strategy": {
            "regime_policy": args.regime,
            "regime_bump": args.regime_bump,
            "mtf_confirmation": args.mtf_gate,
        },
        "sizing": args.sizing,
        "trailing": TrailingConfig(
            enabled=args.trailing,
            trigger_r=args.trail_trigger_r,
            lock_r=args.trail_lock_r,
            distance_r=args.trail_distance_r,
        ),
    }


def _run_variant(args, variant, bars, macro, sentiment, scored, bar_len):
    """Build the strategy for one variant (training it in walk-forward mode) and backtest it."""
    start = audit = None
    kwargs = dict(confidence_threshold=args.threshold, **variant["strategy"])
    if args.mode == "walkforward":
        timeline = sorted(set().union(*(f.index for f in bars.values())))
        start = timeline[int(len(timeline) * args.train_fraction)]
        params = LabelParams(horizon=args.horizon, stop_atr_mult=args.stop_atr, target_atr_mult=args.target_atr)
        purge = params.horizon * bar_len

        def dataset(symbol, frame):
            return build_dataset(
                frame, params, bar_len, (scored or {}).get(symbol), half_life=half_life_from_env(),
                macro_bars=(macro or {}).get(symbol), use_macro=variant["use_macro"],
            )

        datasets = {}
        for symbol, frame in bars.items():
            data = dataset(symbol, frame[frame.index < start])
            datasets[symbol] = data[data.index < start - purge]  # labels must not see the test period
        artifact = train(
            datasets, params, bar_len, threshold=args.threshold, timeframe=args.timeframe,
            use_macro=variant["use_macro"],
        )
        logger.info(f"[{variant['name']}] trained on bars before {start}; trading {start} onward (unseen)")
        strategy = MLStrategy(artifact, **kwargs)

        test = pd.concat([d[d.index >= start] for d in (dataset(s, f) for s, f in bars.items())])
        proba = artifact["pipeline"].predict_proba(test[artifact["feature_columns"]])[:, 1]
        audit = {
            "train_base_rate": artifact["train_base_rate"],
            "test_base_rate": float(test["label"].mean()),
            "exit_threshold_in_use": strategy.exit_threshold,
            "thresholds": threshold_table(proba, test["label"], params),
            "calibration": calibration_table(proba, test["label"]),
        }
    elif args.mode == "model":
        artifact = load_artifact(args.model)
        if artifact is None:
            raise ValueError(f"No usable model at {args.model}; train one or use --mode walkforward")
        data_end = artifact.get("data_end")
        first_bar = min(f.index[0] for f in bars.values())
        if data_end and pd.Timestamp(data_end) >= first_bar:
            logger.warning(
                f"Model was trained on data up to {data_end}, which overlaps this test period. "
                "Results are IN-SAMPLE and overstate performance; prefer --mode walkforward."
            )
        strategy = MLStrategy(artifact, **kwargs)
    else:
        strategy = MLStrategy(artifact=None, model_path="/nonexistent", **kwargs)

    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk_pct,
        slippage_bps=args.slippage_bps,
        commission_per_share=args.commission,
        bar_length=bar_len,
        sizing_method=variant["sizing"],
        kelly_fraction=args.kelly_fraction,
        trailing=variant["trailing"],
        session=args.session,
        spread_bps=args.spread_bps,
    )
    limits = replace(RiskLimits.from_env(), day_trading=bool(args.session and args.session.no_overnight))
    result = Backtester(strategy, RiskManager(limits), config).run(
        bars, sentiment, start=start, macro_bars=macro
    )
    return result, audit


def _print_audit(audit) -> None:
    print(
        f"\n== Threshold audit (out-of-sample bars, before risk limits and costs) ==\n"
        f"Base rate (target hit before stop): train {audit['train_base_rate']:.1%}, "
        f"test {audit['test_base_rate']:.1%}; exit threshold in use {audit['exit_threshold_in_use']:.3f}\n"
    )
    print(audit["thresholds"].to_string(index=False))
    print("\nCalibration (does predicted P match what happened?):")
    print(audit["calibration"].to_string(index=False))
    print(
        "\nNote: picking the threshold that looks best here and reporting its result"
        " overstates performance; confirm any change with a fresh walk-forward over a different period."
    )


COMPARE_METRICS = [
    ("trades", "Trades"),
    ("win_rate_pct", "Win rate %"),
    ("avg_r", "Avg R"),
    ("profit_factor", "Profit factor"),
    ("total_return_pct", "Return %"),
    ("benchmark_return_pct", "Buy & hold %"),
    ("max_drawdown_pct", "Max drawdown %"),
    ("sharpe_daily", "Sharpe (daily)"),
    ("exposure_pct", "Time in market %"),
]


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for key, label in COMPARE_METRICS:
        row = {"metric": label}
        for name, result in results.items():
            row[name] = result.metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_reports(out: Path, result, audit) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result.trades_frame().to_csv(out / "trades.csv", index=False)
    result.equity.rename("equity").to_csv(out / "equity.csv", index_label="time")
    summary = {
        "metrics": result.metrics,
        "signals": result.signals,
        "blocked": result.blocked,
        "gated": result.gated,
        "regime_mix": result.regime_mix,
        "skipped_outside_session": result.skipped_outside_session,
    }
    if audit:
        audit["thresholds"].to_csv(out / "threshold_sweep.csv", index=False)
        audit["calibration"].to_csv(out / "calibration.csv", index=False)
        summary.update(
            train_base_rate=audit["train_base_rate"],
            test_base_rate=audit["test_base_rate"],
            exit_threshold_in_use=audit["exit_threshold_in_use"],
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,AMD,SPY"))
    parser.add_argument("--mode", choices=["walkforward", "model", "heuristic"], default="walkforward")
    parser.add_argument("--model", default=os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--timeframe", default=os.getenv("ML_TIMEFRAME", "1Hour"),
                        help="1m, 5m, 15m, 1h (default) or 1d")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--train-fraction", type=float, default=0.6, help="walkforward: share of history used to train")
    parser.add_argument("--horizon", type=int, default=12, help="walkforward: label horizon in bars")
    parser.add_argument("--stop-atr", type=float, default=float(os.getenv("ATR_STOP_MULT", "1.5")))
    parser.add_argument("--target-atr", type=float, default=float(os.getenv("ATR_TARGET_MULT", "3.0")))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.6")))
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=float(os.getenv("RISK_PER_TRADE_PCT", "1.0")))
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission", type=float, default=0.0, help="Per share")
    parser.add_argument("--news", action="store_true", help="Include Alpaca news sentiment")
    parser.add_argument("--synthetic", action="store_true", help="Generated bars (offline demo)")
    parser.add_argument("--out", help="Directory for trades.csv, equity.csv, summary.json and audit CSVs")

    gates = parser.add_argument_group("regime and multi-timeframe gates")
    gates.add_argument("--regime", choices=REGIME_POLICIES, default=os.getenv("REGIME_FILTER", "penalty"),
                       help="off | suppress (longs only in TRENDING_BULL) | penalty (raise threshold elsewhere)")
    gates.add_argument("--regime-bump", type=float, default=float(os.getenv("REGIME_THRESHOLD_BUMP", "0.10")))
    gates.add_argument("--adx-threshold", type=float, default=float(os.getenv("ADX_TREND_THRESHOLD", "25")))
    gates.add_argument("--mtf", action="store_true",
                       help="Train with higher-timeframe (daily/weekly) trend features")
    gates.add_argument("--mtf-gate", action="store_true", default=_env_bool("MTF_CONFIRMATION", False),
                       help="Only enter longs aligned with the higher-timeframe trend")

    sizing = parser.add_argument_group("sizing and exits")
    sizing.add_argument("--sizing", choices=SIZING_METHODS, default=os.getenv("POSITION_SIZING_METHOD", "fixed"))
    sizing.add_argument("--kelly-fraction", type=float, default=float(os.getenv("KELLY_FRACTION", "0.25")))
    sizing.add_argument("--trailing", action=argparse.BooleanOptionalAction,
                        default=_env_bool("TRAILING_STOP_ENABLED", True))
    sizing.add_argument("--trail-trigger-r", type=float, default=float(os.getenv("TRAILING_STOP_TRIGGER_R", "1.5")))
    sizing.add_argument("--trail-lock-r", type=float, default=float(os.getenv("TRAILING_STOP_LOCK_R", "0")))
    sizing.add_argument("--trail-distance-r", type=float, default=float(os.getenv("TRAILING_STOP_DISTANCE_R", "1.5")))
    intraday = parser.add_argument_group("intraday session rules (1m/5m/15m)")
    intraday.add_argument("--overnight", action=argparse.BooleanOptionalAction, default=None,
                          help="--no-overnight (default for intraday) flattens every position before the close")
    intraday.add_argument("--opening-lockout-minutes", type=int,
                          default=int(os.getenv("OPENING_LOCKOUT_MINUTES", "15")))
    intraday.add_argument("--entry-cutoff-minutes", type=int,
                          default=int(os.getenv("ENTRY_CUTOFF_MINUTES_BEFORE_CLOSE", "15")))
    intraday.add_argument("--flatten-minutes", type=int,
                          default=int(os.getenv("FLATTEN_MINUTES_BEFORE_CLOSE", "10")))
    intraday.add_argument("--spread-bps", type=float, default=None,
                          help="Full bid/ask spread cost in bps, half paid per fill (default 2 intraday, 0 otherwise)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run a baseline (all new layers off) on the same data and print both")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    os.environ["ADX_TREND_THRESHOLD"] = str(args.adx_threshold)
    try:
        args.timeframe = normalize_timeframe(args.timeframe)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    intraday_run = is_intraday(args.timeframe)
    no_overnight = (not args.overnight) if args.overnight is not None else intraday_run
    args.session = SessionConfig(
        opening_lockout_minutes=args.opening_lockout_minutes,
        entry_cutoff_minutes=args.entry_cutoff_minutes,
        flatten_minutes=args.flatten_minutes,
        no_overnight=no_overnight,
    ) if intraday_run else None
    if args.spread_bps is None:
        args.spread_bps = 2.0 if intraday_run else 0.0
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bar_len = bar_length(args.timeframe)
    end = datetime.now(timezone.utc)

    bars = _load_bars(args, symbols, end)
    if not bars:
        logger.error("No usable price history")
        return 1
    sentiment, scored = _load_sentiment(bars, bar_len, end) if (args.news and not args.synthetic) else (None, None)

    macro = None
    if (args.mtf or args.mtf_gate) and not args.synthetic and args.timeframe != "1Day":
        macro = {s: get_history(s, "1Day", args.days + 120, end=end) for s in bars}
        macro = {s: m for s, m in macro.items() if len(m)}  # missing ones are resampled from the primary bars

    variants = [_variant(args, baseline=True)] if args.compare else []
    variants.append(_variant(args, baseline=False))
    results, audits = {}, {}
    for variant in variants:
        try:
            results[variant["name"]], audits[variant["name"]] = _run_variant(
                args, variant, bars, macro, sentiment, scored, bar_len
            )
        except ValueError as exc:
            logger.error(f"[{variant['name']}] {exc}")
            return 1

    suffix = f"{args.mode}, {args.timeframe}, {', '.join(bars)}" + (" (synthetic)" if args.synthetic else "")
    for name, result in results.items():
        print()
        print(format_report(result, f"{name} backtest: {suffix}"))
    main_name = variants[-1]["name"]
    if audits.get(main_name):
        _print_audit(audits[main_name])

    if args.compare:
        print("\n== Baseline vs enhanced (same data, same unseen period) ==")
        print(comparison_table(results).to_string(index=False))

    m = results[main_name].metrics
    if m["trades"] < 30:
        print(f"\n⚠️  Only {m['trades']} trades: too few to judge the strategy. Use more symbols or history.")
    elif m["total_return_pct"] <= m["benchmark_return_pct"] or not m["avg_r"] > 0:
        print("\n⚠️  The strategy did not beat buy-and-hold or lost money per trade. Keep the bot in dry-run.")

    if args.out:
        out = Path(args.out)
        for name, result in results.items():
            _write_reports(out / name if args.compare else out, result, audits.get(name))
        if args.compare:
            comparison_table(results).to_csv(out / "comparison.csv", index=False)
        print(f"\nWrote reports to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
