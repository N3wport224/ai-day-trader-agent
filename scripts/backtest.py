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
from core.ml_strategy import DEFAULT_MODEL_PATH, FEATURE_COLUMNS, MLStrategy, load_artifact  # noqa: E402
from core.ml_training import (  # noqa: E402
    LabelParams,
    build_dataset,
    calibration_table,
    synthetic_bars,
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
from core.risk_manager import RiskLimits, RiskManager  # noqa: E402

logger = logging.getLogger("backtest")


def _load_bars(args, symbols, end):
    bars = {}
    for i, symbol in enumerate(symbols):
        frame = (
            synthetic_bars(n=max(args.days * 7, 1000), seed=i + 1)
            if args.synthetic
            else get_history(symbol, args.timeframe, args.days, end=end)
        )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,AMD,SPY"))
    parser.add_argument("--mode", choices=["walkforward", "model", "heuristic"], default="walkforward")
    parser.add_argument("--model", default=os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--timeframe", default=os.getenv("ML_TIMEFRAME", "1Hour"), choices=["15Min", "1Hour", "1Day"])
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
    parser.add_argument("--out", help="Directory for trades.csv, equity.csv and summary.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bar_len = bar_length(args.timeframe)
    end = datetime.now(timezone.utc)

    bars = _load_bars(args, symbols, end)
    if not bars:
        logger.error("No usable price history")
        return 1
    sentiment, scored = _load_sentiment(bars, bar_len, end) if (args.news and not args.synthetic) else (None, None)

    start = None
    audit = None
    if args.mode == "walkforward":
        timeline = sorted(set().union(*(f.index for f in bars.values())))
        start = timeline[int(len(timeline) * args.train_fraction)]
        params = LabelParams(horizon=args.horizon, stop_atr_mult=args.stop_atr, target_atr_mult=args.target_atr)
        purge = params.horizon * bar_len
        datasets = {}
        for symbol, frame in bars.items():
            history = frame[frame.index < start]
            data = build_dataset(history, params, bar_len, (scored or {}).get(symbol), half_life=half_life_from_env())
            # Drop rows whose labels would look into the test period.
            datasets[symbol] = data[data.index < start - purge]
        try:
            artifact = train(datasets, params, bar_len, threshold=args.threshold, timeframe=args.timeframe)
        except ValueError as exc:
            logger.error(f"Walk-forward training failed: {exc}")
            return 1
        logger.info(f"Trained on bars before {start}; trading {start} onward (never seen by the model)")
        strategy = MLStrategy(artifact, confidence_threshold=args.threshold)

        # Threshold audit on the unseen period: how often did setups at each
        # confidence level actually hit the target before the stop?
        test_sets = [
            build_dataset(frame, params, bar_len, (scored or {}).get(symbol), half_life=half_life_from_env())
            for symbol, frame in bars.items()
        ]
        test = pd.concat([d[d.index >= start] for d in test_sets])
        proba = artifact["pipeline"].predict_proba(test[FEATURE_COLUMNS])[:, 1]
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
            logger.error(f"No usable model at {args.model}; train one or use --mode walkforward")
            return 1
        data_end = artifact.get("data_end")
        first_bar = min(f.index[0] for f in bars.values())
        if data_end and pd.Timestamp(data_end) >= first_bar:
            logger.warning(
                f"Model was trained on data up to {data_end}, which overlaps this test period. "
                "Results are IN-SAMPLE and overstate performance; prefer --mode walkforward."
            )
        strategy = MLStrategy(artifact, confidence_threshold=args.threshold)
    else:
        strategy = MLStrategy(artifact=None, model_path="/nonexistent", confidence_threshold=args.threshold)

    limits = RiskLimits.from_env()
    logger.info(f"Risk limits: {limits}")
    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk_pct,
        slippage_bps=args.slippage_bps,
        commission_per_share=args.commission,
        bar_length=bar_len,
    )
    result = Backtester(strategy, RiskManager(limits), config).run(bars, sentiment, start=start)

    title = f"{args.mode} backtest, {args.timeframe}, {', '.join(bars)}" + (" (synthetic)" if args.synthetic else "")
    print()
    print(format_report(result, title))

    if audit:
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

    m = result.metrics
    if m["trades"] < 30:
        print(f"\n⚠️  Only {m['trades']} trades: too few to judge the strategy. Use more symbols or history.")
    elif m["total_return_pct"] <= m["benchmark_return_pct"] or not m["avg_r"] > 0:
        print("\n⚠️  The strategy did not beat buy-and-hold or lost money per trade. Keep the bot in dry-run.")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.trades_frame().to_csv(out / "trades.csv", index=False)
        result.equity.rename("equity").to_csv(out / "equity.csv", index_label="time")
        summary = {
            "metrics": m,
            "signals": result.signals,
            "blocked": result.blocked,
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
        print(f"\nWrote reports to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
