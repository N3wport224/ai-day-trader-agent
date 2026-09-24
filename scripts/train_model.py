#!/usr/bin/env python3
"""
Train the ML signal model and save it to models/ml_signal.joblib.

Steps:
  1. Download completed OHLCV bars per symbol (Alpaca, Yahoo fallback).
  2. Download and score news (Alpaca News API + FinBERT or lexicon scorer),
     then compute the 24h weighted sentiment as of each bar's close.
  3. Compute causal technical features and triple-barrier labels
     (ATR take-profit hit before ATR stop within --horizon bars).
  4. Train a LightGBM pipeline, report out-of-sample metrics on a purged
     chronological hold-out, refit on all data and save the artifact.

Examples:
  python scripts/train_model.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days 365
  python scripts/train_model.py --symbols AAPL --no-news --timeframe 1Day --days 1500
  python scripts/train_model.py --synthetic      # offline demo, no network

Read the printed metrics before trading on the model: an AUC near 0.5 or a
negative expectancy means the model has no edge on this data.
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

from core.market_history import bar_length, get_history  # noqa: E402
from core.ml_strategy import DEFAULT_MODEL_PATH  # noqa: E402
from core.ml_training import LabelParams, build_dataset, synthetic_bars, train  # noqa: E402
from core.news_sentiment import (  # noqa: E402
    AlpacaNewsClient,
    get_scorer,
    half_life_from_env,
    score_articles,
)

logger = logging.getLogger("train_model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,AMD,SPY"))
    parser.add_argument("--timeframe", default=os.getenv("ML_TIMEFRAME", "1Hour"), choices=["15Min", "1Hour", "1Day"])
    parser.add_argument("--days", type=int, default=365, help="Days of history per symbol")
    parser.add_argument("--horizon", type=int, default=12, help="Bars to wait for target/stop")
    parser.add_argument("--stop-atr", type=float, default=float(os.getenv("ATR_STOP_MULT", "1.5")))
    parser.add_argument("--target-atr", type=float, default=float(os.getenv("ATR_TARGET_MULT", "3.0")))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.6")))
    parser.add_argument("--no-news", action="store_true", help="Train on technical features only")
    parser.add_argument("--synthetic", action="store_true", help="Use generated bars (offline demo)")
    parser.add_argument("--output", default=os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    params = LabelParams(horizon=args.horizon, stop_atr_mult=args.stop_atr, target_atr_mult=args.target_atr)
    bar_len = bar_length(args.timeframe)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    news_client = None if (args.no_news or args.synthetic) else AlpacaNewsClient()
    if news_client is not None and not news_client.configured:
        logger.warning("Alpaca credentials missing; training without news sentiment")
        news_client = None
    scorer = get_scorer() if news_client else None
    if scorer:
        logger.info(f"Scoring news with the {scorer.name} sentiment model")

    end = datetime.now(timezone.utc)
    datasets = {}
    for i, symbol in enumerate(symbols):
        if args.synthetic:
            bars = synthetic_bars(n=max(args.days * 7, 800), seed=i + 1)
        else:
            bars = get_history(symbol, args.timeframe, args.days, end=end)
        if len(bars) < 250:
            logger.warning(f"{symbol}: only {len(bars)} bars, skipping")
            continue

        scored = None
        if news_client:
            try:
                start = bars.index[0].to_pydatetime() - timedelta(days=1)
                articles = news_client.fetch(symbol, start, end, max_articles=20000)
                scored = score_articles(articles, scorer)
                logger.info(f"{symbol}: scored {len(scored)} articles")
            except Exception as exc:
                logger.warning(f"{symbol}: news unavailable ({exc}); sentiment features left missing")

        data = build_dataset(bars, params, bar_len, scored, half_life=half_life_from_env())
        datasets[symbol] = data
        logger.info(f"{symbol}: {len(bars)} bars -> {len(data)} labeled rows, base rate {data['label'].mean():.2%}")

    if not datasets:
        logger.error("No usable data; nothing trained")
        return 1

    try:
        artifact = train(datasets, params, bar_len, threshold=args.threshold, timeframe=args.timeframe)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(artifact, output)
    print(f"\nSaved model to {output}")
    print(json.dumps({k: artifact[k] for k in ("symbols", "timeframe", "label_params", "uses_sentiment", "metrics")},
                     indent=2, default=str))
    metrics = artifact["metrics"]
    if not metrics["auc"] > 0.52 or not metrics["approx_expectancy_r"] > 0:
        print("\n⚠️  Out-of-sample results show little or no edge. Keep the bot in dry-run mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
