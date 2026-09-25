#!/usr/bin/env python3
"""
One-click strategy validation (used by the dashboard's "Validate strategy"
button, also runnable directly):

  1. Walk-forward backtest on real data with the live rules and costs
     (scripts/backtest.py --mode walkforward --folds 4 --promote).
  2. Only if it passes the edge gate: train the model the bot will use, with
     the same settings, so the validated setup and the live setup match.

Exit codes: 0 validated and trained, 2 no edge found (nothing trained), 1 error.

    python scripts/validate_and_train.py --symbols AAPL,MSFT,NVDA --timeframe 5m --days 120 --market
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from core.edge_gate import report_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the strategy out of sample, then train it")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--market", action="store_true")
    parser.add_argument("--out", default="reports/validation")
    parser.add_argument("--trigger", choices=("manual", "scheduled"), default="manual")
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    try:
        code = _run(args)
    except Exception as exc:  # still record the outcome so the dashboard can report it
        print(f"RESULT: validation failed with an error: {exc}", flush=True)
        code = 1
    result = {0: "validated", 2: "no_edge"}.get(code, "error")
    summary = {
        "result": result,
        "exit_code": code,
        "trigger": args.trigger,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "settings": {"symbols": args.symbols.split(","), "timeframe": args.timeframe, "days": args.days,
                     "market": args.market},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return code


def _run(args) -> int:
    import scripts.backtest as backtest
    import scripts.train_model as train_model

    common = ["--symbols", args.symbols, "--timeframe", args.timeframe, "--days", str(args.days)]
    if args.market:
        common.append("--market")

    print("STEP 1/2: walk-forward backtest on real data (this can take several minutes)...", flush=True)
    code = backtest.main([*common, "--mode", "walkforward", "--folds", str(args.folds), "--promote",
                          "--out", args.out])
    if code != 0:
        print("RESULT: the backtest could not run (see messages above).", flush=True)
        return 1
    try:
        report = json.loads(report_path().read_text())
    except (OSError, ValueError):
        report = {}
    if not report.get("passed"):
        print("RESULT: NO EDGE FOUND. The bot will not open positions with this setup. Try other liquid "
              "symbols or more history; don't keep tweaking until something passes.", flush=True)
        for failure in report.get("failures") or []:
            print(f"  - {failure}", flush=True)
        return 2

    print("STEP 2/2: edge confirmed; training the model the bot will use...", flush=True)
    code = train_model.main(common)
    if code != 0:
        print("RESULT: training failed (see messages above).", flush=True)
        return 1
    print("RESULT: VALIDATED AND TRAINED. Start the bot on the Paper Trading tab first.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
