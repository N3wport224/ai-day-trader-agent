#!/usr/bin/env python3
"""
Paper self-test from the terminal (same as the Paper tab's "Test my setup").

    python scripts/selftest.py            # keys, quote, order stream, test order (no trades)
    python scripts/selftest.py --full     # also buys and sells 1 share in the paper account

Order steps need market hours (9:30 to 4:00 ET). Paper account only.
Exit code 0 when nothing failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from core.alpaca_executor import AlpacaExecutor  # noqa: E402
from core.bot_manager import mode_env  # noqa: E402
from core.execution_telemetry import EventLog  # noqa: E402
from core.selftest import run_selftest  # noqa: E402
from core.stream_listener import TradeUpdateStream, stream_enabled  # noqa: E402

ICONS = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "info": "  - "}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the paper trading setup end to end")
    parser.add_argument("--full", action="store_true", help="Also buy and sell 1 share (paper money)")
    parser.add_argument("--symbol", default="SPY")
    args = parser.parse_args(argv)
    try:
        executor = AlpacaExecutor(mode="paper", telemetry=EventLog(mode_env("paper")["EXECUTION_LOG_PATH"]),
                                  price_lookup=lambda s: 0.0)
    except ValueError as exc:
        print(f"FAIL {exc}")
        return 1
    factory = (lambda ex: TradeUpdateStream.for_executor(ex, telemetry=ex.telemetry)) if stream_enabled() else None
    print(f"Running the paper self-test on {args.symbol.upper()}{' (full: 1-share round trip)' if args.full else ''}...")
    result = run_selftest(executor, symbol=args.symbol, full=args.full, stream_factory=factory)
    for step in result["steps"]:
        print(f"[{ICONS.get(step['status'], step['status'])}] {step['label']}: {step['detail']}")
        if step.get("fix") and step["status"] != "ok":
            print(f"        -> {step['fix']}")
    out = PROJECT_ROOT / "logs" / "selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nOverall: {result['overall'].upper()}")
    return 1 if result["overall"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
