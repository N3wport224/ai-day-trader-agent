#!/usr/bin/env python3
"""
Run the trading bot on a schedule.

Examples:
  # Dry run: analyze the watchlist every 15 minutes, place no orders
  python bot.py --symbols AAPL,MSFT,NVDA

  # Trade on Alpaca paper with the risk limits from .env
  python bot.py --symbols AAPL,MSFT,NVDA --portfolio paper --execute

  # One pass and exit (handy for cron)
  python bot.py --once --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from core.portfolio_manager import PortfolioManager  # noqa: E402
from core.trading_bot import TradingBot  # noqa: E402
from core.trading_workflow import TradingWorkflow  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Day Trader scheduled bot")
    parser.add_argument(
        "--symbols",
        default=os.getenv("WATCHLIST", ""),
        help="Comma-separated tickers (default: WATCHLIST from .env)",
    )
    parser.add_argument("--portfolio", "-p", default=os.getenv("BOT_PORTFOLIO", "default"))
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("BOT_INTERVAL_MINUTES", "15")),
        help="Minutes between scans (minimum 1)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit risk-checked bracket orders to Alpaca paper trading",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    portfolio_manager = PortfolioManager(os.getenv("PORTFOLIO_DB_PATH", "data/portfolios.db"))
    if args.execute and not portfolio_manager.get_portfolio(args.portfolio):
        print(
            f"Portfolio '{args.portfolio}' not found. Create it first: python run.py --setup-portfolio",
            file=sys.stderr,
        )
        return 1

    market_clock = None
    if args.execute or os.getenv("ALPACA_API_KEY"):
        from core.alpaca_executor_provider import get_alpaca_executor

        market_clock = get_alpaca_executor().get_clock

    try:
        bot = TradingBot(
            TradingWorkflow(portfolio_manager),
            args.symbols.split(","),
            args.portfolio,
            execute=args.execute,
            interval_seconds=args.interval * 60,
            market_clock=market_clock,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        bot.run(max_cycles=1 if args.once else None)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Bot stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
