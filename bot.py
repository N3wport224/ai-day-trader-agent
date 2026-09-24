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

  # Use the original rule-based pipeline instead of the ML strategy
  python bot.py --strategy classic

  # Intraday day trading (defaults): 5-minute bars, 15-min opening lockout,
  # no entries after 15:45 ET, everything flattened at 15:50 ET
  python bot.py --timeframe 5m --no-overnight --opening-lockout-minutes 15 --execute

  # Swing mode on hourly bars, positions may be held overnight
  python bot.py --timeframe 1h --overnight
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from dataclasses import replace  # noqa: E402

from core.market_history import bar_length, is_intraday, normalize_timeframe  # noqa: E402
from core.portfolio_manager import PortfolioManager  # noqa: E402
from core.session_clock import SessionClock, SessionConfig  # noqa: E402
from core.trading_bot import STRATEGIES, TradingBot, create_workflow  # noqa: E402


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Day Trader scheduled bot")
    parser.add_argument(
        "--symbols",
        default=os.getenv("WATCHLIST", ""),
        help="Comma-separated tickers (default: WATCHLIST from .env)",
    )
    parser.add_argument("--portfolio", "-p", default=os.getenv("BOT_PORTFOLIO", "default"))
    parser.add_argument(
        "--timeframe",
        default=os.getenv("BOT_TIMEFRAME", "5m"),
        help="Bar size: 1m, 5m (default), 15m, 1h or 1d",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Minutes between scans (default: the bar size for intraday, else BOT_INTERVAL_MINUTES or 15)",
    )
    parser.add_argument(
        "--overnight",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="--no-overnight (default for 1m/5m/15m) flattens everything before the close; "
             "--overnight lets positions carry (default for 1h/1d)",
    )
    parser.add_argument("--opening-lockout-minutes", type=int, default=_env_int("OPENING_LOCKOUT_MINUTES", 15),
                        help="No new entries for this long after 9:30 ET")
    parser.add_argument("--entry-cutoff-minutes", type=int,
                        default=_env_int("ENTRY_CUTOFF_MINUTES_BEFORE_CLOSE", 15),
                        help="No new entries this many minutes before the close")
    parser.add_argument("--flatten-minutes", type=int, default=_env_int("FLATTEN_MINUTES_BEFORE_CLOSE", 10),
                        help="Cancel orders and close all positions this many minutes before the close")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit risk-checked bracket orders to Alpaca paper trading",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=os.getenv("BOT_STRATEGY", "ml"),
        help="ml (technical + news sentiment model, default) or classic (rule-based pipeline)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("bot")

    try:
        timeframe = normalize_timeframe(args.timeframe)
        intraday = is_intraday(timeframe)
        if args.overnight is not None:
            no_overnight = not args.overnight
        elif os.getenv("NO_OVERNIGHT") is not None:
            no_overnight = os.getenv("NO_OVERNIGHT", "").strip().lower() in {"1", "true", "yes", "on"}
        else:
            no_overnight = intraday
        session = SessionConfig(
            opening_lockout_minutes=args.opening_lockout_minutes,
            entry_cutoff_minutes=args.entry_cutoff_minutes,
            flatten_minutes=args.flatten_minutes,
            no_overnight=no_overnight,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.interval is None:
        bar_minutes = int(bar_length(timeframe).total_seconds() // 60)
        interval = bar_minutes if intraday else _env_int("BOT_INTERVAL_MINUTES", 15)
    else:
        interval = args.interval

    portfolio_manager = PortfolioManager(os.getenv("PORTFOLIO_DB_PATH", "data/portfolios.db"))
    if args.execute and not portfolio_manager.get_portfolio(args.portfolio):
        print(
            f"Portfolio '{args.portfolio}' not found. Create it first: python run.py --setup-portfolio",
            file=sys.stderr,
        )
        return 1

    session_clock = SessionClock(session)
    market_clock = broker = trailing = None
    if args.execute or os.getenv("ALPACA_API_KEY"):
        from core.alpaca_executor_provider import get_alpaca_executor
        from core.trailing_stops import TrailingStopManager

        broker = get_alpaca_executor()
        market_clock = broker.get_clock
        # Entries obey the session rules at the executor too (the final gate),
        # and intraday entries are checked against the PDT rule.
        broker.session_clock = session_clock
        broker.risk_manager.limits = replace(broker.risk_manager.limits, day_trading=no_overnight)
        # Reconcile every cycle (report-only in dry run); trail stops only when executing.
        trailing = TrailingStopManager(broker) if args.execute else None

    log.info(
        f"Timeframe {timeframe}, scan every {interval} min; "
        f"{'NO overnight: flatten ' + str(session.flatten_minutes) + ' min before close' if no_overnight else 'overnight holding allowed'}; "
        f"opening lockout {session.opening_lockout_minutes} min, entry cutoff {session.entry_cutoff_minutes} min before close"
    )

    try:
        bot = TradingBot(
            create_workflow(portfolio_manager, args.strategy, timeframe=timeframe),
            args.symbols.split(","),
            args.portfolio,
            execute=args.execute,
            interval_seconds=interval * 60,
            market_clock=market_clock,
            broker=broker,
            trailing_manager=trailing,
            session_clock=session_clock,
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
