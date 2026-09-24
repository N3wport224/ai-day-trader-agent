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

  # REAL MONEY (separate live keys, armed on the dashboard; the edge gate
  # cannot be bypassed in live mode)
  python bot.py --mode live --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

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


def _watch_stop_file(bot, log, interval: float = 2.0) -> None:
    """Stop gracefully when the dashboard writes BOT_STOP_FILE (portable:
    Windows has no SIGTERM, only a hard kill)."""
    path = os.getenv("BOT_STOP_FILE")
    if not path:
        return
    import threading
    from pathlib import Path

    stop_file = Path(path)

    def watch() -> None:
        while not bot.stopped:
            if stop_file.exists():
                try:
                    stop_file.unlink()
                except OSError:
                    pass
                log.warning("Stop requested from the dashboard: finishing the current cycle, then stopping")
                bot.stop("dashboard")
                return
            time.sleep(interval)

    threading.Thread(target=watch, name="stop-file-watcher", daemon=True).start()


def apply_mode_paths(mode: str) -> None:
    """Keep each mode's runtime state (heartbeat, telemetry, stops, fills,
    edge monitor) separate, unless explicitly configured."""
    defaults = {
        "HEARTBEAT_PATH": f"logs/{mode}/heartbeat.json",
        "EXECUTION_LOG_PATH": f"logs/{mode}/execution_events.jsonl",
        "TRAILING_STATE_PATH": f"data/{mode}/trailing_state.json",
        "FILL_STATE_PATH": f"data/{mode}/fill_state.json",
        "EDGE_MONITOR_STATE_PATH": f"data/{mode}/edge_monitor.json",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def _edge_gate(bot, strategy_name: str, timeframe: str, log, mode: str = "paper") -> tuple:
    """(block reason or None, warnings) for opening live positions."""
    if mode != "live" and os.getenv("EDGE_GATE", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        log.warning("EDGE_GATE=false: trading WITHOUT a validated out-of-sample edge")
        return None, []
    from core.edge_gate import check_live_setup, strategy_fingerprint

    engine = getattr(bot.workflow, "analysis_runner", None)
    strategy = getattr(engine, "strategy", None)
    if strategy_name != "ml" or strategy is None:
        return "the classic strategy has no walk-forward validation; use --strategy ml", []
    return check_live_setup(strategy_fingerprint(strategy, engine.timeframe), bot.symbols)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Day Trader scheduled bot")
    parser.add_argument(
        "--symbols",
        default=os.getenv("WATCHLIST", ""),
        help="Comma-separated tickers (default: WATCHLIST from .env)",
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "live"),
        default=os.getenv("BOT_MODE", "paper"),
        help="paper (default, fake money) or live (REAL money; needs live keys and arming)",
    )
    parser.add_argument("--portfolio", "-p", default=None,
                        help="Local book for bot trades (default: BOT_PORTFOLIO, or 'live' in live mode)")
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
    parser.add_argument("--reset-edge-monitor", action="store_true",
                        help="Clear an edge-decay pause (after you've reviewed it) and start a fresh live window")
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
    apply_mode_paths(args.mode)
    if args.portfolio is None:
        args.portfolio = "live" if args.mode == "live" else os.getenv("BOT_PORTFOLIO", "default")
    if args.mode == "live":
        log.warning("LIVE MODE: this bot trades REAL MONEY" if args.execute
                    else "LIVE MODE (watch only): reading the live account, no orders")

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
    market_clock = broker = trailing = fills = edge_monitor = None
    live_keys = os.getenv("ALPACA_LIVE_API_KEY") if args.mode == "live" else os.getenv("ALPACA_API_KEY")
    if args.execute or live_keys:
        from core.alpaca_executor_provider import get_executor_for_mode
        from core.trailing_stops import TrailingStopManager

        try:
            broker = get_executor_for_mode(args.mode)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        market_clock = broker.get_clock
        # Entries obey the session rules at the executor too (the final gate),
        # and intraday entries are checked against the PDT rule.
        broker.session_clock = session_clock
        broker.risk_manager.limits = replace(broker.risk_manager.limits, day_trading=no_overnight)
        # Reconcile every cycle (report-only in dry run); trail stops only when executing.
        trailing = TrailingStopManager(broker) if args.execute else None
        if args.execute:
            from core.fill_quality import FillTracker

            from core.edge_monitor import EdgeMonitor

            fills = FillTracker(broker.expected_prices)
            edge_monitor = EdgeMonitor.from_edge_report()
            if args.reset_edge_monitor:
                edge_monitor.reset()
                log.warning("Edge monitor reset: live performance window cleared")
            elif edge_monitor.paused:
                log.error(f"Edge monitor is PAUSED from a previous run: {edge_monitor.paused}. New entries stay "
                          "blocked; review, then restart with --reset-edge-monitor.")

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
            timeframe=timeframe,
            fill_tracker=fills,
            edge_monitor=edge_monitor,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.execute:
        block, warnings = _edge_gate(bot, args.strategy, timeframe, log, mode=args.mode)
        for warning in warnings:
            log.warning(f"Edge gate: {warning}")
        if block:
            bot.standing_entry_block = f"edge gate: {block}"
            log.error(f"EDGE GATE: new entries are BLOCKED for this run: {block}. Exits, stops and the EOD "
                      "flatten still run. See core/edge_gate.py (EDGE_GATE=false overrides at your own risk).")
            bot.telemetry.record("edge_gate", logging.WARNING, passed=False, reason=block)
        elif os.getenv("EDGE_GATE", "true").strip().lower() in {"1", "true", "yes", "on"}:
            log.info("Edge gate: passed (validated walk-forward report matches this setup)")

    def request_stop(signum, frame):
        if bot.stopped:  # second signal: stop waiting
            raise KeyboardInterrupt
        log.warning(f"{signal.Signals(signum).name} received: finishing the current cycle, then stopping "
                    "(send again to force)")
        bot.stop(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _watch_stop_file(bot, log)
    if not args.once:
        from core.keep_awake import keep_awake

        if keep_awake():
            log.info("Keeping this computer awake while the bot runs (KEEP_AWAKE=false to disable)")

    try:
        bot.run(max_cycles=1 if args.once else None)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Bot stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
