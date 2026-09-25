# AI Day Trader Agent

A sophisticated, multi-strategy AI-powered trading agent that combines technical analysis, sentiment analysis, and dividend capture strategies to generate intelligent trade recommendations. Features enhanced signal fusion, comprehensive risk management, and professional-grade architecture.

---

## Features

### 🚀 **Enhanced Multi-Strategy Analysis**
- **Technical Analysis**: RSI, MACD, SMA/EMA with intelligent signal fusion
- **Sentiment Analysis**: Real-time news sentiment using OpenAI GPT-4.1
- **Dividend Capture Strategy**: Advanced dividend timing and capture optimization
- **Signal Fusion**: Intelligent combination of all strategies with priority-based decision making

### 📊 **Advanced Market Data**
- Multi-timeframe candlestick data (1m, 15m, 1h) via **both Twelve Data and Alpha Vantage APIs**
- **500+ data points** for robust technical indicator calculations
- Dual API fallback system for maximum reliability
- Real-time price tracking with moving average comparisons

### 🤖 **AI-Powered Intelligence**
- Discord bot interface for real-time trade analysis
- Enhanced analysis output with detailed technical indicators
- Comprehensive risk management with stop-loss and take-profit calculations
- Position sizing based on volatility and confidence levels

### 🏗️ **Professional Architecture**
- Modular, testable, and maintainable codebase
- Proper Python import structure and module organization
- Comprehensive error handling and logging
- Production-ready configuration management

### 🔄 **Intelligent Rate Limiting**
- Automatic detection and handling of API rate limits
- Exponential backoff with jitter for optimal retry timing
- Request tracking to prevent hitting limits proactively
- Configurable retry attempts and wait times
- Seamless failover between data sources when rate limited

---

## Quick start (dashboard, no command line needed after this)

```bash
pip install -r requirements.txt
python start_dashboard.py
```

Your browser opens the dashboard at http://127.0.0.1:8000/dashboard. From there:

1. **Create your account.** The first visit asks you to create the owner account.
2. **API Keys tab:** paste your Alpaca **paper** keys. It has step-by-step instructions
   for getting them, and a **Test connection** button. Keys are saved to `.env` on this
   computer only, with owner-only permissions, and are never shown again. For safety they
   can only be changed from the computer running the dashboard.
3. **Get Started tab → Validate strategy:** a walk-forward test on real data. The bot
   only places trades if this test finds an edge; if it does, the model is trained
   automatically.
4. **Paper Trading tab** (green, fake money): start or stop the auto-trader, watch it in
   one of two modes (placing paper orders, or watch-only), see positions, activity and
   the day's P&L, place manual paper orders, and use the emergency **Close everything**
   button.
5. **Live Trading tab** (red, real money), optional and only after weeks of paper
   trading. It requires all of these:
   - separate live keys;
   - **arming**: type `I UNDERSTAND THIS USES REAL MONEY` plus your password;
   - a passing validation;
   - a confirmation tick box every time you start the bot.

   **Disarm** stops the live bot instantly. Real-money trading can't be switched on from
   another machine, and the edge gate can't be bypassed in live mode.

The bot runs as its own background process (`bot.py --mode paper|live`), with separate
logs and state per mode under `logs/<mode>/` and `data/<mode>/`. It keeps running if you
close the browser. Stop it from the dashboard.

**Updating.** Open **System check → Software updates**; the sidebar shows **NEW** when
there's a newer version on GitHub.
1. Click **Update now**, then **Restart dashboard**.
2. The update never touches your `.env` keys, `data`, `logs`, `models` or `reports`.
3. Files it replaces are backed up in `backups/`, and **Undo last update** puts them back.
4. Packages are reinstalled only when `requirements.txt` changed.
5. For safety it won't run while a bot or validation is running; stop them first.

Git checkouts update with `git pull --ff-only`. It refuses if you've edited tracked files.

**Tracking results.** The Paper and Live tabs each have a **Performance** card with:
- the account's equity curve (1 day to 1 year);
- the bot's closed trades, with win rate, average R and profit factor shown next to what
  the validation test predicted;
- a plain-language verdict: *on track*, *keep watching*, *behind the test* or *too early
  to tell* (it needs about 20 trades before judging);
- a trade journal with CSV download.

**Staying validated.** The validation expires after 30 days. Once it's 7 days old, the
dashboard re-runs it automatically, with the settings you last used, outside market hours
(weekends, or before 8 am / after 6 pm ET).
- If the strategy still passes, the model is retrained and running bots restart to load
  it (only outside market hours).
- If it no longer passes, bots stop opening trades until it does. Running bots re-check
  the validation every few minutes, so this takes effect mid-run.
- You get an alert either way.
- Turn it off with the checkbox on Get Started (`AUTO_REVALIDATE=false`).

**System check** (sidebar) checks, in one click:
- your keys, the market-data feed and your computer's clock (vs Alpaca);
- the validation and the trained model;
- disk space and the ML library;
- whether bots that should be running are, plus keep-awake, start-at-login and alerts.

Each problem comes with the fix. Run it before your first session.

**Get Started → See the test details** shows every walk-forward test period plus
breakdowns by time of day and market regime. Look for an edge that holds up across
periods, not one lucky stretch.

**Running unattended.** The dashboard remembers which bots you started. If a bot
crashes, or the computer restarts, it is started again with the same settings:
- at most 3 times an hour;
- only if the same safety checks pass (for live: keys, arming, validation);
- never for a bot you stopped yourself.

Turn on **API Keys → Run automatically → Start the dashboard when I log in** and it
comes back by itself after a reboot. On Windows this adds a Startup-folder item; on a
Mac, a login LaunchAgent. Set `AUTO_RESUME_BOTS=false` to turn auto-restart off.

### Running it on your own computer (Windows or Mac)

**One-time setup**
1. **Install Python 3.12** from https://www.python.org/downloads/.
   - Windows: on the first installer screen, tick **"Add python.exe to PATH"**.
   - Mac: use the python.org installer. The built-in `python3` is often too old.
2. **Mac only:** install Homebrew (https://brew.sh), then run `brew install libomp` in
   Terminal. The ML library needs it.
3. **Get the code:** on GitHub, pick the branch and choose **Code → Download ZIP**, then
   unzip it somewhere permanent such as Documents. Alternatively, clone it with GitHub
   Desktop or git.

**Every time**
- Windows: double-click **`start_dashboard.bat`**.
- Mac: double-click **`start_dashboard.command`**. The first time, macOS may say it's
  from an unidentified developer: right-click it, choose **Open**, then **Open** again.

The first launch spends a few minutes installing packages into a private `.venv` folder.
After that the dashboard opens in your browser at http://127.0.0.1:8000/dashboard. Keep
the launcher window open while you trade.

**Keep it running during market hours (9:30 am to 4:00 pm ET)**
- While the bot runs it stops the computer from idle-sleeping, on both Windows and Mac.
  Closing a laptop lid, shutting down, or losing Wi-Fi still stops it, so plug a laptop
  in and leave it open.
- If the computer goes to sleep or offline mid-day, broker-side stop-loss and
  take-profit orders still protect your positions. But the 3:50 pm "close everything"
  won't happen, so day trades would stay open overnight until the bot runs again.
- Windows updates can restart your PC. Set **active hours** to cover the trading day
  (Settings → Windows Update → Advanced options).

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ap6pack/ai-day-trader-agent.git
cd ai-day-trader-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with the following keys:

```
# Primary API Keys
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2
OPENAI_API_KEY=your_openai_key

# Optional notifications
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_GUILD_ID=your_discord_guild_id
DISCORD_CHANNEL_ID=your_discord_channel_id

# Optional market/news fallbacks
MARKET_DATA_PROVIDERS=alpaca,yahoo_finance
ALPACA_DATA_FEED=iex
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
TWELVE_DATA_API_KEY=your_12data_api_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
NEWS_API_KEY=your_newsapi_key

# Optional Trading Configuration
TRADING_CAPITAL=5000.0                    # Your trading capital in dollars
MIN_POSITION_PERCENTAGE=0.02              # Minimum 2% of capital per trade
MAX_POSITION_PERCENTAGE=0.10              # Maximum 10% of capital per trade

# Optional API Tier Configuration
TWELVE_DATA_PREMIUM=auto                  # auto (detect), true (premium), false (free)

# API Rate Limiting Configuration
ALPACA_RATE_LIMIT_WAIT=60                # Seconds to wait when rate limited
ALPACA_MAX_RETRIES=3                     # Max retry attempts
ALPACA_CALLS_PER_MINUTE=180              # Conservative default below Alpaca Basic historical limit

TWELVE_DATA_RATE_LIMIT_WAIT=60           # Seconds to wait when rate limited
TWELVE_DATA_MAX_RETRIES=3                # Max retry attempts
TWELVE_DATA_CALLS_PER_MINUTE=8           # Your plan's limit

ALPHA_VANTAGE_RATE_LIMIT_WAIT=60         # Seconds to wait when rate limited
ALPHA_VANTAGE_MAX_RETRIES=3              # Max retry attempts
ALPHA_VANTAGE_CALLS_PER_MINUTE=5         # Your plan's limit

# Advanced Rate Limiting
API_BACKOFF_FACTOR=2.0                   # Exponential backoff multiplier
API_MAX_BACKOFF_SECONDS=300              # Maximum wait time (5 minutes)
API_JITTER_ENABLED=true                  # Add randomness to prevent thundering herd
```

**Never commit your `.env` file to version control.**  
The `.gitignore` is configured to exclude `.env`, logs, and other sensitive or unnecessary files.

### 4. Trading Capital Configuration

The system uses **portfolio-based position sizing** that adapts to your actual trading capital:

- **Default**: $5,000 trading capital
- **Flexible**: Works whether you own 0, 5, 100, or 4,933 shares of any stock
- **Risk-Managed**: Position sizes scale with signal confidence and market volatility
- **Configurable**: Adjust via environment variables or config files

---

## Usage

### Portfolio Management

#### Setup a Portfolio
```bash
python run.py --setup-portfolio
```
Interactive wizard will guide you through creating a portfolio with trading capital and initial holdings.

#### Portfolio Commands
```bash
# Show portfolio details
python run.py --show-portfolio

# List all portfolios
python run.py --list-portfolios

# Update trading capital
python run.py --update-capital 10000

# Add/update holdings
python run.py --add-holding AAPL 100 --cost 150.00

# Remove holdings
python run.py --remove-holding AAPL

# Record manual trades
python run.py --record-trade AAPL BUY 50 155.00 --strategy technical --confidence 0.75

# Show trade history
python run.py --show-trades --days 30

# Backup database
python run.py --backup --output backups/portfolio_backup.db

# Restore from backup
python run.py --restore backups/portfolio_backup.db

# Analyze entire portfolio
python run.py --analyze-portfolio

# Analyze specific portfolio
python run.py --analyze-portfolio --portfolio my_portfolio
```

### Scheduled Trading Bot

`bot.py` scans a watchlist while the market is open. It is a dry run by
default: it analyzes and logs, and places no orders.

```bash
# Dry run every 15 minutes
python bot.py --symbols AAPL,MSFT,NVDA

# Trade on Alpaca paper with risk limits from .env
python bot.py --symbols AAPL,MSFT,NVDA --portfolio default --execute

# One cycle and exit (for cron)
python bot.py --once --execute
```

#### ML strategy (default for the bot)

The bot scores each symbol with a model that combines candlestick/trend
technicals with news sentiment, then hands actionable signals to the same
executor and risk manager as everything else:

```
bars (Alpaca/Yahoo) -> core/features.py ─┐
news (Alpaca News)  -> core/news_sentiment.py ─┤-> core/ml_strategy.py -> core/ml_signal_engine.py
                                                         -> TradingWorkflow -> RiskManager -> bracket order
```

- **Features** (`core/features.py`): candle body ratio `(C-O)/(H-L)`, upper and
  lower wick ratios, gap %, EMA 20/50/200 spreads and slope, RSI, MACD, ATR,
  trailing returns and volume z-score. All causal: a row only uses bars up
  to that candle's close, and the still-forming bar is dropped.
- **Sentiment** (`core/news_sentiment.py`): Alpaca News headlines scored with
  FinBERT (`pip install -r requirements-ml.txt`) or a built-in lexicon
  fallback, aggregated into a 24-hour time-decayed score in [-1, +1].
  If news is unavailable, the model gets "missing" and keeps working on
  technicals alone.
- **Model** (`core/ml_strategy.py`): LightGBM in a scikit-learn pipeline
  predicts P(ATR take-profit is hit before the ATR stop). BUY at or above
  `ML_CONFIDENCE_THRESHOLD`; SELL (exit a held position) only when P is at
  or below half the training base rate (`ML_EXIT_THRESHOLD`), since target
  hits are rare by design and a low P usually means "no edge", not
  "bearish"; otherwise HOLD. With no trained model it falls back to a
  trend/momentum/sentiment heuristic (`mode=heuristic` in the logs).
- **Sizing and stops**: risk `RISK_PER_TRADE_PCT` of capital between entry
  and a `ATR_STOP_MULT` x ATR stop; target at `ATR_TARGET_MULT` x ATR. Both
  are re-centred on the live price at order time and then subject to every
  risk check below.

Train a model (writes `models/ml_signal.joblib`):

```bash
python scripts/train_model.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days 365
python scripts/train_model.py --no-news --timeframe 1Day --days 1500   # technicals only
python scripts/train_model.py --synthetic                              # offline demo
```

The script labels each bar by whether price hit the take-profit before the
stop within `--horizon` bars, holds out the most recent 20% (with a purge
gap so no training label sees test prices), and prints AUC, base rate, and
win rate / approximate expectancy at your threshold. **If AUC is near 0.5 or
expectancy is negative, the model has no edge: keep the bot in dry-run.**
Retrain after changing `ATR_*_MULT`, the timeframe, or the feature code.
Use `--strategy classic` to run the original rule-based pipeline.

#### Regime filter, multi-timeframe confirmation, sizing and trailing stops

- **Regime filter** (`core/regime.py`): each bar is `TRENDING_BULL`,
  `TRENDING_BEAR`, `CHOPPY` or `UNKNOWN` from ADX/+DI/-DI plus the ATR%
  percentile (ADX >= `ADX_TREND_THRESHOLD`, or 5 below it when volatility is
  expanding). `REGIME_FILTER=suppress` allows longs only in `TRENDING_BULL`;
  `penalty` (default) requires `ML_CONFIDENCE_THRESHOLD + REGIME_THRESHOLD_BUMP`
  elsewhere. Exits are never blocked.
- **Multi-timeframe** (`features.add_macro_features`): each intraday bar gets
  the daily EMA20/EMA50 trend from the last *completed* daily candle (matched
  as of the bar's close), so no daily close leaks into earlier hours.
  `MTF_CONFIRMATION=true` only enters longs aligned with the daily trend;
  `scripts/train_model.py --mtf` also trains the model on these features.
- **Sizing** (`POSITION_SIZING_METHOD`): `fixed` risk per trade;
  `volatility` scales it by median/current ATR% (0.5x-1.5x); `kelly` uses
  fractional Kelly on the model's probability (capped at the fixed risk, zero
  when the model sees no edge), then volatility-scales it.
- **Trailing stops**: once a position is `TRAILING_STOP_TRIGGER_R` in profit
  the bracket stop moves to breakeven, then trails `TRAILING_STOP_DISTANCE_R`
  below the high. The backtester and the live `TrailingStopManager` use the
  same rule; live stops are replaced at Alpaca each cycle (`--execute` only).
- **Reconciliation and recovery**: each cycle starts by reading Alpaca
  positions and open orders. Mismatches with the local book, positions with
  no working stop, and shorts are logged (and holdings synced with
  `RECONCILE_MODE=sync`). If broker state can't be read, the cycle is
  skipped. Order rejections are classified (buying power, margin/PDT, wash
  trade, shares held by bracket legs, invalid prices, ...) with a root cause
  and remediation, written to `logs/execution_events.jsonl`, and retried once
  when there is a safe fix (smaller size, fresh-quote levels, cancel legs
  before an exit).

#### Intraday day-trading mode (1m / 5m bars)

`python bot.py` now defaults to 5-minute bars in day-trading mode:

```bash
python bot.py --timeframe 5m --no-overnight --opening-lockout-minutes 15            # dry run
python bot.py --timeframe 5m --execute                                              # paper orders
python bot.py --timeframe 1h --overnight                                            # swing mode
```

- **Session clock** (`core/session_clock.py`, US/Eastern): opening lockout
  9:30-9:45 (no entries while the opening range forms), entries allowed until
  the cutoff 15 min before the close, then from 10 min before the close the
  bot cancels every working order (bracket legs included) and closes every
  position (`FLATTEN_ORDER_TYPE=market` or a marketable limit). Cutoff and
  flatten are relative to the actual close from Alpaca's clock, so 13:00
  half-days work. The bot wakes up exactly at flatten time; if a close fails
  it retries every cycle until the market closes. The executor enforces the
  same entry windows, so manual orders can't bypass them.
- **Intraday features** (`features.add_intraday_features`): session VWAP
  anchored at 9:30 with 1/2-sigma bands, VWAP distance and z-score, relative
  volume vs the same time-of-day bucket over prior sessions, 15-minute
  opening-range high/low (hidden until the range has formed) and minutes
  since the open. All causal and tested for lookahead on 1m and 5m bars.
  Models trained with `--timeframe 5m` use them automatically, and their
  labels end at the session close (a day trader is flat overnight).
  A model trained on a different bar size is ignored with an error.
- **PDT gate**: in day-trading mode, with equity under $25,000, entries are
  rejected when the account is flagged as a pattern day trader or already
  has 3 day trades in the last 5 business days (`PDT_DAYTRADE_BUFFER` stops
  earlier). Exits are never blocked by it (Alpaca's own PDT protection may
  still reject a same-day close for a flagged account; the flatten failure
  is logged and retried).
- **Intraday drawdown breaker**: once equity is down
  `MAX_INTRADAY_DRAWDOWN_PCT` (2%) from the start of the day (realized +
  unrealized), new entries stop for the rest of the session even if equity
  recovers. It is checked every cycle, not only when a buy is attempted.
- **Data**: 1Min and 5Min bars from Alpaca (paginated) with Yahoo fallback
  (Yahoo keeps ~7 days of 1m and ~60 days of 5m history). Default lookback is
  30 days for 1m and 45 for 5m so EMA200 and RVOL are warmed up. The live bot
  caches history per symbol and fetches only the newest bars each cycle
  (re-fetching the last 3 so revised bars are replaced), with a full reload
  each market day and every 6 hours (`BAR_CACHE=false` turns it off).
- **Stale-data guard**: during regular hours, if the newest completed bar is
  more than `MAX_BAR_AGE_BARS` (3) bars old, e.g. during a feed outage or
  when falling back to delayed data, the symbol gets no signal that cycle.
  Open positions stay protected by their broker-side brackets and the EOD
  flatten.
- **Re-entry cooldown**: after a stop-out, the risk manager rejects a new
  entry in that symbol for `REENTRY_COOLDOWN_MINUTES` (30), so a stop doesn't
  turn straight into a revenge trade. Live, it reads the stop leg's fill time
  from Alpaca's orders (restart-safe); the backtester applies the same rule
  (`--reentry-cooldown-minutes`). `REENTRY_COOLDOWN_STOPS_ONLY=false` applies
  it after every exit.

Backtest the intraday rules (opening lockout, cutoff, 15:50 forced close,
2 bps spread on top of slippage, simulated PDT day-trade count):

```bash
python scripts/backtest.py --timeframe 5m --days 60 --no-overnight --opening-lockout-minutes 15 --compare
```

#### Running unattended: alerts, heartbeat, session report, shutdown

- **Alerts** (`core/alerts.py`): set `ALERT_WEBHOOK_URL` to a Discord or Slack
  incoming webhook to get one-line messages for order submissions, broker
  rejections (with root cause), recoveries, the EOD flatten result (including
  any position that failed to close), drawdown-breaker trips, broker/local
  mismatches, failed trailing-stop updates, the end-of-session report and bot
  start/stop/errors. Sending is asynchronous and throttled, and a broken
  webhook never affects trading. Every event is also in
  `logs/execution_events.jsonl`.
- **Session report**: on the first cycle after the close, the bot records the
  day's P&L (equity vs the previous close), fills and open positions, and
  flags `NOT FLAT` if anything is still open in no-overnight mode.
- **Heartbeat**: `logs/heartbeat.json` is rewritten atomically every cycle
  (phase, positions, errors, entry blocks, seconds to the next cycle). Point
  your monitoring at its modification time; if it goes stale, the bot is down.
- **Fill quality** (`core/fill_quality.py`, execute mode): every fill is
  compared with the price the bot expected (the live quote for market
  entries/exits, the stop price for stop legs, the limit for take-profits)
  and logged as `order_filled` with signed slippage in bps (positive = worse
  for you). Fills worse than `SLIPPAGE_ALERT_BPS` (25) alert, and the session
  report includes mean, notional-weighted and worst slippage plus its dollar
  cost. Compare it with the backtest's `--slippage-bps`/`--spread-bps`: if
  live is consistently worse, the backtest is flattering the strategy.
- **Shutdown**: SIGTERM or Ctrl-C finishes the current cycle and exits
  cleanly (it interrupts the wait between cycles, never an order in flight);
  a second signal forces the exit. A cycle that throws is logged and alerted
  as `bot_error` and the bot keeps running.

Example systemd unit (paper trading, restarts on crash):

```ini
[Service]
WorkingDirectory=/opt/ai-day-trader-agent
ExecStart=/opt/ai-day-trader-agent/venv/bin/python bot.py --timeframe 5m --execute
Restart=on-failure
KillSignal=SIGTERM
TimeoutStopSec=120
```

#### Tactics for a better chance of profit (and fewer ways to lose)

No setting makes a trading bot profitable, and most automated intraday
strategies lose money after costs. These tactics are about only trading
when there's evidence of an edge, cutting costs, and limiting the damage
when there isn't one.

1. **Edge gate: prove it before trading it** (`core/edge_gate.py`). With
   `--execute`, the bot opens **no new positions** unless a walk-forward
   backtest of exactly this setup passed on real data within the last 30
   days. "Exactly this setup" means the same timeframe, ATR stop/target,
   feature set, threshold and entry gates. The minimums: ≥100 trades, ≥3
   folds, profit factor ≥1.2, avg R ≥0.05, ≥60% of folds positive and
   drawdown ≤15%. They're configurable with `EDGE_*`. Exits, stops and the
   EOD flatten always run. `EDGE_GATE=false` overrides it, at your own risk.
   ```bash
   python scripts/backtest.py --timeframe 5m --days 120 --folds 4 --market --promote
   python scripts/train_model.py --timeframe 5m --days 120 --market   # same flags as the promoted run
   python bot.py --timeframe 5m --execute
   ```
   Don't try dozens of settings until one passes. The winner of many tries is
   usually luck, so keep the variants you test few.
2. **Edge-decay monitor** (`core/edge_monitor.py`). Live fills are paired
   into round trips and measured in R. Over the last 30 trades (evaluated
   from 20 on), entries pause if the live profit factor falls below 0.8, or
   if the live mean R is statistically below the backtest's (t < −2). The
   pause persists across restarts until you review it and run
   `bot.py --reset-edge-monitor`.
3. **Market context** (`core/market_context.py`).
   - `--market` trains with relative strength vs SPY over 12 and 48 bars,
     plus SPY's trend and VWAP distance.
   - `--market-filter` (`MARKET_FILTER=true`) pauses longs while SPY is below
     both its EMA50 and its session VWAP. Longs in a falling tape start with
     a headwind.
   - Test both with `--compare` before relying on them.
4. **Lower execution costs**.
   - Entries are marketable limits 10 bps through the ask
     (`ENTRY_ORDER_TYPE`, `ENTRY_LIMIT_OFFSET_BPS`). They cap what a gap or a
     thin book can cost, and the backtester simulates the missed fills.
   - Entries are skipped when the bid/ask spread is wider than 20 bps
     (`MAX_SPREAD_BPS`).
   - Unfilled entries are cancelled after 120 s (`ENTRY_ORDER_TTL_SECONDS`).
   - Slippage is measured on every fill, so you can check the backtest's
     cost assumptions (see Running unattended).
   - With the free IEX feed, quotes are IEX-only and often wider than the
     national best bid/offer; `ALPACA_DATA_FEED=sip` is better if your plan
     includes it.
5. **Portfolio heat and position caps**. Total risk to the stops across all
   open positions is capped at 4% of equity (`MAX_PORTFOLIO_HEAT_PCT`), and
   at most 5 positions can be open at once (`MAX_OPEN_POSITIONS`). This
   matters because correlated positions all stop out together on a market
   drop. Both apply live and in backtests (`--max-heat-pct`,
   `--max-open-positions`).
6. **Already in place**: regime filter, daily-trend confirmation,
   volatility/Kelly sizing, trailing stops, the re-entry cooldown after
   stop-outs, the intraday drawdown breaker, the PDT gate and no overnight
   risk in day-trading mode.

A reasonable path to live trading:
1. Run a walk-forward with `--compare` on 6–12 months of data for 5–10
   liquid symbols.
2. Promote only a setup that passes.
3. Paper trade it for several weeks.
4. Compare live R and slippage with the backtest.
5. Only then consider real money, starting small.

#### Backtesting

`scripts/backtest.py` replays the whole path on history using the live code:
the same features, `MLStrategy` decisions, risk-per-trade sizing and
`RiskManager` checks (with a simulated account), with realistic execution:

- Decide at a bar's close; fill at the **next** bar's open plus slippage.
- Brackets at the signal's ATR distances around the fill price; a gap through
  a level fills at the open, and a bar touching both counts as the stop.
- Intraday: decisions only while the market is open; brackets only trigger in
  bars overlapping the regular session. Queued orders reserve buying power.

```bash
# Walk-forward (default): train on the first 60%, trade only the unseen 40%
python scripts/backtest.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days 730

# Add news sentiment, write trades.csv / equity.csv / summary.json
python scripts/backtest.py --news --out reports/bt

# A saved model (warns if the test period overlaps its training data)
python scripts/backtest.py --mode model

# The no-model heuristic, or an offline demo
python scripts/backtest.py --mode heuristic --timeframe 1Day --days 1500
python scripts/backtest.py --synthetic
```

Benchmark the new layers against the old behaviour on identical data with
`--compare` (baseline = regime off, no MTF, fixed sizing, no trailing):

```bash
python scripts/backtest.py --regime suppress --mtf --mtf-gate --sizing kelly --trailing --compare --out reports/cmp
```

Use `--folds N` for a rolling walk-forward: the unseen period is split into
N consecutive blocks and a fresh model is trained before each one on all
earlier data. The report shows each fold, pooled out-of-sample stats and how
many folds had positive average R; an edge that shows up in only one period
is flagged. Every report also breaks trades down **by entry regime** and
(intraday) **by entry hour**, so you can see whether losses cluster in
CHOPPY/bear regimes or at the open (`by_regime.csv`, `by_entry_hour.csv`).

```bash
python scripts/backtest.py --timeframe 5m --days 120 --folds 4 --compare --out reports/wf
```

The report shows return vs. equal-weight buy-and-hold, max drawdown, daily
Sharpe, trade count, win rate, average R, profit factor, exit reasons, how
often each risk limit blocked an entry, and how many BUY-level signals fell
outside market hours. Walk-forward runs also print a **threshold audit** for
the unseen period (hit rate and approximate expectancy at each
`ML_CONFIDENCE_THRESHOLD`, plus a calibration table), saved as
`threshold_sweep.csv` / `calibration.csv` with `--out`. Treat fewer than ~30 trades as
inconclusive, and only use `--execute` if the walk-forward result beats
buy-and-hold with positive average R after slippage.

Every order, whether from the bot, the CLI (`--paper-trade`), the API or
the dashboard, goes through the same safeguards:

- **Bracket orders**: each BUY is sent with a broker-side stop-loss and
  take-profit, so the position is protected even if the bot stops running.
  The strategy's ATR-based levels are used when they fit the live price,
  otherwise `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`.
- **Risk checks** (`core/risk_manager.py`): kill switch (`TRADING_ENABLED`),
  daily loss limit (`MAX_DAILY_LOSS_PCT`), daily entry limit
  (`MAX_DAILY_TRADES`), per-symbol position cap (`MAX_PORTFOLIO_ALLOCATION`
  of equity), buying power, and `MIN_PRICE`. Orders are shrunk to fit, never
  enlarged. Exits are allowed even after the loss limit trips.
- **No orders while the market is closed**, and sells never exceed the shares
  held (no accidental shorts).

### REST API Server

The AI Day Trader Agent now includes a professional REST API with WebSocket support for real-time updates.

#### API Features
- **JWT Authentication**: Secure token-based authentication
- **Portfolio Management**: Full CRUD operations via REST endpoints
- **Real-time Updates**: WebSocket support for live portfolio updates
- **Trading Analysis**: Run analysis on symbols and portfolios
- **Interactive Documentation**: Available at `http://localhost:8000/docs`

**📚 For complete API documentation, examples, and WebSocket usage, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

### Trade Analysis

#### With Portfolio Context
```bash
# Analyze using default portfolio
python run.py AAPL

# Analyze with specific portfolio
python run.py AAPL --portfolio my_portfolio

# Submit an actionable recommendation to Alpaca paper trading
python run.py AAPL --portfolio my_portfolio --paper-trade

# Record an actionable recommendation locally without submitting an Alpaca order
python run.py AAPL --portfolio my_portfolio --record-paper-trade
```

### Discord Bot

Start the bot:

```bash
python core/discord_bot.py
```

In your Discord server, use:

```
!trade <TICKER>
```

Example:

```
!trade AAPL
```

---

## Security & Compliance

- All API keys and secrets are loaded from environment variables.
- No sensitive data is logged or exposed in error messages.
- License information is included in the LICENSE file (MIT License).
- Follows best practices for modularity, error handling, and input validation.
- `.gitignore` ensures secrets and logs are not tracked by git.

---

## Project Structure

- `core/`: Main pipeline modules
- `utils/`: Logging and formatting utilities
- `config/`: Environment variable loader
- `run.py`: CLI entry point
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes secrets, logs, and unnecessary files

---

## Enhanced Analysis Output

The system now provides comprehensive trading analysis with detailed insights:

```
🤖 AI Day Trader Agent - Enhanced Analysis for APAM

 Analysis Results:
----------------------------------------
**Primary Strategy:** SENTIMENT
**Recommendation:** HOLD
**Confidence:** 50.0%
**Quantity:** 0 shares
**Reason:** Sentiment score: 0.40

**Technical Indicators:**
  Current Price: $40.60
  RSI: 50.99 (Neutral)
  MACD: -0.0529 / Signal: -0.0547 (Bullish)
  SMA(20): $40.61 Below
  EMA(20): $40.54 Above

**All Strategy Signals:**
  Technical: HOLD (strength: 0.20)
  Sentiment: HOLD (score: 0.40)
  Dividend: HOLD (reason: Outside capture window. Next dividend in 46 days)

**Analysis Time:** 2025-06-29 19:58:52
```

---

## License

MIT License. See [LICENSE](LICENSE) file for details.
