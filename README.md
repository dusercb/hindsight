# Hindsight

[![CI](https://github.com/dusercb/hindsight/actions/workflows/ci.yml/badge.svg)](https://github.com/dusercb/hindsight/actions/workflows/ci.yml)

A backtester for premarket momentum setups on Interactive Brokers data, built around one rule: no
decision may use information that wasn't available at the moment it was made.

Most backtests flatter themselves in small ways. They fill orders at prices that never traded, or let a
decision at 08:15 quietly read volume from 08:20. Hindsight replays each premarket session on IBKR's
10-second bars, resolves every candle that crosses a trigger, stop or target from the real trades inside
it, and evaluates every entry condition using only data through the prior completed minute.

## What it does

- **Replays premarket sessions** (04:00–09:30 ET) for a list of ticker/date candidates, fetching and
  caching IBKR bars, trade ticks and daily-volume baselines.
- **Runs a setup state machine.** It tracks the high of day, the first pullback and the breakout, with
  several entry, stop, target and exit modes, repeat setups within a day and scale-out ladders.
- **Fills from real trades.** An entry fills at the first trade after its trigger is crossed, plus order
  latency. A target fills at the crossing trade itself, because it is a limit order already resting at
  the exchange. A stop fills after the latency, because it still has to race to the market.
- **Gates causally.** Relative volume, MACD, EMA/VWAP and red/green-volume checks all read the prior
  completed minute, so a volume surge that hasn't happened yet can't qualify a trade.
- **Sizes positions** from equity, risk per trade and a cap on the share of traded volume, with margin
  funding.
- **Reports in depth:** an interactive HTML report with per-trade candlestick charts, CSV and SQLite
  exports, Monte Carlo resampling, t-statistics, confidence intervals and concentration checks (is the
  result still there without its best three trades?).

## Design notes

- **One decision engine, shared with live trading.** The `signal_core` package exposes the backtester's
  own decision code. The live bot, [Parity](https://github.com/dusercb/parity), imports it instead of
  re-implementing the strategy, so a backtest and a live session make the same decisions from the same
  inputs.
- **Every rejected candidate explains itself.** Each result records which gate blocked which bar, so a
  sweep that produces few trades can be diagnosed rather than guessed at.
- **A cache that heals itself.** IBKR rate-limits tick requests, so ticks are fetched only for bars that
  cross a level, and cached on disk. An empty response is never cached as truth: a gap memo retries
  across runs and retires windows that IBKR will never serve.
- **Configuration fails loudly.** A misspelled key in `config.yaml` stops the run instead of silently
  falling back to a default.

## Running it

Requires Python 3.13 and TWS or IB Gateway with API access enabled. A read-only API connection is
enough: the backtester never sends an order.

```bash
cd ibkr_premarket_backtester
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m premarket_backtester.cli backtest --config config.yaml
```

Candidates go in `data/ticker_dates.csv` as `TICKERS,DATES` rows, with dates as `MM/DD/YYYY`. The
`config.yaml` here contains example settings, and every key is documented in
`premarket_backtester/config.py`. Each run writes a timestamped folder under `results/` holding the
HTML report, the trades and the per-trade charts.

## Tests

```bash
cd ibkr_premarket_backtester
python -m pytest
```

The suite runs fully offline, with IBKR faked, and CI runs it on every push.

## Layout

```
ibkr_premarket_backtester/
  premarket_backtester/   engine, strategy, IBKR data layer, sizing, reporting, Monte Carlo
  signal_core/            the decision code the live bot imports
  tests/                  the pytest suite
  config.yaml             example configuration
  README.md               detailed guide: qualification, entry gates, fills, outputs
```

Python · pandas · NumPy · ib_insync · SQLite · Matplotlib · pytest
