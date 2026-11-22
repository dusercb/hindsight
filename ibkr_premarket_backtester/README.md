# Hindsight: detailed guide

How the engine qualifies candidates, gates entries, resolves fills and reports results. For an overview, see the [main README](../README.md).

Backtesting/paper-research project for testing a premarket **break of HOD after first pullback** setup on user-supplied ticker/date pairs.

This project **does not place live trades**. It connects to IBKR/TWS or IB Gateway in read-only mode (`ibkr.readonly`, default `true` — now honored by `connect()`, so a read-only session physically rejects order submission) and requests historical bars only.

## Strategy rules included

For each ticker/date candidate:

1. Request IBKR historical bars including extended hours.
2. Derive the previous regular-session close from prior-day bars (informational only; see RVOL note below).
3. Build the premarket session from `04:00` to `09:30` New York time.
4. Record price fields for later analysis: previous close, premarket open, premarket high, entry price, exit price.
5. Record stock float if supplied in the input file or optional float reference file.
6. Require a premarket **RVOL (relative volume) spike of at least `min_rvol`, reached ANYWHERE in the session**: cumulative premarket volume vs. this ticker's own baseline regular-session daily volume over its trailing `rvol_lookback_days` — the **median** of those days by default (`rvol_baseline_stat: "median"`, robust to a single prior halt/news day that would inflate a `mean`; set to `"mean"` for the older behavior). Fetched live from IBKR via a cheap, cached daily-bar request; a ticker with too little daily-volume history, e.g. a recent IPO, is marked `INSUFFICIENT_DATA` rather than disqualified. This is a pure yes/no gate -- it doesn't matter *when* in the session RVOL crosses, only that it does. A ticker with flat/normal volume never qualifies, no matter how far the price has moved. (Entry itself additionally re-checks RVOL causally at the bar -- see Entry gates.)
7. Require a premarket move of at least `min_premarket_gain_pct`, measured from the **premarket open** (not the previous session's close, and not tied to when RVOL happened to cross) to the premarket high.
8. Track premarket HOD across the whole session.
9. Wait for the first pullback.
10. Reject the setup if the dip retraces more than `max_pullback_retrace_pct` of **the rip itself** -- the move from its launch low (the lowest low printed while its HOD was still ratcheting) up to that HOD. Think of it as the rip's trending support line: a dip that holds it keeps the setup bullish; a dip that breaks it breaks the formation. This is deliberately a different anchor from the gain% check: a rip often launches from below the open (a flush first, then the real move), and measuring its dips against the move-from-open mis-sizes them. With `first_setup_only: false`, a broken formation only abandons that setup -- the scan keeps looking for a fresh HOD/pullback (measured from the breakdown onward) instead of giving up on the candidate entirely.
11. Simulate a buy trigger at `HOD + $0.01`.
12. Simulate target at `entry + $0.20` and stop at `entry - $0.10`.
13. Every candle that breaks through the trigger, stop, or target is resolved from its real IBKR trade ticks when available (`intrabar_policy: "ticks"`) -- the actual traded price at the crossing, not a bar-derived estimate, offset by `order_latency_seconds` to simulate real order-submission delay. This replaced an earlier design that only used ticks for the rare both-touched-in-one-bar case and otherwise assumed an idealized exact fill at the target/stop level.

### Why RVOL instead of gain-vs-previous-close

Earlier versions of this backtester qualified a candidate purely on "premarket high is N% above yesterday's close." That rewards any stock that happened to be elevated at yesterday's close, not stocks that actually had a sudden move. The RVOL requirement targets sudden moves directly: real volume relative to that ticker's own historical norm, vs. the previous close which says nothing about volume at all.

Two earlier iterations of this design tied the gain% measurement to the moment RVOL crossed the threshold, and both had the same failure mode:

1. First, RVOL compared a bar's volume only to that same morning's own earlier bars (no extra data requests, but as measured on real data it disqualified most sudden-mover candidates, since gradual multi-minute ramps rarely spike 5x above their own immediately preceding average).
2. Switching to a multi-day historical volume baseline (the current RVOL definition, below) fixed that, but introduced a new problem: for tickers with a large average daily volume, crossing 5x of it can take most of the session to accumulate, since volume is cumulative. By the time it finally crossed, the actual price move had frequently already happened -- so anchoring the gain% requirement to "the price right before the spike" measured only whatever was left over after the move was already over, not the move itself. One real example: a ticker ran from $3.40 to $6.66 (96%) well before its RVOL crossed 5x at all; by the time it did, the anchor was already near the peak, and the "gain since spike" was measured at just 1%.

RVOL and gain% are now fully decoupled: RVOL is checked as a pure yes/no gate (did this ticker show 5x volume anywhere this session?), and gain% is independently measured from the premarket open to the premarket high, regardless of when RVOL happened to cross.

Decoupling the *gain% metric* from RVOL timing is a separate concern from *when a trade is allowed to enter* -- the two can't share one check. RVOL crossing 5x eventually doesn't mean an entry on an earlier bar was valid: a live trader watching this ticker wouldn't have known about a volume surge that, as of that bar, hadn't happened yet. So the actual entry decision re-checks RVOL using only cumulative volume through the prior completed minute (see Entry gates below) -- the up-front yes/no check only establishes that a valid entry is *possible somewhere* in the session, never that any specific bar's entry is retroactively valid.

### Entry gates

Once a bar reaches the trigger price, all of these must pass on that same bar (each records a `skipped_*` counter on the trade result if it blocks a bar -- see Diagnostics below). Every gate uses only data through the **prior completed minute** (no lookahead):

- A completed 1-minute MACD bar must exist for the minute right before the trigger.
- **RVOL must have ACTUALLY reached `min_rvol` as of this bar** (cumulative volume through the prior completed minute -- the same causal lookup as MACD). The up-front RVOL check only confirms a valid entry is *possible somewhere* in the session; it does not make an *earlier* bar's entry valid just because volume caught up later. This is a no-lookahead-bias guarantee: a candidate can't trade on a volume surge that, from that bar's point of view, hadn't happened yet.
- **Red/green volume check** (`red_volume_check`, default `"total"`): across the current breakout's bars, cumulative red-candle volume must not outweigh cumulative green-candle volume. The older `"largest_bar"` mode instead compares only the single largest red vs largest green candle. (Neither requires a green candle to exist at all -- an all-red breakout can still trigger if the comparison doesn't apply.)
- **MACD must be bullish**: MACD line > 0 and MACD line > signal line. Periods are configurable via `macd_fast`/`macd_slow`/`macd_signal` (default 12/26/9).
- **Retrace-support gates** (`block_retrace_below_ema9` / `block_retrace_below_ema20` / `block_retrace_below_vwap`, all default `false`): optionally block a trade if the retrace (HOD through this touch) ever traded below the 1-minute EMA9 / EMA20 / session VWAP at *any* point along the way. Each is an independent on/off switch.
- **Entry-time window** (`earliest_entry_time` / `latest_entry_time`, `"HH:MM"` or null): a touch outside the window is skipped, but a *later* touch of the same armed setup inside the window can still fire.
- Fill tolerance: if ticks resolve the fill this doesn't apply -- the real tick price at the crossing is the fill. Only when ticks are unavailable or don't confirm the trigger traded does the bar-based fallback apply: fill price must not exceed the trigger by more than `entry_limit_slippage_cents` (a flat $/share amount).
- If `max_spread_cents` is set, the bid/ask spread at the trigger bar (fetched live from IBKR) must not exceed it.

### More than the first move, and scaling out

- **`max_trades_per_day`** (default `1`): after a completed trade, if the session isn't over, the scan can hunt for a fresh HOD/pullback/trigger -- the 2nd/3rd/... move -- up to this many trades. gain% stays anchored to the *original* premarket open for every move. Each trade records a `setup_index` (1 = first move).
- **Scale-out ladder** (`scale_out`, default `true`): on hitting the planned target, sell `scale_out_sell_pct`% of the remaining position, ratchet the stop to `scale_out_stop_offset` *below* that exit, raise the target by `scale_out_step`, and repeat for up to `scale_out_max_sells` sells (the last one liquidating the remainder). A scaled trade's WIN/LOSS is the sign of its net P&L across all legs; the legs are recorded in `exit_legs`.

### Diagnostics

Every `TradeResult` now records why a qualifying candidate never turned into a trade:

- `rvol_at_spike`, `rvol_spike_time`, `max_rvol_seen`, `daily_avg_volume` -- the RVOL values found (or how close a disqualified candidate got); `rvol_spike_time` is recorded for visibility but doesn't affect qualification. `anchor_price` is the premarket open used for the gain% calc.
- `trigger_touches` -- how many bars actually reached the trigger price.
- `skipped_no_macd_bar`, `skipped_rvol_not_yet_confirmed`, `skipped_red_volume_dominant`, `skipped_macd_not_bullish`, `skipped_fill_exceeds_limit`, `skipped_spread_too_wide`, `skipped_outside_entry_window`, `skipped_retrace_below_ema9`, `skipped_retrace_below_ema20`, `skipped_retrace_below_vwap` -- how many trigger-eligible bars each specific gate blocked. `skipped_rvol_not_yet_confirmed` counts bars where the trigger otherwise fired but RVOL genuinely hadn't confirmed as of that bar yet -- a high count usually means RVOL only confirms well after the setup, so no entry was ever valid without lookahead.
- `invalid_retrace_attempts` -- how many times a pullback breached `max_pullback_retrace_pct` (only >1 when `first_setup_only: false`).

The run summary aggregates these across all candidates (e.g. `skipped_macd_not_bullish_total`), and they're shown as metric cards in the HTML report, so a `NO_TRIGGER` or `DID_NOT_QUALIFY`-heavy run can be explained rather than treated as a black box.

## Folder layout

```text
ibkr_premarket_backtester/
  config.yaml
  requirements.txt
  data/
    ticker_dates.csv          # main input: TICKERS,DATES
    reference_floats.csv      # optional ticker -> float lookup
    raw_tickers_dates.txt     # optional loose input for formatting
    cache/                    # downloaded IBKR bars
  results/                    # generated reports
  premarket_backtester/
    cli.py
    config.py
    engine.py
    ibkr_data.py
    input_tools.py
    models.py
    strategy.py
    reporting.py
  tests/                      # pytest suite (run: python -m pytest)
```

## Setup

1. Open a terminal in this folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open TWS or IB Gateway and log into paper mode.
5. Enable API access in TWS/Gateway settings.
6. Confirm `config.yaml` matches your connection:

```yaml
ibkr:
  host: "127.0.0.1"
  port: 7497
  client_id: 7
  readonly: true
```

Common ports:

- TWS paper: `7497`
- TWS live: `7496`
- IB Gateway paper: often `4002`
- IB Gateway live: often `4001`

The app connects with `readonly` from config (default `true`) and contains no order-sending code. A read-only IBKR session rejects order submission at the API boundary, so this is a hard interlock, not just a convention.

## Input format

Edit `data/ticker_dates.csv` exactly like this:

```csv
TICKERS,DATES
ARTL,03/27/2026
ONCO,03/27/2026
VSA,03/27/2026
ELAB,03/30/2026
```

Dates are `MM/DD/YYYY`.

You may also include float directly in this same file:

```csv
TICKERS,DATES,FLOAT
ARTL,03/27/2026,7.5M
ONCO,03/27/2026,12000000
```

## Stock float data

IBKR historical price bars provide price and volume. They do **not** reliably provide historical float. This project therefore supports an optional float reference file:

```csv
TICKERS,FLOAT
ARTL,7.5M
ONCO,12M
VSA,9000000
```

Place this in:

```text
data/reference_floats.csv
```

The backtest output always includes:

- `float_shares`
- `float_source`
- `float_missing`

If float is not supplied, those columns still exist and `float_missing` will be `True`.

## Price data stored

Each output row includes price fields so you can later group results by price band:

- `prev_close`
- `stock_price` — set to the setup/HOD price when available, otherwise premarket open
- `premarket_open`
- `premarket_high`
- `entry_price`
- `target_price`
- `stop_price`
- `exit_price`

By default, the project **does not filter by stock price**. It tests everything you provide, then records the price data so you can analyse later.

To add price filters later, edit `config.yaml`:

```yaml
strategy:
  min_price: 1.00
  max_price: 20.00
```

Leave both as `null` to test all prices.

## Loose input formatter

If you have tickers and dates in a rough list, put them in:

```text
data/raw_tickers_dates.txt
```

Examples accepted:

```text
ARTL 03/27/2026
03/27/2026 ONCO
VSA, ELAB - 03/30/2026
```

Then run:

```bash
python -m premarket_backtester.cli format-input --raw data/raw_tickers_dates.txt --out data/ticker_dates.csv
```

## Run the backtest

```bash
python -m premarket_backtester.cli backtest --config config.yaml
```



## Bar size

The default historical replay resolution is now:

```yaml
data:
  bar_size: "10 secs"
```

The strategy treats `HOD + $0.01` as a **software trigger** during the replay. There is no simulated native stop-limit order. Once a 10-second bar trades through the trigger, the engine records the trigger and applies the configured slippage model.

`entry_limit_slippage_cents` (default `0.15`) is that model's fill tolerance, as a **flat dollars-per-share amount** above the trigger: the fill can be up to this many cents above the trigger and still count as filled; further than that (a faster gap-through), the software trigger is treated as missed for that bar. Flat cents matches how the order is really placed — e.g. a Webull marketable limit is capped at ~+$0.15 of the market to guarantee a fill — so a $3 and a $15 name get the same real-world buffer, instead of a price-scaled percent that was far too tight on cheap names and too loose on pricey ones.

Widening this does **not** mean paying more on a fill -- the simulated fill price is still `max(trigger, bar open)`, the realistic best-available price on that bar, never the `entry_limit` ceiling itself. Widening the ceiling only raises the chance a bar counts as fillable at all; it mirrors how a real marketable limit order works (IBKR fills at the best available price, the limit is just the worst-case cap you'd accept). Genuinely explosive RVOL movers (a large fraction of what qualifies now -- see the RVOL section above) can run well past a tight window and never come back into it, so setting this too small mostly produces `skipped_fill_exceeds_limit` misses rather than real fills -- though for the most extreme movers (price running by dollars, not cents, within minutes), no cents/percentage-based ceiling alone fully closes that gap; the trigger itself stays frozen at the first breakout unless `first_setup_only: false` lets the scan re-arm on a fresh setup. For context, IBKR's own price-reasonability check on limit orders is roughly 10% away from NBBO for stocks -- this default keeps a comfortable margin under that.

You can later change this to `5 secs`, `15 secs`, `30 secs`, or `1 min`, subject to what IBKR returns for the requested historical period. Smaller bars usually give a better breakout simulation but require more data requests and may hit pacing limits sooner.

## Outputs

The app writes:

```text
results/backtest_trades_YYYYMMDD_HHMMSS.csv
results/summary_YYYYMMDD_HHMMSS.json
results/run_YYYYMMDD_HHMMSS/report_YYYYMMDD_HHMMSS.html
```

Each trade row includes:

- ticker/date
- result
- reason
- previous close
- price fields
- float fields
- premarket high and time
- premarket gain percentage
- pullback low and time
- pullback retracement percentage
- entry/target/stop
- exit price/time
- P&L
- MFE/MAE

## Important limitations

- IBKR historical bars can be rate-limited. Large multi-month tests should run with caching enabled.
- 10-second bars give a better premarket replay than 1-minute candles, but they cannot model intrabar order sequence on their own. With `intrabar_policy: "ticks"` the backtester resolves the ambiguous bars from IBKR historical trade ticks; if ticks are genuinely unavailable for such a bar it falls back to the candle-path heuristic (an ambiguous bar is never an automatic conservative loss unless `intrabar_policy: "conservative"` is explicitly chosen).
- Historical trade bars do not include bid/ask spread. When `max_spread_cents` is set, the backtester fetches it separately via a single live IBKR historical BID_ASK tick request per trigger attempt (same pacing budget as `data.tick_request_pause_s`), rather than from the trade bars.
- Historical float is not available from IBKR bars. Supply float yourself via the input CSV or `reference_floats.csv`.
- IBKR historical tick requests are heavily rate-limited, which is why ticks are only fetched for path-ambiguous bars (never for the whole session), paced by `data.tick_request_pause_s`, and cached on disk.
- Every candidate needs a live daily-bar request to compute its RVOL baseline (the `median` -- or `mean` -- of the trailing `rvol_lookback_days`). This is much cheaper than tick requests -- 1-day bars have generous IBKR duration limits -- and is cached per (ticker, day, `rvol_lookback_days`), so re-running a backtest never re-fetches it.
- The previous regular-session close is still derived and reported for context, but no longer gates a candidate -- the gain% qualification is measured from the premarket open (the session's first print) instead (see Strategy rules above). The previous trading session itself is found from the actual dates present in the fetched bars (not by assuming "day minus 1 calendar day", which breaks on Mondays and any day after a holiday); if the default 2-day fetch window doesn't reach that session at all (e.g. a holiday sitting right next to a weekend), the backtester retries once with a wider window (`data.wide_duration`, default `"6 D"`) before giving up.
- This is a research/backtesting scaffold, not financial advice and not an execution system.

## Changing the strategy

**`config.yaml` is the only file you edit to change settings.** `premarket_backtester/config.py` just defines the schema (and fallback defaults for keys omitted from the yaml) -- editing numbers there has no effect while config.yaml sets the same key, which it does for every strategy knob. To verify what a run actually executed with, check the strategy line echoed to the console at run start, the `config_used.yaml` written into every run folder, or the "Configuration Used" section of the HTML report. A misspelled key or section in config.yaml now fails at startup with an error naming the bad key, instead of silently running with defaults.

Edit `config.yaml`:

```yaml
strategy:
  min_rvol: 5.0
  min_premarket_gain_pct: 20.0
  entry_offset: 0.01
  target_offset: 0.20
  stop_offset: 0.10
  max_pullback_retrace_pct: 45.0
  min_pullback_cents: 0.05
  min_pullback_pct: 2.0
  min_price: null
  max_price: null
```

The pullback rule:

```yaml
max_pullback_retrace_pct: 45.0
```

This means a setup is rejected if the dip retraces more than that percentage of the rip that formed it (measured from the rip's launch low to its HOD -- its trending support line), not of the move-from-open.

## 10-second bars, tick-based fills, and stop-loss accuracy

The project now defaults to `bar_size: "10 secs"`.

A 10-second OHLC bar still contains the bar's **low**, so a dip below the stop is not ignored. If the low is below your stop, the backtester counts the stop as touched. But a bar only tells you a level traded *inside* the interval -- not the exact order of events, and not the precise price at the moment it traded.

**Every candle that breaks through the trigger, stop, or target gets resolved from its real trade ticks when `intrabar_policy: "ticks"`** (the default) **and a tick fetcher is available** -- not just the case where a single bar touches both stop and target:

- **Entry**: instead of guessing the fill from the bar's open, the backtester fetches that bar's ticks and finds the real traded price at the moment price first crossed the trigger. This is the genuine price an order would have gotten, not an estimate.
- **Exit (single-sided)**: a bar that only touches the target, or only the stop, still gets its exit price from the real crossing tick -- so a stop that gapped through its level shows up as a real, worse-than-configured loss, and a target hit shows the actual print rather than an idealized exact fill at the target level.
- **Exit (ambiguous, both touched in one bar)**: example --

  ```text
  High = target touched
  Low  = stop touched
  ```

  Ticks resolve which one actually traded first, same as before.

Because IBKR rate-limits tick requests, ticks are only ever requested for bars that actually break through a level (never for the whole session), consecutive live requests are spaced by `data.tick_request_pause_s` seconds, and every window is cached under `data/cache/ticks/` so it's never requested twice. If the entry bar is also the first exit-checking bar, the already-fetched ticks are reused rather than fetched again.

One caveat on caching: every tick window the strategy requests corresponds to a bar that demonstrably traded (it has volume), so an **empty** tick response is contradictory -- almost always a failed, denied, or paced-out request rather than a quiet market. Empty responses for such windows are therefore never cached as ticks (so a transient IBKR failure can't permanently poison that window into bar-based fills), and any empty tick cache files written before this guard existed are automatically distrusted and re-fetched. Instead, a sidecar "gap memo" (`*_ticks.gap.json`) counts live attempts across runs: the first attempt gets an immediate paced retry, later runs get one attempt each, and once `data.max_empty_tick_attempts` total attempts have all come back empty the window is accepted as a permanent gap in IBKR's tick store and skipped for free from then on (the candle-path heuristic resolves those bars). This keeps transient failures self-healing without every run re-paying paced requests for windows IBKR will simply never serve.

If ticks still can't be fetched, or don't confirm the level actually traded, the bar-based fallback applies:

- **Entry**: fill = `max(trigger, bar open)`, capped by `entry_limit_slippage_cents`.
- **Exit, single-sided**: an idealized exact fill at the target or stop price.
- **Exit, ambiguous**: the candle-path guess (`resolve_intrabar`: a bullish bar most plausibly printed its low early, open -> low -> high -> close; a bearish bar its high early) -- unless `intrabar_policy: "conservative"` is explicitly set, in which case the stop always counts first (understating performance, never overstating it). The resolution used is recorded per trade in `ambiguity_resolution` and the trade's `reason`.

### Order latency

```yaml
strategy:
  order_latency_seconds: 0.1
```

Real orders don't execute the instant a signal fires -- there's a delay between detecting the crossing and the order reaching the market. When ticks resolve a fill, the reported price/time is taken `order_latency_seconds` **after** the crossing tick, not the crossing tick itself; the crossing still decides the outcome (which level was breached first), only the fill price and timestamp shift forward. This only applies when tick data resolves the fill -- the bar-based fallback has no sub-second resolution to model a fractional-second delay against.

The output includes:

- `path_ambiguous` — true when target and stop both touched inside the same bar
- `ambiguous_bar_time` — the bar that was ambiguous
- `ambiguity_resolution` — how an *ambiguous* bar was decided: `ticks`, `heuristic`, or `conservative` (only set when `path_ambiguous` is true)
- `entry_fill_source` / `exit_fill_source` — how the entry/exit fill was actually determined: `ticks` (a real traded price) or `bar` (the idealized/heuristic fallback), set for every trade regardless of ambiguity
- `intrabar_policy` — the rule used for that backtest
- `trigger_time` — when the bar-level signal fired; `entry_time` — the actual fill time (tick-precise and latency-shifted when ticks resolved it; equal to `trigger_time` only when the bar-based fallback was used)
- `trigger_price` — HOD + entry offset
- `fill_price` / `entry_price` — simulated fill (real tick price when available, an idealized estimate otherwise)
- `slippage` — fill minus trigger

For higher precision later, swap the config to 5-second or 1-second bars if IBKR returns them for the date range. The strategy engine is designed so that data-source changes do not require rewriting the strategy.

## Outputs

Each run creates a timestamped folder under `results/` with:

```text
run_YYYYMMDD_HHMMSS/
  backtest_trades_YYYYMMDD_HHMMSS.csv
  summary_YYYYMMDD_HHMMSS.json
  backtest_YYYYMMDD_HHMMSS.sqlite
  report_YYYYMMDD_HHMMSS.html
  report_assets/
    result_counts.png
    cumulative_r.png
    pullback_distribution.png
    price_distribution.png
    float_distribution.png
  trade_folders/
    TICKER_YYYY-MM-DD/
      trade.json
      bars.csv
      chart.png
```

The summary includes win rate, configured reward:risk, realized reward:risk, average R, expectancy, profit factor, result counts, and the number of path-ambiguous trades (split into tick-resolved vs fallback-resolved).

The HTML report opens directly in your browser and includes metric cards, result-count charts, cumulative R, pullback distribution, price distribution, float distribution, and a trade table. Ticker links in the table open each trade chart when the per-trade chart exists.

### Per-trade charts

Each `chart.png` is real OHLC candlesticks (1-minute, resampled from the underlying 10-second bars purely for chart readability -- the backtest itself always runs on the raw bars), not a line plot. When a candidate has a HOD, pullback, trigger, entry, or exit, the chart splits into two panels: a full-session overview (with the zoomed region highlighted), and a detail panel zoomed to the actual setup and trade -- since the whole trade often lasts under a minute out of a multi-hour session, showing only the full session compresses the interesting part to a few pixels. Candidates with nothing to zoom to (rejected before any HOD/pullback formed) get a single full-session view.

Each reference level has distinct styling instead of one uniform dashed line for everything:

- Prev close: thin gray dotted line, full width (background context only)
- Premarket HOD / Pullback low: purple triangle-up / teal triangle-down marker at the exact point each occurred, not a line
- Trigger: orange dashed, spanning from session start to entry (moot afterward)
- Target / Stop: green / red dashed, spanning only from entry to exit (they aren't live risk before entry or after exit)
- Entry: green triangle-up marker at the actual fill price/time
- Exit: colored/shaped by outcome -- green star (WIN), red X (LOSS), gray diamond (other) -- at the actual exit price/time
- The holding period between entry and exit is lightly shaded green/red by outcome
