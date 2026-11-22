from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
import pandas as pd

from .config import AppConfig, parse_hhmm
from .models import Result, TradeResult

# Fetches trade ticks for (ticker, window_start, window_end) -> DataFrame with
# at least "time" and "price" columns, in chronological order.
TickFetcher = Callable[[str, datetime, datetime], pd.DataFrame]

# Fetches a best-effort bid/ask spread (dollars) for (ticker, at) -> float | None.
BidAskFetcher = Callable[[str, datetime], float | None]

# Fetches this ticker's average daily (regular-session) volume over its trailing
# history, as of `day` -> float | None (None if too little history exists).
DailyVolumeFetcher = Callable[[str, date], float | None]

# Fetches this ticker's share float -> float | None (None if unavailable). Only
# called when a float filter is configured and no float was supplied via CSV.
FloatFetcher = Callable[[str], float | None]


def _fmt_time(x) -> str | None:
    if x is None or pd.isna(x):
        return None
    return pd.Timestamp(x).isoformat()


def _round_cent(x: float) -> float:
    return round(float(x) + 1e-9, 2)

def build_1m_bars(df, rule: str = "1min"):
    """Resample OHLCV bars to `rule` (default 1-minute). Used both for the fixed
    1-minute indicator/gate frame AND, with a wider rule, for the chart_timeframe
    "1m"/"2m" scan frames."""
    bars = df.copy()
    bars = bars.set_index("dt_et")

    one_min = (
        bars.resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
        .reset_index()
    )

    return one_min

def add_indicators(one_min: pd.DataFrame, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """Every 1-minute indicator the strategy uses -- EMA9, EMA20, session VWAP,
    and MACD -- computed once, in one place, so the same values drive both the
    decision gates below AND the trade chart's plotted lines (reporting.py
    applies this same function to the chart's resampled candles), rather than
    risking the chart showing something different from what the strategy saw.

    EMA9/EMA20 are plain EWMs of the 1-minute close. VWAP is cumulative from the
    start of this dataframe (the premarket session open), the standard intraday
    definition -- not a rolling window. MACD is the EMA(macd_fast) minus
    EMA(macd_slow) of the close, with its signal line an EMA(macd_signal) of that
    line (defaults 12/26/9, the standard) -- these periods are the only ones the
    strategy exposes as config; EMA9/EMA20 are fixed by name.
    """
    df = one_min.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].astype(float).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["signal"] = df["macd"].ewm(span=macd_signal, adjust=False).mean()
    df["hist"] = df["macd"] - df["signal"]
    return df


def _retrace_breached_indicator(move_bars: pd.DataFrame, one_min: pd.DataFrame, indicator_col: str) -> bool:
    """True if ANY bar across the retrace (move_bars: HOD through the current
    touch) traded below that bar's OWN causal 1-minute indicator value -- the
    same prior-completed-minute lookup used for MACD/RVOL, so this never peeks
    at an indicator value that hadn't actually formed yet. A bar with no prior-
    minute value available (e.g. the session's first minute) can't be judged and
    is treated as not-breached rather than blocking on missing data."""
    bar_minutes = move_bars["dt_et"].dt.floor("min") - pd.Timedelta(minutes=1)
    lookup = one_min.set_index("dt_et")[indicator_col]
    indicator_values = bar_minutes.map(lookup)
    # NaN comparisons are always False in numpy -- bars with no prior-minute
    # value simply never trip this check, which is the desired fail-open.
    return bool((move_bars["low"].to_numpy() < indicator_values.to_numpy()).any())


def _causal_indicator_value(one_min: pd.DataFrame, dt, col: str) -> float | None:
    """The `col` value of the 1-minute bar PRIOR to dt's minute -- the same causal
    (no-lookahead) lag MACD/RVOL/the block_* gates use. None if that prior minute
    isn't present or its value is NaN. Used by the dip_ema_reclaim entry mode to
    place the reclaim trigger at the indicator's last confirmed value."""
    bar_time = pd.Timestamp(dt).floor("min") - pd.Timedelta(minutes=1)
    row = one_min.loc[one_min["dt_et"] == bar_time]
    if row.empty:
        return None
    val = row.iloc[0][col]
    return None if pd.isna(val) else float(val)


def _swing_high_level(premarket: pd.DataFrame, lo_idx: int, i: int) -> float | None:
    """dip_swing_high trigger base: the highest high from the pullback low
    (inclusive) up to the bar BEFORE the current one -- the post-dip resistance
    whose break is the first higher-high off the low. None when no such window
    exists yet (the current bar is the pullback low itself)."""
    if lo_idx is None or lo_idx > i - 1:
        return None
    return float(premarket["high"].iloc[lo_idx:i].max())


def _last_red_high(premarket: pd.DataFrame, hod_idx: int, i: int, ceiling: float | None = None) -> float | None:
    """dip_red_break trigger base: the HIGH of the most recent RED candle
    (close < open) printed in the dip (HOD bar through the bar before the current
    one). The dip is BUILT of red candles -- breaking back above the last one is
    the first proof the sellers who built it have lost control. Each newer red
    candle trails the trigger down with the dip, so the entry stays one candle
    behind the selling instead of anchored to a stale earlier level. None when
    the dip holds no red candle yet (a green drift is not a broken downtrend).

    ceiling (the armed HOD) excludes red candles that printed ABOVE the HOD: a dip
    candle sits below the high it's reversing toward, so a red bar above the HOD
    belongs to a later run-up, not this dip. Without it, a violent post-pullback
    rocket (HIND 2025-09-17: HOD 9.97, ran to 17.5) let the 'last red candle' land
    at 16.72 and the entry chase to 16.80 -- 68% above its own HOD."""
    dip = premarket.iloc[hod_idx:i]
    reds = dip[dip["close"] < dip["open"]]
    if ceiling is not None:
        reds = reds[reds["high"] <= ceiling]
    if reds.empty:
        return None
    return float(reds.iloc[-1]["high"])


def _last_red_low(premarket: pd.DataFrame, hod_idx: int, i: int, ceiling: float | None = None) -> float | None:
    """The LOW of the SAME last red candle _last_red_high selects -- used by
    stop_mode "red_candle_low" so the stop sits at that candle's own low (a tighter
    stop than the whole dip's low). Mirrors _last_red_high's selection exactly."""
    dip = premarket.iloc[hod_idx:i]
    reds = dip[dip["close"] < dip["open"]]
    if ceiling is not None:
        reds = reds[reds["high"] <= ceiling]
    if reds.empty:
        return None
    return float(reds.iloc[-1]["low"])


def _pivot_high_positions(highs: list[float]) -> list[int]:
    """Indices of interior local-maximum highs (a 3-bar pivot: >= both neighbors)."""
    return [k for k in range(1, len(highs) - 1) if highs[k] >= highs[k - 1] and highs[k] >= highs[k + 1]]


def _rip_base(premarket: pd.DataFrame, hod_idx: int, running_low: float, s) -> float:
    """The rip's launch low (what the retrace % is measured against), per
    strategy.rip_anchor. "session_low" is the running minimum (the original, which
    can lock onto a stale early flush); "recent" is the lowest low within
    rip_recent_minutes before the HOD; "swing_low" is the most recent 3-bar pivot
    low before the HOD (the base of the final leg). Recent/swing anchors reflect the
    trend actually being traded; both fall back to running_low with too few bars,
    and can never be below the true running low (a subset min) or above the HOD."""
    if s.rip_anchor == "session_low" or hod_idx is None:
        return running_low
    pre = premarket.iloc[:hod_idx + 1]
    if len(pre) < 3:
        return running_low
    if s.rip_anchor == "recent":
        cutoff = premarket.loc[hod_idx, "dt_et"] - pd.Timedelta(minutes=s.rip_recent_minutes)
        window = pre[pre["dt_et"] >= cutoff]
        return float(window["low"].min()) if not window.empty else running_low
    # swing_low: the last interior pivot low (a 3-bar local minimum) before the HOD.
    lows = pre["low"].tolist()
    pivots = [k for k in range(1, len(lows) - 1) if lows[k] <= lows[k - 1] and lows[k] <= lows[k + 1]]
    return float(lows[pivots[-1]]) if pivots else running_low


def _dip_trendline_level(premarket: pd.DataFrame, hod_idx: int, i: int) -> float | None:
    """dip_trendline trigger base: the value at bar i of a descending least-squares
    line fit through the dip's lower-high pivots -- the HOD (its anchor) plus every
    interior pivot high from hod_idx to i-1. None when there are fewer than two
    pivots or the fit isn't actually descending (slope >= 0), in which case the
    caller falls back to the swing-high level. This is the literal 'draw the
    downtrend line and buy its break' entry, made robust by least-squares over
    pivots instead of hand-picking two points."""
    highs = premarket["high"].iloc[hod_idx:i].tolist()
    if len(highs) < 2:
        return None
    xs = list(range(hod_idx, i))
    piv = [0] + _pivot_high_positions(highs)      # position 0 == the HOD, the anchor
    piv_x = [xs[k] for k in piv]
    piv_y = [highs[k] for k in piv]
    if len(piv_x) < 2:
        return None
    n = len(piv_x)
    mx, my = sum(piv_x) / n, sum(piv_y) / n
    sxx = sum((x - mx) ** 2 for x in piv_x)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(piv_x, piv_y)) / sxx
    if slope >= 0:                                # flat/rising -> no downtrend to break
        return None
    return slope * i + (my - slope * mx)          # project the line to bar i


def add_rvol(one_min: pd.DataFrame, daily_avg_volume: float | None) -> pd.DataFrame:
    """Relative volume per 1-minute premarket bar: CUMULATIVE premarket volume up to
    and including this bar, divided by this ticker's own average daily (regular
    session) volume over its trailing history -- the RVOL definition traders
    actually mean (vs. this ticker's own historical norm, not just that morning).

    Cumulative volume only grows, so this is monotonic: once RVOL crosses the
    threshold it stays crossed for the rest of the session (no separate "spike
    detection" needed beyond finding the first bar where it clears the bar).

    daily_avg_volume of None (e.g. too little daily-volume history for this
    ticker) means RVOL is undefined for the whole session -- it can never qualify.
    """
    df = one_min.copy()
    df["cum_volume"] = df["volume"].astype(float).cumsum()
    if daily_avg_volume is None or daily_avg_volume <= 0:
        df["rvol"] = float("nan")
        return df
    df["rvol"] = df["cum_volume"] / daily_avg_volume
    return df

def previous_close(prev_rth: pd.DataFrame) -> float | None:
    if prev_rth.empty:
        return None
    return float(prev_rth.sort_values("dt_et").iloc[-1]["close"])

def resolve_intrabar(open_, high, low, close, stop, target):
    """Deterministic candle-path guess for a bar whose OHLC touched both stop and
    target, used only when real ticks can't resolve the true order.

    Assumption from candle shape: a bullish bar (close >= open) finishes near its
    high, so its low most plausibly printed early -- path open -> low -> high ->
    close. A bearish bar the reverse: open -> high -> low -> close. The first
    level the assumed path crosses decides the outcome.

    (An earlier version walked the bullish path for every bar regardless of
    direction, which meant any ambiguous bar that didn't OPEN beyond the target
    resolved as a loss -- a conservative default wearing a heuristic label.)
    """
    path = [open_, low, high, close] if close >= open_ else [open_, high, low, close]
    for price in path:
        if price <= stop:
            return "LOSS"
        if price >= target:
            return "WIN"
    # Unreachable for a genuinely ambiguous bar (low <= stop and high >= target
    # guarantee a decision above); kept as a pessimistic backstop.
    return "LOSS"

def _bar_seconds(bar_size: str) -> int:
    """Length of one bar in seconds from an IBKR bar-size string like '10 secs'."""
    n, _, unit = bar_size.strip().partition(" ")
    value = int(n)
    unit = unit.strip().lower()
    if unit.startswith("min"):
        return value * 60
    if unit.startswith("hour"):
        return value * 3600
    return value

def _delayed_fill(times: list, prices: list, cross_idx: int, latency: object | None) -> tuple[object, float]:
    """Given the tick index where a level crossed, return the (time, price) of the
    fill after `latency` (a pd.Timedelta) has elapsed -- the first tick at or after
    cross_time + latency -- simulating real order-submission delay. The crossing
    tick itself decides direction/outcome; this only affects the fill price/time.
    Falls back to the last available tick if the delay runs past the tick window.
    """
    if not latency:
        return times[cross_idx], prices[cross_idx]
    target_time = times[cross_idx] + latency
    fill_idx = next((k for k in range(cross_idx, len(times)) if times[k] >= target_time), len(times) - 1)
    return times[fill_idx], prices[fill_idx]


def _sorted_tick_lists(ticks: pd.DataFrame | None) -> tuple[list, list] | None:
    """(times, prices) in chronological order, or None when there are no ticks."""
    if ticks is None or len(ticks) == 0:
        return None
    t = ticks.sort_values("time", kind="stable")
    return t["time"].tolist(), t["price"].tolist()


def _first_touch_indexed(times: list, prices: list, start_idx: int, stop: float, target: float, latency) -> tuple[str, object, float, int] | None:
    """Like first_touch_from_ticks but over pre-sorted lists from `start_idx`,
    also returning the crossing tick's index -- the scale-out ladder scans the
    same bar's ticks repeatedly against updated stop/target levels.

    Latency applies only to STOP fills: a triggered stop still has to race to
    the market, so its fill is the print `latency` after the crossing. The
    target is a limit order already resting at the exchange -- it fills at the
    crossing print itself (which is >= the target level by definition), not at
    wherever the tape sat moments later. Delaying target fills turned genuine
    spike-through-target sells into break-even phantoms: the crossing printed
    8.16 but the recorded fill was the 7.96 tape 0.1s later."""
    for k in range(start_idx, len(prices)):
        p = prices[k]
        if p <= stop:
            t, fp = _delayed_fill(times, prices, k, latency)
            return "LOSS", t, fp, k
        if p >= target:
            return "WIN", times[k], float(p), k
    return None


def first_touch_from_ticks(ticks: pd.DataFrame | None, stop: float, target: float, entry_cross_price: float | None = None, latency=None) -> tuple[str, object, float] | None:
    """Decide an exit bar (ambiguous or not) from its actual trade ticks.

    Scans ticks in time order and returns ("WIN"|"LOSS", fill_time, fill_price) for
    whichever of target/stop traded first -- fill_price is the REAL traded price,
    not the target/stop level itself, so a stop-loss that gapped through its level
    shows up as worse than the configured stop, and a target hit shows the real
    print rather than an idealized exact fill. If `latency` (a pd.Timedelta) is
    given, STOP fills are taken `latency` after the crossing tick (a triggered
    stop still has to race to the market); target fills always use the crossing
    print itself, being resting limit orders. For the entry bar pass
    entry_cross_price (the trigger): ticks before the trigger crossed predate the
    fill and are ignored, and the fill tick itself cannot also exit the position.

    Returns None when there are no ticks or neither level traded, so the caller
    can fall back to a bar-based policy.
    """
    rows = _sorted_tick_lists(ticks)
    if rows is None:
        return None
    times, prices = rows
    start_idx = 0
    if entry_cross_price is not None:
        fill_idx = next((k for k, p in enumerate(prices) if p >= entry_cross_price), None)
        if fill_idx is None:
            return None
        start_idx = fill_idx + 1
    hit = _first_touch_indexed(times, prices, start_idx, stop, target, latency)
    return None if hit is None else hit[:3]


def first_price_at_or_above(ticks: pd.DataFrame | None, level: float, latency=None) -> tuple[float, object] | None:
    """The real fill price for a software-triggered entry: the first trade tick at
    or above `level` (the trigger), in chronological order, with its price and time.
    If `latency` (a pd.Timedelta) is given, the fill is taken `latency` after the
    crossing tick rather than the crossing tick itself, simulating order-submission
    delay -- the crossing still confirms the trigger genuinely traded.

    Returns None when there are no ticks or the level never actually traded in the
    tick stream (e.g. the bar's high touched it only as a wick/print gap the ticks
    don't confirm), so the caller can fall back to a bar-based fill estimate.
    """
    if ticks is None or len(ticks) == 0:
        return None
    ticks = ticks.sort_values("time", kind="stable")
    times = ticks["time"].tolist()
    prices = ticks["price"].tolist()
    cross_idx = next((k for k, p in enumerate(prices) if p >= level), None)
    if cross_idx is None:
        return None
    fill_time, fill_price = _delayed_fill(times, prices, cross_idx, latency)
    return float(fill_price), fill_time


@dataclass
class ArmedSetup:
    """A dip setup that is ARMED at a completed bar: its trigger level plus the
    frozen structure (HOD, pullback) that evaluate_touch needs. The live path
    receives this via backtest_all_trades(on_armed=...), watches `trigger`
    tick-by-tick, and on the crossing print calls evaluate_touch with the live
    forming bar -- minimal-latency entry, same gate stack as the backtest."""
    trigger: float
    entry_limit: float
    hod_idx: int
    armed_hod: float
    pullback_low: float
    pullback_low_idx: int | None
    armed_bar_index: int
    prior_win_entry: float | None = None


@dataclass
class TouchDecision:
    """The outcome of evaluating one trigger touch against the entry gate stack.
    `kind` is one of:
      * "defer"  -- keep the setup armed, do not condemn the HOD (a data gap or
                    an unconfirmed close). `counter`, if set, is a skip counter to
                    bump (skipped_no_macd_bar).
      * "reject" -- the touch broke the level but failed a QUALITY gate; the
                    caller runs the stale-HOD terminate/reset via `counter`+`reason`.
      * "fire"   -- a valid entry; the fill/stop/target fields are populated.
    `entry_fill_source` is carried on "fire" AND on the two rejects that occur
    AFTER the fill was resolved (skipped_fill_exceeds_limit / skipped_spread_too_
    wide), so the caller records the same telemetry the inline code did."""
    kind: str
    counter: str | None = None
    reason: str | None = None
    entry_fill_source: str | None = None
    fill: float | None = None
    fill_time: object = None
    entry_bar_pos: int | None = None
    stop: float | None = None
    target: float | None = None
    risk_per_share: float | None = None
    volume_at_entry: float | None = None
    # The entry bar's fetched ticks, so the exit simulation can REUSE them (the
    # "reuse the entry bar's already-fetched ticks" optimization) instead of
    # re-requesting -- and, crucially, resolve exits from the entry cross point
    # exactly as before. None when the tick path wasn't taken.
    entry_ticks: object = None


def evaluate_touch(*, ticker: str, i: int, row, high: float, low: float, open_: float,
                   trigger: float, entry_limit: float, premarket: pd.DataFrame,
                   one_min: pd.DataFrame, move_bars: pd.DataFrame, hod_idx: int,
                   pullback_low: float, pullback_low_idx: int | None, armed_hod: float,
                   s, prior_win_entry: float | None, earliest_t, latest_t,
                   tick_fetcher, bid_ask_fetcher, latency, bar_len) -> TouchDecision:
    """The dip strategy's per-touch entry decision -- the exact gate stack, fill
    resolution, and stop/target computation that used to live inline in the scan
    loop, lifted verbatim into a pure function so BOTH the backtest loop and the
    live path evaluate a trigger touch identically. No side effects: the caller
    owns the mutable scan state (HOD/pullback reset via _stale_hod_break) and the
    TradeResult recording. Behavior is byte-identical to the pre-extraction inline
    code (the 342-test suite is the contract)."""
    # Close-confirmed breakout (A/B): the bar must CLOSE at/above the trigger in
    # the top of its range; entry then lands on the NEXT bar's open. A weak or
    # last-bar close DEFERS (level stays armed).
    entry_bar_pos = i
    if s.require_breakout_close_confirm:
        close_ = float(row["close"])
        bar_range = high - low
        close_pos = (close_ - low) / bar_range if bar_range > 0 else 1.0
        if close_ < trigger or close_pos < s.breakout_close_min_frac:
            return TouchDecision("defer")
        if i + 1 >= len(premarket):
            return TouchDecision("defer")
        entry_bar_pos = i + 1

    # Time-of-day filter.
    entry_clock = row["dt_et"].time()
    if (earliest_t is not None and entry_clock < earliest_t) or (latest_t is not None and entry_clock > latest_t):
        return TouchDecision("reject", counter="skipped_outside_entry_window", reason="outside the entry-time window")

    # Prior completed 1-minute bar (causal) for the RVOL/volume/indicator/MACD gates.
    bar_time = row["dt_et"].floor("min") - pd.Timedelta(minutes=1)
    macd_row = one_min.loc[one_min["dt_et"] == bar_time]
    if macd_row.empty:
        return TouchDecision("defer", counter="skipped_no_macd_bar")

    rvol_so_far = macd_row.iloc[0]["rvol"]
    if pd.isna(rvol_so_far) or rvol_so_far < s.min_rvol:
        return TouchDecision("reject", counter="skipped_rvol_not_yet_confirmed", reason="RVOL below min_rvol at the break")

    volume_so_far = macd_row.iloc[0]["cum_volume"]
    if s.min_premarket_volume is not None and volume_so_far < s.min_premarket_volume:
        return TouchDecision("reject", counter="skipped_premarket_volume_not_yet_confirmed", reason="premarket volume below min_premarket_volume at the break")

    greens = move_bars[move_bars.close > move_bars.open]
    reds = move_bars[move_bars.close < move_bars.open]
    if not greens.empty and not reds.empty:
        if s.red_volume_check == "largest_bar":
            red_dominant = reds.volume.max() >= greens.volume.max()
        else:
            red_dominant = reds.volume.sum() >= greens.volume.sum()
        if red_dominant:
            return TouchDecision("reject", counter="skipped_red_volume_dominant", reason="red volume dominant in the move")

    if s.require_red_dip_candle and pullback_low_idx is not None:
        dip_bars = premarket.loc[hod_idx:pullback_low_idx]
        if not (dip_bars["close"] < dip_bars["open"]).any():
            return TouchDecision("reject", counter="skipped_no_red_dip_candle", reason="dip had no red 10-second candle")

    if s.block_retrace_below_ema9 and _retrace_breached_indicator(move_bars, one_min, "ema9"):
        return TouchDecision("reject", counter="skipped_retrace_below_ema9", reason="retrace pierced EMA9")
    if s.block_retrace_below_ema20 and _retrace_breached_indicator(move_bars, one_min, "ema20"):
        return TouchDecision("reject", counter="skipped_retrace_below_ema20", reason="retrace pierced EMA20")
    if s.block_retrace_below_vwap and _retrace_breached_indicator(move_bars, one_min, "vwap"):
        return TouchDecision("reject", counter="skipped_retrace_below_vwap", reason="retrace pierced VWAP")

    if s.block_buy_below_ema9 and trigger < macd_row.iloc[0]["ema9"]:
        return TouchDecision("reject", counter="skipped_buy_below_ema9", reason="trigger below EMA9")
    if s.block_buy_below_ema20 and trigger < macd_row.iloc[0]["ema20"]:
        return TouchDecision("reject", counter="skipped_buy_below_ema20", reason="trigger below EMA20")
    if s.block_buy_below_vwap and trigger < macd_row.iloc[0]["vwap"]:
        return TouchDecision("reject", counter="skipped_buy_below_vwap", reason="trigger below VWAP")

    if prior_win_entry is not None:
        vwap_required = prior_win_entry * (1 + s.continuation_vwap_above_entry_pct / 100.0)
        if macd_row.iloc[0]["vwap"] < vwap_required:
            return TouchDecision("reject", counter="skipped_continuation_vwap",
                                 reason=f"VWAP not yet {s.continuation_vwap_above_entry_pct:g}% above the prior winning entry {prior_win_entry:.2f}")

    macd = macd_row.iloc[0]["macd"]
    signal = macd_row.iloc[0]["signal"]
    if macd <= 0 or macd <= signal:
        return TouchDecision("reject", counter="skipped_macd_not_bullish", reason="MACD not bullish at the break")

    # Fill resolution (bar_confirm / ticks / bar fallback), capped by entry_limit.
    fill = None
    fill_time = None
    entry_ticks = None
    entry_fill_source = None
    if s.require_breakout_close_confirm:
        entry_row = premarket.iloc[entry_bar_pos]
        fill = _round_cent(float(entry_row["open"]))
        fill_time = entry_row["dt_et"]
        entry_fill_source = "bar_confirm"
    if fill is None and tick_fetcher is not None:
        bar_start = pd.Timestamp(row["dt_et"])
        try:
            entry_ticks = tick_fetcher(ticker, bar_start.to_pydatetime(), (bar_start + bar_len).to_pydatetime())
            resolved = first_price_at_or_above(entry_ticks, trigger, latency=latency)
        except Exception:
            resolved = None
        if resolved is not None:
            fill = _round_cent(resolved[0])
            fill_time = resolved[1]
            entry_fill_source = "ticks"
    if fill is None:
        fill = _round_cent(max(trigger, open_))
        entry_fill_source = "bar"
    if fill > entry_limit:
        return TouchDecision("reject", counter="skipped_fill_exceeds_limit",
                             reason="price gapped through the trigger beyond the fill limit",
                             entry_fill_source=entry_fill_source)

    if s.max_spread_cents is not None and bid_ask_fetcher is not None:
        spread = bid_ask_fetcher(ticker, row["dt_et"].to_pydatetime())
        if spread is not None and spread * 100.0 > s.max_spread_cents:
            return TouchDecision("reject", counter="skipped_spread_too_wide",
                                 reason="bid/ask spread too wide", entry_fill_source=entry_fill_source)

    # Stop/target: offset or percent, then the structural stop override, then the
    # max-risk gate, then the R-multiple target.
    if s.offset_mode == "percent":
        target = _round_cent(fill * (1 + s.target_offset_pct / 100.0))
        stop = _round_cent(fill * (1 - s.stop_offset_pct / 100.0))
    else:
        target = _round_cent(fill + s.target_offset)
        stop = _round_cent(fill - s.stop_offset)
    if s.stop_mode == "pullback_low":
        stop = _round_cent(pullback_low)
    elif s.stop_mode == "red_candle_low":
        red_low = _last_red_low(premarket, hod_idx, i, ceiling=armed_hod)
        stop = _round_cent(red_low if red_low is not None else pullback_low)
    risk_per_share = fill - stop
    if s.max_risk_pct is not None and fill > 0 and risk_per_share / fill * 100.0 > s.max_risk_pct:
        return TouchDecision("reject", counter="skipped_risk_too_wide",
                             reason="entry-to-stop risk exceeds max_risk_pct",
                             entry_fill_source=entry_fill_source)
    if s.target_mode == "r_multiple":
        target = _round_cent(fill + s.target_r_multiple * risk_per_share)

    return TouchDecision("fire", entry_fill_source=entry_fill_source, fill=fill,
                         fill_time=fill_time, entry_bar_pos=entry_bar_pos, stop=stop,
                         target=target, risk_per_share=risk_per_share,
                         volume_at_entry=float(macd_row.iloc[0]["cum_volume"]),
                         entry_ticks=entry_ticks)


def backtest_one(ticker: str, day: date, bars: pd.DataFrame, premarket: pd.DataFrame, prev_rth: pd.DataFrame, config: AppConfig, float_shares: float | None = None, tick_fetcher: TickFetcher | None = None, bid_ask_fetcher: BidAskFetcher | None = None, daily_volume_fetcher: DailyVolumeFetcher | None = None, float_fetcher: FloatFetcher | None = None) -> TradeResult:
    """The day's first trade (or its terminal non-trade outcome) only. A thin
    single-result wrapper around backtest_all_trades, kept because most callers
    only care about the first setup; see backtest_all_trades for every move of
    the day when strategy.max_trades_per_day > 1."""
    return backtest_all_trades(
        ticker, day, bars, premarket, prev_rth, config,
        float_shares=float_shares, tick_fetcher=tick_fetcher, bid_ask_fetcher=bid_ask_fetcher,
        daily_volume_fetcher=daily_volume_fetcher, float_fetcher=float_fetcher,
    )[0]


def backtest_all_trades(ticker: str, day: date, bars: pd.DataFrame, premarket: pd.DataFrame, prev_rth: pd.DataFrame, config: AppConfig, float_shares: float | None = None, tick_fetcher: TickFetcher | None = None, bid_ask_fetcher: BidAskFetcher | None = None, daily_volume_fetcher: DailyVolumeFetcher | None = None, float_fetcher: FloatFetcher | None = None, on_armed: Callable | None = None) -> list[TradeResult]:
    """Replay one ticker/date using only information available up to each 10-second bar.
    Always returns a non-empty list: one TradeResult per completed trade, or a single
    informational result (DID_NOT_QUALIFY/INSUFFICIENT_DATA/NO_TRIGGER/etc.) if none
    ever completed.

    Qualification: a ticker must show a premarket RVOL spike of at least min_rvol
    ANYWHERE in the session (cumulative volume vs. its own average daily volume over
    its trailing history, fetched via daily_volume_fetcher) before it's considered at
    all -- if it never happens, no valid entry could exist regardless of price action.
    But RVOL having crossed *eventually* doesn't make an *earlier* bar's entry valid:
    the actual trigger evaluation re-checks RVOL using only cumulative volume through
    the prior completed minute (the same causal lookup used for MACD) and blocks entry
    until the stock has genuinely shown the volume surge as of that point in time --
    no backtest lookahead. The premarket gain% is measured independently, from the
    premarket open to the premarket high (not the previous session's close, and not
    tied to when RVOL crossed -- otherwise a late RVOL crossing on a high-average-
    volume ticker would misreport a move that already happened as a non-move). This
    gain% anchor is the ORIGINAL premarket open for every move of the day, not just
    the first -- "is this move worth trading" is always judged against the session's
    actual first print.

    Setup: the dip/retrace check is measured against the rip itself -- the move from
    its launch low (the lowest low printed while its HOD was still ratcheting) to
    that HOD -- treating max_pullback_retrace_pct as the rip's trending support
    line. A dip holding it keeps the setup bullish; a dip breaching it breaks the
    formation and invalidates the setup (the whole candidate, with first_setup_only).
    This is deliberately NOT the same anchor as gain% above: a rip often launches
    from below the open (a flush first, then the move), and measuring its dips
    against the move-from-open mis-sizes them.

    Fills: every candle that breaks through the trigger, stop, or target is resolved
    from its real IBKR trade ticks when a tick_fetcher is supplied (intrabar_policy
    "ticks") -- the actual traded price at the crossing, not a bar-derived estimate,
    and offset by order_latency_seconds to simulate real order-submission delay.
    OHLC bars tell us whether a level traded inside a 10-second interval, but not
    the exact order of events or the precise fill price; the bar-based fallback
    (used only when ticks are unavailable or don't confirm the level traded) still
    assumes an idealized exact fill at trigger/target/stop, capped for entries by
    entry_limit_slippage_cents. When a single bar touches both stop and target with no
    ticks to resolve it, intrabar_policy governs the fallback ("heuristic" candle-path
    guess, or "conservative" stop-first).

    Multiple trades per day: strategy.max_trades_per_day caps how many completed
    trades one ticker/day can produce. After a trade completes, if the session
    isn't over and the cap allows another, the hunt resumes right where that trade
    exited, looking for a brand-new HOD/pullback/trigger -- the 2nd/3rd/... move.
    Only the HOD/pullback/rip state resets per move (the same reset already used
    when abandoning an invalidated retrace mid-search); the qualification facts
    established once above (RVOL-ever-reached, the daily-volume baseline, the
    gain% anchor) are session-level and shared by every move.
    """
    s = config.strategy
    r = config.risk
    bar_len = pd.Timedelta(seconds=_bar_seconds(config.data.bar_size))
    latency = pd.Timedelta(seconds=s.order_latency_seconds) if s.order_latency_seconds else None
    tr = TradeResult(ticker=ticker.upper(), date=day.isoformat(), result=Result.ERROR.value)
    tr.float_shares = float_shares
    tr.float_source = "input/reference_csv" if float_shares is not None else "missing"
    tr.float_missing = float_shares is None
    tr.intrabar_policy = s.intrabar_policy
    try:
        # Parsed once per candidate (not per move): a malformed "HH:MM" string
        # should surface as one clean ERROR result via the except below, not
        # blow up repeatedly inside the per-move scan.
        earliest_t = time(*parse_hhmm(s.earliest_entry_time, "earliest_entry_time", "config")) if s.earliest_entry_time else None
        latest_t = time(*parse_hhmm(s.latest_entry_time, "latest_entry_time", "config")) if s.latest_entry_time else None

        tr.bars = int(len(premarket))
        if premarket.empty or len(premarket) < 5:
            tr.result = Result.INSUFFICIENT_DATA.value
            tr.reason = "No/too little premarket bar data returned by IBKR."
            return [tr]

        # Previous close is informational only now (qualification is anchored to the
        # premarket open), so a missing one no longer blocks the candidate.
        pc = previous_close(prev_rth)
        tr.prev_close = pc if pc and pc > 0 else None

        premarket = premarket.sort_values("dt_et").reset_index(drop=True)
        # Raw (native-bar_size) frame is kept for the 1-minute indicator/gate frame
        # and the candle_trail minute lookup, which are ALWAYS 1-minute regardless
        # of the chart timeframe the setup is scanned on.
        raw_premarket = premarket
        if s.chart_timeframe in ("1m", "2m"):
            # Trade off the 1- or 2-minute chart: resample the fetched bars to that
            # timeframe and run the WHOLE setup/entry detection (HOD, dip, red-break,
            # trigger) on them. Fills stay tick-resolved; the tick window widens to
            # the bar so a bar-level fetch still spans the right interval.
            tf_minutes = 1 if s.chart_timeframe == "1m" else 2
            premarket = build_1m_bars(premarket, f"{tf_minutes}min").sort_values("dt_et").reset_index(drop=True)
            bar_len = pd.Timedelta(minutes=tf_minutes)
        first_price = float(premarket.iloc[0]["open"])
        tr.premarket_open = first_price
        tr.stock_price = first_price
        tr.premarket_volume = int(premarket.get("volume", pd.Series(dtype=float)).fillna(0).sum()) if "volume" in premarket else None

        if s.min_price is not None and first_price < s.min_price:
            tr.result = Result.SKIPPED_PRICE_FILTER.value
            tr.reason = f"Opening premarket price {first_price:.2f} below configured min_price {s.min_price:.2f}."
            return [tr]
        if s.max_price is not None and first_price > s.max_price:
            tr.result = Result.SKIPPED_PRICE_FILTER.value
            tr.reason = f"Opening premarket price {first_price:.2f} above configured max_price {s.max_price:.2f}."
            return [tr]

        # Share-float band. Only engaged when configured (no filter = no extra
        # requests). Float from the input/reference CSV wins; otherwise fetch it
        # from IBKR (Reuters snapshot). Checked before the RVOL baseline fetch so an
        # over-float name is rejected without paying for the daily-bar request too.
        if s.min_float is not None or s.max_float is not None:
            fl = float_shares
            if fl is None and float_fetcher is not None:
                fl = float_fetcher(ticker)
                if fl is not None:
                    # Reassign the closure var (not just tr) so the per-move
                    # _seeded_result() copies below inherit the fetched float too.
                    float_shares = fl
                    tr.float_shares = fl
                    tr.float_source = "fetched"
                    tr.float_missing = False
            if fl is None:
                # A float filter is required but we couldn't determine float (no CSV
                # value and no IBKR fundamental data). Don't silently trade a name we
                # can't confirm is in-band -- surface it as a data gap instead.
                tr.result = Result.INSUFFICIENT_DATA.value
                tr.reason = "Float unavailable (no CSV value and IBKR fundamental snapshot returned none -- fundamental-data subscription?); cannot apply the configured float filter."
                return [tr]
            if s.max_float is not None and fl > s.max_float:
                tr.result = Result.SKIPPED_FLOAT_FILTER.value
                tr.reason = f"Float {fl:,.0f} shares above configured max_float {s.max_float:,.0f}."
                return [tr]
            if s.min_float is not None and fl < s.min_float:
                tr.result = Result.SKIPPED_FLOAT_FILTER.value
                tr.reason = f"Float {fl:,.0f} shares below configured min_float {s.min_float:,.0f}."
                return [tr]

        daily_avg_volume = daily_volume_fetcher(ticker, day) if daily_volume_fetcher is not None else None
        if daily_avg_volume is None:
            tr.result = Result.INSUFFICIENT_DATA.value
            tr.reason = f"Not enough daily-volume history to compute an RVOL baseline (need >= {s.rvol_min_history_days} trading days)."
            return [tr]
        tr.daily_avg_volume = daily_avg_volume

        # ALWAYS built from the raw (native) bars, never the resampled scan frame:
        # the 1-minute indicators/gates and the candle_trail minute lookup must stay
        # true 1-minute regardless of chart_timeframe (a 2-minute scan frame cannot
        # be down-resampled back to 1-minute).
        one_min = build_1m_bars(raw_premarket)
        one_min = add_rvol(one_min, daily_avg_volume)
        one_min = add_indicators(one_min, s.macd_fast, s.macd_slow, s.macd_signal)

        # Per-minute low + fat-green flag, for exit_mode "candle_trail_1min" (the
        # 1-minute trailing manager). Keyed by the minute-start timestamp. A candle
        # is fat-green when it closes up, its body is >= fatgreen_body_min of its
        # range, and it closes in the top third of that range.
        def _is_fatgreen(o, h, l, c) -> bool:
            rng = h - l
            return bool(rng > 0 and c > o and (c - o) >= s.fatgreen_body_min * rng and (c - l) / rng >= 2.0 / 3.0)
        minute_info = {
            pd.Timestamp(m.dt_et).floor("min"): (float(m.low), _is_fatgreen(m.open, m.high, m.low, m.close))
            for m in one_min.itertuples()
        }

        tr.max_rvol_seen = float(one_min["rvol"].max()) if one_min["rvol"].notna().any() else None

        # Cheap short-circuit: cumulative premarket volume only grows through the
        # session, so if it never reaches min_premarket_volume even by the CLOSE
        # of premarket (tr.premarket_volume, the full-session total), it certainly
        # hadn't reached it at any earlier touch either -- no per-touch scan below
        # could ever have passed the causal check. This is a valid necessary-
        # condition pre-filter (unlike a hard requirement based on the full-session
        # total), the same shortcut RVOL's "ever reached" check below already
        # relies on; the causal per-touch check further down (using cumulative
        # volume through the prior completed minute, like RVOL's) is what actually
        # decides whether a given touch may trade.
        if s.min_premarket_volume is not None and (tr.premarket_volume or 0) < s.min_premarket_volume:
            tr.result = Result.DID_NOT_QUALIFY.value
            tr.disqualify_reason = "premarket_volume"
            tr.reason = (
                f"Premarket session volume {tr.premarket_volume:,.0f} shares never reached configured "
                f"min_premarket_volume ({s.min_premarket_volume:,.0f}) even by session close."
            )
            return [tr]

        # RVOL is a pure yes/no qualifying gate: did this ticker show a genuine
        # volume surge ANYWHERE in the premarket session? It no longer anchors
        # gain%/HOD tracking to the moment it happened -- that lagged the actual
        # price move for tickers with a large daily-volume baseline (crossing 5x a
        # big number can take most of the session, by which point the real move
        # was often already over, leaving little gain% "left" to measure).
        spike_matches = one_min.index[one_min["rvol"] >= s.min_rvol]
        if len(spike_matches) == 0:
            tr.result = Result.DID_NOT_QUALIFY.value
            tr.disqualify_reason = "rvol"
            peak = f" (peak {tr.max_rvol_seen:.2f}x)" if tr.max_rvol_seen is not None else ""
            tr.reason = f"Premarket cumulative volume never reached {s.min_rvol:.1f}x this ticker's baseline daily volume ({s.rvol_baseline_stat} of trailing days){peak}."
            return [tr]
        spike_idx = spike_matches[0]
        tr.rvol_at_spike = float(one_min.loc[spike_idx, "rvol"])
        tr.rvol_spike_time = _fmt_time(one_min.loc[spike_idx, "dt_et"])

        # Two distinct reference points feed two distinct checks:
        #   - anchor_price (the premarket open): the gain% gate -- is the session
        #     up at least min_premarket_gain_pct from its first print? Unchanged,
        #     and shared by every move of the day (see backtest_all_trades docstring).
        #   - rip_low (the low the current move launched from -- the lowest low
        #     printed while its HOD was still ratcheting): the dip/retrace check.
        #     The dip must hold the rip's trending support: retracing more than
        #     max_pullback_retrace_pct of the rip (launch low -> HOD) means the
        #     formation broke down and the setup is no longer bullish. Earlier
        #     versions measured the retrace against the move-from-open, which
        #     mis-sized dips whenever the rip launched from anywhere other than
        #     the open (e.g. a flush below the open first, then the real move).
        anchor_price = first_price
        tr.anchor_price = _round_cent(anchor_price)
    except Exception as e:
        tr.result = Result.ERROR.value
        tr.reason = repr(e)
        return [tr]

    def _seeded_result() -> TradeResult:
        """A fresh per-move TradeResult, pre-populated with the session-level
        facts established above (shared by every move -- see docstring)."""
        mtr = TradeResult(ticker=ticker.upper(), date=day.isoformat(), result=Result.ERROR.value)
        mtr.float_shares = float_shares
        mtr.float_source = tr.float_source
        mtr.float_missing = tr.float_missing
        mtr.intrabar_policy = s.intrabar_policy
        mtr.entry_mode = s.entry_mode
        mtr.bars = tr.bars
        mtr.prev_close = tr.prev_close
        mtr.premarket_open = tr.premarket_open
        mtr.premarket_volume = tr.premarket_volume
        mtr.daily_avg_volume = tr.daily_avg_volume
        mtr.max_rvol_seen = tr.max_rvol_seen
        mtr.rvol_at_spike = tr.rvol_at_spike
        mtr.rvol_spike_time = tr.rvol_spike_time
        mtr.anchor_price = tr.anchor_price
        return mtr

    # The TRUE high of day: the running causal maximum of every bar's high seen so
    # far this session. It persists across setup resets AND across moves
    # (max_trades_per_day), because a HOD is by definition the highest price traded
    # today -- so no setup, first or later, may arm a "HOD" below a level price has
    # already printed. Without this, a reset (stale break / invalid retrace /
    # pre-window discard, all only reachable with first_setup_only False or an
    # entry window) rewound the HOD tracker to -inf and let the scan re-arm a lower
    # local peak as a fresh HOD -- e.g. AGPU 2026-04-22 peaked at 14.47 @ 08:05,
    # faded, then armed a 12.20 bounce @ 08:10 and traded ITS break: not a HOD, a
    # random peak below one that already traded.
    session_high = -math.inf

    def _scan(start_i: int, prior_win_entry: float | None = None) -> tuple[TradeResult, int | None]:
        """Hunt for one HOD/pullback/trigger/trade starting at bar position start_i.
        Returns (result, resume_index): resume_index is where the NEXT move's hunt
        should start, or None when there's nothing left to look at (the session
        ended, or this attempt found no trade at all).

        prior_win_entry: set only when this scan is a WIN-continuation hunt
        (trade_winners_qty) and continuation_vwap_above_entry_pct is configured --
        the prior winning trade's entry price, which the session VWAP must have
        lifted above by that % before a continuation break may fill."""
        nonlocal session_high
        mtr = _seeded_result()

        # Seed the HOD at the true session high so far (not -inf): the FIRST move
        # starts at -inf (nothing has traded yet), but a later move -- and every
        # reset below -- must clear the running session high before a new HOD arms.
        hod = session_high
        hod_idx = None
        pullback_seen = False
        pullback_low = math.inf
        pullback_low_idx = None
        running_low = float(premarket.iloc[start_i]["low"])
        rip_low = running_low
        armed_hod = None

        def _stale_hod_break(counter: str, reason: str):
            """A trigger touch broke the armed HOD but failed an entry gate, so no
            valid entry filled -- the first pullback is spent (a re-break of a
            level price already blew past is no longer a FIRST pullback). Bumps the
            gate's own skip counter, then either terminates the candidate as
            STALE_HOD (first_setup_only) or resets state to hunt a fresh HOD/
            pullback -- the exact terminate-vs-reset dichotomy the invalid-retrace
            path uses. Returns (mtr, None) to return from _scan when terminating,
            or None meaning 'state reset -- continue the loop'."""
            nonlocal hod, hod_idx, armed_hod, pullback_seen, pullback_low, pullback_low_idx, running_low
            setattr(mtr, counter, getattr(mtr, counter) + 1)
            mtr.stale_hod_breaks += 1
            if s.first_setup_only:
                mtr.result = Result.STALE_HOD.value
                mtr.reason = (
                    f"Armed HOD {_round_cent(armed_hod):.2f} broke without a valid entry "
                    f"({reason}); the first-pullback setup is spent (first_setup_only)."
                )
                mtr.premarket_high = _round_cent(hod)
                mtr.stock_price = _round_cent(hod)
                mtr.premarket_high_time = _fmt_time(premarket.loc[hod_idx, "dt_et"])
                mtr.premarket_gain_pct = (hod - anchor_price) / anchor_price * 100.0
                mtr.rip_low = _round_cent(rip_low)
                mtr.trigger_price = trigger
                if pullback_low_idx is not None:
                    mtr.pullback_low = _round_cent(pullback_low)
                    mtr.pullback_low_time = _fmt_time(premarket.loc[pullback_low_idx, "dt_et"])
                    mtr.pullback_retrace_pct = (armed_hod - pullback_low) / (armed_hod - rip_low) * 100.0
                return mtr, None
            hod, hod_idx, armed_hod = session_high, None, None  # a fresh HOD must clear the true session high
            pullback_seen = False
            pullback_low, pullback_low_idx = math.inf, None
            running_low = low
            return None

        for i, row in premarket.iloc[start_i:].iterrows():
            high = float(row["high"])
            low = float(row["low"])
            open_ = float(row["open"])
            running_low = min(running_low, low)
            session_high = max(session_high, high)  # the true high of day, causal

            if high > hod and not pullback_seen:
                hod = high
                hod_idx = i
                armed_hod = hod
                # While the HOD is still ratcheting, the rip's base re-anchors to
                # the lowest low seen so far. Once a pullback registers, the
                # setup freezes (hod stops ratcheting), and so does its base.
                rip_low = running_low
                continue

            if hod_idx is None:
                continue

            # A HOD that formed before earliest_entry_time can never have been
            # legitimately traded on time -- any touch before the window is
            # rejected outright (the time-of-day filter below), so a touch that
            # finally lands inside the window is really just "whichever stale
            # level got re-tested first," not a fresh signal. PBM 2026-04-20 armed
            # its HOD at 04:00, got touched 591 times over the next 4+ hours
            # (590 rejected purely for landing before the window), and the entry
            # only happened because that was the first touch to land after 07:00
            # -- nothing about volume or momentum was actually fresh at that
            # point. Once the clock reaches the window with the current HOD
            # still dated before it, drop it and hunt for a genuinely new one,
            # the same reset already used when a retrace invalidates a setup.
            hod_clock = premarket.loc[hod_idx, "dt_et"].time()
            entry_clock_now = row["dt_et"].time()
            if earliest_t is not None and entry_clock_now >= earliest_t and hod_clock < earliest_t:
                hod, hod_idx, armed_hod = session_high, None, None  # a fresh HOD must clear the true session high
                pullback_seen = False
                pullback_low, pullback_low_idx = math.inf, None
                running_low = low
                continue

            gain_pct = (hod - anchor_price) / anchor_price * 100.0
            if gain_pct < s.min_premarket_gain_pct:
                continue

            # Re-anchor the rip base to the configured rip_anchor now that the setup
            # is actually being evaluated. "session_low" keeps the running-minimum
            # base frozen at HOD-ratchet time (the original); "recent"/"swing_low"
            # pull the base up to the recent leg (bars up to the HOD, so it's stable
            # once the HOD freezes) so a stale early flush no longer masks a
            # fully-retraced spike.
            if s.rip_anchor != "session_low":
                rip_low = _rip_base(premarket, hod_idx, running_low, s)
            rip = hod - rip_low
            if rip <= 0:
                continue

            pullback_from_hod = hod - low
            pullback_pct_of_price = pullback_from_hod / hod * 100.0
            retrace_pct = pullback_from_hod / rip * 100.0

            if retrace_pct > s.max_pullback_retrace_pct:
                mtr.invalid_retrace_attempts += 1
                if s.first_setup_only:
                    mtr.result = Result.INVALID_RETRACE.value
                    mtr.reason = f"Pullback retraced {retrace_pct:.1f}% of the rip ({_round_cent(rip_low):.2f} -> {_round_cent(hod):.2f}); max is {s.max_pullback_retrace_pct:.1f}%."
                    mtr.premarket_high = _round_cent(hod)
                    mtr.stock_price = _round_cent(hod)
                    mtr.premarket_high_time = _fmt_time(premarket.loc[hod_idx, "dt_et"])
                    mtr.premarket_gain_pct = gain_pct
                    mtr.rip_low = _round_cent(rip_low)
                    mtr.pullback_low = _round_cent(low)
                    mtr.pullback_low_time = _fmt_time(row["dt_et"])
                    mtr.pullback_retrace_pct = retrace_pct
                    return mtr, None
                # Abandon just this setup and keep scanning for a fresh HOD/pullback.
                # The broken formation's base is dead with it: the next move is
                # measured from this breakdown onward, not from the old rip's low.
                hod, hod_idx, armed_hod = session_high, None, None  # a fresh HOD must clear the true session high
                pullback_seen = False
                pullback_low, pullback_low_idx = math.inf, None
                running_low = low
                continue

            pullback_big_enough = pullback_from_hod >= s.min_pullback_cents and (
                s.min_pullback_pct is None or pullback_pct_of_price >= s.min_pullback_pct
            )
            if not pullback_seen and pullback_big_enough:
                pullback_seen = True
                pullback_low = low
                pullback_low_idx = i
                armed_hod = hod
                continue

            if pullback_seen:
                if low < pullback_low:
                    pullback_low = low
                    pullback_low_idx = i
                    retrace_pct = (armed_hod - pullback_low) / (armed_hod - rip_low) * 100.0
                    if retrace_pct > s.max_pullback_retrace_pct:
                        mtr.invalid_retrace_attempts += 1
                        if s.first_setup_only:
                            mtr.result = Result.INVALID_RETRACE.value
                            mtr.reason = f"Pullback deepened to {retrace_pct:.1f}% retracement of the rip ({_round_cent(rip_low):.2f} -> {_round_cent(armed_hod):.2f}); max is {s.max_pullback_retrace_pct:.1f}%."
                            mtr.premarket_high = _round_cent(armed_hod)
                            mtr.stock_price = _round_cent(armed_hod)
                            mtr.premarket_high_time = _fmt_time(premarket.loc[hod_idx, "dt_et"])
                            mtr.premarket_gain_pct = (armed_hod - anchor_price) / anchor_price * 100.0
                            mtr.rip_low = _round_cent(rip_low)
                            mtr.pullback_low = _round_cent(pullback_low)
                            mtr.pullback_low_time = _fmt_time(premarket.loc[pullback_low_idx, "dt_et"])
                            mtr.pullback_retrace_pct = retrace_pct
                            return mtr, None
                        hod, hod_idx, armed_hod = session_high, None, None  # a fresh HOD must clear the true session high
                        pullback_seen = False
                        pullback_low, pullback_low_idx = math.inf, None
                        running_low = low
                        continue

                # Entry trigger LEVEL for the active entry_mode (config.entry_mode).
                # Every mode adds entry_offset and then runs the identical gate
                # stack below; only which price break arms the entry differs.
                #   hod_break      -> the armed HOD (a confirmed new-high breakout)
                #   dip_swing_high -> highest high since the pullback low
                #   dip_trendline  -> descending pivot line projected to this bar
                #                     (fallback: swing-high), floored at the dip low
                #   dip_ema_reclaim-> the reclaim indicator's last confirmed value,
                #                     only once the dip has pierced below it
                # The dip_* modes fire below the HOD, so they need a formed pullback
                # low that is strictly earlier than this bar (never on the very bar
                # that just printed a new low). base None -> no entry level this bar
                # yet; skip it without counting a trigger touch.
                base: float | None = None
                if s.entry_mode == "hod_break":
                    base = armed_hod
                elif pullback_low_idx is not None and pullback_low_idx < i:
                    if s.entry_mode == "dip_swing_high":
                        base = _swing_high_level(premarket, pullback_low_idx, i)
                    elif s.entry_mode == "dip_red_break":
                        base = _last_red_high(premarket, hod_idx, i, ceiling=armed_hod)
                    elif s.entry_mode == "dip_trendline":
                        base = _dip_trendline_level(premarket, hod_idx, i)
                        if base is None:
                            base = _swing_high_level(premarket, pullback_low_idx, i)
                        if base is not None:
                            base = max(base, pullback_low)  # resistance can't sit below the dip
                    else:  # dip_ema_reclaim
                        ind_val = _causal_indicator_value(one_min, row["dt_et"], s.dip_reclaim_indicator)
                        if ind_val is not None and _retrace_breached_indicator(premarket.iloc[hod_idx:i], one_min, s.dip_reclaim_indicator):
                            base = ind_val
                    # A dip / reversal entry is by definition AT OR BELOW the HOD it
                    # breaks toward -- never above it. Capping every dip mode's trigger
                    # at the armed HOD stops a post-pullback run-up from floating the
                    # entry up above the setup's own high (HIND 2025-09-17: entered
                    # 16.80 on a frozen 9.97 HOD). When price has already gapped past
                    # the HOD, the capped fill then exceeds entry_limit and is rejected
                    # (STALE_HOD) rather than chasing the top.
                    if base is not None:
                        base = min(base, armed_hod)
                if base is None:
                    continue

                trigger = _round_cent(base + s.entry_offset)
                entry_limit = _round_cent(trigger + s.entry_limit_slippage_cents)

                # Bars comprising this specific breakout: from its HOD, through the
                # pullback, to the current bar (used for the red/green volume check
                # below). Not the same as anchor_price (the premarket open, used for
                # gain%) or rip_low (the move's launch low, used for the retrace calc).
                move_bars = premarket.iloc[hod_idx:i + 1]

                # Observer hook (byte-identical, decision-free): report the setup
                # ARMED at this bar -- trigger + the frozen structure the live path
                # needs to watch the trigger tick-by-tick and later call
                # evaluate_touch on the crossing print. Fires whether or not this
                # bar crosses; the live path keeps the latest. Never influences the
                # backtest (a pure callback).
                if on_armed is not None:
                    on_armed(ArmedSetup(trigger=trigger, entry_limit=entry_limit,
                                        hod_idx=hod_idx, armed_hod=armed_hod,
                                        pullback_low=pullback_low, pullback_low_idx=pullback_low_idx,
                                        armed_bar_index=i, prior_win_entry=prior_win_entry))

                if high >= trigger:
                    mtr.trigger_touches += 1

                    decision = evaluate_touch(
                        ticker=ticker, i=i, row=row, high=high, low=low, open_=open_,
                        trigger=trigger, entry_limit=entry_limit, premarket=premarket,
                        one_min=one_min, move_bars=move_bars, hod_idx=hod_idx,
                        pullback_low=pullback_low, pullback_low_idx=pullback_low_idx,
                        armed_hod=armed_hod, s=s, prior_win_entry=prior_win_entry,
                        earliest_t=earliest_t, latest_t=latest_t, tick_fetcher=tick_fetcher,
                        bid_ask_fetcher=bid_ask_fetcher, latency=latency, bar_len=bar_len)
                    # entry_fill_source is telemetry set during fill resolution, so it
                    # is recorded for a fire AND for the post-fill rejects (fill limit /
                    # spread / risk), exactly as the inline code did -- before the stale.
                    if decision.entry_fill_source is not None:
                        mtr.entry_fill_source = decision.entry_fill_source
                    if decision.kind == "defer":
                        if decision.counter:
                            setattr(mtr, decision.counter, getattr(mtr, decision.counter) + 1)
                        continue
                    if decision.kind == "reject":
                        res = _stale_hod_break(decision.counter, decision.reason)
                        if res is not None:
                            return res
                        continue
                    # fire: unpack the entry the gate stack produced
                    fill = decision.fill
                    fill_time = decision.fill_time
                    entry_bar_pos = decision.entry_bar_pos
                    stop = decision.stop
                    target = decision.target
                    risk_per_share = decision.risk_per_share
                    entry_ticks = decision.entry_ticks   # for the exit sim's reuse
                    mtr.trigger_time = _fmt_time(row["dt_et"])
                    mtr.trigger_price = trigger
                    # entry_time is the actual fill time (tick-precise, latency-shifted
                    # when ticks resolved it); trigger_time is when the bar-level signal
                    # fired. They're the same only when the bar-based fallback was used.
                    mtr.entry_time = _fmt_time(fill_time) if fill_time is not None else mtr.trigger_time
                    mtr.entry_price = fill
                    mtr.volume_at_entry = decision.volume_at_entry
                    mtr.fill_price = fill
                    mtr.entry_limit_price = entry_limit
                    mtr.slippage = _round_cent(fill - trigger)
                    mtr.target_price = target
                    mtr.stop_price = stop
                    mtr.premarket_high = _round_cent(armed_hod)
                    mtr.stock_price = _round_cent(armed_hod)
                    mtr.premarket_high_time = _fmt_time(premarket.loc[hod_idx, "dt_et"])
                    mtr.premarket_gain_pct = (armed_hod - anchor_price) / anchor_price * 100.0
                    mtr.rip_low = _round_cent(rip_low)
                    mtr.pullback_low = _round_cent(pullback_low)
                    mtr.pullback_low_time = _fmt_time(premarket.loc[pullback_low_idx, "dt_et"])
                    mtr.pullback_retrace_pct = (armed_hod - pullback_low) / (armed_hod - rip_low) * 100.0

                    # Exit simulation runs from the ENTRY bar forward. With close
                    # confirmation on, that's the bar after the confirmation bar
                    # (entry_bar_pos = i + 1); otherwise it's the touch bar itself.
                    after = premarket.iloc[entry_bar_pos:].copy()
                    max_high = fill
                    min_low = fill
                    last_bar_pos = entry_bar_pos  # last bar position the exit simulation actually consumed
                    # Exit ladder state. With scale_out off the first target sell
                    # closes the whole position -- the original single-exit behavior.
                    # With it on, each target hit banks sell_frac of the REMAINING
                    # shares, the stop ratchets to scale_out_stop_offset BELOW the hit
                    # target (a stop AT it would fill on the very next print -- price
                    # is already trading right there), and the target moves up by
                    # scale_out_step. Sell number scale_out_max_sells liquidates the
                    # remainder; a ratcheted stop or session end may end it earlier.
                    sell_frac = s.scale_out_sell_pct / 100.0
                    cur_stop, cur_target = stop, target
                    remaining = 1.0
                    legs: list[dict] = []
                    # 1-minute trailing manager (exit_mode "candle_trail_1min"): the
                    # stop trails up to the prior COMPLETED 1-minute candle's low, and
                    # the first fat-green 1-minute close DROPS the fixed target (it
                    # becomes +inf, so only the trailing stop / session end can exit).
                    # The scale-out ladder is disabled in this mode.
                    candle_trail = s.exit_mode == "candle_trail_1min"
                    use_scale = s.scale_out and not candle_trail
                    # Runner engine: on a target hit with no prior fat green, sell this
                    # fraction and let the rest run (1.0 = all-out at target, no runner).
                    trail_partial = s.trail_partial_at_target if candle_trail else 1.0
                    target_active = True
                    entry_minute = pd.Timestamp(after.iloc[0]["dt_et"]).floor("min") if len(after) else None
                    folded_upto = entry_minute  # next minute-start not yet folded into the trail

                    def bank_leg(kind: str, frac: float, price: float, when) -> None:
                        legs.append({"kind": kind, "frac": round(frac, 6), "price": _round_cent(price), "time": _fmt_time(when)})

                    for j, ex in after.iterrows():
                        if remaining <= 1e-9:
                            break
                        last_bar_pos = j
                        bh, bl = float(ex["high"]), float(ex["low"])
                        max_high = max(max_high, bh)
                        min_low = min(min_low, bl)
                        if candle_trail and folded_upto is not None:
                            # Fold every 1-minute candle that has fully completed
                            # before this bar's minute: trail the stop up to its low
                            # (never down), and drop the target on the first fat green
                            # (excluding the partial entry minute).
                            this_minute = pd.Timestamp(ex["dt_et"]).floor("min")
                            while folded_upto < this_minute:
                                info = minute_info.get(folded_upto)
                                if info is not None:
                                    lo, fatgreen = info
                                    cur_stop = max(cur_stop, _round_cent(lo))
                                    if fatgreen and folded_upto > entry_minute:
                                        target_active = False
                                folded_upto = folded_upto + pd.Timedelta(minutes=1)
                            cur_target = target if target_active else float("inf")
                        touched_stop = bl <= cur_stop
                        touched_target = bh >= cur_target
                        if not (touched_stop or touched_target):
                            continue

                        # Every candle that breaks through the stop/target level gets
                        # resolved from its real trade ticks when available -- not just
                        # the rarer case where a single bar touches both. This finds the
                        # actual traded price at the crossing (so a gapped-through stop
                        # shows real slippage, not an idealized exact fill), not merely
                        # which level came first.
                        bar_ambiguous = touched_stop and touched_target
                        if bar_ambiguous and not mtr.path_ambiguous:
                            mtr.path_ambiguous = True
                            mtr.ambiguous_bar_time = _fmt_time(ex["dt_et"])
                        note = None
                        tick_rows = None
                        if tick_fetcher is not None:
                            try:
                                # Reuse the entry bar's already-fetched ticks instead of
                                # re-requesting the same window when this is that same bar.
                                if j == i and entry_ticks is not None:
                                    ticks = entry_ticks
                                else:
                                    bar_start = pd.Timestamp(ex["dt_et"])
                                    ticks = tick_fetcher(ticker, bar_start.to_pydatetime(), (bar_start + bar_len).to_pydatetime())
                                tick_rows = _sorted_tick_lists(ticks)
                            except Exception as te:
                                tick_rows = None
                                note = f"tick fetch failed: {te!r}"

                        banked_from_ticks = False
                        if tick_rows is not None:
                            times_l, prices_l = tick_rows
                            # On the entry bar, ticks before the trigger crossed
                            # predate the fill and must not count as exits.
                            scan_idx = 0
                            if j == i:
                                fill_idx = next((k for k, p in enumerate(prices_l) if p >= trigger), None)
                                scan_idx = None if fill_idx is None else fill_idx + 1
                            while scan_idx is not None and remaining > 1e-9:
                                hit = _first_touch_indexed(times_l, prices_l, scan_idx, cur_stop, cur_target, latency)
                                if hit is None:
                                    break
                                kind, exit_ts, exit_fp, k_idx = hit
                                banked_from_ticks = True
                                mtr.exit_fill_source = "ticks"
                                if bar_ambiguous and mtr.ambiguity_resolution is None:
                                    mtr.ambiguity_resolution = "ticks"
                                mtr.exit_time = _fmt_time(exit_ts)
                                if kind == "LOSS":
                                    bank_leg("stop", remaining, exit_fp, exit_ts)
                                    remaining = 0.0
                                    break
                                if candle_trail and target_active and trail_partial < 1.0 - 1e-9:
                                    # Runner: bank the partial at the 1:1, then drop the
                                    # target, jump the remainder's stop to breakeven, and
                                    # let it trail out from here.
                                    frac = remaining * trail_partial
                                    bank_leg("target", frac, exit_fp, exit_ts)
                                    remaining -= frac
                                    target_active = False
                                    cur_target = float("inf")
                                    cur_stop = max(cur_stop, fill)
                                    scan_idx = k_idx + 1
                                    continue
                                last_sell = not use_scale or mtr.scale_outs + 1 >= s.scale_out_max_sells
                                frac = remaining if last_sell else remaining * sell_frac
                                bank_leg("target", frac, exit_fp, exit_ts)
                                remaining -= frac
                                if use_scale:
                                    mtr.scale_outs += 1
                                if remaining > 1e-9:
                                    # Scale out: the ladder keeps scanning this same
                                    # bar's later ticks against the raised levels.
                                    cur_stop = _round_cent(cur_target - s.scale_out_stop_offset)
                                    cur_target = _round_cent(cur_target + s.scale_out_step)
                                    scan_idx = k_idx + 1
                        if banked_from_ticks:
                            continue
                        if tick_fetcher is not None and note is None:
                            note = "neither stop nor target traded in the tick stream"

                        # Bar-based fallback: no ticks (or they didn't confirm a level
                        # traded). Levels ratchet pessimistically WITHIN the bar: after
                        # a banked target the same bar's low is re-checked against the
                        # raised stop -- OHLC can't order the prints, so the remainder
                        # is assumed stopped rather than granted the best path.
                        mtr.exit_fill_source = "bar"
                        while remaining > 1e-9:
                            t_stop = bl <= cur_stop
                            t_target = bh >= cur_target
                            if not (t_stop or t_target):
                                break
                            if t_stop and t_target:
                                if s.intrabar_policy == "conservative":
                                    # Only when explicitly configured: an ambiguous bar
                                    # is never an automatic loss under any other policy.
                                    outcome = "LOSS"
                                    res_kind = "conservative"
                                else:
                                    # "ticks" (with ticks unavailable) and "heuristic"
                                    # both fall back to the candle-path guess.
                                    outcome = resolve_intrabar(
                                        open_=float(ex["open"]),
                                        high=bh,
                                        low=bl,
                                        close=float(ex["close"]),
                                        stop=cur_stop,
                                        target=cur_target,
                                    )
                                    res_kind = "heuristic"
                                if mtr.path_ambiguous and mtr.ambiguity_resolution is None:
                                    mtr.ambiguity_resolution = res_kind
                                if note is not None and not mtr.reason:
                                    mtr.reason = f"Ambiguous bar: {note}; resolved via {res_kind} policy."
                            else:
                                outcome = "LOSS" if t_stop else "WIN"
                                if note is not None and not mtr.reason:
                                    mtr.reason = f"Exit bar: {note}; used bar-based ({outcome}) fill."
                            mtr.exit_time = _fmt_time(ex["dt_et"])
                            if outcome == "LOSS":
                                bank_leg("stop", remaining, cur_stop, ex["dt_et"])
                                remaining = 0.0
                            elif candle_trail and target_active and trail_partial < 1.0 - 1e-9:
                                # Runner (bar fallback): bank the partial at the 1:1,
                                # drop the target, remainder rides breakeven+ on the trail.
                                frac = remaining * trail_partial
                                bank_leg("target", frac, cur_target, ex["dt_et"])
                                remaining -= frac
                                target_active = False
                                cur_target = float("inf")
                                cur_stop = max(cur_stop, fill)
                                break
                            else:
                                last_sell = not use_scale or mtr.scale_outs + 1 >= s.scale_out_max_sells
                                frac = remaining if last_sell else remaining * sell_frac
                                bank_leg("target", frac, cur_target, ex["dt_et"])
                                remaining -= frac
                                if use_scale:
                                    mtr.scale_outs += 1
                                if remaining > 1e-9:
                                    cur_stop = _round_cent(cur_target - s.scale_out_stop_offset)
                                    cur_target = _round_cent(cur_target + s.scale_out_step)

                    plain_session_exit = False
                    if remaining > 1e-9:
                        last = after.iloc[-1]
                        # A session-end exit is a mark at the close, not a level
                        # crossing, but it's still a real fill and deserves the same
                        # tick-quality price as every other exit: the actual last
                        # trade print in the closing bar, not just the bar's OHLC
                        # aggregate close. This was the one exit path that never
                        # attempted a tick fetch at all (every stop/target touch
                        # already does).
                        exit_fp = float(last["close"])
                        exit_src = "bar"
                        if tick_fetcher is not None:
                            try:
                                bar_start = pd.Timestamp(last["dt_et"])
                                closing_ticks = tick_fetcher(ticker, bar_start.to_pydatetime(), (bar_start + bar_len).to_pydatetime())
                                tick_rows = _sorted_tick_lists(closing_ticks)
                            except Exception:
                                tick_rows = None
                            if tick_rows is not None:
                                _, prices_l = tick_rows
                                if prices_l:
                                    exit_fp = _round_cent(prices_l[-1])
                                    exit_src = "ticks"
                        bank_leg("session_end", remaining, exit_fp, last["dt_et"])
                        mtr.exit_fill_source = exit_src
                        mtr.exit_time = _fmt_time(last["dt_et"])
                        if len(legs) == 1:
                            # Nothing ever touched: the classic exit-at-open case.
                            plain_session_exit = True
                            mtr.result = Result.EXIT_AT_OPEN.value if s.exit_at_regular_open else Result.NO_TRIGGER.value
                            mtr.reason = "Neither target nor stop hit before regular open; exited at last premarket close."
                        elif not mtr.reason:
                            mtr.reason = f"Scaled out {mtr.scale_outs}x; remainder exited at last premarket close."
                    elif s.scale_out and mtr.scale_outs > 0 and not mtr.reason:
                        last_leg = legs[-1]
                        if last_leg["kind"] == "target":
                            mtr.reason = f"Scaled out {mtr.scale_outs}x; ladder completed at {last_leg['price']:.2f}."
                        else:
                            mtr.reason = f"Scaled out {mtr.scale_outs}x; remainder stopped at {last_leg['price']:.2f}."
                    # No more session left after an exit at the last bar; otherwise the
                    # next move's hunt (if allowed) resumes right after this exit bar.
                    resume_idx = None if remaining > 1e-9 else (last_bar_pos + 1 if last_bar_pos + 1 < len(premarket) else None)

                    weighted_exit = sum(leg["frac"] * leg["price"] for leg in legs)
                    mtr.exit_price = _round_cent(weighted_exit)
                    mtr.exit_legs = legs
                    mtr.mfe = _round_cent(max_high - fill)
                    mtr.mae = _round_cent(fill - min_low)
                    mtr.pnl_per_share = _round_cent((weighted_exit - fill) - (2 * r.fees_per_share))
                    mtr.pnl_total = _round_cent(mtr.pnl_per_share * r.shares)
                    mtr.pnl_pct = round(mtr.pnl_per_share / fill * 100.0, 3) if fill else None
                    mtr.r_multiple = round(mtr.pnl_per_share / risk_per_share, 3) if risk_per_share else None
                    if not plain_session_exit:
                        # Classified by the REALIZED P&L, not by which level nominally
                        # triggered the exit ("this was a stop leg" != "this lost money").
                        # A stop crossing still has to race to the market
                        # (order_latency_seconds): the crossing print can be well past
                        # the stop level while the actual delayed fill lands back at
                        # breakeven or even in profit after a sharp reversal. Labeling
                        # that a LOSS just because a stop-kind leg fired produced
                        # $0.00-P&L trades reported as losses (e.g. SPRC 2026-06-04:
                        # entry 14.90 -> exit 14.90 via a delayed stop fill, labeled
                        # LOSS despite zero realized P&L). >= 0 treats an exact
                        # breakeven as not-a-loss; there's no dedicated scratch/
                        # breakeven Result value, and introducing one would ripple
                        # through every win-rate/EV stat downstream.
                        mtr.result = Result.WIN.value if mtr.pnl_per_share >= 0 else Result.LOSS.value
                    if candle_trail:
                        # Runner telemetry + a descriptive reason for the log/HTML.
                        mtr.trail_activated = not target_active
                        mtr.trail_stop_final = _round_cent(cur_stop)
                        last_kind = legs[-1]["kind"] if legs else ""
                        if not target_active:
                            where = "held to the open" if last_kind == "session_end" else f"trailed out at {mtr.trail_stop_final:.2f}"
                            mtr.reason = f"Runner: fat-green dropped the 1:1 target; {where}."
                        elif last_kind == "target":
                            mtr.reason = "Base hit: 1:1 target banked (no fat-green runner)."
                        elif last_kind == "stop":
                            mtr.reason = f"Trailing stop hit at {mtr.trail_stop_final:.2f} before any fat-green runner."
                    return mtr, resume_idx

        # Ran through the remaining premarket bars with no valid triggered entry --
        # relative to THIS scan's own vantage (start_i onward), not the whole session,
        # so a later move's failed hunt isn't judged against an earlier move's peak.
        remaining_bars = premarket.iloc[start_i:]
        max_gain = (float(remaining_bars["high"].max()) - anchor_price) / anchor_price * 100.0
        mtr.premarket_high = _round_cent(float(remaining_bars["high"].max()))
        mtr.stock_price = mtr.premarket_high
        max_idx = remaining_bars["high"].idxmax()
        mtr.premarket_high_time = _fmt_time(premarket.loc[max_idx, "dt_et"])
        mtr.premarket_gain_pct = max_gain
        if max_gain < s.min_premarket_gain_pct:
            mtr.result = Result.DID_NOT_QUALIFY.value
            mtr.disqualify_reason = "gain_pct"
            mtr.reason = f"Max premarket gain (from premarket open) was {max_gain:.1f}%, below {s.min_premarket_gain_pct:.1f}%."
        else:
            mtr.result = Result.NO_TRIGGER.value
            mtr.reason = "Qualified, but no valid first-pullback HOD break triggered."
        return mtr, None

    trades: list[TradeResult] = []
    next_start = 0

    def _may_continue() -> tuple[bool, float | None]:
        """May another move be hunted, and (second value) is it a win-continuation
        that must clear the continuation VWAP gate? Two independent grants:
        max_trades_per_day re-hunts after ANY completed trade; trade_winners_qty
        extends the day only while the last trade keeps WINNING -- a loss ends it."""
        n = len(trades)
        if n == 0:
            return True, None
        last = trades[-1]
        won = last.result == Result.WIN.value
        anchor = last.entry_price if (won and s.continuation_vwap_above_entry_pct is not None) else None
        if n < max(1, s.max_trades_per_day):
            return True, anchor
        if won and n < max(1, s.trade_winners_qty):
            return True, anchor
        return False, None

    while next_start is not None:
        allowed, prior_win_entry = _may_continue()
        if not allowed:
            break
        try:
            mtr, next_start = _scan(next_start, prior_win_entry)
        except Exception as e:
            if trades:
                # A trade (or more) already completed this day; a later move's
                # hunt blowing up must not discard them -- just stop looking.
                break
            mtr = _seeded_result()
            mtr.result = Result.ERROR.value
            mtr.reason = repr(e)
            trades.append(mtr)
            break
        if mtr.entry_price is None:
            if not trades:
                # No trade at all today: this informational result (DID_NOT_QUALIFY/
                # INVALID_RETRACE/NO_TRIGGER/...) is the whole answer, exactly as a
                # single-move backtest_one always returned it.
                mtr.setup_index = 1
                trades.append(mtr)
            else:
                # Post-trade informational results are discarded (as always), but
                # keep the win-continuation diagnostic visible on the day's last
                # trade row -- otherwise a VWAP-blocked continuation would leave
                # no trace anywhere in the CSV/summary.
                trades[-1].skipped_continuation_vwap += mtr.skipped_continuation_vwap
            break
        mtr.setup_index = len(trades) + 1
        trades.append(mtr)
    return trades
