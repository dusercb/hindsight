"""Symmetrical-pennant strategy: a SEPARATE, bidirectional setup that trades
whichever side breaks -- LONG on an upside break of the descending resistance
line, SHORT on a downside break of the ascending support line. Kept out of
strategy.py so the validated long-only HOD-first-pullback engine is untouched.

This file has two halves:
  1. Pure pennant GEOMETRY (pivots, line fits, convergence) -- trivially testable.
  2. `backtest_wedge`, the direction-aware trade lifecycle (qualification reused
     from strategy thresholds, a volatility-persistence gate, the bidirectional
     break, a short-side borrow gate, and a direction-aware fixed-R exit).

Direction is explicit everywhere: `direction="short"` mirrors the sign of P&L /
R / MFE / MAE and flips which way the stop and target sit. Sizing (whole shares,
the risk/notional/volume caps, fees, compounding) is applied later by
engine.apply_position_sizing, exactly as for the dip strategy.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd

from .config import AppConfig, parse_hhmm
from .models import Result, TradeResult
from .strategy import (
    add_indicators, add_rvol, build_1m_bars, first_price_at_or_above,
    previous_close, resolve_intrabar, _bar_seconds, _delayed_fill, _fmt_time,
    _round_cent, _sorted_tick_lists,
)


def pivot_positions(values: list[float], strength: int, kind: str) -> list[int]:
    """Indices of k-strength pivots: a 'high' whose value is >= all `strength`
    neighbours on each side, or a 'low' that is <= them. strength=1 is the plain
    3-bar pivot. Endpoints (no full neighbourhood) are never pivots."""
    n = len(values)
    out: list[int] = []
    for i in range(strength, n - strength):
        window = values[i - strength:i + strength + 1]
        if kind == "high" and values[i] >= max(window):
            out.append(i)
        elif kind == "low" and values[i] <= min(window):
            out.append(i)
    return out


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Least-squares (slope, intercept) through the points, or None when the x's
    are degenerate (all identical). Slope/intercept are in the x-coordinate given
    (bar positions), so a fitted line projects directly to any bar index."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Theil-Sen robust line: slope = MEDIAN of all pairwise slopes, intercept =
    median of (y - slope*x). Used for the pennant trendlines instead of least
    squares because one violent flush/spike pivot (routine on these names) drags an
    LSQ slope far from the line a trader would draw; the median shrugs it off.
    Deterministic, and O(k^2) over at most a few dozen pivots."""
    n = len(xs)
    if n < 2:
        return None
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(n) for j in range(i + 1, n) if xs[j] != xs[i]
    ]
    if not slopes:
        return None
    slopes.sort()
    slope = slopes[len(slopes) // 2] if len(slopes) % 2 else 0.5 * (slopes[len(slopes) // 2 - 1] + slopes[len(slopes) // 2])
    resid = sorted(y - slope * x for x, y in zip(xs, ys))
    intercept = resid[len(resid) // 2] if len(resid) % 2 else 0.5 * (resid[len(resid) // 2 - 1] + resid[len(resid) // 2])
    return slope, intercept


@dataclass(frozen=True)
class Pennant:
    """A fitted symmetrical pennant over the TIME window [x0, x1] (x = seconds, the
    caller's clock): a DESCENDING resistance line and an ASCENDING support line that
    converge, each shifted outward so it CONTAINS every bar in the window -- the
    resistance sits on/above every high, the support on/below every low, exactly the
    trendline a trader draws (not a regression midline through the pivots). Fitting
    in time (not bar index) matters because premarket bars with no trades are simply
    missing from the frame: an index-space line redrawn between two timestamps bends
    away from the prices wherever bars are absent."""
    x0: float
    x1: float
    slope_r: float
    intercept_r: float
    slope_s: float
    intercept_s: float
    touches_r: int
    touches_s: int
    convergence_pct: float

    def resistance_at(self, x: float) -> float:
        return self.slope_r * x + self.intercept_r

    def support_at(self, x: float) -> float:
        return self.slope_s * x + self.intercept_s


def detect_pennant(xs: list[float], highs: list[float], lows: list[float], *,
                   strength: int, min_touches: int, touch_tol_frac: float,
                   min_convergence_pct: float, max_convergence_pct: float = 100.0,
                   containment_outliers: int = 1) -> Pennant | None:
    """Fit a symmetrical pennant over one window of bars (xs = each bar's time in
    seconds on the caller's clock; the window should start at the pole top / local
    high the flag hangs from). The lines are the ones a TRADER would draw:

      * slopes come from THEIL-SEN (median of pairwise pivot slopes), so one
        violent flush/spike pivot can't tilt a line the way least squares does;
      * each line is then shifted OUTWARD to sit on price, but allowing up to
        `containment_outliers` bars to poke through -- a rogue wick through an
        otherwise clean trendline is accepted, exactly as in manual charting,
        rather than letting that wick own the whole line;
      * a pivot "touches" its line when within touch_tol_frac of the window's
        MEDIAN bar range (spike-proof, unlike the total high-low range); each
        line needs >= min_touches touches to count as a real trendline;
      * the channel must be open at the window end, and its NARROWING must be
        between min_convergence_pct (a real squeeze) and max_convergence_pct
        (not already past the apex -- a spent coil isn't tradeable).
    """
    n = len(xs)
    if n < 5 or not (len(xs) == len(highs) == len(lows)):
        return None
    piv_h = pivot_positions(highs, strength, "high")
    piv_l = pivot_positions(lows, strength, "low")
    if len(piv_h) < min_touches or len(piv_l) < min_touches:
        return None
    res = theil_sen([xs[k] for k in piv_h], [highs[k] for k in piv_h])
    sup = theil_sen([xs[k] for k in piv_l], [lows[k] for k in piv_l])
    if res is None or sup is None:
        return None
    slope_r, int_r = res
    slope_s, int_s = sup
    # Symmetrical: resistance descends, support rises (shifting preserves slope).
    if not (slope_r < 0 < slope_s):
        return None
    # Outward shift to the (containment_outliers+1)-th most extreme residual: the
    # line sits ON the price envelope while up to that many rogue wicks poke
    # through instead of dictating the line.
    k_out = max(0, min(containment_outliers, n - 1))
    resid_h = sorted((h - (slope_r * x + int_r) for x, h in zip(xs, highs)), reverse=True)
    resid_l = sorted(low - (slope_s * x + int_s) for x, low in zip(xs, lows))
    int_r += resid_h[k_out]
    int_s += resid_l[k_out]
    ranges = sorted(h - low for h, low in zip(highs, lows))
    median_bar_range = ranges[len(ranges) // 2]
    if median_bar_range <= 0:
        return None
    tol = touch_tol_frac * median_bar_range
    touches_r = sum(1 for k in piv_h if abs((slope_r * xs[k] + int_r) - highs[k]) <= tol)
    touches_s = sum(1 for k in piv_l if abs(lows[k] - (slope_s * xs[k] + int_s)) <= tol)
    if touches_r < min_touches or touches_s < min_touches:
        return None
    x0, x1 = xs[0], xs[-1]
    width_0 = (slope_r * x0 + int_r) - (slope_s * x0 + int_s)
    width_1 = (slope_r * x1 + int_r) - (slope_s * x1 + int_s)
    if width_0 <= 0 or width_1 <= 0:
        return None  # lines already crossed -> the coil is spent, not tradeable
    convergence = (width_0 - width_1) / width_0 * 100.0
    if not (min_convergence_pct <= convergence <= max_convergence_pct):
        return None
    return Pennant(x0=x0, x1=x1, slope_r=slope_r, intercept_r=int_r,
                   slope_s=slope_s, intercept_s=int_s,
                   touches_r=touches_r, touches_s=touches_s,
                   convergence_pct=round(convergence, 2))


# --- direction-aware trade lifecycle -------------------------------------------

DailyVolumeFetcher = Callable[[str, date], float | None]
TickFetcher = Callable[[str, datetime, datetime], "pd.DataFrame"]


def _first_price_at_or_below(ticks, level: float, latency=None) -> tuple[float, object] | None:
    """Short-entry mirror of strategy.first_price_at_or_above: the first trade tick
    at or BELOW `level` (a downside break), with its price and time. None when there
    are no ticks or the level never traded down to."""
    if ticks is None or len(ticks) == 0:
        return None
    ticks = ticks.sort_values("time", kind="stable")
    times = ticks["time"].tolist()
    prices = ticks["price"].tolist()
    idx = next((k for k, p in enumerate(prices) if p <= level), None)
    if idx is None:
        return None
    if latency:
        target_time = times[idx] + latency
        fill_idx = next((k for k in range(idx, len(times)) if times[k] >= target_time), len(times) - 1)
        return float(prices[fill_idx]), times[fill_idx]
    return float(prices[idx]), times[idx]


def _first_touch_directional_indexed(times, prices, start_idx, direction, stop, target, latency):
    """Direction-aware exit resolution from a bar's real ticks, applying
    order-submission latency to the legs that CANNOT rest in premarket (so their
    fill is the tape `latency` after the crossing -- i.e. real slippage), and the
    crossing print itself to the one leg that can rest:

      long  -> stop is a sell BELOW (can't rest)  -> latency;
               target is a sell ABOVE -> a resting limit -> crossing print, no latency.
      short -> stop is a buy ABOVE (can't rest)   -> latency;
               target is a buy BELOW (a cover, also a buy, can't rest) -> latency too.

    Returns ("LOSS"|"WIN", exit_time, exit_price, crossing_index) for whichever
    traded first, or None if neither level traded in this tick slice. The index is
    returned so the candle-trail runner can keep scanning the SAME bar's later
    ticks against updated levels after banking a partial (like the dip's
    _first_touch_indexed)."""
    for k in range(start_idx, len(prices)):
        p = prices[k]
        if direction == "long":
            if p <= stop:
                t, fp = _delayed_fill(times, prices, k, latency)
                return "LOSS", t, fp, k
            if p >= target:
                return "WIN", times[k], float(p), k   # resting sell limit: crossing print
        else:  # short
            if p >= stop:
                t, fp = _delayed_fill(times, prices, k, latency)
                return "LOSS", t, fp, k
            if p <= target:
                t, fp = _delayed_fill(times, prices, k, latency)
                return "WIN", t, fp, k                # cover is a buy: can't rest -> latency
    return None


def _first_touch_directional(times, prices, start_idx, direction, stop, target, latency):
    """The non-indexed wrapper (the bracket exit's original interface)."""
    hit = _first_touch_directional_indexed(times, prices, start_idx, direction, stop, target, latency)
    return None if hit is None else hit[:3]


def _resolve_bar(direction: str, open_, high, low, close, stop, target) -> str:
    """Which of stop/target a single bar hit FIRST, when it touched both, from the
    candle-shape path heuristic. Long uses strategy.resolve_intrabar directly; short
    mirrors it (stop sits ABOVE, target BELOW)."""
    if direction == "long":
        return resolve_intrabar(open_, high, low, close, stop, target)
    path = [open_, low, high, close] if close >= open_ else [open_, high, low, close]
    for price in path:
        if price >= stop:
            return "LOSS"
        if price <= target:
            return "WIN"
    return "LOSS"


def _recent_atr(one_min, upto_ts, window_min: float) -> float | None:
    """Mean 1-minute high-low range over the trailing `window_min` minutes, using
    only COMPLETED minutes strictly before upto_ts's minute (causal). None when no
    such bar exists."""
    upto_min = pd.Timestamp(upto_ts).floor("min")
    cutoff = upto_min - pd.Timedelta(minutes=window_min)
    win = one_min[(one_min["dt_et"] > cutoff) & (one_min["dt_et"] < upto_min)]
    if win.empty:
        return None
    return float((win["high"] - win["low"]).mean())


def _windowed_rvol(one_min, break_ts, window_min: float, daily_avg_volume: float | None) -> float | None:
    """The stock's trailing-window volume RATE as a multiple of its own norm:
    shares printed in the `window_min` minutes ending at the prior COMPLETED
    minute (causal, same convention as every other gate), divided by what this
    stock normally prints in that many minutes (daily_avg_volume / 390 RTH
    minutes). This is what 'the market is still looking' actually means -- unlike
    cumulative session RVOL, which only ever grows, so an early pumper keeps its
    5am badge all morning no matter how dead the tape has gone. None when there's
    no baseline or no completed minute yet."""
    if daily_avg_volume is None or daily_avg_volume <= 0:
        return None
    hi = pd.Timestamp(break_ts).floor("min") - pd.Timedelta(minutes=1)
    upto_hi = one_min[one_min["dt_et"] <= hi]
    if upto_hi.empty:
        return None
    cum_hi = float(upto_hi["cum_volume"].iloc[-1])
    lo = hi - pd.Timedelta(minutes=window_min)
    upto_lo = one_min[one_min["dt_et"] <= lo]
    cum_lo = float(upto_lo["cum_volume"].iloc[-1]) if not upto_lo.empty else 0.0
    per_minute_norm = daily_avg_volume / 390.0
    return (cum_hi - cum_lo) / (window_min * per_minute_norm)


def _persistence_ok(w, one_min, break_ts, rip_range: float, daily_avg_volume: float | None) -> tuple[bool, str]:
    """The attention-persistence gate: is the market STILL in this name as of the
    break, or is it a forgotten early pumper? 'rvol' -> the trailing
    persistence_window_min minutes must have printed >= min_sustained_rvol x the
    stock's own normal volume rate (declining coil volume passes -- textbook
    pennant -- but a dead tape does not); 'atr' -> recent 1-min ATR >=
    min_atr_pct_of_rip of the session's range so far (price still moves);
    'both' -> both. Returns (ok, reason-when-not)."""
    if w.persistence_mode in ("rvol", "both"):
        rate = _windowed_rvol(one_min, break_ts, w.persistence_window_min, daily_avg_volume)
        if rate is None or math.isnan(rate) or rate < w.min_sustained_rvol:
            return False, "recent volume rate below min_sustained_rvol"
    if w.persistence_mode in ("atr", "both"):
        atr = _recent_atr(one_min, break_ts, w.persistence_window_min)
        if atr is None or rip_range <= 0 or atr < w.min_atr_pct_of_rip * rip_range:
            return False, "recent ATR below rip fraction"
    return True, ""


@dataclass
class WedgeArmedSetup:
    """A pennant that is ARMED at a completed bar: both break triggers plus the
    frozen geometry evaluate_wedge_touch needs. The live path receives this via
    backtest_wedge(on_armed=...), watches long_trig/short_trig tick-by-tick, and
    on the crossing print calls evaluate_wedge_touch with the live forming bar --
    minimal-latency entry, same gate stack as the backtest.

    Freezing the geometry is what makes the live path CORRECT, not merely fast: a
    pennant re-fitted on a part-formed candle can pick the opposite break side
    (see complete_tf_frame in the bot). Arm on whole candles, then only ask "has
    this specific level been crossed" -- a causal question ticks can answer."""
    long_trig: float
    short_trig: float
    res_i: float
    sup_i: float
    anchor: int
    armed_bar_index: int
    # The fitted lines themselves. The caller's exit walk reads the pennant, and
    # carrying it (rather than only the two levels) keeps the armed setup a
    # complete description of the geometry the decision was made on.
    pen: object = None


@dataclass
class WedgeTouchDecision:
    """evaluate_wedge_touch's return. `kind` is one of:
      * "defer" -- no valid entry at this bar; the pennant stays armed for a later,
                   more convincing touch (every gate in the wedge defers rather
                   than condemning). `counter`, if set, is a TradeResult counter
                   the caller bumps; `borrow_fee_blocked` records a short refused
                   on borrow cost.
      * "fire"  -- a valid entry; the fill/stop/target fields are populated."""
    kind: str
    direction: str | None = None
    trigger: float | None = None
    fill: float | None = None
    fill_time: object = None
    stop: float | None = None
    target: float | None = None
    entry_limit: float | None = None
    risk_per_share: float | None = None
    entry_bar_pos: int | None = None
    entry_ticks: object = None
    entry_fill_source: str | None = None
    counter: str | None = None
    borrow_fee_blocked: float | None = None
    # The causal prior-minute row (and its rvol) this break was judged on. The
    # caller's exit walk reads both, so they travel with the decision rather than
    # being recomputed -- recomputing risks the two paths quietly disagreeing.
    mrow: object = None
    rvol_so_far: float | None = None


def arm_wedge(xs, highs, lows, i: int, w) -> WedgeArmedSetup | None:
    """Fit the pennant governing bar `i` and evaluate its two break levels AT that
    bar. None when no pennant governs it. Shared by the backtest scan loop and the
    live trigger-watch so both arm identically.

    Note what `i` may be: the detector only ever reads bars[anchor:i] -- STRICTLY
    before i -- and the lines are then evaluated at xs[i]. So the levels for a bar
    are fully determined before that bar prints, which is exactly what lets the
    live path arm a candle that is still forming and watch its levels tick by
    tick. Passing the forming bar's index is therefore correct, not a lookahead;
    passing the last COMPLETE index instead would silently use a stale point on a
    sloping line (the resistance descends, so the trigger would sit too high)."""
    w_lo = max(0, i - w.pivot_lookback_bars)
    if i <= w_lo:
        return None
    # Anchor the pennant at the POLE TOP: the highest high within the trailing
    # lookback. A flag hangs off its pole -- anchoring there keeps detection
    # consistent bar to bar (the window only changes when a new high prints)
    # instead of a sliding window re-fitting different lines every bar.
    anchor = max(range(w_lo, i), key=lambda k: highs[k])
    if i - anchor < w.min_pennant_bars:
        return None  # the coil hasn't lasted long enough to be a pattern yet
    pen = detect_pennant(xs[anchor:i], highs[anchor:i], lows[anchor:i],
                         strength=w.pivot_strength, min_touches=w.min_pivot_touches,
                         touch_tol_frac=w.touch_tolerance,
                         min_convergence_pct=w.min_convergence_pct,
                         max_convergence_pct=w.max_convergence_pct,
                         containment_outliers=w.containment_outliers)
    if pen is None:
        return None
    res_i, sup_i = pen.resistance_at(xs[i]), pen.support_at(xs[i])
    return WedgeArmedSetup(long_trig=_round_cent(res_i + w.breakout_offset),
                           short_trig=_round_cent(sup_i - w.breakout_offset),
                           res_i=res_i, sup_i=sup_i, anchor=anchor, armed_bar_index=i, pen=pen)


def evaluate_wedge_touch(*, ticker: str, i: int, row, n: int, premarket,
                         armed: WedgeArmedSetup, highs, lows, volumes,
                         one_min, first_price: float, s, w,
                         daily_avg_volume: float | None, borrow_fee: float | None,
                         tick_fetcher, latency, bar_len) -> WedgeTouchDecision:
    """The wedge's per-touch entry decision -- the exact gate stack, fill
    resolution and stop/target computation that used to live inline in the scan
    loop, lifted verbatim into a pure function so BOTH the backtest loop and the
    live path evaluate a break identically. No side effects: the caller owns the
    TradeResult recording and the exit walk. Behavior is byte-identical to the
    pre-extraction inline code (the 347-test suite is the contract)."""
    anchor, res_i, sup_i = armed.anchor, armed.res_i, armed.sup_i
    long_trig, short_trig = armed.long_trig, armed.short_trig
    high, low, open_ = float(row["high"]), float(row["low"]), float(row["open"])
    long_break = w.sides in ("both", "long") and high >= long_trig
    short_break = w.sides in ("both", "short") and low <= short_trig
    if not long_break and not short_break:
        return WedgeTouchDecision("defer")
    prefetched_ticks = None
    if long_break and short_break:
        # Both lines broke inside ONE bar. Bars alone cannot say which side
        # broke FIRST -- that ordering is future information at bar level --
        # so resolve it from the bar's real ticks when available: whichever
        # trigger actually crossed first wins. Only without ticks (or when
        # neither crossing shows on the tape) fall back to the larger-breach
        # heuristic, the same class of candle-shape guess as resolve_intrabar.
        if tick_fetcher is not None:
            try:
                bstart = pd.Timestamp(row["dt_et"])
                prefetched_ticks = tick_fetcher(ticker, bstart.to_pydatetime(), (bstart + bar_len).to_pydatetime())
            except Exception:
                prefetched_ticks = None
        first_side = None
        if prefetched_ticks is not None and len(prefetched_ticks):
            tick_prices = prefetched_ticks.sort_values("time", kind="stable")["price"].tolist()
            li = next((k for k, p in enumerate(tick_prices) if p >= long_trig), None)
            si = next((k for k, p in enumerate(tick_prices) if p <= short_trig), None)
            if li is not None and (si is None or li < si):
                first_side = "long"
            elif si is not None:
                first_side = "short"
        if first_side is None:
            first_side = "long" if (high - long_trig) >= (short_trig - low) else "short"
        if first_side == "long":
            short_break = False
        else:
            long_break = False
    direction = "long" if long_break else "short"
    trigger = long_trig if long_break else short_trig

    # Causal gain gate: the session-so-far move (running high through the
    # prior COMPLETED bar vs the premarket open) must already show the
    # qualifying gain AT this break -- the same per-touch convention the dip
    # engine uses. The full-session check in qualification above is only the
    # cheap necessary-condition short-circuit; entering here on a gain the
    # session only reaches later would be trading on future information.
    running_high = max(highs[:i])
    gain_so_far = (running_high - first_price) / first_price * 100.0
    if gain_so_far < s.min_premarket_gain_pct:
        # not a qualified mover YET; a later break may be
        return WedgeTouchDecision("defer", counter="skipped_gain_not_yet_confirmed")

    # Causal RVOL-so-far (prior completed minute), the same lookup the dip uses.
    bar_time = row["dt_et"].floor("min") - pd.Timedelta(minutes=1)
    mrow = one_min.loc[one_min["dt_et"] == bar_time]
    if mrow.empty:
        return WedgeTouchDecision("defer")   # can't assess this break yet -> defer
    rvol_so_far = mrow.iloc[0]["rvol"]
    if pd.isna(rvol_so_far) or rvol_so_far < s.min_rvol:
        return WedgeTouchDecision("defer")

    rip_range = max(highs[:i + 1]) - min(lows[:i + 1])
    ok, _preason = _persistence_ok(w, one_min, row["dt_et"], rip_range, daily_avg_volume)
    if not ok:
        return WedgeTouchDecision("defer")   # the market has moved on here; a later, livelier break may still qualify

    # Breakout-QUALITY gates (additive, default OFF; each DEFERS rather than
    # condemning the setup, so a later, more convincing touch of the same lines
    # can still fire). See WedgeConfig for the full rationale.
    if w.require_breakout_volume_expansion:
        coil_vols = volumes[anchor:i]
        coil_avg_vol = sum(coil_vols) / len(coil_vols) if coil_vols else 0.0
        if coil_avg_vol <= 0 or volumes[i] < w.breakout_volume_multiple * coil_avg_vol:
            return WedgeTouchDecision("defer")   # no real volume expansion on the break -- not yet convincing
    if w.require_vwap_alignment:
        vwap_now = mrow.iloc[0]["vwap"]
        if pd.isna(vwap_now) or (direction == "long" and trigger < vwap_now) or (direction == "short" and trigger > vwap_now):
            return WedgeTouchDecision("defer")   # breaking against the session VWAP -- not a trend-aligned break

    # Short-side borrow gate.
    if direction == "short" and w.max_borrow_fee_pct is not None:
        if borrow_fee is not None and borrow_fee > w.max_borrow_fee_pct:
            # un-shortable here; keep scanning (a long break may still come)
            return WedgeTouchDecision("defer", borrow_fee_blocked=borrow_fee)

    # Close-confirmed breakout (a SEPARATE wedge-only toggle -- see
    # WedgeConfig.require_breakout_close_confirm). A bar's high/low merely
    # touching the trigger is not enough: it must CLOSE beyond it, in the
    # favourable fraction of its own range -- a held break, not a rejected
    # wick/throw-over (the classic pennant failure mode). A weak/unconfirmed
    # close DEFERS (the setup stays alive for a later bar), the same
    # non-condemning treatment the dip strategy uses for its own
    # require_breakout_close_confirm.
    if w.require_breakout_close_confirm:
        close_ = float(row["close"])
        bar_range = high - low
        close_pos = (close_ - low) / bar_range if bar_range > 0 else 1.0
        if direction == "long":
            confirmed = close_ >= trigger and close_pos >= w.breakout_close_min_frac
        else:
            confirmed = close_ <= trigger and (1.0 - close_pos) >= w.breakout_close_min_frac
        if not confirmed:
            return WedgeTouchDecision("defer")   # weak/unconfirmed close: defer, keep the setup alive

    # Entry timing. Gates that read END-OF-BAR information (a confirmed close;
    # the break bar's total volume for the expansion gate) cannot honestly fill
    # INSIDE that same bar -- the information they conditioned on doesn't exist
    # until the bar completes, and filling intrabar off it would be a lookahead.
    # Any such gate therefore shifts the entry to the NEXT bar's open (a market-
    # on-open style fill), the treatment close-confirm always used. With both
    # gates off, the original tick-resolved intrabar trigger-cross fill applies.
    enter_on_next_open = w.require_breakout_close_confirm or w.require_breakout_volume_expansion
    entry_bar_pos = i
    if enter_on_next_open:
        if i + 1 >= n:
            return WedgeTouchDecision("defer")   # confirmed on the last bar; no next-bar open to enter on
        entry_bar_pos = i + 1

    entry_ticks = None
    if enter_on_next_open:
        # Confirmed-breakout fill: the gate resolved on the completed break
        # bar, so this is a market-on-open style entry on its successor, not a
        # trigger-cross. The exit sim stays tick-precise; only the entry skips
        # tick resolution here.
        entry_row = premarket.iloc[entry_bar_pos]
        fill = _round_cent(float(entry_row["open"]))
        fill_time = entry_row["dt_et"]
        entry_fill_source = "bar_confirm"
    else:
        if tick_fetcher is not None and prefetched_ticks is not None:
            entry_ticks = prefetched_ticks   # the double-break resolver already fetched this bar
        elif tick_fetcher is not None:
            try:
                bstart = pd.Timestamp(row["dt_et"])
                entry_ticks = tick_fetcher(ticker, bstart.to_pydatetime(), (bstart + bar_len).to_pydatetime())
            except Exception:
                entry_ticks = None
        if direction == "long":
            resolved = first_price_at_or_above(entry_ticks, trigger, latency=latency) if entry_ticks is not None else None
            fill = _round_cent(resolved[0]) if resolved else _round_cent(max(trigger, open_))
        else:
            resolved = _first_price_at_or_below(entry_ticks, trigger, latency=latency) if entry_ticks is not None else None
            fill = _round_cent(resolved[0]) if resolved else _round_cent(min(trigger, open_))
        fill_time = resolved[1] if resolved else row["dt_et"]
        entry_fill_source = "ticks" if (entry_ticks is not None and resolved) else "bar"

    if direction == "long":
        stop = _round_cent(sup_i)
        entry_limit = _round_cent(trigger + w.entry_limit_slippage_cents)
        if fill > entry_limit:
            return WedgeTouchDecision("defer")
    else:
        stop = _round_cent(res_i)
        entry_limit = _round_cent(trigger - w.entry_limit_slippage_cents)
        if fill < entry_limit:
            return WedgeTouchDecision("defer")

    risk_per_share = abs(fill - stop)
    if risk_per_share <= 0:
        return WedgeTouchDecision("defer")
    if w.min_risk_pct_of_price is not None and (risk_per_share / fill * 100.0) < w.min_risk_pct_of_price:
        return WedgeTouchDecision("defer")   # stop sits unrealistically close to the fill -- defer

    if w.target_mode == "measured_move":
        # The classical pennant target: the pole's height (its top -- the
        # anchor high -- minus the lowest low in the pole_lookback_bars before
        # it) projected from the breakout point, scaled by measured_move_fraction.
        pole_lo = max(0, anchor - w.pole_lookback_bars)
        pole_base = min(lows[pole_lo:anchor + 1])
        pole_height = highs[anchor] - pole_base
        if pole_height <= 0:
            return WedgeTouchDecision("defer")
        mm = w.measured_move_fraction * pole_height
        target = _round_cent(fill + mm) if direction == "long" else _round_cent(fill - mm)
    else:
        target = _round_cent(fill + w.target_r_multiple * risk_per_share) if direction == "long" \
            else _round_cent(fill - w.target_r_multiple * risk_per_share)

    return WedgeTouchDecision(
        "fire", direction=direction, trigger=trigger, fill=fill, fill_time=fill_time,
        stop=stop, target=target, entry_limit=entry_limit, risk_per_share=risk_per_share,
        entry_bar_pos=entry_bar_pos, entry_ticks=entry_ticks, entry_fill_source=entry_fill_source,
        mrow=mrow, rvol_so_far=rvol_so_far)


class _RetestExit:
    """Failed-retest early exit for the wedge (WedgeConfig.exit_retest_failure) --
    a way to LEAVE a working position instead of holding it into the open when the
    momentum quietly turns.

    Short: a swing low forms, price BOUNCES off it (>= retest_bounce_min_pct), then
    a RED candle comes back down and RETESTS the low (within retest_tolerance_pct)
    but does NOT break it -- the low held. The moment a LATER candle trades back
    above that attempt candle's HIGH, cover. Mirrored for a long (swing high, GREEN
    attempt candle, exit below its LOW). If price instead extends the trade (a new
    low for a short / new high for a long), any armed retest is discarded and the
    swing re-bases -- the position is still working, so we keep holding.

    Fed one COMPLETED candle at a time (premarket is already on the wedge's own
    timeframe here), and the exit can only fire on a candle strictly AFTER the
    attempt candle, so it is fully causal -- no peeking at a bar's own future.
    """

    def __init__(self, direction: str, entry_fill: float,
                 bounce_min_pct: float, retest_tol_pct: float):
        self.long = direction == "long"
        self.bounce = bounce_min_pct / 100.0
        self.tol = retest_tol_pct / 100.0
        self.best = entry_fill          # best extreme in the trade's favour so far
        self.counter = entry_fill       # furthest counter-move since `best` re-based
        self.bounced = False
        self.armed = None               # attempt candle's break level, once armed

    def step(self, o: float, h: float, l: float, c: float):
        """Advance one completed candle; return the exit price if a candle after
        the attempt has broken its level, else None."""
        if self.long:
            if self.armed is not None and l <= self.armed:
                return self.armed                         # broke the attempt low -> exit
            if h > self.best:                             # new high: trade extending
                self.best, self.counter, self.bounced, self.armed = h, h, False, None
                return None
            self.counter = min(self.counter, l)
            if not self.bounced and self.counter <= self.best * (1 - self.bounce):
                self.bounced = True
            if self.bounced and self.armed is None and c > o and h >= self.best * (1 - self.tol):
                self.armed = l                            # green candle retested the high but held
            return None
        # short
        if self.armed is not None and h >= self.armed:
            return self.armed                             # broke the attempt high -> exit
        if l < self.best:                                 # new low: trade extending
            self.best, self.counter, self.bounced, self.armed = l, l, False, None
            return None
        self.counter = max(self.counter, h)
        if not self.bounced and self.counter >= self.best * (1 + self.bounce):
            self.bounced = True
        if self.bounced and self.armed is None and c < o and l <= self.best * (1 + self.tol):
            self.armed = h                                # red candle retested the low but held
        return None


def backtest_wedge(ticker: str, day: date, bars, premarket, prev_rth, config: AppConfig,
                   float_shares: float | None = None, tick_fetcher: TickFetcher | None = None,
                   bid_ask_fetcher=None, daily_volume_fetcher: DailyVolumeFetcher | None = None,
                   float_fetcher=None, borrow_fee: float | None = None,
                   on_armed: Callable | None = None) -> list[TradeResult]:
    """Replay one ticker/day for the symmetrical-pennant strategy. Same call shape
    as strategy.backtest_all_trades so the engine can route to either. Qualification
    thresholds (price/float/rvol/gain/entry-window) come from config.strategy; the
    pennant geometry, persistence, borrow gate, and R-target come from config.wedge.
    Returns one TradeResult (a trade, or a terminal non-trade outcome)."""
    s, w, r = config.strategy, config.wedge, config.risk
    latency = pd.Timedelta(seconds=s.order_latency_seconds) if s.order_latency_seconds else None
    tr = TradeResult(ticker=ticker.upper(), date=day.isoformat(), result=Result.ERROR.value)
    tr.entry_mode = "wedge_break"
    tr.float_shares = float_shares
    tr.intrabar_policy = s.intrabar_policy
    try:
        earliest_t = time(*parse_hhmm(s.earliest_entry_time, "earliest_entry_time", "config")) if s.earliest_entry_time else None
        latest_t = time(*parse_hhmm(s.latest_entry_time, "latest_entry_time", "config")) if s.latest_entry_time else None
        tr.bars = int(len(premarket))
        if premarket.empty or len(premarket) < 5:
            tr.result = Result.INSUFFICIENT_DATA.value
            tr.reason = "No/too little premarket bar data returned by IBKR."
            return [tr]
        pc = previous_close(prev_rth)
        tr.prev_close = pc if pc and pc > 0 else None
        premarket = premarket.sort_values("dt_et").reset_index(drop=True)
        raw_premarket = premarket
        # The wedge scans its OWN timeframe (w.timeframe: 10s/1m/2m, A/B-toggleable,
        # independent of strategy.chart_timeframe) -- pennants are a visual pattern
        # and the candle size IS part of the pattern definition.
        bar_len = pd.Timedelta(seconds=_bar_seconds(config.data.bar_size))
        if w.timeframe in ("1m", "2m"):
            tf = 1 if w.timeframe == "1m" else 2
            premarket = build_1m_bars(premarket, f"{tf}min").sort_values("dt_et").reset_index(drop=True)
            bar_len = pd.Timedelta(minutes=tf)
        if len(premarket) < w.min_pennant_bars + 5:
            tr.result = Result.INSUFFICIENT_DATA.value
            tr.reason = f"Too few {w.timeframe} bars to form a pennant."
            return [tr]
        first_price = float(premarket.iloc[0]["open"])
        tr.premarket_open = first_price
        tr.stock_price = first_price
        tr.premarket_volume = int(premarket.get("volume", pd.Series(dtype=float)).fillna(0).sum()) if "volume" in premarket else None
        if s.min_price is not None and first_price < s.min_price:
            tr.result = Result.SKIPPED_PRICE_FILTER.value
            tr.reason = f"Opening price {first_price:.2f} below min_price {s.min_price:.2f}."
            return [tr]
        if s.max_price is not None and first_price > s.max_price:
            tr.result = Result.SKIPPED_PRICE_FILTER.value
            tr.reason = f"Opening price {first_price:.2f} above max_price {s.max_price:.2f}."
            return [tr]
        if (s.min_float is not None or s.max_float is not None):
            fl = float_shares if float_shares is not None else (float_fetcher(ticker) if float_fetcher else None)
            if fl is None:
                tr.result = Result.INSUFFICIENT_DATA.value
                tr.reason = "Float unavailable; cannot apply the configured float filter."
                return [tr]
            if (s.max_float is not None and fl > s.max_float) or (s.min_float is not None and fl < s.min_float):
                tr.result = Result.SKIPPED_FLOAT_FILTER.value
                tr.reason = f"Float {fl:,.0f} outside the configured band."
                return [tr]
        daily_avg_volume = daily_volume_fetcher(ticker, day) if daily_volume_fetcher is not None else None
        if daily_avg_volume is None:
            tr.result = Result.INSUFFICIENT_DATA.value
            tr.reason = "No daily-volume baseline for RVOL."
            return [tr]
        tr.daily_avg_volume = daily_avg_volume
        one_min = add_indicators(add_rvol(build_1m_bars(raw_premarket), daily_avg_volume), s.macd_fast, s.macd_slow, s.macd_signal)
        tr.max_rvol_seen = float(one_min["rvol"].max()) if one_min["rvol"].notna().any() else None
        if tr.max_rvol_seen is None or tr.max_rvol_seen < s.min_rvol:
            tr.result = Result.DID_NOT_QUALIFY.value
            tr.disqualify_reason = "rvol"
            tr.reason = f"Premarket RVOL never reached {s.min_rvol:.1f}x."
            return [tr]
        # Full-session gain check -- a cheap NECESSARY-CONDITION short-circuit only
        # (a running max only grows, so "never reached the gain" is decidable from
        # the whole session; same ledger pattern as the session-wide RVOL check).
        # The entry itself re-checks the gain CAUSALLY at each break attempt below
        # -- without that re-check a break could trade on a gain the session only
        # reaches later, i.e. on future information.
        premarket_high = float(premarket["high"].max())
        gain_pct = (premarket_high - first_price) / first_price * 100.0
        tr.premarket_high = _round_cent(premarket_high)
        tr.premarket_gain_pct = gain_pct
        if gain_pct < s.min_premarket_gain_pct:
            tr.result = Result.DID_NOT_QUALIFY.value
            tr.disqualify_reason = "gain_pct"
            tr.reason = f"Premarket gain {gain_pct:.1f}% below {s.min_premarket_gain_pct:.1f}%."
            return [tr]
    except Exception as e:
        tr.result = Result.ERROR.value
        tr.reason = repr(e)
        return [tr]

    highs = premarket["high"].tolist()
    lows = premarket["low"].tolist()
    volumes = premarket["volume"].tolist()
    n = len(premarket)
    # Each bar's x in SECONDS since the session start -- the detector fits in time,
    # not bar index, so gaps (missing no-trade bars) can't bend the lines away from
    # the prices the chart draws at real timestamps.
    t_session = premarket["dt_et"].iloc[0]
    xs = [(t - t_session).total_seconds() for t in premarket["dt_et"]]
    borrow_blocked_fee: float | None = None

    for i in range(w.min_pennant_bars, n):
        row = premarket.iloc[i]
        clk = row["dt_et"].time()
        if earliest_t is not None and clk < earliest_t:
            continue
        if latest_t is not None and clk > latest_t:
            break  # entry window closed

        armed = arm_wedge(xs, highs, lows, i, w)
        if armed is None:
            continue
        anchor, pen = armed.anchor, armed.pen   # pole top + fitted lines, for the exit walk
        w_lo = max(0, i - w.pivot_lookback_bars)   # same window arm_wedge anchored in
        # The live trigger-watch arms from this same call, one bar ahead (the
        # forming candle), and then watches these levels tick by tick.
        if on_armed is not None:
            on_armed(armed)
        decision = evaluate_wedge_touch(
            ticker=ticker, i=i, row=row, n=n, premarket=premarket, armed=armed,
            highs=highs, lows=lows, volumes=volumes, one_min=one_min,
            first_price=first_price, s=s, w=w, daily_avg_volume=daily_avg_volume,
            borrow_fee=borrow_fee, tick_fetcher=tick_fetcher, latency=latency,
            bar_len=bar_len)
        if decision.kind != "fire":
            if decision.counter is not None:
                setattr(tr, decision.counter, getattr(tr, decision.counter, 0) + 1)
            if decision.borrow_fee_blocked is not None:
                borrow_blocked_fee = decision.borrow_fee_blocked
            continue
        direction = decision.direction
        trigger = decision.trigger
        fill, fill_time = decision.fill, decision.fill_time
        stop, target = decision.stop, decision.target
        entry_limit = decision.entry_limit
        risk_per_share = decision.risk_per_share
        entry_bar_pos = decision.entry_bar_pos
        entry_ticks = decision.entry_ticks
        tr.entry_fill_source = decision.entry_fill_source
        mrow, rvol_so_far = decision.mrow, decision.rvol_so_far
        high, low, open_ = float(row["high"]), float(row["low"]), float(row["open"])
        # Direction-aware exit walk from the break bar. Every bar that touches the
        # stop or target is resolved from its REAL ticks (so a gapped-through stop
        # shows true slippage, and the order-latency fill models a software-triggered
        # order that can't rest in premarket -- see _first_touch_directional). Only
        # when ticks are unavailable / don't confirm does it fall back to a bar-level
        # fill at the level, with the intrabar policy for a both-touched bar.
        #
        # Hold-to-open limiters (each null = off): precomputed deadlines the walk
        # checks at every later bar's OPEN, using only completed-bar information --
        # a market order placed the moment the condition is knowable, never a level
        # picked from that bar's own future path.
        hold_until = pd.Timestamp(fill_time) + pd.Timedelta(minutes=w.max_hold_minutes) if w.max_hold_minutes is not None else None
        flat_by_t = time(*parse_hhmm(w.latest_exit_time, "wedge.latest_exit_time", "config")) if w.latest_exit_time else None
        stale_after = pd.Timestamp(fill_time) + pd.Timedelta(minutes=w.stale_exit_minutes) if w.stale_exit_minutes is not None else None
        max_high = fill
        min_low = fill
        exit_price = None
        exit_time = None
        exit_src = "bar"
        exit_kind = None
        forced_exit = None
        plain_session_exit = False
        if w.exit_mode == "candle_trail_1min":
            # 1-minute trailing manager: the dip strategy's PROVEN candle_trail_1min
            # runner, mirrored direction-aware for the wedge (see WedgeConfig.
            # exit_mode). The stop trails to the prior COMPLETED 1-minute candle's
            # favourable extreme (low for a long, high for a short -- ratchets
            # toward profit only); the first FAT-BODIED 1-minute close in the
            # trade's direction (green for a long, red for a short; same candle
            # definition as the dip's, strategy.fatgreen_body_min) DROPS the fixed
            # target and lets the runner ride the trail; hitting the original
            # target before any runner candle banks trail_partial_at_target of the
            # position and jumps the remainder's stop to breakeven. Everything is
            # causal: only minutes fully completed before the current scan bar's
            # minute are folded, the identical convention to the dip's manager.
            fg_min = s.fatgreen_body_min

            def _fat(o, h, low_, c, up: bool) -> bool:
                rng = h - low_
                if rng <= 0:
                    return False
                if up:
                    return c > o and (c - o) >= fg_min * rng and (c - low_) / rng >= 2.0 / 3.0
                return c < o and (o - c) >= fg_min * rng and (h - c) / rng >= 2.0 / 3.0

            minute_info = {
                pd.Timestamp(m.dt_et).floor("min"): (float(m.low), float(m.high),
                                                     _fat(m.open, m.high, m.low, m.close, True),
                                                     _fat(m.open, m.high, m.low, m.close, False))
                for m in one_min.itertuples()
            }
            cur_stop, cur_target = stop, target
            remaining = 1.0
            legs: list[dict] = []
            target_active = True
            entry_minute = pd.Timestamp(fill_time).floor("min")
            folded_upto = entry_minute
            no_target = float("inf") if direction == "long" else float("-inf")

            def bank_leg(kind: str, frac: float, price: float, when) -> None:
                legs.append({"kind": kind, "frac": round(frac, 6), "price": _round_cent(price), "time": _fmt_time(when)})

            for j in range(entry_bar_pos, n):
                if remaining <= 1e-9:
                    break
                ex = premarket.iloc[j]
                eh, el, eo, ec = float(ex["high"]), float(ex["low"]), float(ex["open"]), float(ex["close"])
                # Forced premarket exit -- identical semantics to the bracket walk
                # (clock + best-excursion-through-the-PRIOR-bar only; never on the
                # entry bar). Closes whatever fraction is still open at this open.
                if j > entry_bar_pos:
                    bar_ts = ex["dt_et"]
                    if hold_until is not None and bar_ts >= hold_until:
                        forced_exit = "time_stop"
                    elif flat_by_t is not None and bar_ts.time() >= flat_by_t:
                        forced_exit = "flat_by_time"
                    elif stale_after is not None and bar_ts >= stale_after:
                        span = abs(target - fill)
                        best = (max_high - fill) if direction == "long" else (fill - min_low)
                        if span > 0 and best < w.stale_exit_min_progress * span:
                            forced_exit = "stale_progress"
                    if forced_exit is not None:
                        bank_leg(forced_exit, remaining, eo, bar_ts)
                        remaining = 0.0
                        exit_time = bar_ts
                        exit_src = "bar"
                        exit_kind = forced_exit
                        break
                max_high = max(max_high, eh)
                min_low = min(min_low, el)
                # Fold every 1-minute candle fully completed before this bar's
                # minute: trail the stop (profit-direction only), and drop the
                # target on the first fat candle in the trade's favour (the entry
                # minute itself never counts -- it's a partial candle for us).
                this_minute = pd.Timestamp(ex["dt_et"]).floor("min")
                while folded_upto < this_minute:
                    info = minute_info.get(folded_upto)
                    if info is not None:
                        m_lo, m_hi, m_fatgreen, m_fatred = info
                        if direction == "long":
                            cur_stop = max(cur_stop, _round_cent(m_lo))
                            if m_fatgreen and folded_upto > entry_minute:
                                target_active = False
                        else:
                            cur_stop = min(cur_stop, _round_cent(m_hi))
                            if m_fatred and folded_upto > entry_minute:
                                target_active = False
                    folded_upto = folded_upto + pd.Timedelta(minutes=1)
                cur_target = target if target_active else no_target
                touched_stop = (el <= cur_stop) if direction == "long" else (eh >= cur_stop)
                touched_tgt = (eh >= cur_target) if direction == "long" else (el <= cur_target)
                if not (touched_stop or touched_tgt):
                    continue

                bar_ambiguous = touched_stop and touched_tgt
                if bar_ambiguous and not tr.path_ambiguous:
                    tr.path_ambiguous = True
                tick_rows = None
                if tick_fetcher is not None:
                    try:
                        if j == i and entry_ticks is not None:
                            ex_ticks = entry_ticks   # reuse the entry bar's already-fetched ticks
                        else:
                            bs = pd.Timestamp(ex["dt_et"])
                            ex_ticks = tick_fetcher(ticker, bs.to_pydatetime(), (bs + bar_len).to_pydatetime())
                        tick_rows = _sorted_tick_lists(ex_ticks)
                    except Exception:
                        tick_rows = None

                banked_from_ticks = False
                if tick_rows is not None:
                    times_l, prices_l = tick_rows
                    scan_idx = 0
                    if j == i:
                        # On the entry bar, ticks up to and including the entry
                        # cross predate the fill and can't be exits.
                        if direction == "long":
                            fidx = next((k for k, p in enumerate(prices_l) if p >= trigger), None)
                        else:
                            fidx = next((k for k, p in enumerate(prices_l) if p <= trigger), None)
                        scan_idx = None if fidx is None else fidx + 1
                    while scan_idx is not None and remaining > 1e-9:
                        hit = _first_touch_directional_indexed(times_l, prices_l, scan_idx, direction, cur_stop, cur_target, latency)
                        if hit is None:
                            break
                        kind, exit_ts, exit_fp, k_idx = hit
                        banked_from_ticks = True
                        exit_src = "ticks"
                        exit_time = exit_ts
                        if bar_ambiguous and tr.ambiguity_resolution is None:
                            tr.ambiguity_resolution = "ticks"
                        if kind == "LOSS":
                            leg_kind = "stop" if cur_stop == stop else "trail_stop"
                            bank_leg(leg_kind, remaining, exit_fp, exit_ts)
                            remaining = 0.0
                            exit_kind = leg_kind
                            break
                        if target_active and w.trail_partial_at_target < 1.0 - 1e-9:
                            # Runner: bank the partial at the target, then drop it,
                            # jump the remainder's stop to breakeven, keep scanning
                            # this same bar's later ticks against the new levels.
                            frac = remaining * w.trail_partial_at_target
                            bank_leg("target", frac, exit_fp, exit_ts)
                            remaining -= frac
                            target_active = False
                            cur_target = no_target
                            cur_stop = max(cur_stop, fill) if direction == "long" else min(cur_stop, fill)
                            scan_idx = k_idx + 1
                            continue
                        bank_leg("target", remaining, exit_fp, exit_ts)
                        remaining = 0.0
                        exit_kind = "target"
                        break
                if banked_from_ticks:
                    continue
                # No tick confirmation. On the entry bar, DON'T bar-exit (the touch
                # may be the pre-entry break action); defer to the next bar.
                if j == i:
                    continue
                exit_src = "bar"
                exit_time = ex["dt_et"]
                if touched_stop and touched_tgt:
                    outcome = _resolve_bar(direction, eo, eh, el, ec, cur_stop, cur_target)
                    if tr.ambiguity_resolution is None:
                        tr.ambiguity_resolution = "heuristic"
                else:
                    outcome = "LOSS" if touched_stop else "WIN"
                if outcome == "LOSS":
                    leg_kind = "stop" if cur_stop == stop else "trail_stop"
                    bank_leg(leg_kind, remaining, cur_stop, ex["dt_et"])
                    remaining = 0.0
                    exit_kind = leg_kind
                elif target_active and w.trail_partial_at_target < 1.0 - 1e-9:
                    # Runner (bar fallback): bank the partial at the target level,
                    # drop the target, remainder rides breakeven+ on the trail (the
                    # same shape as the dip's bar-fallback runner branch).
                    frac = remaining * w.trail_partial_at_target
                    bank_leg("target", frac, cur_target, ex["dt_et"])
                    remaining -= frac
                    target_active = False
                    cur_target = no_target
                    cur_stop = max(cur_stop, fill) if direction == "long" else min(cur_stop, fill)
                else:
                    bank_leg("target", remaining, cur_target, ex["dt_et"])
                    remaining = 0.0
                    exit_kind = "target"

            if remaining > 1e-9:
                # Session end: whatever fraction is still open marks at the last
                # premarket close (the same convention as the bracket walk).
                exit_bar = premarket.iloc[-1]
                bank_leg("session_end", remaining, float(exit_bar["close"]), exit_bar["dt_et"])
                remaining = 0.0
                exit_time = exit_bar["dt_et"]
                exit_src = "bar"
                if len(legs) == 1:
                    # Nothing ever banked: the classic exit-at-open case.
                    exit_kind = "session_end"
                    plain_session_exit = True
                else:
                    exit_kind = "session_end"
            exit_price = sum(leg["frac"] * leg["price"] for leg in legs)
            tr.exit_legs = legs
            tr.trail_activated = not target_active
            tr.trail_stop_final = _round_cent(cur_stop)
        else:
            rf = (_RetestExit(direction, fill, w.retest_bounce_min_pct, w.retest_tolerance_pct)
                  if w.exit_retest_failure else None)
            # Traditional structural/momentum exit: per-minute indicator lookup for
            # an exit when price RECLAIMS the chosen 1-min indicator against the
            # position (or MACD hist turns against it). The indicators are causal
            # (EMA/VWAP through the completed minute), so a close-vs-indicator check
            # at each completed candle has no lookahead.
            reclaim = w.exit_on_reclaim
            ind_by_min = None
            if reclaim:
                ind_by_min = {pd.Timestamp(m.dt_et).floor("min"):
                              (float(m.ema9), float(m.ema20), float(m.vwap), float(m.hist))
                              for m in one_min.itertuples()}
            for j in range(entry_bar_pos, n):
                ex = premarket.iloc[j]
                eh, el, eo, ec = float(ex["high"]), float(ex["low"]), float(ex["open"]), float(ex["close"])
                # Forced premarket exit? Decided AT THIS BAR'S OPEN from information
                # already complete (the clock, and best excursion through the PRIOR
                # bar -- max_high/min_low are updated below, after this check). Never
                # on the entry bar itself: the deadlines are minutes-scale and the
                # position only just opened.
                if j > entry_bar_pos:
                    bar_ts = ex["dt_et"]
                    if hold_until is not None and bar_ts >= hold_until:
                        forced_exit = "time_stop"
                    elif flat_by_t is not None and bar_ts.time() >= flat_by_t:
                        forced_exit = "flat_by_time"
                    elif stale_after is not None and bar_ts >= stale_after:
                        span = abs(target - fill)
                        best = (max_high - fill) if direction == "long" else (fill - min_low)
                        if span > 0 and best < w.stale_exit_min_progress * span:
                            forced_exit = "stale_progress"
                    if forced_exit is not None:
                        exit_price = eo
                        exit_time = bar_ts
                        exit_src = "bar"
                        exit_kind = forced_exit
                        break
                max_high = max(max_high, eh)
                min_low = min(min_low, el)
                # Failed-retest early exit: a swing extreme in the trade's favour
                # was retested and HELD, and this candle broke back through the
                # attempt candle -- leave rather than hold to the open. Checked
                # before stop/target: its level is the tighter, earlier one.
                if rf is not None:
                    rf_level = rf.step(eo, eh, el, ec)
                    if rf_level is not None and j > entry_bar_pos:
                        exit_price = rf_level
                        exit_time = ex["dt_et"]
                        exit_src = "bar"
                        exit_kind = "retest_failure"
                        break
                # Structural/momentum reclaim exit: close crosses back through the
                # chosen indicator against the position (or MACD hist turns against
                # it) -> leave at this candle's close.
                if ind_by_min is not None and j > entry_bar_pos:
                    ind = ind_by_min.get(pd.Timestamp(ex["dt_et"]).floor("min"))
                    if ind is not None:
                        ema9v, ema20v, vwapv, histv = ind
                        if reclaim == "macd":
                            against = histv > 0 if direction == "short" else histv < 0
                        else:
                            lvl = {"vwap": vwapv, "ema9": ema9v, "ema20": ema20v}[reclaim]
                            against = ec > lvl if direction == "short" else ec < lvl
                        if against:
                            exit_price = ec
                            exit_time = ex["dt_et"]
                            exit_src = "bar"
                            exit_kind = "reclaim_" + reclaim
                            break
                touched_stop = (el <= stop) if direction == "long" else (eh >= stop)
                touched_tgt = (eh >= target) if direction == "long" else (el <= target)
                if not (touched_stop or touched_tgt):
                    continue

                tick_rows = None
                if tick_fetcher is not None:
                    try:
                        if j == i and entry_ticks is not None:
                            ex_ticks = entry_ticks   # reuse the entry bar's already-fetched ticks
                        else:
                            bs = pd.Timestamp(ex["dt_et"])
                            ex_ticks = tick_fetcher(ticker, bs.to_pydatetime(), (bs + bar_len).to_pydatetime())
                        tick_rows = _sorted_tick_lists(ex_ticks)
                    except Exception:
                        tick_rows = None

                outcome = None
                if tick_rows is not None:
                    times_l, prices_l = tick_rows
                    scan_idx = 0
                    if j == i:
                        # On the entry bar, ticks up to and including the entry cross
                        # predate the fill and can't be exits.
                        if direction == "long":
                            fidx = next((k for k, p in enumerate(prices_l) if p >= trigger), None)
                        else:
                            fidx = next((k for k, p in enumerate(prices_l) if p <= trigger), None)
                        scan_idx = None if fidx is None else fidx + 1
                    if scan_idx is not None:
                        hit = _first_touch_directional(times_l, prices_l, scan_idx, direction, stop, target, latency)
                        if hit is not None:
                            outcome, exit_time, exit_price = hit
                            exit_src = "ticks"
                if outcome is None:
                    # No tick confirmation. On the entry bar, DON'T bar-exit (the touch may
                    # be the pre-entry break action); defer to the next bar.
                    if j == i:
                        continue
                    exit_src = "bar"
                    if touched_stop and touched_tgt:
                        outcome = _resolve_bar(direction, eo, eh, el, ec, stop, target)
                        tr.path_ambiguous = True
                        tr.ambiguity_resolution = "heuristic"
                    else:
                        outcome = "LOSS" if touched_stop else "WIN"
                    exit_price = stop if outcome == "LOSS" else target
                    exit_time = ex["dt_et"]
                exit_kind = "target" if outcome == "WIN" else "stop"
                break
            else:
                exit_bar = premarket.iloc[-1]
                exit_price = float(exit_bar["close"])
                exit_time = exit_bar["dt_et"]
                exit_kind = "session_end"
                plain_session_exit = True

        tr.direction = direction
        # Pennant trendline geometry for the chart: the two lines' values at the
        # anchor (pole top) and at the break bar, plus those timestamps. The fit is
        # linear in TIME, so the chart's straight line between these two points
        # reproduces the fitted line exactly at every timestamp in between.
        tr.wedge_line_start_time = _fmt_time(premarket.loc[anchor, "dt_et"])
        tr.wedge_line_end_time = _fmt_time(row["dt_et"])
        tr.wedge_res_start = _round_cent(pen.resistance_at(xs[anchor]))
        tr.wedge_res_end = _round_cent(pen.resistance_at(xs[i]))
        tr.wedge_sup_start = _round_cent(pen.support_at(xs[anchor]))
        tr.wedge_sup_end = _round_cent(pen.support_at(xs[i]))
        tr.trigger_price = trigger
        tr.entry_time = _fmt_time(fill_time)
        tr.entry_price = fill
        tr.fill_price = fill
        tr.entry_limit_price = entry_limit
        tr.slippage = _round_cent(fill - trigger)
        tr.stop_price = stop
        tr.target_price = target
        tr.exit_price = _round_cent(exit_price)
        tr.exit_time = _fmt_time(exit_time)
        tr.exit_fill_source = exit_src
        tr.volume_at_entry = float(mrow.iloc[0]["cum_volume"])
        # Volume traded across the pivot-lookback WINDOW (pole + coil): the bars
        # from w_lo through the entry bar, on the wedge's own timeframe. Base for
        # the wedge.max_entry_pct_of_lookback_volume size cap -- pattern-local
        # liquidity, distinct from volume_at_entry (whole-session cumulative).
        tr.lookback_volume = float(sum(volumes[w_lo:i + 1]))
        tr.rvol_at_spike = float(rvol_so_far)
        tr.premarket_high_time = _fmt_time(premarket.loc[premarket["high"].idxmax(), "dt_et"])
        tr.anchor_price = _round_cent(first_price)
        if direction == "short" and w.max_borrow_fee_pct is not None:
            tr.borrow_fee = borrow_fee
            tr.borrow_fee_source = "reference_csv" if borrow_fee is not None else "assumed"
        pnl = (exit_price - fill) if direction == "long" else (fill - exit_price)
        tr.pnl_per_share = _round_cent(pnl - 2 * r.fees_per_share)
        tr.pnl_pct = round(tr.pnl_per_share / fill * 100.0, 3) if fill else None
        tr.r_multiple = round(tr.pnl_per_share / risk_per_share, 3) if risk_per_share else None
        tr.mfe = _round_cent((max_high - fill) if direction == "long" else (fill - min_low))
        tr.mae = _round_cent((fill - min_low) if direction == "long" else (max_high - fill))
        tr.exit_trigger = exit_kind
        if plain_session_exit:
            tr.result = Result.EXIT_AT_OPEN.value if s.exit_at_regular_open else Result.NO_TRIGGER.value
            tr.reason = f"{direction.title()} pennant break; neither target nor stop hit before the open."
        elif forced_exit is not None:
            # A hold-to-open limiter closed the position in premarket: a normal
            # realized WIN/LOSS by its own P&L, never an EXIT_AT_OPEN.
            tr.result = Result.WIN.value if tr.pnl_per_share >= 0 else Result.LOSS.value
            if forced_exit == "time_stop":
                label = f"time stop after {w.max_hold_minutes:g} min unresolved"
            elif forced_exit == "flat_by_time":
                label = f"flat-by {w.latest_exit_time} ahead of the open"
            else:
                label = (f"stale after {w.stale_exit_minutes:g} min "
                         f"(<{w.stale_exit_min_progress:.0%} of the way to target)")
            tr.reason = f"{direction.title()} pennant break closed early -- {label}."
        elif exit_kind == "retest_failure":
            # Failed-retest early exit: a realized WIN/LOSS by its own P&L, closed
            # in premarket rather than held to the open.
            tr.result = Result.WIN.value if tr.pnl_per_share >= 0 else Result.LOSS.value
            extreme = "low" if direction == "short" else "high"
            tr.reason = (f"{direction.title()} pennant break closed early -- failed retest of the "
                         f"swing {extreme}; broke back through the attempt candle.")
        elif exit_kind and exit_kind.startswith("reclaim_"):
            tr.result = Result.WIN.value if tr.pnl_per_share >= 0 else Result.LOSS.value
            sig = exit_kind.split("_", 1)[1]
            tr.reason = (f"{direction.title()} pennant break closed early -- price reclaimed "
                         f"{sig.upper()} against the position.")
        else:
            tr.result = Result.WIN.value if tr.pnl_per_share >= 0 else Result.LOSS.value
            tr.reason = (f"{direction.title()} break of the {'resistance' if direction == 'long' else 'support'} line "
                         f"(pennant {pen.convergence_pct:.0f}% converged, {pen.touches_r}/{pen.touches_s} touches).")
            if w.exit_mode == "candle_trail_1min" and tr.exit_legs:
                # Runner telemetry appended for the log/HTML, mirroring the dip's.
                last_kind = tr.exit_legs[-1]["kind"]
                if tr.trail_activated:
                    where = "held to the open" if last_kind == "session_end" else f"trailed out at {tr.trail_stop_final:.2f}"
                    tr.reason += f" Runner: target dropped; {where}."
                elif last_kind == "target":
                    tr.reason += " Base hit: target banked in full (no runner)."
                elif last_kind in ("stop", "trail_stop"):
                    tr.reason += f" {'Trailing stop' if last_kind == 'trail_stop' else 'Stop'} hit at {tr.trail_stop_final:.2f}."
        tr.setup_index = 1
        return [tr]

    if borrow_blocked_fee is not None:
        tr.result = Result.SKIPPED_BORROW_FEE.value
        tr.direction = "short"
        tr.borrow_fee = borrow_blocked_fee
        tr.borrow_fee_source = "reference_csv"
        tr.reason = f"Short pennant break rejected: borrow fee {borrow_blocked_fee:g}% exceeds max_borrow_fee_pct {w.max_borrow_fee_pct:g}%."
        return [tr]
    tr.result = Result.NO_WEDGE.value
    tr.reason = "Qualified, but no symmetrical-pennant break triggered in the entry window."
    return [tr]
