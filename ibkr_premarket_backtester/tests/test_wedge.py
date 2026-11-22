"""Symmetrical-pennant strategy: the pure detector geometry, and the direction-
aware trade lifecycle (bidirectional break, short-side math signs, persistence and
borrow gates)."""
from pathlib import Path

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig,
    StrategyConfig, WedgeConfig,
)
from datetime import date

from premarket_backtester.wedge import (
    _first_touch_directional, backtest_wedge, detect_pennant, pivot_positions,
)


# --- tick-exit latency / slippage rules ----------------------------------------

_LAT = pd.Timedelta(seconds=0.3)


def _ticks(prices, step=0.1):
    base = pd.Timestamp("2026-04-09 08:00:00", tz="America/New_York")
    return [base + pd.Timedelta(seconds=step * i) for i in range(len(prices))], list(prices)


def test_tick_exit_latency_rules_per_direction():
    # LONG target is a RESTING sell limit -> fills at the crossing print, NO latency.
    t, p = _ticks([10.0, 10.5, 11.0, 11.2, 11.4, 11.6, 11.8])
    o = _first_touch_directional(t, p, 0, "long", 9.0, 11.0, _LAT)
    assert o[0] == "WIN" and o[2] == 11.0

    # LONG stop is a software-triggered sell -> fills ~0.3s later (worse than 9.00).
    t, p = _ticks([10.0, 9.5, 9.0, 8.9, 8.7, 8.5, 8.4])   # crosses at idx2 -> fill idx5
    o = _first_touch_directional(t, p, 0, "long", 9.0, 11.0, _LAT)
    assert o[0] == "LOSS" and o[2] == 8.5

    # SHORT target is a COVER (a buy) -> also can't rest -> latency too (the key
    # asymmetry vs long): fills 0.3s later, not at the 9.00 level.
    t, p = _ticks([10.0, 9.5, 9.0, 8.95, 8.9, 8.85, 8.8])
    o = _first_touch_directional(t, p, 0, "short", 11.0, 9.0, _LAT)
    assert o[0] == "WIN" and o[2] == 8.85

    # SHORT stop is a buy-to-cover ABOVE -> latency.
    t, p = _ticks([10.0, 10.5, 11.0, 11.1, 11.2, 11.3, 11.4])
    o = _first_touch_directional(t, p, 0, "short", 11.0, 9.0, _LAT)
    assert o[0] == "LOSS" and o[2] == 11.3


# --- pure detector -------------------------------------------------------------

_XS = [float(10 * i) for i in range(15)]     # evenly spaced bars, 10s apart
_HIGHS = [10, 9.5, 9.6, 9.0, 9.1, 8.6, 8.7, 8.2, 8.3, 7.9, 8.0, 7.6, 7.7, 7.3, 7.4]
_LOWS = [5.0, 5.4, 5.2, 5.8, 5.6, 6.1, 5.9, 6.3, 6.1, 6.5, 6.3, 6.7, 6.5, 6.9, 6.7]

# Strict containment (no outlier poke-through) for the pure-geometry tests;
# touch_tol_frac is a fraction of the window's MEDIAN bar range.
_KW = dict(strength=1, min_touches=3, touch_tol_frac=0.75, min_convergence_pct=30,
           containment_outliers=0)


def test_pivot_positions_finds_local_extrema():
    assert pivot_positions([1, 3, 2, 4, 1], 1, "high") == [1, 3]
    assert pivot_positions([5, 2, 4, 1, 6], 1, "low") == [1, 3]


def test_detect_symmetrical_pennant():
    p = detect_pennant(_XS, _HIGHS, _LOWS, **_KW)
    assert p is not None
    assert p.slope_r < 0 < p.slope_s          # resistance down, support up
    assert p.convergence_pct > 30
    assert p.resistance_at(_XS[-1]) > p.support_at(_XS[-1]) > 0


def test_lines_contain_every_bar_like_a_drawn_trendline():
    """The core visual-fidelity property: after the containment shift, the
    resistance sits ON/ABOVE every high and the support ON/BELOW every low --
    the line a trader would draw, not a regression through the middle."""
    p = detect_pennant(_XS, _HIGHS, _LOWS, **_KW)
    for x, h, low in zip(_XS, _HIGHS, _LOWS):
        assert p.resistance_at(x) >= h - 1e-9
        assert p.support_at(x) <= low + 1e-9
    # And the lines genuinely TOUCH price (not shoved arbitrarily far out).
    assert min(p.resistance_at(x) - h for x, h in zip(_XS, _HIGHS)) < 1e-9
    assert min(low - p.support_at(x) for x, low in zip(_XS, _LOWS)) < 1e-9


def test_uneven_time_spacing_still_fits_in_time_space():
    """Bars with gaps (missing no-trade bars): the fit runs on real timestamps, so
    a straight line drawn between the window's endpoint VALUES reproduces the fit
    at every interior bar -- the chart/algo disconnect the index-space fit had."""
    xs = [0.0, 10.0, 20.0, 90.0, 100.0, 110.0, 180.0, 190.0, 200.0]   # two long gaps
    highs = [9.8, 10.0, 9.7, 9.2, 9.3, 8.9, 8.3, 8.4, 8.0]            # pivots at 1, 4, 7 (descending)
    lows = [7.2, 7.0, 7.3, 7.5, 7.35, 7.6, 7.8, 7.65, 7.9]            # pivots at 1, 4, 7 (ascending)
    p = detect_pennant(xs, highs, lows, strength=1, min_touches=2, touch_tol_frac=0.75,
                       min_convergence_pct=20, containment_outliers=0)
    assert p is not None
    # Linear-in-time: interpolating between the endpoints equals the line at an
    # interior bar even across the gaps.
    v0, v1 = p.resistance_at(xs[0]), p.resistance_at(xs[-1])
    for x in xs:
        interp = v0 + (v1 - v0) * (x - xs[0]) / (xs[-1] - xs[0])
        assert abs(interp - p.resistance_at(x)) < 1e-9


def test_detect_rejects_parallel_and_insufficient_convergence():
    parallel_lows = [5.0 - 0.19 * i for i in range(15)]     # support falling, not rising
    assert detect_pennant(_XS, _HIGHS, parallel_lows, **_KW) is None
    # A real pennant but demanding 99% convergence it doesn't reach:
    assert detect_pennant(_XS, _HIGHS, _LOWS, strength=1, min_touches=3, touch_tol_frac=0.75,
                          min_convergence_pct=99, containment_outliers=0) is None
    # Too many touches required:
    assert detect_pennant(_XS, _HIGHS, _LOWS, strength=1, min_touches=20, touch_tol_frac=0.75,
                          min_convergence_pct=30, containment_outliers=0) is None
    # Past the apex: the same pennant rejected when its convergence exceeds the cap.
    assert detect_pennant(_XS, _HIGHS, _LOWS, strength=1, min_touches=3, touch_tol_frac=0.75,
                          min_convergence_pct=30, max_convergence_pct=50, containment_outliers=0) is None


def test_rogue_spike_pokes_through_with_outlier_budget_else_rejects():
    """The manual-charting rule: one violent wick through an otherwise clean
    trendline is ACCEPTED (the line hugs the rest of price), but with a strict
    containment budget the wick owns the line, nothing else touches it, and the
    pattern is rightly rejected."""
    spiked = list(_HIGHS)
    spiked[7] = 14.0
    kw = dict(strength=1, min_touches=3, touch_tol_frac=0.5, min_convergence_pct=10)
    assert detect_pennant(_XS, spiked, _LOWS, **kw, containment_outliers=0) is None
    p = detect_pennant(_XS, spiked, _LOWS, **kw, containment_outliers=1)
    assert p is not None
    assert p.resistance_at(_XS[7]) < 14.0            # the spike pokes THROUGH the line
    contained = sum(1 for x, h in zip(_XS, spiked) if p.resistance_at(x) >= h - 1e-9)
    assert contained >= len(_XS) - 1                 # ...and everything else is under it


# --- integration: direction-aware trade lifecycle ------------------------------

def _cfg(**wedge_over) -> AppConfig:
    wcfg = WedgeConfig(enabled=True, sides="both", timeframe="10s", pivot_lookback_bars=26,
                       min_pennant_bars=8, pivot_strength=1, min_pivot_touches=3,
                       touch_tolerance=0.25, min_convergence_pct=20.0, persistence_mode="rvol",
                       min_sustained_rvol=1.0, target_r_multiple=2.0, max_borrow_fee_pct=20.0,
                       breakout_offset=0.10)   # so a mere touch of a boundary isn't a break
    wcfg = WedgeConfig(**{**wcfg.__dict__, **wedge_over})
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("x.csv"), reference_csv=None, cache_dir=Path("c"), output_dir=Path("o")),
        session=SessionConfig(),
        strategy=StrategyConfig(earliest_entry_time=None, latest_entry_time=None, min_rvol=2.5,
                                min_premarket_gain_pct=28.0, min_price=1.0, max_price=100.0),
        risk=RiskConfig(fees_per_share=0.0),
        reporting=ReportingConfig(),
        wedge=wcfg,
    )


def _pennant_session(break_dir: str) -> pd.DataFrame:
    """A synthetic premarket: a rip (3.00 -> 4.00, +33% gain), a converging zigzag
    pennant, then a decisive break in `break_dir`, then follow-through to the target."""
    rows = [(3.00, 3.30, 2.98, 3.28), (3.28, 3.60, 3.25, 3.58), (3.58, 3.90, 3.55, 3.88), (3.88, 4.00, 3.85, 3.75)]
    # Converging zigzag over ~24 bars: lower highs, higher lows.
    for k in range(12):
        hi = 3.92 - 0.018 * k       # descending pivot highs
        lo = 3.42 + 0.012 * k       # ascending pivot lows
        rows.append((lo + 0.02, hi, lo, hi - 0.03))   # up-bar to the high (pivot high)
        rows.append((hi - 0.03, hi - 0.02, lo, lo + 0.02))   # down-bar to the low (pivot low)
    # Break bar breaks ONE boundary decisively but does NOT reach the opposite
    # (stop) level; follow-through bars then carry price to the target.
    if break_dir == "long":
        rows.append((3.70, 4.00, 3.68, 3.98))                          # breaks resistance up
        rows += [(3.98, 4.30, 3.96, 4.28), (4.28, 4.70, 4.25, 4.65)]   # run up through the target
    else:
        rows.append((3.55, 3.57, 3.30, 3.32))                          # breaks support down
        rows += [(3.32, 3.34, 3.00, 3.02), (3.02, 3.04, 2.70, 2.72),
                 (2.72, 2.74, 2.30, 2.32)]                             # run well through the target
    base = pd.Timestamp("2026-04-09 08:00:00", tz="America/New_York")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 2000.0
    df["dt_et"] = [base + pd.Timedelta(seconds=10 * i) for i in range(len(df))]
    return df


_EMPTY_PREV = pd.DataFrame({"dt_et": pd.Series([], dtype="datetime64[ns, America/New_York]"), "close": pd.Series([], dtype=float)})


def _run(cfg, pm, **kw):
    return backtest_wedge("TEST", date(2026, 4, 9), pm, pm, _EMPTY_PREV, cfg,
                          daily_volume_fetcher=lambda t, d: 100.0, **kw)[0]


def test_long_break_goes_long_with_correct_sign():
    tr = _run(_cfg(), _pennant_session("long"))
    assert tr.entry_price is not None, tr.reason
    assert tr.direction == "long"
    assert tr.result == "WIN"                       # ran up to the target
    assert tr.exit_price > tr.entry_price           # long profits on the way up
    assert tr.pnl_per_share > 0 and tr.r_multiple > 0


def test_short_break_goes_short_with_correct_sign():
    tr = _run(_cfg(), _pennant_session("short"))
    assert tr.entry_price is not None, tr.reason
    assert tr.direction == "short"
    assert tr.result == "WIN"                       # ran down to the target
    assert tr.exit_price < tr.entry_price           # SHORT profits when price falls
    assert tr.pnl_per_share > 0 and tr.r_multiple > 0
    # stop sits ABOVE entry, target BELOW, for a short.
    assert tr.stop_price > tr.entry_price > tr.target_price
    assert tr.mfe >= 0 and tr.mae >= 0


def test_sides_long_only_ignores_a_down_break():
    tr = _run(_cfg(sides="long"), _pennant_session("short"))
    assert tr.entry_price is None                   # the down-break is not taken
    assert tr.result in ("NO_WEDGE", "EXIT_AT_OPEN", "NO_TRIGGER")


def test_borrow_gate_rejects_expensive_short():
    over = _run(_cfg(max_borrow_fee_pct=20.0), _pennant_session("short"), borrow_fee=45.0)
    assert over.result == "SKIPPED_BORROW_FEE" and over.borrow_fee == 45.0
    # Within the cap -> the short trades, tagged from the reference source.
    ok = _run(_cfg(max_borrow_fee_pct=20.0), _pennant_session("short"), borrow_fee=8.0)
    assert ok.entry_price is not None and ok.direction == "short"
    assert ok.borrow_fee == 8.0 and ok.borrow_fee_source == "reference_csv"
    # Unknown borrow (no data) -> assumed borrowable, still trades, flagged "assumed".
    unk = _run(_cfg(max_borrow_fee_pct=20.0), _pennant_session("short"))
    assert unk.entry_price is not None and unk.borrow_fee_source == "assumed"


def _ohlc_tick_fetcher(pm):
    """A tick_fetcher that returns each bar's O/H/L/C as four ticks, so the exit
    loop resolves from 'ticks' rather than bars."""
    idx = {pd.Timestamp(r.dt_et): (r.open, r.high, r.low, r.close) for r in pm.itertuples()}

    def fetch(ticker, start, end):
        ts = pd.Timestamp(start)
        if ts in idx:
            o, h, l, c = idx[ts]
            times = [ts + pd.Timedelta(milliseconds=100 * k) for k in range(4)]
            return pd.DataFrame({"time": times, "price": [o, h, l, c]})
        return pd.DataFrame({"time": [], "price": []})

    return fetch


def test_short_exit_resolves_from_ticks_when_available():
    pm = _pennant_session("short")
    # Wide slippage cap: the crude 4-tick fetcher + 0.3s latency lands the fill a few
    # cents from the trigger, which the default 15c cap would (correctly) reject.
    tr = _run(_cfg(entry_limit_slippage_cents=1.00), pm, tick_fetcher=_ohlc_tick_fetcher(pm))
    assert tr.entry_price is not None and tr.direction == "short"
    assert tr.exit_fill_source == "ticks"            # the tick path fired, not the bar fallback
    # internal consistency: a short makes money iff it covered below entry.
    assert (tr.pnl_per_share > 0) == (tr.exit_price < tr.entry_price)
    assert (tr.r_multiple > 0) == (tr.pnl_per_share > 0)


def test_persistence_rvol_gate_blocks_when_rvol_too_low():
    # Require sustained RVOL above what this session reaches by the break -> no trade.
    tr = _run(_cfg(persistence_mode="rvol", min_sustained_rvol=1e9), _pennant_session("short"))
    assert tr.entry_price is None and tr.result == "NO_WEDGE"


def test_breakout_volume_expansion_gate():
    """A real breakout is validated by volume: require the break bar's OWN volume
    to exceed a multiple of the coil's own (uniform, in this fixture) average.
    Off (default) ignores volume entirely; on, a weak-volume break defers (and
    here never gets a second, stronger-volume chance within the fixture's short
    follow-through, so it ends NO_WEDGE), while a volume-pumped break still fires.
    The gate reads the break bar's TOTAL volume -- end-of-bar information -- so a
    passing break enters on the NEXT bar's open (never intrabar off future info);
    the wide slippage cap keeps that later, worse fill inside the entry limit,
    which is not what this test is about."""
    pm = _pennant_session("long")
    assert _run(_cfg(), pm).entry_price is not None   # off: uniform 2000-share volume, trades

    break_idx = pm.index[-3]   # the bar that actually breaks resistance (3 rows follow it)
    weak = pm.copy()
    weak.loc[break_idx, "volume"] = 500.0   # well below 1.5x the 2000-share coil average
    tr = _run(_cfg(require_breakout_volume_expansion=True, breakout_volume_multiple=1.5,
                   entry_limit_slippage_cents=1.00), weak)
    assert tr.entry_price is None and tr.result == "NO_WEDGE"

    strong = pm.copy()
    strong.loc[break_idx, "volume"] = 10_000.0   # well above the multiple
    tr2 = _run(_cfg(require_breakout_volume_expansion=True, breakout_volume_multiple=1.5,
                    entry_limit_slippage_cents=1.00), strong)
    assert tr2.entry_price is not None
    # The entry is the confirmation bar's successor's OPEN, not the intrabar cross.
    assert tr2.entry_fill_source == "bar_confirm"
    assert tr2.entry_price == pytest.approx(3.98)


def test_vwap_alignment_gate_blocks_a_break_against_the_session_vwap():
    """require_vwap_alignment requires the trigger to sit on the favourable side of
    the causal session VWAP. A huge early spike (with huge volume) drags VWAP well
    above the modest breakout that follows -- off (default), the break still
    trades; on, it's correctly rejected as not trend-aligned."""
    def with_early_spike(pm):
        before = pm["dt_et"].iloc[0] - pd.Timedelta(seconds=10)
        spike = pd.DataFrame([{"open": 3.0, "high": 20.0, "low": 3.0, "close": 20.0,
                               "volume": 200_000.0, "dt_et": before}])
        return pd.concat([spike, pm], ignore_index=True)

    pm = with_early_spike(_pennant_session("long"))
    assert _run(_cfg(), pm).entry_price is not None
    tr = _run(_cfg(require_vwap_alignment=True), pm)
    assert tr.entry_price is None and tr.result == "NO_WEDGE"


def _pennant_session_weak_then_strong_close() -> pd.DataFrame:
    """Like _pennant_session('long'), but the first touch of resistance WICKS
    through (high 4.00 > the 3.80-ish trigger) yet closes weakly (close 3.81, low
    in its own range) -- and only the FOLLOWING bar closes strongly enough to
    confirm (close 4.02, top of its range)."""
    rows = [(3.00, 3.30, 2.98, 3.28), (3.28, 3.60, 3.25, 3.58), (3.58, 3.90, 3.55, 3.88), (3.88, 4.00, 3.85, 3.75)]
    for k in range(12):
        hi = 3.92 - 0.018 * k
        lo = 3.42 + 0.012 * k
        rows.append((lo + 0.02, hi, lo, hi - 0.03))
        rows.append((hi - 0.03, hi - 0.02, lo, lo + 0.02))
    rows.append((3.70, 4.00, 3.68, 3.81))   # weak-close touch: wicks above trigger, closes low
    rows.append((3.81, 4.05, 3.80, 4.02))   # strong-close confirm bar
    rows.append((4.02, 4.40, 3.98, 4.35))   # entry lands here (this bar's OPEN)
    rows.append((4.35, 4.70, 4.30, 4.65))
    base = pd.Timestamp("2026-04-09 08:00:00", tz="America/New_York")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 2000.0
    df["dt_et"] = [base + pd.Timedelta(seconds=10 * i) for i in range(len(df))]
    return df


def test_close_confirm_defers_a_weak_close_then_enters_on_the_next_bars_open():
    pm = _pennant_session_weak_then_strong_close()
    # Off (default, wick-based): the weak bar itself triggers immediately.
    off = _run(_cfg(), pm)
    assert off.entry_price == pytest.approx(3.80) and off.entry_fill_source == "bar"

    # On: the weak close DEFERS (no trade on that touch); the confirming bar's
    # successor supplies the fill, at THAT bar's open -- not the trigger level.
    on = _run(_cfg(require_breakout_close_confirm=True), pm)
    assert on.entry_price == pytest.approx(4.02)
    assert on.entry_fill_source == "bar_confirm"
    assert on.entry_time == "2026-04-09T08:05:00-04:00"


def test_measured_move_target_projects_the_pole_height():
    """target_mode 'measured_move': pole height = the anchor (pole-top) high minus
    the lowest low in pole_lookback_bars before it. For the long fixture the pole
    top is the initial rip's high (4.00) and its base the session's opening low
    (2.98) -> pole height 1.02; fraction scales it linearly."""
    tr_full = _run(_cfg(target_mode="measured_move", measured_move_fraction=1.0), _pennant_session("long"))
    assert tr_full.entry_price == pytest.approx(3.80)
    assert tr_full.target_price == pytest.approx(3.80 + 1.02)

    tr_half = _run(_cfg(target_mode="measured_move", measured_move_fraction=0.5), _pennant_session("long"))
    assert tr_half.target_price == pytest.approx(3.80 + 0.51)


# --- exit_mode "candle_trail_1min": the dip's runner engine, direction-aware ----

def _mins(*minutes):
    """Each (o, h, l, c) minute becomes six flat 10s bars [o, h, l, c, c, c] --
    the resulting 1-minute candle has exactly that OHLC, while the 10s scan frame
    touches levels in a controlled order (open first, then high, then low)."""
    rows = []
    for (o, h, low_, c) in minutes:
        rows += [(v, v, v, v) for v in (o, h, low_, c, c, c)]
    return rows


def _trail_session(break_dir: str, minutes) -> pd.DataFrame:
    """The standard pennant fixture up to and including its break bar (entry at
    ~08:04:40), one filler bar to finish the entry minute, then fully-controlled
    1-minute candles from 08:05 (via _mins). No default follow-through rows."""
    rows = [(3.00, 3.30, 2.98, 3.28), (3.28, 3.60, 3.25, 3.58), (3.58, 3.90, 3.55, 3.88), (3.88, 4.00, 3.85, 3.75)]
    for k in range(12):
        hi = 3.92 - 0.018 * k
        lo = 3.42 + 0.012 * k
        rows.append((lo + 0.02, hi, lo, hi - 0.03))
        rows.append((hi - 0.03, hi - 0.02, lo, lo + 0.02))
    if break_dir == "long":
        rows.append((3.70, 4.00, 3.68, 3.98))     # break bar (fills at 3.80)
        rows.append((3.98, 3.98, 3.98, 3.98))     # filler: completes the 08:04 minute
    else:
        rows.append((3.55, 3.57, 3.30, 3.32))     # break bar (fills at 3.46)
        rows.append((3.32, 3.32, 3.32, 3.32))
    rows += _mins(*minutes)
    base = pd.Timestamp("2026-04-09 08:00:00", tz="America/New_York")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 2000.0
    df["dt_et"] = [base + pd.Timedelta(seconds=10 * i) for i in range(len(df))]
    return df


def test_trail_long_ratchets_to_prior_minute_lows_and_trails_out_in_profit():
    """No fat candle, target never reached: the stop ratchets up to each completed
    1-minute low (3.90 then 3.95) and the reversal exits at the trailed level --
    a PROFITABLE structural exit where the bracket would have held to the open."""
    pm = _trail_session("long", [
        (3.98, 4.02, 3.90, 4.00),    # 08:05 -- low 3.90 (not fat: body 0.02/0.12)
        (4.00, 4.06, 3.95, 4.04),    # 08:06 -- low 3.95
        (4.04, 4.05, 3.85, 3.88),    # 08:07 -- reversal through the trailed stop
    ])
    tr = _run(_cfg(exit_mode="candle_trail_1min"), pm)
    assert tr.entry_price == pytest.approx(3.80)
    assert tr.exit_price == pytest.approx(3.95)          # the 08:06 minute's low
    assert tr.result == "WIN" and tr.pnl_per_share > 0   # trailed out IN PROFIT
    assert tr.exit_trigger == "trail_stop"
    assert tr.exit_legs is not None and len(tr.exit_legs) == 1
    assert tr.trail_stop_final == pytest.approx(3.95)
    # Same session under the bracket: nothing hit -> held to the open.
    br = _run(_cfg(exit_mode="bracket"), pm)
    assert br.result == "EXIT_AT_OPEN"


def test_trail_short_ratchets_to_prior_minute_highs():
    """Direction mirror: the short's stop trails DOWN to completed 1-minute highs
    (3.35 then 3.28); the bounce covers at the trailed level below entry."""
    pm = _trail_session("short", [
        (3.30, 3.35, 3.20, 3.22),    # 08:05 -- high 3.35
        (3.22, 3.28, 3.10, 3.12),    # 08:06 -- high 3.28
        (3.12, 3.40, 3.11, 3.38),    # 08:07 -- bounce through the trailed stop
    ])
    tr = _run(_cfg(exit_mode="candle_trail_1min"), pm)
    assert tr.direction == "short" and tr.entry_price == pytest.approx(3.46)
    assert tr.exit_price == pytest.approx(3.28)          # covered at the 08:06 high
    assert tr.result == "WIN" and tr.pnl_per_share > 0
    assert tr.exit_trigger == "trail_stop"


def test_trail_partial_banks_at_target_and_remainder_rides_breakeven_plus():
    """Target hit before any fat candle: trail_partial_at_target (0.5) banks at
    the target, the remainder's stop jumps to breakeven and keeps ratcheting (to
    the 08:05 minute low 3.95); the reversal exits the remainder there."""
    pm = _trail_session("long", [
        (3.98, 4.30, 3.95, 4.05),    # 08:05 -- hits the 2R target 4.28 intrabar
        (4.05, 4.06, 3.60, 3.62),    # 08:06 -- hard reversal
    ])
    tr = _run(_cfg(exit_mode="candle_trail_1min", trail_partial_at_target=0.5), pm)
    assert tr.exit_legs is not None and [leg["kind"] for leg in tr.exit_legs] == ["target", "trail_stop"]
    assert tr.exit_legs[0]["price"] == pytest.approx(4.28)
    assert tr.exit_legs[1]["price"] == pytest.approx(3.95)   # breakeven+ (ratcheted past 3.80)
    assert tr.exit_price == pytest.approx(0.5 * 4.28 + 0.5 * 3.95, abs=0.01)  # cent-rounded blend
    assert tr.result == "WIN" and tr.trail_activated


def test_trail_fat_green_drops_the_target_and_rides_the_runner():
    """A fat-bodied green minute (body 0.21/range 0.23, closes at the top) DROPS
    the fixed target: the 4.40 print that would have banked the 4.28 target sails
    by untouched, and the runner trails out at the 08:06 minute low instead."""
    pm = _trail_session("long", [
        (3.98, 4.20, 3.97, 4.19),    # 08:05 -- FAT GREEN
        (4.19, 4.40, 4.15, 4.38),    # 08:06 -- through the (dropped) target
        (4.38, 4.39, 4.10, 4.12),    # 08:07 -- trails out at 4.15
    ])
    tr = _run(_cfg(exit_mode="candle_trail_1min"), pm)
    assert tr.trail_activated                             # the runner engaged
    assert all(leg["kind"] != "target" for leg in tr.exit_legs)   # target never banked
    assert tr.exit_price == pytest.approx(4.15)           # the 08:06 minute's low
    assert tr.result == "WIN" and tr.pnl_per_share > 0


def test_trail_partial_then_session_end_is_a_win_not_exit_at_open():
    """A banked partial makes it a realized trade: the remainder marking at the
    session's last close is a WIN with a session_end leg -- never EXIT_AT_OPEN
    (that label is only for positions where NOTHING ever banked)."""
    pm = _trail_session("long", [
        (3.98, 4.30, 3.95, 4.05),    # 08:05 -- partial banks at 4.28
        (4.05, 4.06, 4.02, 4.03),    # 08:06 -- quiet; session ends here
    ])
    tr = _run(_cfg(exit_mode="candle_trail_1min", trail_partial_at_target=0.5), pm)
    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "session_end"]
    assert tr.result == "WIN"                             # NOT EXIT_AT_OPEN
    assert tr.exit_trigger == "session_end"
    assert tr.exit_price == pytest.approx(0.5 * 4.28 + 0.5 * 4.03, abs=0.01)  # cent-rounded blend


def test_min_risk_pct_of_price_defers_a_near_degenerate_stop():
    """The long fixture's natural entry-to-stop risk is ~6.3% of price (3.80 vs
    stop 3.56). A floor below that doesn't interfere; a floor above it defers that
    specific touch (a near-zero-risk stop would produce an absurd, unrealistic
    R-multiple for a modest dollar move -- a measurement artifact, not edge)."""
    pm = _pennant_session("long")
    below = _run(_cfg(min_risk_pct_of_price=1.0), pm)
    assert below.entry_price == pytest.approx(3.80) and below.stop_price == pytest.approx(3.56)

    above = _run(_cfg(min_risk_pct_of_price=10.0), pm)
    assert above.entry_price != pytest.approx(3.80)   # that touch was deferred, not taken

    assert _run(_cfg(min_risk_pct_of_price=None), pm).entry_price == pytest.approx(3.80)  # off = no floor


def test_causal_gain_gate_defers_a_break_before_the_gain_is_real():
    """No lookahead: the gain% qualification must hold AS OF the break, not just
    somewhere in the session. Raising the premarket open to 3.20 leaves the coil
    and its break identical, but the running high at the break (4.00) is only
    +25% -- below the 28% gate -- while the session's LATER follow-through
    (4.70, +47%) still passes the full-session screen. The old code qualified on
    that future high and traded the break; now the break defers."""
    pm = _pennant_session("long")
    pm.loc[0, ["open", "high", "low", "close"]] = [3.20, 3.30, 3.18, 3.28]
    tr = _run(_cfg(), pm)
    assert tr.entry_price is None
    assert tr.skipped_gain_not_yet_confirmed >= 1
    # The unmodified session (open 3.00 -> running high 4.00 = +33% at the break)
    # still trades: the gate defers unqualified breaks, it doesn't kill setups.
    assert _run(_cfg(), _pennant_session("long")).entry_price is not None


# --- hold-to-open limiters ------------------------------------------------------

def _drift_session() -> pd.DataFrame:
    """A long pennant break whose aftermath DRIFTS: after the break (fill 3.80,
    stop ~3.56, 2R target ~4.28) price meanders 3.72-3.88 for five minutes until
    the session simply ends -- the hold-into-the-open case the limiters bound."""
    rows = [(3.00, 3.30, 2.98, 3.28), (3.28, 3.60, 3.25, 3.58), (3.58, 3.90, 3.55, 3.88), (3.88, 4.00, 3.85, 3.75)]
    for k in range(12):
        hi = 3.92 - 0.018 * k
        lo = 3.42 + 0.012 * k
        rows.append((lo + 0.02, hi, lo, hi - 0.03))
        rows.append((hi - 0.03, hi - 0.02, lo, lo + 0.02))
    rows.append((3.70, 4.00, 3.68, 3.98))            # the break bar (fill 3.80)
    rows += [(3.80, 3.88, 3.72, 3.80)] * 30          # five minutes of going nowhere
    base = pd.Timestamp("2026-04-09 08:00:00", tz="America/New_York")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 2000.0
    df["dt_et"] = [base + pd.Timedelta(seconds=10 * i) for i in range(len(df))]
    return df


def test_unresolved_break_rides_to_the_open_by_default():
    tr = _run(_cfg(), _drift_session())
    assert tr.entry_price is not None
    assert tr.result == "EXIT_AT_OPEN" and tr.exit_trigger == "session_end"


def test_max_hold_minutes_time_stops_an_unresolved_break():
    tr = _run(_cfg(max_hold_minutes=2.0), _drift_session())
    assert tr.entry_price is not None
    # Entry fills at the 08:04:40 break bar; the first bar at/after +2min is
    # 08:06:40, and the exit is that bar's OPEN -- never into the session end.
    assert tr.exit_trigger == "time_stop" and tr.result in ("WIN", "LOSS")
    assert tr.exit_price == pytest.approx(3.80)
    assert tr.exit_time.endswith("08:06:40-04:00")


def test_latest_exit_time_goes_flat_ahead_of_the_open():
    tr = _run(_cfg(latest_exit_time="08:06"), _drift_session())
    assert tr.exit_trigger == "flat_by_time" and tr.result in ("WIN", "LOSS")
    assert tr.exit_time.endswith("08:06:00-04:00")


def test_stale_progress_exit_cuts_a_dead_drift_but_spares_real_progress():
    # Best excursion after entry is 0.20 of the 0.48 entry-to-target span (~42%).
    # Demanding 90% progress after a minute cuts the drift...
    cut = _run(_cfg(stale_exit_minutes=1.0, stale_exit_min_progress=0.9), _drift_session())
    assert cut.exit_trigger == "stale_progress" and cut.result in ("WIN", "LOSS")
    # ...while a 30% bar is already met, so the trade keeps its time and still
    # ends as the default hold-to-open.
    keep = _run(_cfg(stale_exit_minutes=1.0, stale_exit_min_progress=0.3), _drift_session())
    assert keep.exit_trigger == "session_end" and keep.result == "EXIT_AT_OPEN"


def test_windowed_persistence_expires_a_forgotten_pumper():
    """The gate is a WINDOWED volume rate: identical geometry, but a tape that
    stops printing in the minutes before the break is a forgotten pumper and must
    not trade -- even though its CUMULATIVE session RVOL (which only grows, and
    which the old gate used) still towers over the threshold."""
    cfg = _cfg(persistence_mode="rvol", min_sustained_rvol=2.0, persistence_window_min=1.0)
    pm = _pennant_session("short")
    alive = _run(cfg, pm)
    assert alive.entry_price is not None            # normal tape: the market is looking
    dead_pm = pm.copy()
    tz = "America/New_York"
    mask = (dead_pm["dt_et"] >= pd.Timestamp("2026-04-09 08:02:00", tz=tz)) & \
           (dead_pm["dt_et"] <= pd.Timestamp("2026-04-09 08:03:59", tz=tz))
    dead_pm.loc[mask, "volume"] = 0.0               # the tape dies just before the break
    dead = _run(cfg, dead_pm)
    assert dead.entry_price is None and dead.result == "NO_WEDGE"
