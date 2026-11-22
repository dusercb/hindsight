"""Configuration SCHEMA and loader -- not the file you edit to tune the backtester.

config.yaml (project root) is the single source of truth for settings: the CLI
loads it via load_config() below, and any key present there always wins. The
dataclass defaults in this module are only fallbacks for keys *omitted* from
the yaml -- and since config.yaml sets every strategy key explicitly, editing
the numbers here normally changes nothing. Edit config.yaml instead.

To verify what a run actually executed with, check the console echo at run
start or the config_used.yaml written into every run folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date as date_cls
from pathlib import Path
from typing import Any
import os
import yaml

from .calendar_blackout import parse_recurring_range


def parse_hhmm(value: str, field_name: str, source: object) -> tuple[int, int]:
    """Parse a strict 'HH:MM' 24-hour string (the same format already used by
    session.premarket_start/regular_open) -- shared by config validation (fails
    loudly at load time) and strategy/wedge code (which trusts the string is
    well-formed but re-parses it the same way, since tests can build a config
    dataclass directly without going through load_config). field_name should be
    the full dotted key ("strategy.latest_entry_time") when called from
    validation, so the error names the exact yaml key."""
    try:
        h_str, m_str = str(value).split(":")
        h, m = int(h_str), int(m_str)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
        return h, m
    except (ValueError, AttributeError):
        raise ValueError(f"{source}: {field_name} must be a 'HH:MM' 24-hour time string, got {value!r}") from None


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17
    readonly: bool = True


@dataclass(frozen=True)
class DataConfig:
    input_csv: Path
    reference_csv: Path | None
    cache_dir: Path
    output_dir: Path
    use_cache: bool = True
    bar_size: str = "10 secs"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    # Minimum seconds between live historical-tick requests (IBKR pacing limits).
    tick_request_pause_s: float = 3.0
    # Minimum seconds between live historical BAR requests (sync fetch and the
    # async pre-warmer share one clock). 0 = unpaced, the original behavior.
    # IBKR's documented small-bar pacing is ~60 requests/10 min; days of unpaced
    # mass ingest got the account soft-throttled to the point where every
    # request queued past the timeout (an 11-hour run yielded 6 fetches).
    # Deliberate pacing keeps a big ingest under the radar instead of
    # re-triggering the penalty box.
    bar_request_pause_s: float = 0.0
    # Max seconds to wait for any single IBKR request before treating it as
    # unavailable. ib_insync's default is to wait FOREVER, so one silently
    # unanswered request (dropped connection, stalled gateway) used to hang the
    # entire run with no indication of what it was stuck on.
    request_timeout_s: float = 60.0
    # Total live attempts (accumulated ACROSS runs) allowed for a tick window that
    # keeps coming back empty even though its bar shows trades. Transient failures
    # heal within this budget; after it, the window is accepted as a permanent gap
    # in IBKR's tick store and never requested again (the candle-path heuristic
    # resolves those bars) -- without a cap, every run re-paid two paced requests
    # per gap window forever.
    max_empty_tick_attempts: int = 4
    # Same self-healing budget for a candidate's BAR fetch coming back empty
    # (delisted/renamed ticker, no data for that day): empty bar responses were
    # never cached, so every run re-paid a live IBKR request per dead candidate
    # (253 of them in one measured run). Transient failures (dropped connection,
    # timeout) heal within the budget; after it, the candidate is accepted as
    # having no data and marked INSUFFICIENT_DATA for free. A later successful
    # fetch always clears the memo.
    max_empty_bar_attempts: int = 3
    # Autonomous throttle/disconnect recovery. When IBKR's history farm stops
    # serving (soft-throttle after heavy ingest, or TWS's nightly restart window
    # refusing connections), the run does NOT abort: it pauses, reconnects,
    # cools down with escalating backoff, and re-probes a liquid ticker until the
    # farm answers real data again -- then continues exactly where it left off.
    # This is the total time it will keep trying before giving up gracefully
    # (partial results stay --resume-able). Default 6h comfortably covers a
    # throttle cooldown plus the ~1h nightly TWS restart. Set 0 to disable
    # auto-recovery (revert to abort-on-throttle).
    recovery_max_total_wait_s: float = 21600.0
    # Fallback fetch duration when the default "2 D" bar window doesn't reach back far
    # enough to find a previous trading session (e.g. a holiday sitting next to a weekend).
    wide_duration: str = "6 D"


@dataclass(frozen=True)
class SessionConfig:
    timezone: str = "America/New_York"
    premarket_start: str = "04:00"
    regular_open: str = "09:30"


@dataclass(frozen=True)
class StrategyConfig:
    # Minimum relative volume before a ticker is considered to have had a "sudden
    # move" worth evaluating at all: cumulative premarket volume so far vs. this
    # ticker's own average REGULAR-SESSION daily volume over its trailing history
    # (the standard RVOL definition -- not just that morning's own bars).
    min_rvol: float = 5.0
    # Trailing trading days used to compute the RVOL baseline. Requires one cheap,
    # cached, live IBKR daily-bar request per candidate (generous IBKR duration
    # limits for 1-day bars, unlike the 10-second bars used elsewhere).
    rvol_lookback_days: int = 20
    # Statistic over those days: "median" (robust -- one prior halt/news day
    # printing 33M shares must not inflate a 66K-share/day ticker's baseline 35x
    # and stall qualification until the move is over, as happened to ARTL) or
    # "mean" (conservative: a big prior day raises the bar for the whole window).
    rvol_baseline_stat: str = "median"
    # Minimum days of daily-volume history required before RVOL can be computed at
    # all (e.g. a recently listed ticker won't have this) -- below it, RVOL is
    # unavailable and the candidate is marked INSUFFICIENT_DATA, not disqualified.
    rvol_min_history_days: int = 0
    # Binary "was the spike worth trading?" check: minimum premarket gain, measured
    # from the premarket OPEN (the session's first print) to the running premarket
    # high at the moment of evaluation. Independent of RVOL (and of when RVOL
    # crossed), and NOT measured from the previous session's close.
    min_premarket_gain_pct: float = 20.0
    entry_offset: float = 0.01
    # How the target/stop distances from the entry fill are measured:
    #   "cents"   = flat dollar offsets (target_offset / stop_offset), the original
    #               behavior -- a fixed 0.50/0.25 no matter the share price.
    #   "percent" = a percentage OF THE ENTRY FILL (target_offset_pct /
    #               stop_offset_pct). A 25c stop is ~9% of a $2.64 stock but <2% of
    #               a $12 one, so a flat-cents stop is real structure on cheap names
    #               and pure noise on expensive ones (backtest run_20260709_112928:
    #               <$5 names won 55%, >=$8 names 14%, on the same 0.50/0.25). Percent
    #               mode gives every price a proportional stop so higher-priced movers
    #               stay tradeable instead of being noise-stopped or price-banned.
    # R-multiple and all risk math use the ACTUAL per-share risk (fill - stop) in
    # either mode, so switching modes keeps R meaningful.
    offset_mode: str = "cents"
    target_offset: float = 0.20  # used when offset_mode == "cents"
    stop_offset: float = 0.10    # used when offset_mode == "cents"
    target_offset_pct: float = 5.0  # used when offset_mode == "percent"
    stop_offset_pct: float = 2.5    # used when offset_mode == "percent"
    # WHERE the protective stop sits:
    #   "offset"       = below the fill by stop_offset / stop_offset_pct (per
    #                    offset_mode above) -- the original behavior.
    #   "pullback_low" = at the setup's own pullback low: the dip's low IS the
    #                    thesis, so price trading back below it means the
    #                    reversal failed -- that low is the maximum loss,
    #                    wherever the entry filled. Risk per share then varies
    #                    with the structure (deep dip = wide stop); R-multiples
    #                    always divide by the real (fill - stop) so they stay
    #                    honest. Natural partner of the dip_* entry modes.
    #   "red_candle_low" = the LOW of the last red candle whose high was broken
    #                    (dip_red_break only) -- a TIGHTER stop than pullback_low
    #                    (the red candle's own low, not the whole dip's low), so R
    #                    is smaller. Falls back to pullback_low when there is no
    #                    identifiable red candle (non-dip_red_break modes).
    stop_mode: str = "offset"
    # HOW the profit target is placed:
    #   "offset"     = above the fill by target_offset / target_offset_pct (per
    #                  offset_mode) -- the original behavior.
    #   "r_multiple" = derived from the KNOWN risk: entry and max loss are both
    #                  fixed at fill time, so the sell is target_r_multiple times
    #                  that risk above the entry (target = fill + R * (fill -
    #                  stop)). Keeps a constant reward:risk on every trade even
    #                  when stop_mode = "pullback_low" makes the dollar risk vary.
    target_mode: str = "offset"
    target_r_multiple: float = 2.0
    # Which candle timeframe the WHOLE setup is detected and traded on (HOD arming,
    # pullback, red-candle break, the trigger). Fills stay tick-resolved either way.
    #   "10s" = the native 10-second bars (the original behavior).
    #   "1m"  = resample to 1-minute candles first, so the HOD/dip/red-break are
    #           read off the 1-minute chart instead. Coarser, fewer setups, less noise.
    #   "2m"  = resample to 2-minute candles -- coarser still.
    # The 1-minute indicators/gates (MACD/RVOL/VWAP) stay 1-minute regardless.
    chart_timeframe: str = "10s"
    # HOW the position is managed AFTER entry:
    #   "standard"          = the fixed target (target_mode) + fixed stop (stop_mode),
    #                         with the optional scale_out ladder -- the original behavior.
    #   "candle_trail_1min" = a 1-minute trailing manager: the stop ratchets up to the
    #                         prior COMPLETED 1-minute candle's low (never down), and the
    #                         first time a FAT-BODY GREEN 1-minute candle closes, the fixed
    #                         target is DROPPED and the trade rides the trailing stop until
    #                         it's hit (or the regular open). Until that first fat green, the
    #                         fixed target is still live. Pairs naturally with stop_mode
    #                         "red_candle_low" but works with any initial stop. scale_out is
    #                         ignored in this mode.
    exit_mode: str = "standard"
    # A 1-minute candle counts as "fat-body green" (which drops the target under
    # exit_mode "candle_trail_1min") when it closes up (close > open), its body is at
    # least this fraction of its high-low range, AND it closes in the top third of
    # that range -- the same strong-close signal that separated winners in the
    # candlestick analysis. Must be in [0, 1].
    fatgreen_body_min: float = 0.70
    # candle_trail_1min runner engine. When the fixed 1:1 target is hit BEFORE any
    # fat-green candle fires (the common case on fast movers, where the target is
    # reached in under a minute so the fat-green runner never engages), sell this
    # FRACTION of the position at the target and let the REST run: the target is
    # dropped, the remaining shares' stop jumps to breakeven, and it trails on the
    # 1-minute lows until stopped. 1.0 = the original all-out exit at the target (no
    # runner unless a fat green fired first). 0.5 = bank half at 1R, run half. This
    # is what makes candle_trail a genuine runner engine on 10-second charts, not
    # just a loss-protecting stop. Must be in (0, 1]. Only used by candle_trail_1min.
    trail_partial_at_target: float = 1.0
    # WHICH price break the entry triggers on. Every mode reuses the identical
    # upstream machinery -- stock qualification (gain%/RVOL/float/price), HOD
    # arming, pullback detection, the retrace-invalidation and all entry gates
    # (MACD, RVOL-so-far, volume, red/green, indicator gates, time window). ONLY
    # the price level whose break fires the entry differs:
    #   "hod_break"     = buy the break of the armed HOD + entry_offset (the
    #                     original strategy -- a confirmed new-high breakout).
    #   "dip_swing_high"= buy the break of the highest high printed since the
    #                     pullback low (the first higher-high off the low). Fires
    #                     EARLIER than the HOD reclaim -> tighter stop, better R:R,
    #                     but less confirmation.
    #   "dip_red_break" = buy the break of the LAST RED candle's high in the dip.
    #                     The dip is built of red candles; breaking back above the
    #                     most recent one is the first proof the sellers who built
    #                     it lost control. Each newer red candle trails the trigger
    #                     down with the dip. A dip with no red candle yet produces
    #                     no trigger. Pairs naturally with stop_mode "pullback_low"
    #                     + target_mode "r_multiple" (entry = last-red break, max
    #                     loss = the dip low, sell = R x that known risk).
    #   "dip_trendline" = fit a descending line through the pullback's lower-high
    #                     pivots (HOD -> pullback low) and buy the break of that
    #                     projected line. Falls back to dip_swing_high when the dip
    #                     has too few pivots to define a line. The literal
    #                     "break of the downtrend" entry.
    #   "dip_ema_reclaim" = buy when price reclaims dip_reclaim_indicator (the
    #                     1-minute EMA9/EMA20/VWAP) after the dip pierced below it
    #                     -- a moving-average expression of the same downtrend break,
    #                     reusing the indicators already computed for the block_*
    #                     gates and drawn on every chart. Requires a prior breach; a
    #                     dip that never lost the indicator produces no reclaim entry.
    # The dip_* modes enter below the HOD, so they turn some setups that never
    # reclaim the HOD (today's INVALID_RETRACE / STALE_HOD / NO_TRIGGER) into
    # trades -- more entries, at the cost of less confirmation. A/B them against
    # hod_break on the same candidate set. A failed entry gate still spends the
    # setup (STALE_HOD / rescan) exactly as in hod_break.
    entry_mode: str = "hod_break"
    # Which 1-minute indicator "dip_ema_reclaim" reclaims: "ema9", "ema20" or
    # "vwap". Ignored by every other entry_mode.
    dip_reclaim_indicator: str = "ema9"
    # Close-confirmed breakout entry (A/B toggle, applies to EVERY entry_mode).
    #   False = the original behavior: fill intrabar the moment a bar's HIGH
    #           crosses the trigger (a wick through the level fills).
    #   True  = don't fill on the touch. Only once a bar CLOSES at/above the
    #           trigger AND in the top (1 - breakout_close_min_frac) of its own
    #           range -- i.e. a strong-bodied breakout candle that HELD into its
    #           close, not a rejected wick -- enter on the NEXT bar's open. An
    #           unconfirmed (weak/wicky) break is DEFERRED, not condemned: the
    #           HOD stays armed and a later bar can still print a confirming
    #           close (same treatment as a missing MACD bar, never STALE_HOD).
    # Rationale (run_20260714_105120, in-sample): breaks whose candle closed in
    # the bottom third of its range won 18% (-0.59R); top third won 77% (+0.58R).
    # Costs a bar of timing and a slightly worse fill (next open, above the
    # trigger), and skips rockets that gap away without a second bar -- an A/B,
    # not a free win. Every downstream gate/stop/target/exit path is unchanged.
    require_breakout_close_confirm: bool = False
    # The breakout candle's close must sit at or above this fraction of its
    # high-low range for the break to confirm (only when the toggle above is on).
    # 0.667 = "top third". 1.0 demands a close exactly on the high; 0.5 = top
    # half (a looser filter that in-sample still lifted win% to ~70). Must be in
    # [0, 1].
    breakout_close_min_frac: float = 0.667
    # MACD periods for the 1-minute bullish-momentum entry gate (fast/slow EMAs
    # of the close, then the signal EMA of that line). 12/26/9 is the standard.
    # The gate itself is fixed: an entry needs MACD both above zero AND above its
    # signal line; only these periods are configurable. Each must be an integer
    # >= 1, and fast should be < slow for MACD to mean the usual thing.
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Scale-out ladder. When ON and the planned exit (target) is hit: sell
    # scale_out_sell_pct % of the REMAINING shares at that exit, the stop
    # ratchets to scale_out_stop_offset BELOW that exit (a stop placed AT the
    # exit would fill on the very next print -- price is already trading
    # there), and the planned exit moves up by scale_out_step. The ladder
    # completes in at most scale_out_max_sells sells: the final sell liquidates
    # the whole remainder (defaults: 50% -> 25% -> final 25%). The remainder
    # exits earlier at the ratcheted stop (or session end). A scaled trade's
    # WIN/LOSS is the sign of its NET P&L across all legs. OFF = single
    # all-out exit at target/stop, the original behavior.
    scale_out: bool = True
    scale_out_sell_pct: float = 50.0
    scale_out_step: float = 0.10
    scale_out_max_sells: int = 3
    scale_out_stop_offset: float = 0.01
    # WHERE the rip's base (the "launch low" the retrace % is measured against) is
    # anchored. The gain gate delays retrace evaluation until the HOD is up
    # min_premarket_gain_pct, by which time "session_low" can lock onto a stale
    # early-session flush far below any real trend -- so a chart that grinds up
    # choppily from a morning flush and then spikes-and-fails scores a shallow
    # retrace and passes the trend filter it should fail (ADVB 2025-12-10: rip_low
    # 7.36 from 08:00, HOD 10.80 at 08:53 -> a ~100%-of-the-recent-spike dip read as
    # 29%). The alternatives anchor the rip to the RECENT leg instead:
    #   "session_low" = lowest low since the scan start / last reset (the original).
    #   "recent"      = lowest low within rip_recent_minutes before the HOD.
    #   "swing_low"   = the most recent 3-bar pivot low before the HOD (the base of
    #                   the final leg). Both fall back to session_low when too few
    #                   bars exist. Never below the true running low, never above HOD.
    rip_anchor: str = "session_low"
    # Lookback window (minutes before the HOD) for rip_anchor "recent".
    rip_recent_minutes: float = 5.0
    # The rip's trending-support line: max % of the rip (base -> HOD, see rip_anchor)
    # that the dip may retrace. A dip holding this line keeps the setup bullish;
    # breaching it breaks the formation and invalidates the setup. Measured against
    # the rip itself, NOT the move-from-open used by the gain% check.
    max_pullback_retrace_pct: float = 45.0
    # Retrace-support gates: block a trade if the retrace (HOD through the
    # current touch) ever traded below this 1-minute indicator's value at ANY
    # point, not just at the touch itself -- a dip that pierced trend support
    # even briefly, then recovered, still counts as pierced. Each indicator is
    # its own on/off switch; all default off (no filtering). Indicator values
    # use the same causal prior-completed-minute lookup as MACD/RVOL (no
    # lookahead). The same EMA9/EMA20/VWAP series are drawn on every trade chart
    # (grey/dark blue/yellow) regardless of whether these gates are enabled.
    block_retrace_below_ema9: bool = False
    block_retrace_below_ema20: bool = False
    block_retrace_below_vwap: bool = False
    # Point-in-time entry gates: block THIS touch if the trigger price itself is
    # below the indicator's value as of the prior completed minute (same causal
    # lookup as MACD/RVOL -- no lookahead), regardless of whether the retrace
    # ever pierced it earlier. Distinct from block_retrace_below_* above (which
    # judges the whole HOD-to-touch path, not just the touch): a retrace can dip
    # below EMA9 and fully reclaim it by the time of the touch, which the
    # retrace-path gates would still block but these would allow. A later touch
    # of the same setup can still fire once price reclaims the indicator -- this
    # doesn't invalidate the setup, just this attempt. All default off.
    block_buy_below_vwap: bool = False
    block_buy_below_ema9: bool = False
    block_buy_below_ema20: bool = False
    # How the red/green volume check compares selling vs buying inside the move
    # window (HOD -> current bar) before allowing an entry:
    #   "total"       = cumulative red volume must NOT outweigh cumulative green
    #                   volume (majority flow -- one heavy pullback flush bar
    #                   can't veto the first re-break by itself; that flush IS
    #                   the setup on fast movers like ARTL)
    #   "largest_bar" = original stricter rule: the single largest red bar must
    #                   be smaller than the single largest green bar
    red_volume_check: str = "total"
    # Require the dip after the HOD to contain at least one RED 10-second candle
    # (a bar that closed below its open) somewhere between the HOD and the pullback
    # low -- evidence of a genuine down-tick, not just a sideways drift or a
    # one-print wobble. Off by default. Evaluated at the trigger touch; a dip with
    # no red candle rejects the entry (skipped_no_red_dip_candle) and, like every
    # other trigger-touch gate, marks the HOD-break stale (STALE_HOD under
    # first_setup_only, else rescan). Uses the raw 10-second bars the backtest runs
    # on, not the 1-minute resample.
    require_red_dip_candle: bool = False
    # Minimum dip size for a pullback to register: at least min_pullback_cents
    # deep, and -- when min_pullback_pct is set -- also at least that % of the
    # HOD price. Set min_pullback_pct to null to size pullbacks by the flat
    # cents floor alone.
    min_pullback_cents: float = 0.05
    min_pullback_pct: float | None = None
    min_price: float = 2.0
    max_price: float = 20.0
    # Share-float band (in shares): a candidate whose float is above max_float or
    # below min_float is skipped (SKIPPED_FLOAT_FILTER). Both null (default) = no
    # float filter and no extra data requests. Float is taken from the input /
    # reference_floats.csv when supplied; otherwise it is fetched from IBKR's
    # Reuters company snapshot (reqFundamentalData "ReportSnapshot" -> TotalFloat),
    # cached per ticker. [BACKTEST CAVEAT] that snapshot is the ticker's CURRENT
    # float, not its float as of the historical backtest date -- offerings/splits
    # since then aren't reflected. For most low-float movers it's close enough, but
    # supply historically-accurate values via reference_floats.csv when precision
    # matters. Requires an IBKR fundamental-data subscription; if unavailable and no
    # CSV float is supplied, the candidate is marked INSUFFICIENT_DATA (can't confirm
    # the float criterion) rather than silently traded.
    min_float: float | None = None
    max_float: float | None = None
    # Minimum cumulative premarket volume, in shares, that must have ACTUALLY
    # printed as of a given trigger touch (through the prior completed minute --
    # the same causal, lagged lookup used for the RVOL-so-far check) before that
    # touch may fill. An absolute floor, independent of RVOL (which is only
    # relative to the ticker's own daily-average baseline and says nothing about
    # how thin that baseline itself is). A touch that fails this is skipped, not
    # disqualifying: a LATER touch of the same setup can still fire once enough
    # volume has printed (mtr.skipped_premarket_volume_not_yet_confirmed counts
    # these, same pattern as skipped_rvol_not_yet_confirmed). A cheap upfront
    # short-circuit also disqualifies the whole candidate (DID_NOT_QUALIFY,
    # disqualify_reason "premarket_volume") when cumulative volume never reaches
    # this floor even by session close -- a valid necessary-condition check since
    # cumulative volume only grows, same shortcut RVOL's "ever reached" check uses.
    #
    # IMPORTANT: this must be checked against volume AS OF THE TOUCH, not the
    # session's eventual full-day total (tr.premarket_volume by itself)  -- most
    # premarket volume is back-loaded toward the open, so a full-session number
    # is mostly volume that HASN'T HAPPENED YET at the time of an early pullback
    # (median trade in the 59-trade sample below had only printed ~9% of that
    # day's eventual total by entry). An earlier version of this filter used the
    # full-session total directly and was, in effect, a look-ahead bug.
    #
    # Default 50,000: once corrected to the causal (as-of-touch) volume, the
    # win-rate/R-multiple signal is much weaker and non-monotonic in the 59-trade
    # sample (run_20260708_143121) than the flawed full-session version suggested
    # -- e.g. a 100k-150k floor would have DROPPED trades with strongly positive
    # combined R, i.e. cut real winners, not junk. Only one trade in the whole
    # sample was genuinely dubious by this measure (7,827 shares printed by entry,
    # a marginal +0.28R). 50,000 clears that single outlier without touching
    # anything else observed so far -- a sanity floor, not a tuned optimum;
    # revisit once more trades accumulate. See also risk.max_pct_of_volume_at_
    # entry, which caps position SIZE against the ticker's actual volume rather
    # than screening the candidate out entirely. Defaults to null here (like
    # min_float/max_float above) -- config.yaml sets the tuned 50,000 value; this
    # fallback only matters for code that builds a StrategyConfig directly (e.g.
    # tests) without going through config.yaml.
    min_premarket_volume: float | None = None
    # Max bid/ask spread allowed at the trigger bar. Requires a live IBKR
    # historical BID_ASK tick request per trigger attempt; leave null to skip
    # the spread check entirely (no extra requests).
    max_spread_cents: float | None = None
    # Max entry-to-stop distance, as a % of the entry fill, that a trade may carry.
    # Risk per share = fill - stop; this rejects a touch when (fill - stop) / fill *
    # 100 exceeds the cap -- the structural stop (pullback_low) sits too far below
    # the entry to be worth taking, which under position_pct_of_equity sizing also
    # means the trade would risk an outsized slice of equity. Like the other
    # trigger-touch gates it marks the break stale (STALE_HOD under first_setup_only,
    # else rescans) and is recorded as skipped_risk_too_wide. Especially relevant
    # with require_breakout_close_confirm on, which raises the entry (wider stop);
    # in-sample (run_20260714_105120) trades with risk > ~7.5% of entry underperformed.
    # Leave null to disable (the default) -- a filter for the analysis layer to sweep,
    # not a live setting until the data supports a threshold.
    max_risk_pct: float | None = None
    # Only allow a trigger touch to fill within this time-of-day window ("HH:MM",
    # 24-hour, inclusive, in session.timezone). A touch outside the window is
    # skipped like any other gate -- a LATER touch of the same armed setup can
    # still fire if it lands inside the window. Leave either null to not bound
    # that side. Backtest evidence: entries before 07:00 ET have a materially
    # worse track record than later ones (thin premarket liquidity, less
    # reliable follow-through) -- this is a filter to test that hypothesis on
    # more data, not a threshold tuned to fit one run.
    earliest_entry_time: str | None = None
    latest_entry_time: str | None = None
    # True (default): only ever evaluate the first HOD/pullback setup of the day;
    # if it breaches max_pullback_retrace_pct, give up on this candidate entirely.
    # False: abandon just that setup and keep scanning the rest of the premarket
    # session for a fresh HOD and pullback. Applies to EACH move's own hunt (see
    # max_trades_per_day below), so it still governs "give up entirely" the
    # moment any move's retrace search fails -- no further moves are attempted
    # after that, even if max_trades_per_day would otherwise allow more.
    first_setup_only: bool = True
    # How many trades to take from ONE ticker/day: 1 (default) = only the first
    # move, today's behavior. N > 1: after a completed trade (win or loss), if
    # the premarket session isn't over, keep hunting for a brand-new HOD/pullback/
    # trigger -- the 2nd move, 3rd, etc. -- up to N total trades. Every move is
    # judged by the exact same gates (gain% still anchored to the ORIGINAL
    # premarket open, MACD/RVOL-so-far at its own trigger, red/green volume,
    # entry limit); only the HOD/pullback/rip state resets per move, the same
    # reset already used when abandoning an invalidated retrace mid-search.
    max_trades_per_day: int = 1
    # Press winners: total trades allowed from one ticker/day WHILE each completed
    # trade keeps winning. 1 (default) = off: the first trade ends the day
    # whatever its result. 3 = a winning first trade unlocks the hunt for a
    # second, genuinely fresh HOD/pullback (the new HOD must exceed the true
    # session high); a winning second unlocks a third. Any LOSS ends the day on
    # the spot. Unlike max_trades_per_day above (which re-hunts after ANY
    # completed trade, win or lose), this only ever adds trades on strength.
    trade_winners_qty: int = 1
    # Extra volume-confirmation gate for those win-continuation entries: at the
    # continuation break, the session VWAP (causal, prior completed minute) must
    # sit at least this % ABOVE the prior winning trade's entry price. VWAP is
    # volume-weighted, so it only lifts that far this fast when heavy volume
    # keeps printing at higher prices -- proof the run is still under real
    # accumulation (a volatile, heavily traded long), not a thinning drift
    # poking marginal new highs. 0.0 = VWAP merely above the prior entry;
    # null = no VWAP condition on continuations. Deliberately DIFFERENT from
    # block_buy_below_vwap (trigger vs. VWAP at any entry): this one anchors to
    # the prior entry price and applies only to win-continuation moves.
    continuation_vwap_above_entry_pct: float | None = None
    exit_at_regular_open: bool = True
    # Fill tolerance for the simulated software-triggered premarket limit order, in
    # DOLLARS/share (flat cents, not a percent): how far ABOVE the trigger the fill
    # may be and still count as filled -- a fill worse than trigger + this is
    # rejected (the tape gapped through too fast for a marketable limit to catch).
    # Flat cents models how these orders are really placed (e.g. Webull caps a
    # marketable limit at ~+$0.15 of the market to guarantee a fill), so a $3 and a
    # $15 name get the SAME real-world buffer rather than a price-scaled percent
    # (which was far too tight on cheap names and too loose on pricey ones). The
    # actual simulated fill is still max(trigger, bar open) / the real tick price --
    # this only decides whether a fill happened at all, not at what price.
    entry_limit_slippage_cents: float = 0.15
    # Simulated order latency: entry and STOP fills are taken from the tick this
    # many seconds after the trigger/stop actually crossed, not the crossing tick
    # itself -- modeling the real delay between a signal firing and an order
    # reaching the market. Target fills are exempt: the target is a limit order
    # already resting at the exchange, so it fills at the crossing print itself.
    # Only meaningful (and only applied) when tick data resolves the fill; the
    # 10-second-bar fallback has no sub-second resolution to model this against.
    order_latency_seconds: float = 0.1
    # What to do when a bar touches both stop and target ("path-ambiguous"):
    #   "ticks"        - fetch IBKR historical ticks for that bar and use the actual
    #                    order in which the levels traded. Used broadly: every entry
    #                    fill and every single-sided stop/target touch is resolved
    #                    from real ticks when this is set, not just the rarer
    #                    both-touched-in-one-bar case. If ticks are genuinely
    #                    unavailable for the bar, falls back to the "heuristic"
    #                    candle-path guess below -- an ambiguous bar is never an
    #                    automatic conservative loss unless explicitly configured.
    #   "heuristic"    - candle-path guess via resolve_intrabar (ambiguous bars only):
    #                    a bullish bar most plausibly printed its low early
    #                    (open -> low -> high -> close), a bearish bar its high early.
    #   "conservative" - always count the stop first on an ambiguous bar (a loss)
    intrabar_policy: str = "ticks"


@dataclass(frozen=True)
class RiskConfig:
    # All-in position sizing: each trade commits this % of the CURRENT account
    # value (equity compounds trade to trade, in chronological order), buying as
    # many whole shares as that affords at the entry price. Staying below 100
    # leaves headroom for slippage. Set to null to use the fixed `shares` count
    # per trade instead.
    position_pct_of_equity: float | None = 50.0
    # Fixed shares per trade -- only used when position_pct_of_equity is null.
    shares: int = 100
    # Broker commission per share, charged on BOTH sides of the trade (entry +
    # exit): pnl_per_share is reduced by 2 x fees_per_share, so a round trip of
    # `shares` costs 2 x fees_per_share x shares off that trade's P&L. 0 = free.
    fees_per_share: float = 0.0
    # Flat platform/subscription cost in DOLLARS charged to the account once per
    # calendar month over the backtest's trading span (first trade month through
    # last, gaps included), deducted from compounding equity so it also shrinks
    # later position sizes. Reported as monthly_fees_total / net_pnl_after_fees in
    # the summary. null (default) = no monthly fee. Distinct from fees_per_share
    # above, which is a per-share broker commission on each fill.
    monthly_fees: float | None = None
    starting_equity: float = 25_000.0
    # Second, independent cap on share count: at most this % of volume_at_entry
    # (the ticker's own CUMULATIVE premarket volume as of the actual entry --
    # THIS session's real volume, causally, same underlying quantity
    # strategy.min_premarket_volume's per-touch check uses, not the
    # daily-average RVOL baseline) -- whichever of this and
    # position_pct_of_equity/shares affords fewer shares wins. This caps
    # position SIZE rather than screening the trade out, and scales
    # automatically as position sizing grows (bigger starting_equity or
    # position_pct_of_equity): a fill that would eat an
    # unrealistic slice of the day's actual liquidity gets shrunk instead of
    # traded at a size the backtest's tick-based fill model can't really vouch
    # for (it doesn't model market impact from the order itself). In the same
    # 59-trade sample used for min_premarket_volume, only a handful of trades
    # exceeded 1% participation (one reached 7.6%) -- 2% leaves normal trades
    # untouched while clipping those outliers. Defaults to null here (config.yaml
    # sets the tuned 2.0) -- this fallback only matters for code building a
    # RiskConfig directly without config.yaml.
    max_pct_of_volume_at_entry: float | None = None
    # Fixed-fractional RISK sizing (additive; null = off, current notional/shares
    # behavior unchanged). When set, the position is capped so the entry-to-stop
    # loss (entry_price - stop_price, the trade's real per-share risk at fill) is at
    # most this % of the CURRENT account value -- a constant dollar risk per trade
    # regardless of how wide or tight the structural stop is. It is applied as a
    # CEILING on top of the existing sizing: the final share count is the MINIMUM of
    # (a) this risk cap, (b) the position_pct_of_equity/shares notional size, and
    # (c) the max_pct_of_volume_at_entry cap. So when the stop is tight (risk isn't
    # the binding constraint) the notional % naturally wins -- you never buy MORE
    # than position_pct_of_equity allows just because the stop is close; and a wide
    # stop shrinks the position instead of blowing an outsized hole. A trade with no
    # stop_price, or a non-positive risk (stop >= entry, which shouldn't happen),
    # skips the risk cap and falls back to the notional size. Example: 2.0 risks 2%
    # of equity per trade. Which cap actually bound each trade is recorded on
    # TradeResult.size_basis ("risk"/"notional"/"volume"/"shares"/"funding") for
    # transparency.
    risk_pct_of_equity: float | None = None
    # FUNDING model for positions that are open AT THE SAME TIME (two tickers'
    # entries overlapping -- and, under wedge.concurrent_with_dip, two strategies'
    # on the same name). Sizing caps above are always computed against realized
    # account equity, so each trade's RISK stays constant; this decides whether
    # that size can actually be BOUGHT while other positions hold capital:
    #   null = unlimited parallel funding -- every trade sizes as if it were the
    #          only one (the original behavior; overlap is not tracked).
    #   1.0  = cash-only: committed notional across all open positions may not
    #          exceed 1x equity -- a second entry only gets whatever cash the
    #          first left over.
    #   2.0  = margin: committed notional may reach 2x equity (Reg-T-style), so
    #          a second entry keeps its full risk-derived size, funded on margin,
    #          until the cap binds. TradeResult.margin_used / leverage_at_entry
    #          record what each entry actually borrowed; a funding-capped trade
    #          gets size_basis "funding".
    max_leverage: float | None = None
    # SIZE-TO-RISK. When true, the per-trade cash cap (position_pct_of_equity) is
    # lifted to the leverage ceiling (max_leverage x equity) so the RISK cap
    # (risk_pct_of_equity) governs: a tight structural stop is funded on margin to
    # reach the full target risk instead of being shortchanged by cash. Exposure is
    # still bounded by max_leverage (single-trade AND concurrent, via the funding
    # walk) exactly as before -- a trade needing more than max_leverage to hit the
    # risk target is capped there (size_basis "notional"/"funding"). Requires
    # risk_pct_of_equity and max_leverage set. false (default) = the classic
    # cash-notional cap; config.yaml opts in.
    size_to_risk: bool = False


@dataclass(frozen=True)
class ReportingConfig:
    write_trades_csv: bool = True
    write_summary_json: bool = True
    write_sqlite: bool = True
    write_trade_folders: bool = True
    write_charts: bool = True
    write_html_report: bool = True
    # WHICH results get a trade folder (trade.json + bars.csv + chart.png).
    # Folders were written for every candidate: one measured run rendered 1,613
    # matplotlib charts (~0.9s each, ~24 minutes, 600MB) of which only 24 were
    # trades. Every candidate always keeps its full row in the trades CSV/SQLite
    # regardless of this setting -- this gates only the per-candidate folder.
    #   "trades" = entries that actually filled (WIN/LOSS/EXIT_AT_OPEN) -- fastest.
    #   "setups" = trades plus formed-but-untraded setups (STALE_HOD,
    #              INVALID_RETRACE, NO_TRIGGER) -- keeps the charts that diagnose
    #              why a setup was rejected (the AGPU stale-HOD investigation used
    #              exactly such a chart) while skipping price/float/data rejects
    #              that never had a setup to look at.
    #   "all"    = every candidate, the original behavior.
    trade_folders_for: str = "setups"
    verbose: bool = True


@dataclass(frozen=True)
class MonteCarloConfig:
    # Off by default -- a single backtest run is one realized ordering of the
    # trades; Monte Carlo re-samples that trade population many times to show the
    # RANGE of equity outcomes the same edge could plausibly produce, and how
    # much of the headline result was sequence/luck. Each simulation re-runs the
    # exact same position sizing (compounding position_pct_of_equity, whole
    # shares, the max_pct_of_volume_at_entry cap) on a resampled trade order, so
    # the distribution reflects THIS backtester's real sizing, not a generic
    # R-multiple shuffle.
    enabled: bool = False
    # How many simulated equity paths to draw. 10k gives smooth percentiles in
    # ~a second; more only tightens the tails.
    iterations: int = 10_000
    # "resample"  = bootstrap: draw N trades WITH replacement from the realized
    #               set (N = number of actual trades). Captures both a different
    #               ORDER and a different MIX (some trades repeated, some absent)
    #               -- the standard "what else could this edge have produced?"
    #               view, and the more conservative of the two.
    # "reshuffle" = permute the SAME trades into a random order (each used once).
    #               Isolates pure SEQUENCE risk (does an early losing streak sink
    #               a compounding account?) while holding the trade mix fixed.
    mode: str = "resample"
    # Set an integer for reproducible runs; null draws a fresh random stream each
    # time.
    random_seed: int | None = None
    # Equity falling to or below this % of starting equity, at any point in a
    # simulated path, counts that path as "ruin" for the risk-of-ruin stat.
    ruin_threshold_pct: float = 50.0


@dataclass(frozen=True)
class BlackoutConfig:
    # Trading-blackout calendar: days the bot skips entirely -- no scan, no trade.
    # Every candidate whose date falls in a blackout is recorded
    # SKIPPED_BLACKOUT_DATE and never fetched, so thin, drifty holiday sessions
    # can't pollute the edge or the equity curve. The rules resolve PER YEAR (see
    # calendar_blackout.py), because Thanksgiving and the year-end window move.
    enabled: bool = False
    # Saturday/Sunday. Premarket data rarely exists then anyway; explicit toggle.
    skip_weekends: bool = False
    # The Mon-Fri trading week containing US Thanksgiving (the 4th Thursday of
    # November, computed per year) -- the holiday is a market close and the whole
    # week is thin and unrepresentative.
    thanksgiving_week: bool = False
    # Recurring month/day ranges (inclusive), the SAME every year, each "MM-DD..MM-DD".
    # A start that falls AFTER its end WRAPS the New Year -- "12-20..01-01" blacks
    # out Dec 20 through Jan 1. Lists (not tuples) so config_used.yaml round-trips.
    recurring_ranges: list = field(default_factory=list)
    # One-off explicit dates, ISO "YYYY-MM-DD" (an ad-hoc event / known bad day).
    dates: list = field(default_factory=list)


@dataclass(frozen=True)
class WedgeConfig:
    # A SEPARATE strategy: a symmetrical pennant that trades WHICHEVER side breaks
    # -- long on an upside break of the descending resistance line, SHORT on a
    # downside break of the ascending support line. Runs in the same premarket
    # entry window as `strategy` (its "keep volatility into the session" idea =
    # the name is still active by that window). Off by default; when on, the
    # engine routes candidates through `wedge.backtest_wedge` INSTEAD of the
    # HOD-first-pullback engine -- unless concurrent_with_dip is also on, in
    # which case BOTH engines scan every candidate in the same pass and the
    # sizing walk arbitrates their shared capital (see risk.max_leverage).
    #
    # SHARED vs OWN config (the authoritative annotated list lives at the top of
    # config.yaml's wedge: section): the wedge reads qualification from
    # `strategy` (min/max_price, min/max_float, min_rvol + the rvol_* baseline
    # keys, min_premarket_gain_pct, earliest/latest_entry_time), execution
    # plumbing from `strategy` (order_latency_seconds, intrabar_policy,
    # exit_at_regular_open, macd_* for the shared 1-minute indicators), ALL of
    # `risk` (sizing/funding/fees), data.bar_size and `session`. It deliberately
    # OVERRIDES five same-named knobs with its own values below (timeframe,
    # entry_limit_slippage_cents, target_mode/target_r_multiple,
    # require_breakout_close_confirm/breakout_close_min_frac,
    # max_pct_of_volume_at_entry) so either strategy can be A/B'd without moving
    # the other; strategy.max_spread_cents and the dip's setup machinery
    # (entry_mode, offsets, pullback/retrace, scale_out, trail, block_* gates,
    # max_trades_per_day, ...) are ignored here.
    enabled: bool = False
    # Run the wedge ALONGSIDE the dip strategy instead of replacing it: every
    # candidate is scanned by BOTH engines in the same pass, both can enter (even
    # simultaneously -- the same ticker/day can hold a dip trade and a wedge trade
    # at once), and the position-sizing walk arbitrates the shared capital via
    # risk.max_leverage (a concurrent entry is funded on margin when cash is tied
    # up in the other strategy's position). False = the original either/or
    # routing: wedge.enabled picks WHICH single strategy runs.
    concurrent_with_dip: bool = False
    # Which break to take: "both" | "long" | "short".
    sides: str = "both"
    # Candle timeframe the pennant is DETECTED and traded on: "10s" | "1m" | "2m"
    # (A/B-toggleable, independent of strategy.chart_timeframe). Pennants are a
    # visual pattern -- the candle size is part of the definition. "1m" is what a
    # day trader actually draws on; "10s" fits 5x more (noisier) coils.
    timeframe: str = "1m"
    # --- pennant geometry. Trendlines are fit through the pivots in TIME space and
    # then shifted OUTWARD to CONTAIN every bar (resistance on/above all highs,
    # support on/below all lows) -- a drawn trendline, not a regression midline.
    # The window is anchored at the POLE TOP (highest high in the lookback). ---
    pivot_lookback_bars: int = 30       # max scan-bars back the anchor may sit
    min_pennant_bars: int = 8           # min scan-bars from anchor to break (coil duration)
    pivot_strength: int = 1             # k-bar pivot: a high/low >= its k neighbours each side
    min_pivot_touches: int = 3          # >= this many pivots must TOUCH each line
    # A pivot "touches" its line when within this fraction of the window's MEDIAN
    # bar range (median, so one rogue flush/spike bar can't inflate the tolerance
    # and let a corrupted line pass the touch test).
    touch_tolerance: float = 0.75
    # How many bars may POKE THROUGH each containment line instead of dictating it
    # -- a wick through an otherwise clean trendline is accepted, exactly as in
    # manual charting. 0 = strict containment (every bar inside the channel).
    containment_outliers: int = 1
    min_convergence_pct: float = 30.0   # channel must NARROW >= this % across the window
    # ...but not PAST the apex: a coil that has already converged more than this is
    # spent (the ambiguous post-apex chop pros skip), so no fresh break is taken.
    max_convergence_pct: float = 85.0
    breakout_offset: float = 0.0        # dollars beyond the line to arm the break
    # Marketable-limit fill cap, in DOLLARS/share (not percent): the fill may be at
    # most this far BEYOND the trigger (long: trigger+X, short: trigger-X). A faster
    # gap-through doesn't fill -> no trade, modelling a real +$0.15 marketable limit
    # placed to ENSURE a fill (0.15 = 15 cents). Flat cents, so a $3 name and a $15
    # name get the same real-world buffer instead of a price-scaled percent.
    entry_limit_slippage_cents: float = 0.15
    # --- breakout QUALITY confirmation gates. All additive, default OFF, and each
    # is a per-touch DEFER (a failing touch doesn't kill the setup -- a later, more
    # convincing touch of the same lines can still fire). These encode the two
    # things a real pennant trader checks before taking a break that geometry alone
    # can't see: was there actual buying/selling interest behind it, and did it
    # HOLD rather than wick-and-reject. ---
    # A real breakout is validated by volume: a break on a limp tape is noise, not
    # conviction. Requires the breakout bar's OWN volume to be >= this many times
    # the average bar volume printed during the coil itself (anchor -> break) --
    # i.e. real expansion relative to the pattern's own (declining, by definition)
    # baseline, not the whole session's average.
    require_breakout_volume_expansion: bool = False
    breakout_volume_multiple: float = 1.5
    # Close-confirmed breakout -- a SEPARATE wedge-only toggle, deliberately not
    # shared with strategy.require_breakout_close_confirm (the dip's). A bar's
    # high/low merely touching the trigger is not enough: the bar must CLOSE beyond
    # it, in the favourable (1 - breakout_close_min_frac) fraction of its own range
    # -- a held break, not the classic pennant failure mode (a throw-over/throw-
    # under that instantly reverses). Entry then lands on the NEXT bar's open
    # (bar-based fill; the exit sim stays tick-precise). Off (default) keeps the
    # original intrabar-touch fill.
    require_breakout_close_confirm: bool = False
    breakout_close_min_frac: float = 0.667
    # Trend alignment at the break: the trigger must sit on the favourable side of
    # the session VWAP (above it for a long, below for a short), using the SAME
    # causal prior-completed-minute VWAP lookup every other gate uses. Its OWN
    # independent wedge toggle -- deliberately NOT wired to the dip strategy's
    # block_buy_below_vwap/ema9/ema20 (those stay strategy-only and untouched).
    require_vwap_alignment: bool = False
    # --- exit. "r_multiple" (default): target = entry +/- target_r_multiple x
    # (entry-to-stop). "measured_move": the classical pennant target -- project the
    # POLE's height (the anchor/pole-top high minus the lowest low in the
    # pole_lookback_bars before it) from the breakout point, scaled by
    # measured_move_fraction (1.0 = the full pole; premarket's compressed window
    # rarely runs the full measured move, so < 1.0 is often more realistic). ---
    target_r_multiple: float = 2.0
    target_mode: str = "r_multiple"     # "r_multiple" | "measured_move"
    pole_lookback_bars: int = 20
    measured_move_fraction: float = 1.0
    # Minimum entry-to-stop distance, as a % of the fill price -- a FLOOR (the
    # mirror of the dip strategy's max_risk_pct ceiling). The wedge's stop sits at
    # the OPPOSITE pennant line, a structural level that can occasionally land a
    # cent or two from the fill (a near-degenerate, unrealistically tight stop --
    # real slippage/spread would blow through it). Such trades also produce
    # absurd R-multiples (a $0.01 stop turns a modest dollar move into a 20+R
    # "win") that are a measurement artifact, not edge. Deferred like every other
    # per-touch gate (continue -- a later, better-formed touch can still fire).
    # null = off.
    min_risk_pct_of_price: float | None = 0.5
    # --- attention-persistence gate (the selection edge): is the market STILL in
    # this name at the break, or is it a forgotten early pumper? "rvol" = the
    # trailing persistence_window_min minutes must have printed at least
    # min_sustained_rvol x the stock's OWN normal volume rate (daily baseline /
    # 390 RTH minutes) -- a WINDOWED rate, not cumulative session RVOL, which only
    # grows and therefore can never expire an early pump. Declining coil volume
    # still passes (textbook pennant); a dead tape does not. "atr" = recent 1-min
    # ATR >= min_atr_pct_of_rip of the session range (price still moves).
    # "both" = require both. ---
    persistence_mode: str = "both"      # "rvol" | "atr" | "both"
    persistence_window_min: float = 20.0   # trailing minutes measured as of the break
    min_sustained_rvol: float = 2.0     # windowed volume rate >= this x the stock's norm
    min_atr_pct_of_rip: float = 0.15    # recent ATR >= this fraction of the initial rip's range
    # --- short-side borrow gate (a SHORT whose annualized borrow fee exceeds this
    # is rejected SKIPPED_BORROW_FEE; longs ignore it). null = off. ---
    max_borrow_fee_pct: float | None = 20.0
    # --- independent volume-participation cap for WEDGE trades: at most this % of
    # volume_at_entry (cumulative session volume through the entry), overriding the
    # shared risk.max_pct_of_volume_at_entry for entry_mode "wedge_break" only.
    # A pennant is entered INTO a declining-volume coil, so the session total
    # overstates what the tape is printing right now -- a stricter cap keeps a
    # large share size from being the market in a quiet coil. null = inherit the
    # shared risk cap. ---
    max_pct_of_volume_at_entry: float | None = None
    # --- WEDGE entry-size cap against the LOOKBACK-WINDOW volume: the position is
    # capped at this % of the cumulative volume traded across the pivot-lookback
    # window (pole + coil, w_lo..entry bar) rather than the whole-session
    # volume_at_entry the cap above uses. This measures the liquidity the PATTERN
    # itself printed, so a large order can't exceed a slice of what actually
    # traded during the setup. Independent of and stacks with the caps above
    # (whichever binds smallest wins). null = disabled. ---
    max_entry_pct_of_lookback_volume: float | None = None
    # --- hold-to-open limiters: toggles that bound how long an unresolved wedge
    # position may ride before the regular open (EXIT_AT_OPEN -- the "neither
    # target nor stop hit" outcome). Each is independently toggleable, all default
    # OFF (null), and each exits at the first bar OPEN at/after its condition is
    # known -- a market order placed the moment the deadline/staleness passes,
    # using only completed-bar information (causal). A forced exit is a normal
    # WIN/LOSS by its own P&L, tagged on TradeResult.exit_trigger ("time_stop" /
    # "flat_by_time" / "stale_progress"); only a position still open at session
    # end remains EXIT_AT_OPEN ("session_end"). ---
    # Hard time stop: exit this many minutes after the entry fill when neither
    # target nor stop has resolved the trade. null = off.
    max_hold_minutes: float | None = None
    # Flat-by clock time ("HH:MM", 24-hour, session timezone): close any open
    # wedge position at the first bar at/after this time -- e.g. "09:15"
    # guarantees no wedge position is carried into the 09:30 open. null = off.
    latest_exit_time: str | None = None
    # Stale-progress exit: from this many minutes after entry onward, exit unless
    # the trade's best excursion so far has covered at least
    # stale_exit_min_progress of the entry-to-target distance -- a coil break
    # that went nowhere is a dead thesis, not a hold. null = off.
    stale_exit_minutes: float | None = None
    # Fraction of the entry-to-target distance that counts as "real progress" for
    # the stale-progress exit above. Must be in [0, 1].
    stale_exit_min_progress: float = 0.25
    # --- post-entry exit MANAGEMENT (the bar-logic alternative to the clock
    # limiters above). "bracket" = the original fixed stop + fixed target held
    # until one fills or the session ends. "candle_trail_1min" = the dip
    # strategy's PROVEN runner engine, mirrored direction-aware for the wedge:
    #   * the stop trails to the prior COMPLETED 1-minute candle's extreme --
    #     its LOW for a long (ratchets up only), its HIGH for a short (ratchets
    #     down only) -- so a stalling/reversing move is walked out by price
    #     structure, bar by bar, instead of held to the open;
    #   * the first FAT-BODIED 1-minute close in the trade's favour (green for a
    #     long / red for a short, body >= strategy.fatgreen_body_min of its
    #     range, closing in the outer third -- the same candle definition the
    #     dip uses) DROPS the fixed target: a move that strong is a runner, ride
    #     the trail instead of capping it;
    #   * at the original (measured-move) target with no runner yet:
    #     trail_partial_at_target of the position banks there, the remainder's
    #     stop jumps to breakeven and rides the trail.
    # Exits stay tick-resolved wherever ticks exist, exactly as the bracket's.
    # This addresses the hold-to-open problem STRUCTURALLY (a trade that stops
    # progressing gets trailed out well before the open), where the clock
    # limiters above are blunt timeouts; both can be enabled together. ---
    exit_mode: str = "bracket"          # "bracket" | "candle_trail_1min"
    # Fraction of the position banked when the original target is hit before any
    # fat-bodied runner candle (candle_trail_1min only). 1.0 = everything exits
    # at the target (the trail then only manages the stop side).
    trail_partial_at_target: float = 0.5
    # FAILED-RETEST early exit (reduces hold-to-open). When on, the wedge also
    # leaves when a swing extreme in the trade's favour is retested and HOLDS: for a
    # short -- a swing low forms, price bounces off it by >= retest_bounce_min_pct,
    # a RED candle later retests within retest_tolerance_pct of the low but doesn't
    # break it, and a still-later candle trades back above that attempt candle's
    # HIGH -> cover (mirrored for a long: swing high, GREEN attempt candle, exit
    # below its LOW). If the trade instead extends (new low short / new high long),
    # any armed retest is discarded and the swing re-bases. bracket exit_mode only
    # for now; exit_trigger "retest_failure". false (default) = off.
    exit_retest_failure: bool = False
    retest_bounce_min_pct: float = 0.5      # min bounce off the extreme (% of it) to arm a retest
    retest_tolerance_pct: float = 1.0       # how close to the extreme (% of it) counts as a retest
    # Traditional structural / momentum exit (A/B). Exit when price RECLAIMS the
    # chosen 1-minute indicator against the position on a completed candle (a short
    # exits when the close crosses back ABOVE it; a long, below), or -- for "macd"
    # -- when the MACD histogram turns against the position. null = off.
    # "vwap" | "ema9" | "ema20" | "macd". bracket exit_mode only; exit_trigger
    # "reclaim_<signal>". The hold-to-open structural analysis says vwap/ema are the
    # signals that hold while momentum (macd) is noise near the open.
    exit_on_reclaim: str | None = None


@dataclass(frozen=True)
class AppConfig:
    ibkr: IBKRConfig
    data: DataConfig
    session: SessionConfig
    strategy: StrategyConfig
    risk: RiskConfig
    reporting: ReportingConfig
    # Defaulted so existing AppConfig(...) construction (and tests) that predates
    # this section keep working; load_config always passes it explicitly.
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    blackout: BlackoutConfig = field(default_factory=BlackoutConfig)
    wedge: WedgeConfig = field(default_factory=WedgeConfig)


def _merge_env_ibkr(raw: dict[str, Any]) -> dict[str, Any]:
    ib = raw.setdefault("ibkr", {})
    ib["host"] = os.getenv("IBKR_HOST", ib.get("host", "127.0.0.1"))
    ib["port"] = int(os.getenv("IBKR_PORT", ib.get("port", 7497)))
    ib["client_id"] = int(os.getenv("IBKR_CLIENT_ID", ib.get("client_id", 17)))
    return raw


_SECTION_TYPES = {
    "ibkr": IBKRConfig,
    "data": DataConfig,
    "session": SessionConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "reporting": ReportingConfig,
    "monte_carlo": MonteCarloConfig,
    "blackout": BlackoutConfig,
    "wedge": WedgeConfig,
}


def _check_keys(section: str, raw_section: dict[str, Any], source: Path) -> None:
    """Fail loudly on a misspelled key instead of silently running with the default
    -- a typo'd knob that quietly does nothing looks exactly like 'my config change
    had no effect'."""
    valid = {f.name for f in fields(_SECTION_TYPES[section])}
    unknown = sorted(set(raw_section) - valid)
    if unknown:
        raise ValueError(
            f"{source}: unknown key(s) {unknown} in section '{section}'. "
            f"Valid keys: {sorted(valid)}"
        )


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown_sections = sorted(set(raw) - set(_SECTION_TYPES))
    if unknown_sections:
        raise ValueError(
            f"{path}: unknown top-level section(s) {unknown_sections}. "
            f"Valid sections: {sorted(_SECTION_TYPES)}"
        )
    raw = _merge_env_ibkr(raw)
    for section in _SECTION_TYPES:
        _check_keys(section, raw.get(section, {}) or {}, path)
    root = path.parent
    data_raw = raw.get("data", {})
    stat = (raw.get("strategy", {}) or {}).get("rvol_baseline_stat", StrategyConfig.rvol_baseline_stat)
    if stat not in ("mean", "median"):
        raise ValueError(f"{path}: strategy.rvol_baseline_stat must be 'mean' or 'median', got {stat!r}")
    red_mode = (raw.get("strategy", {}) or {}).get("red_volume_check", StrategyConfig.red_volume_check)
    if red_mode not in ("total", "largest_bar"):
        raise ValueError(f"{path}: strategy.red_volume_check must be 'total' or 'largest_bar', got {red_mode!r}")
    offset_mode = (raw.get("strategy", {}) or {}).get("offset_mode", StrategyConfig.offset_mode)
    if offset_mode not in ("cents", "percent"):
        raise ValueError(f"{path}: strategy.offset_mode must be 'cents' or 'percent', got {offset_mode!r}")
    for field_name in ("target_offset_pct", "stop_offset_pct"):
        value = (raw.get("strategy", {}) or {}).get(field_name, getattr(StrategyConfig, field_name))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{path}: strategy.{field_name} must be a positive number (percent of entry), got {value!r}")
    entry_mode = (raw.get("strategy", {}) or {}).get("entry_mode", StrategyConfig.entry_mode)
    if entry_mode not in ("hod_break", "dip_swing_high", "dip_red_break", "dip_trendline", "dip_ema_reclaim"):
        raise ValueError(f"{path}: strategy.entry_mode must be one of 'hod_break', 'dip_swing_high', 'dip_red_break', 'dip_trendline', 'dip_ema_reclaim', got {entry_mode!r}")
    stop_mode = (raw.get("strategy", {}) or {}).get("stop_mode", StrategyConfig.stop_mode)
    if stop_mode not in ("offset", "pullback_low", "red_candle_low"):
        raise ValueError(f"{path}: strategy.stop_mode must be 'offset', 'pullback_low' or 'red_candle_low', got {stop_mode!r}")
    chart_tf = (raw.get("strategy", {}) or {}).get("chart_timeframe", StrategyConfig.chart_timeframe)
    if chart_tf not in ("10s", "1m", "2m"):
        raise ValueError(f"{path}: strategy.chart_timeframe must be '10s', '1m' or '2m', got {chart_tf!r}")
    exit_mode = (raw.get("strategy", {}) or {}).get("exit_mode", StrategyConfig.exit_mode)
    if exit_mode not in ("standard", "candle_trail_1min"):
        raise ValueError(f"{path}: strategy.exit_mode must be 'standard' or 'candle_trail_1min', got {exit_mode!r}")
    fg = (raw.get("strategy", {}) or {}).get("fatgreen_body_min", StrategyConfig.fatgreen_body_min)
    if isinstance(fg, bool) or not isinstance(fg, (int, float)) or not (0.0 <= fg <= 1.0):
        raise ValueError(f"{path}: strategy.fatgreen_body_min must be a number in [0, 1], got {fg!r}")
    tpt = (raw.get("strategy", {}) or {}).get("trail_partial_at_target", StrategyConfig.trail_partial_at_target)
    if isinstance(tpt, bool) or not isinstance(tpt, (int, float)) or not (0.0 < tpt <= 1.0):
        raise ValueError(f"{path}: strategy.trail_partial_at_target must be a number in (0, 1], got {tpt!r}")
    ra = (raw.get("strategy", {}) or {}).get("rip_anchor", StrategyConfig.rip_anchor)
    if ra not in ("session_low", "recent", "swing_low"):
        raise ValueError(f"{path}: strategy.rip_anchor must be 'session_low', 'recent' or 'swing_low', got {ra!r}")
    rrm = (raw.get("strategy", {}) or {}).get("rip_recent_minutes", StrategyConfig.rip_recent_minutes)
    if isinstance(rrm, bool) or not isinstance(rrm, (int, float)) or rrm <= 0:
        raise ValueError(f"{path}: strategy.rip_recent_minutes must be a positive number of minutes, got {rrm!r}")
    target_mode = (raw.get("strategy", {}) or {}).get("target_mode", StrategyConfig.target_mode)
    if target_mode not in ("offset", "r_multiple"):
        raise ValueError(f"{path}: strategy.target_mode must be 'offset' or 'r_multiple', got {target_mode!r}")
    target_r = (raw.get("strategy", {}) or {}).get("target_r_multiple", StrategyConfig.target_r_multiple)
    if isinstance(target_r, bool) or not isinstance(target_r, (int, float)) or target_r <= 0:
        raise ValueError(f"{path}: strategy.target_r_multiple must be a positive number, got {target_r!r}")
    reclaim_ind = (raw.get("strategy", {}) or {}).get("dip_reclaim_indicator", StrategyConfig.dip_reclaim_indicator)
    if reclaim_ind not in ("ema9", "ema20", "vwap"):
        raise ValueError(f"{path}: strategy.dip_reclaim_indicator must be 'ema9', 'ema20' or 'vwap', got {reclaim_ind!r}")
    close_frac = (raw.get("strategy", {}) or {}).get("breakout_close_min_frac", StrategyConfig.breakout_close_min_frac)
    if isinstance(close_frac, bool) or not isinstance(close_frac, (int, float)) or not (0.0 <= close_frac <= 1.0):
        raise ValueError(f"{path}: strategy.breakout_close_min_frac must be a number in [0, 1], got {close_frac!r}")
    folders_for = (raw.get("reporting", {}) or {}).get("trade_folders_for", ReportingConfig.trade_folders_for)
    if folders_for not in ("trades", "setups", "all"):
        raise ValueError(f"{path}: reporting.trade_folders_for must be 'trades', 'setups' or 'all', got {folders_for!r}")
    max_trades = (raw.get("strategy", {}) or {}).get("max_trades_per_day", StrategyConfig.max_trades_per_day)
    if not isinstance(max_trades, int) or isinstance(max_trades, bool) or max_trades < 1:
        raise ValueError(f"{path}: strategy.max_trades_per_day must be an integer >= 1, got {max_trades!r}")
    winners_qty = (raw.get("strategy", {}) or {}).get("trade_winners_qty", StrategyConfig.trade_winners_qty)
    if not isinstance(winners_qty, int) or isinstance(winners_qty, bool) or winners_qty < 1:
        raise ValueError(f"{path}: strategy.trade_winners_qty must be an integer >= 1, got {winners_qty!r}")
    cont_vwap = (raw.get("strategy", {}) or {}).get("continuation_vwap_above_entry_pct", StrategyConfig.continuation_vwap_above_entry_pct)
    if cont_vwap is not None and (isinstance(cont_vwap, bool) or not isinstance(cont_vwap, (int, float)) or cont_vwap <= -100):
        raise ValueError(f"{path}: strategy.continuation_vwap_above_entry_pct must be a percentage > -100 or null, got {cont_vwap!r}")
    for field_name in ("earliest_entry_time", "latest_entry_time"):
        value = (raw.get("strategy", {}) or {}).get(field_name, getattr(StrategyConfig, field_name))
        if value is not None:
            parse_hhmm(value, f"strategy.{field_name}", path)
    for field_name in ("macd_fast", "macd_slow", "macd_signal"):
        value = (raw.get("strategy", {}) or {}).get(field_name, getattr(StrategyConfig, field_name))
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{path}: strategy.{field_name} must be an integer >= 1, got {value!r}")
    strat_raw = raw.get("strategy", {}) or {}
    min_fl = strat_raw.get("min_float", StrategyConfig.min_float)
    max_fl = strat_raw.get("max_float", StrategyConfig.max_float)
    for field_name, value in (("min_float", min_fl), ("max_float", max_fl)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"{path}: strategy.{field_name} must be a positive number of shares or null, got {value!r}")
    if min_fl is not None and max_fl is not None and min_fl > max_fl:
        raise ValueError(f"{path}: strategy.min_float ({min_fl}) must not exceed strategy.max_float ({max_fl}).")
    min_pmv = strat_raw.get("min_premarket_volume", StrategyConfig.min_premarket_volume)
    if min_pmv is not None and (isinstance(min_pmv, bool) or not isinstance(min_pmv, (int, float)) or min_pmv <= 0):
        raise ValueError(f"{path}: strategy.min_premarket_volume must be a positive number of shares or null, got {min_pmv!r}")
    max_risk = strat_raw.get("max_risk_pct", StrategyConfig.max_risk_pct)
    if max_risk is not None and (isinstance(max_risk, bool) or not isinstance(max_risk, (int, float)) or max_risk <= 0):
        raise ValueError(f"{path}: strategy.max_risk_pct must be a positive percent of the entry price or null, got {max_risk!r}")
    risk_raw = raw.get("risk", {}) or {}
    max_vol_pct = risk_raw.get("max_pct_of_volume_at_entry", RiskConfig.max_pct_of_volume_at_entry)
    if max_vol_pct is not None and (isinstance(max_vol_pct, bool) or not isinstance(max_vol_pct, (int, float)) or not (0 < max_vol_pct <= 100)):
        raise ValueError(f"{path}: risk.max_pct_of_volume_at_entry must be a number in (0, 100] or null, got {max_vol_pct!r}")
    monthly_fees = risk_raw.get("monthly_fees", RiskConfig.monthly_fees)
    if monthly_fees is not None and (isinstance(monthly_fees, bool) or not isinstance(monthly_fees, (int, float)) or monthly_fees < 0):
        raise ValueError(f"{path}: risk.monthly_fees must be a non-negative dollar amount or null, got {monthly_fees!r}")
    risk_pct = risk_raw.get("risk_pct_of_equity", RiskConfig.risk_pct_of_equity)
    if risk_pct is not None and (isinstance(risk_pct, bool) or not isinstance(risk_pct, (int, float)) or risk_pct <= 0):
        raise ValueError(f"{path}: risk.risk_pct_of_equity must be a positive percent of equity to risk per trade or null, got {risk_pct!r}")
    max_lev = risk_raw.get("max_leverage", RiskConfig.max_leverage)
    if max_lev is not None and (isinstance(max_lev, bool) or not isinstance(max_lev, (int, float)) or max_lev < 1.0):
        raise ValueError(f"{path}: risk.max_leverage must be a number >= 1.0 (1.0 = cash-only, 2.0 = 2x margin) or null (unlimited parallel funding), got {max_lev!r}")
    size_to_risk = risk_raw.get("size_to_risk", RiskConfig.size_to_risk)
    if not isinstance(size_to_risk, bool):
        raise ValueError(f"{path}: risk.size_to_risk must be true or false, got {size_to_risk!r}")
    if size_to_risk and (risk_pct is None or max_lev is None):
        raise ValueError(f"{path}: risk.size_to_risk requires both risk_pct_of_equity and max_leverage set -- it sizes to the risk target, funded on margin up to the leverage ceiling.")
    mc_raw = raw.get("monte_carlo", {}) or {}
    mc_iters = mc_raw.get("iterations", MonteCarloConfig.iterations)
    if not isinstance(mc_iters, int) or isinstance(mc_iters, bool) or mc_iters < 1:
        raise ValueError(f"{path}: monte_carlo.iterations must be an integer >= 1, got {mc_iters!r}")
    mc_mode = mc_raw.get("mode", MonteCarloConfig.mode)
    if mc_mode not in ("resample", "reshuffle"):
        raise ValueError(f"{path}: monte_carlo.mode must be 'resample' or 'reshuffle', got {mc_mode!r}")
    mc_seed = mc_raw.get("random_seed", MonteCarloConfig.random_seed)
    if mc_seed is not None and (isinstance(mc_seed, bool) or not isinstance(mc_seed, int)):
        raise ValueError(f"{path}: monte_carlo.random_seed must be an integer or null, got {mc_seed!r}")
    mc_ruin = mc_raw.get("ruin_threshold_pct", MonteCarloConfig.ruin_threshold_pct)
    if isinstance(mc_ruin, bool) or not isinstance(mc_ruin, (int, float)) or not (0 <= mc_ruin < 100):
        raise ValueError(f"{path}: monte_carlo.ruin_threshold_pct must be a number in [0, 100), got {mc_ruin!r}")
    bl_raw = raw.get("blackout", {}) or {}
    for spec in bl_raw.get("recurring_ranges", []) or []:
        try:
            parse_recurring_range(spec)
        except ValueError as e:
            raise ValueError(f"{path}: blackout.recurring_ranges entry {e}") from None
    for d in bl_raw.get("dates", []) or []:
        try:
            date_cls.fromisoformat(str(d))
        except ValueError:
            raise ValueError(f"{path}: blackout.dates entry must be ISO 'YYYY-MM-DD', got {d!r}") from None
    wg_raw = raw.get("wedge", {}) or {}
    wg_sides = wg_raw.get("sides", WedgeConfig.sides)
    if wg_sides not in ("both", "long", "short"):
        raise ValueError(f"{path}: wedge.sides must be 'both', 'long', or 'short', got {wg_sides!r}")
    wg_tf = wg_raw.get("timeframe", WedgeConfig.timeframe)
    if wg_tf not in ("10s", "1m", "2m"):
        raise ValueError(f"{path}: wedge.timeframe must be '10s', '1m', or '2m', got {wg_tf!r}")
    wg_minbars = wg_raw.get("min_pennant_bars", WedgeConfig.min_pennant_bars)
    if not isinstance(wg_minbars, int) or isinstance(wg_minbars, bool) or wg_minbars < 3:
        raise ValueError(f"{path}: wedge.min_pennant_bars must be an integer >= 3, got {wg_minbars!r}")
    wg_tol = wg_raw.get("touch_tolerance", WedgeConfig.touch_tolerance)
    if isinstance(wg_tol, bool) or not isinstance(wg_tol, (int, float)) or not (0 < wg_tol <= 3):
        raise ValueError(f"{path}: wedge.touch_tolerance must be a positive fraction of the median bar range (<= 3), got {wg_tol!r}")
    wg_out = wg_raw.get("containment_outliers", WedgeConfig.containment_outliers)
    if not isinstance(wg_out, int) or isinstance(wg_out, bool) or wg_out < 0:
        raise ValueError(f"{path}: wedge.containment_outliers must be an integer >= 0, got {wg_out!r}")
    wg_minc = wg_raw.get("min_convergence_pct", WedgeConfig.min_convergence_pct)
    wg_maxc = wg_raw.get("max_convergence_pct", WedgeConfig.max_convergence_pct)
    if isinstance(wg_maxc, bool) or not isinstance(wg_maxc, (int, float)) or not (0 < wg_maxc <= 100) or wg_maxc <= wg_minc:
        raise ValueError(f"{path}: wedge.max_convergence_pct must be a percent in (0, 100] above min_convergence_pct, got {wg_maxc!r}")
    wg_pmode = wg_raw.get("persistence_mode", WedgeConfig.persistence_mode)
    if wg_pmode not in ("rvol", "atr", "both"):
        raise ValueError(f"{path}: wedge.persistence_mode must be 'rvol', 'atr', or 'both', got {wg_pmode!r}")
    wg_vol = wg_raw.get("max_pct_of_volume_at_entry", WedgeConfig.max_pct_of_volume_at_entry)
    if wg_vol is not None and (isinstance(wg_vol, bool) or not isinstance(wg_vol, (int, float)) or not (0 < wg_vol <= 100)):
        raise ValueError(f"{path}: wedge.max_pct_of_volume_at_entry must be a number in (0, 100] or null, got {wg_vol!r}")
    wg_lbvol = wg_raw.get("max_entry_pct_of_lookback_volume", WedgeConfig.max_entry_pct_of_lookback_volume)
    if wg_lbvol is not None and (isinstance(wg_lbvol, bool) or not isinstance(wg_lbvol, (int, float)) or not (0 < wg_lbvol <= 100)):
        raise ValueError(f"{path}: wedge.max_entry_pct_of_lookback_volume must be a number in (0, 100] or null, got {wg_lbvol!r}")
    wg_borrow = wg_raw.get("max_borrow_fee_pct", WedgeConfig.max_borrow_fee_pct)
    if wg_borrow is not None and (isinstance(wg_borrow, bool) or not isinstance(wg_borrow, (int, float)) or wg_borrow < 0):
        raise ValueError(f"{path}: wedge.max_borrow_fee_pct must be a non-negative percent or null, got {wg_borrow!r}")
    wg_bvm = wg_raw.get("breakout_volume_multiple", WedgeConfig.breakout_volume_multiple)
    if isinstance(wg_bvm, bool) or not isinstance(wg_bvm, (int, float)) or wg_bvm <= 0:
        raise ValueError(f"{path}: wedge.breakout_volume_multiple must be a positive number, got {wg_bvm!r}")
    wg_bccf = wg_raw.get("breakout_close_min_frac", WedgeConfig.breakout_close_min_frac)
    if isinstance(wg_bccf, bool) or not isinstance(wg_bccf, (int, float)) or not (0 <= wg_bccf <= 1):
        raise ValueError(f"{path}: wedge.breakout_close_min_frac must be a fraction in [0, 1], got {wg_bccf!r}")
    wg_tgtmode = wg_raw.get("target_mode", WedgeConfig.target_mode)
    if wg_tgtmode not in ("r_multiple", "measured_move"):
        raise ValueError(f"{path}: wedge.target_mode must be 'r_multiple' or 'measured_move', got {wg_tgtmode!r}")
    wg_pole = wg_raw.get("pole_lookback_bars", WedgeConfig.pole_lookback_bars)
    if not isinstance(wg_pole, int) or isinstance(wg_pole, bool) or wg_pole < 1:
        raise ValueError(f"{path}: wedge.pole_lookback_bars must be an integer >= 1, got {wg_pole!r}")
    wg_erf = wg_raw.get("exit_retest_failure", WedgeConfig.exit_retest_failure)
    if not isinstance(wg_erf, bool):
        raise ValueError(f"{path}: wedge.exit_retest_failure must be true or false, got {wg_erf!r}")
    for _k in ("retest_bounce_min_pct", "retest_tolerance_pct"):
        _v = wg_raw.get(_k, getattr(WedgeConfig, _k))
        if isinstance(_v, bool) or not isinstance(_v, (int, float)) or _v <= 0:
            raise ValueError(f"{path}: wedge.{_k} must be a positive percent, got {_v!r}")
    wg_recl = wg_raw.get("exit_on_reclaim", WedgeConfig.exit_on_reclaim)
    if wg_recl is not None and wg_recl not in ("vwap", "ema9", "ema20", "macd"):
        raise ValueError(f"{path}: wedge.exit_on_reclaim must be null or one of 'vwap','ema9','ema20','macd', got {wg_recl!r}")
    wg_mmf = wg_raw.get("measured_move_fraction", WedgeConfig.measured_move_fraction)
    if isinstance(wg_mmf, bool) or not isinstance(wg_mmf, (int, float)) or wg_mmf <= 0:
        raise ValueError(f"{path}: wedge.measured_move_fraction must be a positive number, got {wg_mmf!r}")
    wg_minrisk = wg_raw.get("min_risk_pct_of_price", WedgeConfig.min_risk_pct_of_price)
    if wg_minrisk is not None and (isinstance(wg_minrisk, bool) or not isinstance(wg_minrisk, (int, float)) or wg_minrisk < 0):
        raise ValueError(f"{path}: wedge.min_risk_pct_of_price must be a non-negative percent or null, got {wg_minrisk!r}")
    if int(wg_raw.get("min_pivot_touches", WedgeConfig.min_pivot_touches)) < 2:
        raise ValueError(f"{path}: wedge.min_pivot_touches must be >= 2 (a line needs two points).")
    for field_name in ("max_hold_minutes", "stale_exit_minutes"):
        value = wg_raw.get(field_name, getattr(WedgeConfig, field_name))
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"{path}: wedge.{field_name} must be a positive number of minutes or null, got {value!r}")
    wg_flat = wg_raw.get("latest_exit_time", WedgeConfig.latest_exit_time)
    if wg_flat is not None:
        parse_hhmm(wg_flat, "wedge.latest_exit_time", path)
    wg_stale_frac = wg_raw.get("stale_exit_min_progress", WedgeConfig.stale_exit_min_progress)
    if isinstance(wg_stale_frac, bool) or not isinstance(wg_stale_frac, (int, float)) or not (0.0 <= wg_stale_frac <= 1.0):
        raise ValueError(f"{path}: wedge.stale_exit_min_progress must be a fraction in [0, 1], got {wg_stale_frac!r}")
    wg_exit = wg_raw.get("exit_mode", WedgeConfig.exit_mode)
    if wg_exit not in ("bracket", "candle_trail_1min"):
        raise ValueError(f"{path}: wedge.exit_mode must be 'bracket' or 'candle_trail_1min', got {wg_exit!r}")
    wg_tpart = wg_raw.get("trail_partial_at_target", WedgeConfig.trail_partial_at_target)
    if isinstance(wg_tpart, bool) or not isinstance(wg_tpart, (int, float)) or not (0.0 < wg_tpart <= 1.0):
        raise ValueError(f"{path}: wedge.trail_partial_at_target must be a fraction in (0, 1], got {wg_tpart!r}")
    return AppConfig(
        ibkr=IBKRConfig(**raw.get("ibkr", {})),
        data=DataConfig(
            input_csv=(root / data_raw.get("input_csv", "data/ticker_dates.csv")).resolve(),
            reference_csv=((root / data_raw["reference_csv"]).resolve() if data_raw.get("reference_csv") else None),
            cache_dir=(root / data_raw.get("cache_dir", "data/cache")).resolve(),
            output_dir=(root / data_raw.get("output_dir", "results")).resolve(),
            use_cache=bool(data_raw.get("use_cache", True)),
            bar_size=data_raw.get("bar_size", "10 secs"),
            what_to_show=data_raw.get("what_to_show", "TRADES"),
            use_rth=bool(data_raw.get("use_rth", False)),
            tick_request_pause_s=float(data_raw.get("tick_request_pause_s", 3.0)),
            bar_request_pause_s=float(data_raw.get("bar_request_pause_s", 0.0)),
            request_timeout_s=float(data_raw.get("request_timeout_s", 60.0)),
            max_empty_tick_attempts=int(data_raw.get("max_empty_tick_attempts", 4)),
            max_empty_bar_attempts=int(data_raw.get("max_empty_bar_attempts", 3)),
            recovery_max_total_wait_s=float(data_raw.get("recovery_max_total_wait_s", 21600.0)),
            wide_duration=data_raw.get("wide_duration", "6 D"),
        ),
        session=SessionConfig(**raw.get("session", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        reporting=ReportingConfig(**raw.get("reporting", {})),
        monte_carlo=MonteCarloConfig(**mc_raw),
        blackout=BlackoutConfig(**bl_raw),
        wedge=WedgeConfig(**wg_raw),
    )


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Plain-serializable snapshot of a resolved AppConfig, section by section --
    what run_backtest writes to each run folder as config_used.yaml so any run
    records exactly which values it executed with."""
    def plain(v: Any) -> Any:
        return str(v) if isinstance(v, Path) else v

    out: dict[str, Any] = {}
    for section_name, section_type in _SECTION_TYPES.items():
        section = getattr(cfg, section_name)
        out[section_name] = {f.name: plain(getattr(section, f.name)) for f in fields(section_type)}
    return out
