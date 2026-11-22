from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from enum import Enum
from typing import Any


class Result(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    NO_TRIGGER = "NO_TRIGGER"
    INVALID_RETRACE = "INVALID_RETRACE"
    # The armed HOD was broken (a trigger touch) but no valid entry filled -- the
    # first-pullback opportunity is spent, so under first_setup_only the candidate
    # is done (a re-break of a level price already blew past is a stale, no-longer-
    # first pullback). With first_setup_only False the scan instead resets and
    # hunts for a fresh HOD/pullback (the second setup), so this result only
    # surfaces under first_setup_only.
    STALE_HOD = "STALE_HOD"
    DID_NOT_QUALIFY = "DID_NOT_QUALIFY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SKIPPED_PRICE_FILTER = "SKIPPED_PRICE_FILTER"
    SKIPPED_FLOAT_FILTER = "SKIPPED_FLOAT_FILTER"
    # The candidate's date fell in a configured trading blackout (holiday window,
    # Thanksgiving week, weekend, or an explicit date) -- skipped before any data
    # fetch, so the bot never scans or trades that day. See calendar_blackout.py.
    SKIPPED_BLACKOUT_DATE = "SKIPPED_BLACKOUT_DATE"
    # Wedge strategy: a symmetrical pennant never formed / never broke, or a SHORT
    # was rejected because the name's borrow fee exceeded max_borrow_fee_pct.
    NO_WEDGE = "NO_WEDGE"
    INVALID_WEDGE = "INVALID_WEDGE"
    SKIPPED_PERSISTENCE = "SKIPPED_PERSISTENCE"
    SKIPPED_BORROW_FEE = "SKIPPED_BORROW_FEE"
    EXIT_AT_OPEN = "EXIT_AT_OPEN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Candidate:
    ticker: str
    day: date
    float_shares: float | None = None
    notes: str | None = None


@dataclass
class TradeResult:
    ticker: str
    date: str
    result: str
    reason: str = ""
    prev_close: float | None = None
    stock_price: float | None = None
    premarket_open: float | None = None
    regular_open_price: float | None = None
    float_shares: float | None = None
    float_source: str | None = None
    float_missing: bool = False
    premarket_high: float | None = None
    premarket_high_time: str | None = None
    premarket_gain_pct: float | None = None
    premarket_volume: int | None = None
    pullback_low: float | None = None
    pullback_low_time: str | None = None
    pullback_retrace_pct: float | None = None
    trigger_time: str | None = None
    trigger_price: float | None = None
    entry_time: str | None = None
    entry_price: float | None = None
    fill_price: float | None = None
    entry_limit_price: float | None = None
    slippage: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    exit_time: str | None = None
    exit_price: float | None = None
    # WHAT actually closed the position (wedge trades): "target" / "stop" /
    # "session_end" (held into the regular open -- the EXIT_AT_OPEN case), or one
    # of the hold-to-open limiters -- "time_stop" (wedge.max_hold_minutes),
    # "flat_by_time" (wedge.latest_exit_time), "stale_progress"
    # (wedge.stale_exit_minutes). None for dip trades (their exit story lives in
    # exit_legs / trail_* fields).
    exit_trigger: str | None = None
    # How many scale-out legs were banked before the final exit (0 = the whole
    # position exited in one piece).
    scale_outs: int = 0
    # Every exit leg in order: {"kind": "target"|"stop"|"session_end",
    # "frac": fraction of the ORIGINAL position, "price": fill, "time": ISO}.
    # exit_price is the frac-weighted average of these.
    exit_legs: list | None = None
    # exit_mode "candle_trail_1min" telemetry. trail_activated = a fat-green
    # 1-minute candle fired and DROPPED the fixed target (the trade became a
    # "runner"); False means it stayed a "base hit" managed against the live 1:1
    # target. trail_stop_final = the highest the trailing stop ratcheted to (the
    # final 1-minute-low trail level). Both stay defaults under exit_mode
    # "standard".
    trail_activated: bool = False
    trail_stop_final: float | None = None
    # Position size actually bought for this trade. With risk.position_pct_of_equity
    # set, this is floor(equity * pct / entry_price) from the account's compounding
    # equity at the time of the trade (chronological), filled in by the engine's
    # sizing pass; otherwise the fixed risk.shares. Then further capped by
    # risk.max_pct_of_volume_at_entry against volume_at_entry, if configured.
    shares: int | None = None
    # Which cap actually bound this trade's share count (set by the sizing pass):
    # "notional" (position_pct_of_equity), "shares" (fixed count), "risk"
    # (risk_pct_of_equity's entry-to-stop cap), "volume"
    # (max_pct_of_volume_at_entry), or "funding" (risk.max_leverage -- the account
    # couldn't fund the full size while other positions held its capital). Lets a
    # run show when risk sizing was the binding constraint vs. when the notional %
    # or liquidity cap won instead.
    size_basis: str | None = None
    # Concurrency context, stamped by the sizing pass's event walk. How many OTHER
    # positions were still open at the moment this trade entered (0 = it had the
    # account to itself)...
    open_positions_at_entry: int = 0
    # ...the notional dollars of this entry funded BEYOND available cash (0.0 when
    # cash covered it; only ever > 0 while another position holds capital)...
    margin_used: float | None = None
    # ...and the account's gross leverage the moment this entry filled: total
    # committed notional (all open positions incl. this one) / realized equity.
    # <= 1.0 means fully cash-funded; risk.max_leverage caps it when set.
    leverage_at_entry: float | None = None
    # Account value after this trade closed (starting_equity compounded through
    # all trades up to and including this one, in chronological order; realized
    # at each trade's EXIT, so an overlapping trade never sizes against P&L that
    # hadn't happened yet when it entered).
    equity_after: float | None = None
    pnl_per_share: float = 0.0
    pnl_total: float = 0.0
    # Per-trade return as a percentage of the entry price.
    pnl_pct: float | None = None
    r_multiple: float | None = None
    mfe: float | None = None
    mae: float | None = None
    bars: int = 0
    intrabar_policy: str | None = None
    # Which entry_mode armed this trade's trigger: "hod_break" (break of the HOD),
    # "dip_swing_high", "dip_trendline" or "dip_ema_reclaim" (break of the dip's
    # downtrend, below the HOD), or "wedge_break" (symmetrical-pennant breakout).
    # trigger_price reflects the actual level used.
    entry_mode: str | None = None
    # Trade side. "long" for every dip/HOD trade (the default keeps them byte-
    # identical); the wedge strategy sets "short" when the pennant breaks DOWN.
    # Drives the sign of pnl/R/MFE/MAE and which way the stop/target/trail sit.
    direction: str = "long"
    # Short-side borrow context (wedge shorts only): the name's annualized borrow
    # fee %% and where it came from ("reference_csv" / "assumed" when no data). A
    # short whose fee exceeds max_borrow_fee_pct is rejected (SKIPPED_BORROW_FEE).
    borrow_fee: float | None = None
    borrow_fee_source: str | None = None
    # Symmetrical-pennant geometry (wedge trades only), so the chart can redraw the
    # two converging trendlines: the resistance (descending highs) and support
    # (ascending lows) line VALUES at the window start and at the break bar, plus
    # those two timestamps. None for every non-wedge trade.
    wedge_line_start_time: str | None = None
    wedge_line_end_time: str | None = None
    wedge_res_start: float | None = None
    wedge_res_end: float | None = None
    wedge_sup_start: float | None = None
    wedge_sup_end: float | None = None
    path_ambiguous: bool = False
    ambiguous_bar_time: str | None = None
    # How a path-ambiguous bar was decided: "ticks", "heuristic" or "conservative".
    ambiguity_resolution: str | None = None
    # How the entry/exit fill price was actually determined: "ticks" (the real
    # traded price from IBKR historical ticks for that bar) or "bar" (a bar-based
    # estimate, used when ticks were unavailable or didn't confirm the level traded).
    # Populated for every fill, not just ambiguous ones -- distinct from
    # ambiguity_resolution, which only applies when a bar touched both stop and target.
    entry_fill_source: str | None = None
    exit_fill_source: str | None = None

    # Qualification context. RVOL values are informational for when/how strongly
    # volume confirmed; the entry itself re-checks RVOL causally per bar.
    rvol_at_spike: float | None = None
    rvol_spike_time: str | None = None
    max_rvol_seen: float | None = None
    # Cumulative volume traded so far this session (premarket, through the prior
    # completed minute) as of the actual entry -- the raw number behind
    # rvol_so_far at the entry bar, not just the ratio to the daily baseline.
    volume_at_entry: float | None = None
    # WEDGE only: cumulative volume traded across the pivot-lookback WINDOW (the
    # pole + coil, w_lo..entry bar), as opposed to the whole session's
    # volume_at_entry above. The base for wedge.max_entry_pct_of_lookback_volume
    # -- a pattern-local liquidity measure (what actually traded during the setup)
    # rather than the session total. None for non-wedge trades / when disabled.
    lookback_volume: float | None = None
    # The premarket open: the anchor the gain% qualification is measured from.
    anchor_price: float | None = None
    # The low the traded (or invalidated) setup's rip launched from -- the anchor
    # its pullback_retrace_pct is measured against (NOT the same as anchor_price:
    # a rip often launches from below the open, e.g. a flush first, then the move).
    rip_low: float | None = None
    # Which gate disqualified a DID_NOT_QUALIFY candidate: "premarket_volume"
    # (this session's own cumulative premarket volume never reached
    # min_premarket_volume even by session close -- a cheap necessary-condition
    # short-circuit; the causal per-touch check, skipped_premarket_volume_not_
    # yet_confirmed, is what actually gates a given touch), "rvol" (cumulative
    # premarket volume never reached min_rvol relative to the ticker's own
    # daily-average baseline) or "gain_pct" (max premarket gain from the open
    # stayed below min_premarket_gain_pct).
    disqualify_reason: str | None = None
    # This ticker's average daily (regular-session) volume over its trailing
    # history -- the RVOL baseline used for this candidate.
    daily_avg_volume: float | None = None

    # Diagnostics: how many times each internal entry gate blocked a trigger-eligible
    # bar, and how many bars actually reached the trigger price at all. Lets a
    # NO_TRIGGER/DID_NOT_QUALIFY result be explained instead of being a black box.
    trigger_touches: int = 0
    skipped_no_macd_bar: int = 0
    # Trigger touches where RVOL hadn't ACTUALLY reached min_rvol yet as of that bar
    # (using only volume up to the prior completed minute) -- RVOL confirming
    # somewhere later that session doesn't retroactively make an earlier entry valid.
    skipped_rvol_not_yet_confirmed: int = 0
    # Trigger touches where cumulative premarket volume (through the prior
    # completed minute) hadn't ACTUALLY reached min_premarket_volume yet -- same
    # causal pattern as skipped_rvol_not_yet_confirmed above, just against an
    # absolute floor instead of a relative-to-baseline one.
    skipped_premarket_volume_not_yet_confirmed: int = 0
    # Wedge break attempts where the session-so-far gain (running high through the
    # prior completed bar vs the premarket open) hadn't ACTUALLY reached
    # min_premarket_gain_pct yet as of the break -- the causal counterpart of the
    # full-session gain short-circuit, same pattern as the RVOL counters above.
    skipped_gain_not_yet_confirmed: int = 0
    skipped_red_volume_dominant: int = 0
    # Trigger touches rejected because the dip (HOD through the pullback low) held
    # no red 10-second candle -- a genuine down-tick during the pullback -- only
    # counted when strategy.require_red_dip_candle is enabled.
    skipped_no_red_dip_candle: int = 0
    skipped_macd_not_bullish: int = 0
    skipped_fill_exceeds_limit: int = 0
    skipped_spread_too_wide: int = 0
    # Trigger touches outside [earliest_entry_time, latest_entry_time]; a later
    # touch of the same armed setup inside the window can still fire.
    skipped_outside_entry_window: int = 0
    # Trigger touches where the retrace (HOD through that touch) traded below
    # the respective 1-minute EMA9/EMA20/VWAP at some point along the way --
    # only counted when the matching block_retrace_below_* gate is enabled.
    skipped_retrace_below_ema9: int = 0
    skipped_retrace_below_ema20: int = 0
    skipped_retrace_below_vwap: int = 0
    # Trigger touches where the trigger PRICE ITSELF was below the respective
    # 1-minute EMA9/EMA20/VWAP as of the prior completed minute -- only counted
    # when the matching block_buy_below_* gate is enabled. Distinct from
    # skipped_retrace_below_* above (which judges the whole retrace path).
    skipped_buy_below_ema9: int = 0
    skipped_buy_below_ema20: int = 0
    skipped_buy_below_vwap: int = 0
    # Trigger touches rejected because the entry-to-stop distance (fill - stop, as a
    # % of the fill) exceeded strategy.max_risk_pct -- the structural stop sits too
    # far below the entry to be worth the risk. Only counted when max_risk_pct is set.
    skipped_risk_too_wide: int = 0
    # Win-continuation touches (trade_winners_qty) rejected because the session
    # VWAP hadn't lifted continuation_vwap_above_entry_pct above the prior
    # winning entry yet -- the volume-confirmation gate for pressing winners.
    skipped_continuation_vwap: int = 0
    # Times a pullback breached max_pullback_retrace_pct; only >1 when
    # first_setup_only is False and the scan kept looking for a fresh setup.
    invalid_retrace_attempts: int = 0
    # Times the armed HOD was broken by a trigger touch that failed an entry gate
    # (RVOL/volume/red/MACD/indicator/window/spread/fill) -- the first-pullback
    # opportunity spent on a re-break of an already-broken level. Terminates the
    # candidate (STALE_HOD) under first_setup_only; else the scan resets to hunt a
    # fresh setup, and this can climb >1.
    stale_hod_breaks: int = 0
    # Which move-of-the-day this trade is: 1 = the first setup traded that day,
    # 2 = the second, etc. Only ever > 1 when strategy.max_trades_per_day allowed
    # re-entering after a completed trade -- purely informational (disambiguates
    # per-day trade-folder names); qualification and sizing don't depend on it.
    setup_index: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
