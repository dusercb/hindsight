from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import AppConfig, BlackoutConfig, DataConfig, RiskConfig, SessionConfig, StrategyConfig, WedgeConfig, config_to_dict
from .calendar_blackout import blackout_reason
from .ibkr_data import IBKRHistoricalData, split_sessions
from .input_tools import load_borrow_fees, load_candidates
from .models import Result, TradeResult
from .montecarlo import MIN_TRADES as MC_MIN_TRADES, run_monte_carlo
from . import quant
from .routing import active_strategies
# Position sizing lives in its own pure module (no IBKR imports) so signal_core
# can export it to the live bot; engine re-imports it here, keeping every
# existing caller (tests, the progress readout below) source-compatible.
from .sizing import _capped_shares, apply_position_sizing
from .reporting import (
    TRADE_CHARTS_DIRNAME,
    jsonify_cells,
    trade_folder_name,
    wants_trade_folder,
    write_html_report,
    write_sqlite,
    write_trade_folder,
)
from .strategy import backtest_all_trades
from .wedge import backtest_wedge
from .wedge_chart import write_wedge_trade_folder

console = Console()


def _stdout_can_render(sample: str) -> bool:
    """Fancy glyphs crash rich's legacy-Windows writer when stdout isn't UTF-8
    (piped to a file, old conhost, cp1252) -- probe once and fall back to ASCII."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sample.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_FANCY = _stdout_can_render("▲▼◆·…▐→")
_UP, _DOWN, _FLAT, _DOT, _BAR, _ELLIPSIS, _ARROW = (
    ("▲", "▼", "◆", "·", "▐", "…", "→") if _FANCY else ("+", "-", "o", ".", "|", "...", "->")
)
_SPINNER = "dots" if _FANCY else "line"  # braille dots on UTF-8, retro |/-\ elsewhere

# Per-result glyph + style for the scrolling run log. Deliberately CRT-flavored:
# a phosphor-green hit, a red miss, dim amber for everything that never traded.
_RESULT_GLYPHS = {
    "WIN": (_UP, "bold green"),
    "LOSS": (_DOWN, "bold red"),
    "EXIT_AT_OPEN": (_FLAT, "bold yellow"),
    "NO_TRIGGER": (_DOT, "dim"),
    "INVALID_RETRACE": (_DOT, "dim yellow"),
    "STALE_HOD": (_DOT, "dim yellow"),
    "DID_NOT_QUALIFY": (_DOT, "dim"),
    "INSUFFICIENT_DATA": ("?", "dim cyan"),
    "SKIPPED_PRICE_FILTER": (_DOT, "dim"),
    "SKIPPED_FLOAT_FILTER": (_DOT, "dim"),
    "SKIPPED_BLACKOUT_DATE": (_DOT, "dim magenta"),
    "NO_WEDGE": (_DOT, "dim"),
    "INVALID_WEDGE": (_DOT, "dim yellow"),
    "SKIPPED_PERSISTENCE": (_DOT, "dim"),
    "SKIPPED_BORROW_FEE": (_DOT, "dim magenta"),
    "ERROR": ("!", "bold white on red"),
}


def _result_line(r: TradeResult, live: tuple[float, int, float] | None = None) -> str:
    glyph, style = _RESULT_GLYPHS.get(r.result, (_DOT, "dim"))
    head = f"[{style}]{glyph} {r.ticker:<6} {r.date}  {r.result:<20}[/]"
    if r.entry_price is not None:
        fills = f"{r.entry_fill_source or '?'}/{r.exit_fill_source or '?'}"
        # R-multiple and scale-out activity surfaced live, not just in the report --
        # the actualized risk:reward for THIS trade, and whether the ladder fired.
        r_tag = f" [dim]{r.r_multiple:+.2f}R[/]" if r.r_multiple is not None else ""
        scale_tag = f" [dim]({r.scale_outs}x scale)[/]" if r.scale_outs > 0 else ""
        side_tag = " [magenta]SHORT[/]" if getattr(r, "direction", "long") == "short" else ""
        detail = f"[{style}]{r.pnl_per_share:+.2f}/sh[/]{r_tag}{scale_tag}{side_tag}  [dim]entry {r.entry_price:.2f} {_ARROW} exit {r.exit_price:.2f} ({fills})[/]"
        # Account-relative sizing for this trade, when the live walk supplies it:
        # position as % of account, entry-to-stop risk as % of account, and the
        # account change this trade produced. These are ratios (shares scale with
        # equity) so they're near-invariant to the exact equity level.
        if live is not None:
            equity_before, shares, pnl_total = live
            if equity_before:
                pct_acct = shares * r.entry_price / equity_before * 100.0
                acct_chg = pnl_total / equity_before * 100.0
                risk_txt = ""
                if r.stop_price is not None and r.entry_price > r.stop_price:
                    risk_txt = f" {_DOT} risk {shares * (r.entry_price - r.stop_price) / equity_before * 100.0:.1f}%"
                chg_style = "green" if pnl_total >= 0 else "red"
                detail += f" [dim]{_DOT} {pct_acct:.0f}% of acct{risk_txt} {_DOT}[/] [{chg_style}]{acct_chg:+.1f}% acct[/]"
    else:
        reason = (r.reason or "").strip()
        detail = f"[dim]{reason[:88]}{_ELLIPSIS if len(reason) > 88 else ''}[/]"
    return f"  {head} {detail}"


def _acct_field(equity: float, starting: float) -> str:
    """The live account readout for the progress bar's bottom line: current value
    plus the running change from starting equity, in dollars and percent."""
    diff = equity - starting
    pct = (diff / starting * 100.0) if starting else 0.0
    colour = "green" if diff >= 0 else "red"
    sign = "+" if diff >= 0 else "-"
    return f"[bold]${equity:,.0f}[/] [{colour}]{sign}${abs(diff):,.0f} {sign}{abs(pct):.1f}%[/]"


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name=_SPINNER, style="bold cyan"),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(bar_width=22, style="grey35", complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[bold green]{task.fields[wins]}" + _UP + "[/][dim]/[/][bold red]{task.fields[losses]}" + _DOWN + "[/]"),
        TextColumn("{task.fields[acct]}"),
        TextColumn("[dim]{task.fields[status]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _max_streak(values: list[str], target: str) -> int:
    best = cur = 0
    for v in values:
        if v == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _cohort_stats(cohort: list[TradeResult]) -> dict:
    """Win rate / avg realized R-multiple / avg $-per-share for one cohort of
    trades (e.g. scaled vs non-scaled) -- the ACTUALIZED numbers from what really
    happened (r_multiple is each trade's own weighted-exit P&L ÷ stop_offset,
    already reflecting every scale-out leg), not the configured target."""
    if not cohort:
        return {"count": 0, "win_rate_pct": None, "avg_r_multiple": None, "avg_pnl_per_share": None}
    wins = sum(1 for r in cohort if r.result == "WIN")
    r_multiples = [r.r_multiple for r in cohort if r.r_multiple is not None]
    return {
        "count": len(cohort),
        "win_rate_pct": round(wins / len(cohort) * 100, 2),
        "avg_r_multiple": round(sum(r_multiples) / len(r_multiples), 3) if r_multiples else None,
        "avg_pnl_per_share": round(sum(r.pnl_per_share for r in cohort) / len(cohort), 4),
    }


def _win_rate_by_price_bucket(trades: list[TradeResult]) -> dict:
    """Win rate split by ENTRY PRICE band. run_20260709_112928 showed a flat-cents
    stop won 55% on <$5 names and 14% on >=$8 ones -- this surfaces that split
    directly so the price/offset-mode interaction is visible per run, not just in
    ad-hoc analysis. Buckets are fixed and coarse; small-N bands show their count."""
    bands = [("<$5", 0.0, 5.0), ("$5-$8", 5.0, 8.0), (">=$8", 8.0, float("inf"))]
    out = {}
    for label, lo, hi in bands:
        rs = [r for r in trades if r.entry_price is not None and lo <= r.entry_price < hi]
        if rs:
            out[label] = {"count": len(rs), "win_rate_pct": round(sum(1 for r in rs if r.result == "WIN") / len(rs) * 100, 2)}
    return out


def _trail_cohort(cohort: list) -> dict:
    """WIN%/avg-R for a set of candle_trail trades (runners vs base hits)."""
    n = len(cohort)
    if n == 0:
        return {"count": 0, "win_rate_pct": None, "avg_r": None, "avg_pnl_per_share": None}
    wins = sum(1 for t in cohort if t.result == "WIN")
    rs = [t.r_multiple for t in cohort if t.r_multiple is not None]
    return {
        "count": n,
        "win_rate_pct": round(wins / n * 100, 1),
        "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
        "avg_pnl_per_share": round(sum(t.pnl_per_share for t in cohort) / n, 4),
    }


def summarize(results: list[TradeResult], starting_equity: float, stop_offset: float, target_offset: float, candidate_count: int | None = None, offset_mode: str = "cents", target_offset_pct: float | None = None, stop_offset_pct: float | None = None, entry_mode: str = "hod_break", stop_mode: str = "offset", target_mode: str = "offset", target_r_multiple: float | None = None, monthly_fees_total: float = 0.0, exit_mode: str = "standard") -> dict:
    """candidate_count is the true number of input ticker/date lines processed --
    pass it explicitly whenever a candidate can produce more than one TradeResult
    (strategy.max_trades_per_day > 1), since len(results) then overcounts it.
    Defaults to len(results) for callers where that distinction doesn't apply.

    offset_mode/target_offset_pct/stop_offset_pct describe how target/stop were
    measured, so the configured-R:R string and theoretical EV render correctly in
    percent mode (where a single per-share cents figure is meaningless -- there the
    per-share EV is left null and only the R-multiple EV, a pure ratio, is shown)."""
    trades = [r for r in results if r.entry_price is not None]
    wins = [r for r in trades if r.result == "WIN"]
    losses = [r for r in trades if r.result == "LOSS"]
    pnl = sum(r.pnl_total for r in trades)
    gross_win = sum(r.pnl_total for r in wins)
    gross_loss = abs(sum(r.pnl_total for r in losses))
    rr = None
    if wins and losses:
        rr = round((sum(r.pnl_per_share for r in wins) / len(wins)) / abs(sum(r.pnl_per_share for r in losses) / len(losses)), 3)
    win_rate_frac = len(wins) / len(trades) if trades else 0.0
    # Expected Value: a forward-looking projection from the CONFIGURED risk:reward
    # (not the realized one) and the OBSERVED win rate, assuming every non-win is a
    # full loss. Distinct from expectancy_per_share/avg_r_multiple, which are the
    # actual realized average outcome (including fees, slippage, and fills that
    # landed short of a clean target/stop).
    # Configured reward:risk ratio and its string form, mode-aware. The RATIO is
    # identical whether measured in cents or percent (10%/5% == 0.50/0.25 == 2:1),
    # so the R-multiple EV works in both modes; the per-share (cents) EV only means
    # something in cents mode and is left null under percent. With target_mode
    # "r_multiple" the configured R:R IS the knob itself (target = R x the real
    # per-trade risk), and no single per-share figure exists at all.
    if target_mode == "r_multiple" and target_r_multiple:
        reward_r = float(target_r_multiple)
        risk_reward_configured = f"+{target_r_multiple:g}R/-1R" + (" (stop @ pullback low)" if stop_mode == "pullback_low" else "")
        expected_value_per_share = None
    elif offset_mode == "percent" and target_offset_pct and stop_offset_pct:
        reward_r = target_offset_pct / stop_offset_pct
        risk_reward_configured = f"+{target_offset_pct:g}%/-{stop_offset_pct:g}%"
        expected_value_per_share = None
    else:
        reward_r = (target_offset / stop_offset) if stop_offset else None
        risk_reward_configured = f"+{target_offset:.2f}/-{stop_offset:.2f}" if stop_offset else None
        expected_value_per_share = (win_rate_frac * target_offset - (1 - win_rate_frac) * stop_offset) if trades else None
    # NOTE: this parametric figure is retained ONLY as a labelled model baseline
    # ("what a fixed R:R bracket at this win rate would pay"), never as the
    # headline expectancy. It assumes every win pays exactly reward_r and every
    # non-win costs exactly 1R -- both false under a trailing exit, and doubly
    # so under concurrent mode where the wedge uses measured-move targets and
    # reward_r here comes from config.strategy. On run_20260723_221310 it read
    # -0.163R against a realized +0.353R. The report shows realized expectancy
    # (quant.per_strategy[*].expectancy_r); this stays for the comparison only.
    expected_value_r = (win_rate_frac * reward_r - (1 - win_rate_frac)) if (trades and reward_r is not None) else None
    results_ordered = [r.result for r in trades]
    # Distinct (ticker, date) pairs, not rows: a candidate that took 3 trades
    # (max_trades_per_day > 1) is still ONE qualified candidate, not three.
    qualified_pairs = {(r.ticker, r.date) for r in results if r.result not in {"DID_NOT_QUALIFY", "INSUFFICIENT_DATA", "SKIPPED_PRICE_FILTER", "SKIPPED_FLOAT_FILTER", "SKIPPED_BLACKOUT_DATE", "ERROR"}}

    # Scaled vs non-scaled: does the ladder actually help, or just add legs?
    scaled_stats = _cohort_stats([r for r in trades if r.scale_outs > 0])
    non_scaled_stats = _cohort_stats([r for r in trades if r.scale_outs == 0])

    # Runner vs base-hit split for exit_mode "candle_trail_1min": how often did the
    # fat-green runner engage (dropping the 1:1 target), and how did each cohort do?
    # Only meaningful in that exit mode -- omitted otherwise.
    trail_stats = None
    if exit_mode == "candle_trail_1min":
        trail_stats = {
            "runners": _trail_cohort([t for t in trades if t.trail_activated]),
            "base_hits": _trail_cohort([t for t in trades if not t.trail_activated]),
        }

    # Win rate by entry hour (ET, from each trade's own entry_time) -- a coarse
    # session-timing breakdown; small-N hours are visible via their own count.
    hourly: dict[int, list[TradeResult]] = {}
    for r in trades:
        if not r.entry_time:
            continue
        try:
            hr = pd.Timestamp(r.entry_time).hour
        except (ValueError, TypeError):
            continue
        hourly.setdefault(hr, []).append(r)
    win_rate_by_hour = {
        hr: {"count": len(rs), "win_rate_pct": round(sum(1 for r in rs if r.result == "WIN") / len(rs) * 100, 2)}
        for hr, rs in sorted(hourly.items())
    }

    # Position-sizing aggregates: how many shares the trades actually bought, and
    # which cap bound each (notional %, the risk cap, the volume cap, or fixed
    # shares) -- so a run makes the sizing regime visible, especially when
    # risk_pct_of_equity is on and should be the binding constraint on wide stops.
    total_shares = sum(r.shares for r in trades if r.shares)
    avg_shares = round(total_shares / len(trades)) if trades else 0
    size_basis_counts: dict[str, int] = {}
    for r in trades:
        if r.size_basis:
            size_basis_counts[r.size_basis] = size_basis_counts.get(r.size_basis, 0) + 1

    # Wedge exit taxonomy: what actually closed each position -- target / stop /
    # session_end (the EXIT_AT_OPEN hold) / one of the hold-to-open limiters
    # (time_stop, flat_by_time, stale_progress). Only wedge trades set
    # exit_trigger, so this stays empty on dip-only runs; on a wedge run it makes
    # "how often did we still ride into the open?" a one-line answer.
    exit_trigger_counts: dict[str, int] = {}
    for r in trades:
        if r.exit_trigger:
            exit_trigger_counts[r.exit_trigger] = exit_trigger_counts.get(r.exit_trigger, 0) + 1

    # The quant block: per-strategy cohorts, hold times, exit-at-open tracking,
    # realized equity risk, leverage/concurrency, significance, and strategy
    # interaction -- see quant.py for why each exists.
    qb = quant.build(results, starting_equity)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates": candidate_count if candidate_count is not None else len(results),
        "qualified_or_traded": len(qualified_pairs),
        "trades_taken": len(trades),
        "total_shares_traded": total_shares,
        "avg_shares_per_trade": avg_shares,
        "size_basis_counts": size_basis_counts,
        "exit_trigger_counts": exit_trigger_counts,
        # Concurrency / funding: how often entries overlapped in time, how many
        # needed margin dollars beyond free cash, and the account's peak gross
        # leverage -- the numbers that show what a live margin account would
        # actually have had to fund (see risk.max_leverage).
        "concurrent_entries": sum(1 for r in trades if r.open_positions_at_entry > 0),
        "max_open_positions": (max(r.open_positions_at_entry for r in trades) + 1) if trades else 0,
        "margin_funded_entries": sum(1 for r in trades if (r.margin_used or 0.0) > 0.0),
        "margin_used_total": round(sum(r.margin_used or 0.0 for r in trades), 2),
        "peak_gross_leverage": round(max((r.leverage_at_entry or 0.0) for r in trades), 3) if trades else 0.0,
        "wins": len(wins),
        "losses": len(losses),
        # WIN RATE BY REALIZED PROFIT, not by the Result label. The label bucket
        # excludes EXIT_AT_OPEN, but those trades are mostly PROFITABLE (in
        # run_20260723_221310: 13 of 17 green, +1.18R mean -- the best cohort in
        # the run), so labelling them non-wins understated the rate by ~9 points
        # (41.86 -> 51.16) and fed a misleading parametric EV. The label-based
        # count stays visible as wins/losses above, and win_rate_labeled_pct
        # below, so the two definitions can be compared rather than silently
        # swapped.
        "win_rate_pct": qb["combined"]["win_rate_pct"] if qb.get("combined") else 0.0,
        "win_rate_labeled_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        # The strategies that ACTUALLY traded, from the rows themselves. The old
        # scalar entry_mode came from config.strategy and labelled a concurrent
        # run "dip_red_break" even when 68% of its trades were wedge breaks.
        "strategies_traded": sorted({r.entry_mode for r in trades if r.entry_mode}),
        "quant": qb,
        "entry_mode": entry_mode,
        "offset_mode": offset_mode,
        "stop_mode": stop_mode,
        "target_mode": target_mode,
        "exit_mode": exit_mode,
        "trail": trail_stats,
        "risk_reward_configured": risk_reward_configured,
        "realized_avg_reward_to_risk": rr,
        "avg_r_multiple": round(sum((r.r_multiple or 0) for r in trades) / len(trades), 3) if trades else 0.0,
        "expected_value_r": round(expected_value_r, 3) if expected_value_r is not None else None,
        "expected_value_per_share": round(expected_value_per_share, 4) if expected_value_per_share is not None else None,
        "net_pnl": round(pnl, 2),
        "net_pnl_pct": round(pnl / starting_equity * 100.0, 2) if starting_equity else None,
        # Monthly platform fees are a flat cost outside trade P&L: net_pnl stays the
        # gross trade result, while the account's real ending value nets them out.
        "monthly_fees_total": round(monthly_fees_total, 2),
        "net_pnl_after_fees": round(pnl - monthly_fees_total, 2),
        "ending_equity": round(starting_equity + pnl - monthly_fees_total, 2),
        "avg_pnl_per_trade": round(pnl / len(trades), 2) if trades else 0.0,
        "expectancy_per_share": round(sum(r.pnl_per_share for r in trades) / len(trades), 4) if trades else 0.0,
        "expectancy_pct": round(sum((r.pnl_pct or 0.0) for r in trades) / len(trades), 3) if trades else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else None,
        "avg_pullback_retrace_pct": round(pd.Series([r.pullback_retrace_pct for r in trades if r.pullback_retrace_pct is not None]).mean(), 2) if trades else None,
        "trades_with_scale_outs": sum(1 for r in trades if r.scale_outs > 0),
        "scaled_trades_count": scaled_stats["count"],
        "scaled_trades_win_rate_pct": scaled_stats["win_rate_pct"],
        "scaled_trades_avg_r_multiple": scaled_stats["avg_r_multiple"],
        "scaled_trades_avg_pnl_per_share": scaled_stats["avg_pnl_per_share"],
        "non_scaled_trades_count": non_scaled_stats["count"],
        "non_scaled_trades_win_rate_pct": non_scaled_stats["win_rate_pct"],
        "non_scaled_trades_avg_r_multiple": non_scaled_stats["avg_r_multiple"],
        "non_scaled_trades_avg_pnl_per_share": non_scaled_stats["avg_pnl_per_share"],
        "win_rate_by_hour": win_rate_by_hour,
        "win_rate_by_price_bucket": _win_rate_by_price_bucket(trades),
        "path_ambiguous_bars": sum(1 for r in trades if r.path_ambiguous),
        "ambiguous_resolved_by_ticks": sum(1 for r in trades if r.ambiguity_resolution == "ticks"),
        "ambiguous_resolved_by_fallback": sum(1 for r in trades if r.path_ambiguous and r.ambiguity_resolution != "ticks"),
        "largest_win_streak": _max_streak(results_ordered, "WIN"),
        "largest_losing_streak": _max_streak(results_ordered, "LOSS"),
        "result_counts": pd.Series([r.result for r in results]).value_counts().to_dict() if results else {},
        # Diagnostics: why did candidates that reached a trigger-eligible bar not trade?
        "insufficient_rvol_history": sum(1 for r in results if r.result == "INSUFFICIENT_DATA" and "RVOL baseline" in (r.reason or "")),
        "did_not_qualify_premarket_volume": sum(1 for r in results if r.disqualify_reason == "premarket_volume"),
        "did_not_qualify_rvol": sum(1 for r in results if r.disqualify_reason == "rvol"),
        "did_not_qualify_gain_pct": sum(1 for r in results if r.disqualify_reason == "gain_pct"),
        "trigger_touches_total": sum(r.trigger_touches for r in results),
        "skipped_no_macd_bar_total": sum(r.skipped_no_macd_bar for r in results),
        "skipped_rvol_not_yet_confirmed_total": sum(r.skipped_rvol_not_yet_confirmed for r in results),
        "skipped_premarket_volume_not_yet_confirmed_total": sum(r.skipped_premarket_volume_not_yet_confirmed for r in results),
        "skipped_gain_not_yet_confirmed_total": sum(r.skipped_gain_not_yet_confirmed for r in results),
        "skipped_red_volume_dominant_total": sum(r.skipped_red_volume_dominant for r in results),
        "skipped_macd_not_bullish_total": sum(r.skipped_macd_not_bullish for r in results),
        "skipped_fill_exceeds_limit_total": sum(r.skipped_fill_exceeds_limit for r in results),
        "skipped_spread_too_wide_total": sum(r.skipped_spread_too_wide for r in results),
        "skipped_outside_entry_window_total": sum(r.skipped_outside_entry_window for r in results),
        "skipped_retrace_below_ema9_total": sum(r.skipped_retrace_below_ema9 for r in results),
        "skipped_retrace_below_ema20_total": sum(r.skipped_retrace_below_ema20 for r in results),
        "skipped_retrace_below_vwap_total": sum(r.skipped_retrace_below_vwap for r in results),
        "skipped_buy_below_ema9_total": sum(r.skipped_buy_below_ema9 for r in results),
        "skipped_buy_below_ema20_total": sum(r.skipped_buy_below_ema20 for r in results),
        "skipped_buy_below_vwap_total": sum(r.skipped_buy_below_vwap for r in results),
        "skipped_risk_too_wide_total": sum(r.skipped_risk_too_wide for r in results),
        "skipped_continuation_vwap_total": sum(r.skipped_continuation_vwap for r in results),
        "skipped_no_red_dip_candle_total": sum(r.skipped_no_red_dip_candle for r in results),
        "invalid_retrace_attempts_total": sum(r.invalid_retrace_attempts for r in results),
        "stale_hod_breaks_total": sum(r.stale_hod_breaks for r in results),
    }


def get_bars_with_prev_session(data: IBKRHistoricalData, ticker: str, day, config: AppConfig):
    """Fetch a candidate's bars and split them, retrying with a wider window if the
    default fetch didn't reach back far enough to find the previous trading session."""
    bars = data.get_day_bars(ticker, day)
    premarket, prev_rth = split_sessions(
        bars, day, config.session.timezone, config.session.premarket_start, config.session.regular_open
    )
    if prev_rth.empty and not premarket.empty:
        # Likely a holiday sitting next to a weekend, pushing the previous session
        # further back than the default window reaches. Retry once, wider.
        wide_bars = data.get_day_bars(ticker, day, duration=config.data.wide_duration)
        wide_premarket, wide_prev_rth = split_sessions(
            wide_bars, day, config.session.timezone, config.session.premarket_start, config.session.regular_open
        )
        if not wide_prev_rth.empty:
            return wide_bars, wide_premarket, wide_prev_rth
    return bars, premarket, prev_rth


# Resume: which config keys actually change a candidate's RESULT rows. risk is
# deliberately excluded (position sizing is recomputed over the merged results
# every run), as are reporting/ibkr/monte_carlo and the data-fetch plumbing
# (cache, pacing, timeouts -- they change speed, never outcomes). wedge IS
# included: its keys (enabled/concurrent_with_dip/geometry/exits) decide which
# rows exist at all. wedge.max_pct_of_volume_at_entry is technically re-sized
# every run like the risk section, but one sizing-only key doesn't justify
# splitting the section's comparability check.
_RESUME_MATCH_SECTIONS = {"strategy": StrategyConfig, "session": SessionConfig, "blackout": BlackoutConfig, "wedge": WedgeConfig}
_RESUME_MATCH_DATA_KEYS = ("bar_size", "what_to_show", "use_rth")


def resume_config_mismatches(prev_snapshot: dict, cur_snapshot: dict) -> list[str]:
    """Result-affecting differences between a previous run's config_used.yaml and
    the current resolved config. A key MISSING from the old snapshot (a knob added
    to the codebase since that run) counts as matching only when the current value
    still equals the schema default -- the old run necessarily ran the default
    behavior, so a default-valued new knob preserves comparability."""
    diffs: list[str] = []
    for section, schema in _RESUME_MATCH_SECTIONS.items():
        prev_s = prev_snapshot.get(section, {}) or {}
        cur_s = cur_snapshot.get(section, {}) or {}
        for k in sorted(set(prev_s) | set(cur_s)):
            if k not in prev_s and cur_s.get(k) == getattr(schema, k, None):
                continue
            if prev_s.get(k) != cur_s.get(k):
                diffs.append(f"{section}.{k}: {prev_s.get(k)!r} -> {cur_s.get(k)!r}")
    prev_d = prev_snapshot.get("data", {}) or {}
    cur_d = cur_snapshot.get("data", {}) or {}
    for k in _RESUME_MATCH_DATA_KEYS:
        if prev_d.get(k, getattr(DataConfig, k, None)) != cur_d.get(k):
            diffs.append(f"data.{k}: {prev_d.get(k)!r} -> {cur_d.get(k)!r}")
    return diffs


def _trade_result_from_row(row: dict) -> TradeResult:
    """Rebuild a TradeResult from a trades-CSV row: NaN/'' revert to the field's
    dataclass default, ints/bools re-coerce (CSV round-trips them as floats/
    strings), and exit_legs' JSON string parses back to its list."""
    tr = TradeResult(ticker=str(row.get("ticker", "")), date=str(row.get("date", "")), result=str(row.get("result", "")))
    for name, f in TradeResult.__dataclass_fields__.items():
        if name in ("ticker", "date", "result") or name not in row:
            continue
        v = row[name]
        if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v == ""):
            continue
        if name == "exit_legs":
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
        elif isinstance(f.default, bool):
            v = str(v).strip().lower() in ("true", "1", "1.0")
        elif isinstance(f.default, int):
            v = int(float(v))
        setattr(tr, name, v)
    return tr


def _append_partial_rows(path: Path, rows: list[TradeResult]) -> None:
    """Crash-safe incremental results: append each candidate's rows the moment
    they exist, so a killed/crashed/aborted run still leaves a --resume-able
    artifact. Without this, the trades CSV only landed at the very END of a run
    -- which made resume useless for exactly the interrupted runs it was built
    for (four runs in a row died mid-flight and left nothing to continue from).
    Never allowed to break the run itself."""
    if not rows:
        return
    try:
        fieldnames = list(TradeResult.__dataclass_fields__)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(jsonify_cells([r.to_dict()])[0])
    except Exception:
        pass


def load_resume_results(resume_dir: Path, cur_snapshot: dict, candidate_keys: set[tuple[str, str]]) -> tuple[dict[tuple[str, str], list[TradeResult]], list[str]]:
    """Everything needed to continue a previous run: its non-ERROR result rows
    keyed by (TICKER, date-iso), plus any result-affecting config differences.
    ERROR rows (timeouts, drops) and candidates the run never reached are left
    out so the caller reprocesses them; rows for candidates no longer in the
    current input are dropped. Reads the run's final trades CSV when it exists,
    else the incremental partial checkpoint an interrupted run left behind."""
    prev_cfg = resume_dir / "config_used.yaml"
    if not prev_cfg.exists():
        raise FileNotFoundError(f"{resume_dir} has no config_used.yaml -- not a run folder?")
    prev_snapshot = yaml.safe_load(prev_cfg.read_text(encoding="utf-8")) or {}
    mismatches = resume_config_mismatches(prev_snapshot, cur_snapshot)
    csvs = sorted(resume_dir.glob("backtest_trades_*.csv"))
    finals = [p for p in csvs if "partial" not in p.name]
    src = finals[-1] if finals else (csvs[-1] if csvs else None)
    if src is None:
        raise FileNotFoundError(
            f"{resume_dir} has no backtest_trades_*.csv (final or partial) to resume from -- that run died "
            f"before recording any results (its cached bars still make a fresh run fast)."
        )
    df = pd.read_csv(src)
    carried: dict[tuple[str, str], list[TradeResult]] = {}
    for _, row in df.iterrows():
        key = (str(row["ticker"]).upper(), str(row["date"]))
        if key not in candidate_keys or str(row["result"]) == Result.ERROR.value:
            continue
        carried.setdefault(key, []).append(_trade_result_from_row(row.to_dict()))
    return carried, mismatches


def _copy_carried_trade_folders(carried: dict, resume_dir: Path, run_dir: Path, config: AppConfig) -> None:
    """Bring the previous run's folder artifacts (chart/json/bars) over for the
    carried candidates so the merged report's links resolve without re-rendering
    a single chart. Missing source folders (gated out or chart-less) just skip."""
    if not config.reporting.write_trade_folders:
        return
    charts_dir = run_dir / TRADE_CHARTS_DIRNAME
    for rows in carried.values():
        for r in rows:
            safe = trade_folder_name(r.ticker, r.date, r.setup_index, r.entry_mode)
            src = resume_dir / "trade_folders" / safe
            dst = run_dir / "trade_folders" / safe
            if not src.exists() or dst.exists():
                continue
            shutil.copytree(src, dst)
            chart = dst / "chart.png"
            if chart.exists() and config.reporting.write_charts:
                charts_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(chart, charts_dir / f"{safe}.png")
                if r.result in (Result.WIN.value, Result.LOSS.value):
                    sub = charts_dir / ("wins" if r.result == Result.WIN.value else "losses")
                    sub.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(chart, sub / f"{safe}.png")


# Auto-recovery tuning. _THROTTLE_TRIP: timeouts among the last 15 candidates
# that mean "the farm is throttling us, not just one dead ticker" -> enter
# recovery. _RECOVERY_BACKOFF: escalating cooldown (seconds) between health
# probes while waiting for the farm to cool / TWS to come back. _MAX_RECOVERY_
# ROUNDS/_MAX_TRANSIENT_RETRIES bound how long ONE candidate is chased before it
# is finally recorded ERROR (still --resume-able) so a single poison candidate
# can't loop forever.
_THROTTLE_TRIP = 6
_RECOVERY_BACKOFF = [120, 300, 600, 900]
_MAX_RECOVERY_ROUNDS_PER_CANDIDATE = 6
_MAX_TRANSIENT_RETRIES = 2


def _await_farm_recovery(data: IBKRHistoricalData, config: AppConfig, status) -> bool:
    """Wait out an IBKR history-farm throttle or a dropped/nightly-restarted TWS
    connection until a health probe on a liquid ticker returns real data again.
    Reconnects as needed, cools down with escalating backoff, and keeps the
    connection serviced. Returns True once the farm answers, or False if it never
    recovered within config.data.recovery_max_total_wait_s (the run then stops
    with its partial checkpoint intact for --resume). Never raises."""
    cap = config.data.recovery_max_total_wait_s
    if cap <= 0:
        return False
    waited = 0.0
    attempt = 0
    while waited < cap:
        if not data.ib.isConnected():
            status("connection down -- reconnecting to TWS")
            try:
                data.disconnect()
            except Exception:
                pass
            try:
                data.connect()
            except Exception:
                pass
        if data.ib.isConnected():
            status("probing IBKR data-farm health")
            if data.probe_farm_health():
                status("IBKR data farm healthy -- resuming")
                return True
        delay = _RECOVERY_BACKOFF[min(attempt, len(_RECOVERY_BACKOFF) - 1)]
        status(f"IBKR throttled/unreachable -- cooling down {delay // 60}m (waited {int(waited // 60)}m of {int(cap // 60)}m)")
        try:
            data.ib.sleep(delay)   # keeps the event loop/connection serviced
        except Exception:
            time.sleep(delay)      # last resort if the loop is unusable
        waited += delay
        attempt += 1
    return False


def run_backtest(config: AppConfig, resume_from: Path | None = None) -> tuple[Path, Path | None]:
    config.data.output_dir.mkdir(parents=True, exist_ok=True)
    config.data.cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(config.data.input_csv, config.data.reference_csv)
    console.print(f"[bold]Loaded {len(candidates)} ticker/date candidates[/bold]")
    # Optional per-ticker borrow fees (data/borrow_fees.csv) for the wedge strategy's
    # short-side gate. Absent (the usual case) -> {} and shorts are treated as
    # borrowable-but-assumed; see load_borrow_fees / config.wedge.max_borrow_fee_pct.
    borrow_fees = load_borrow_fees(config.data.input_csv.parent / "borrow_fees.csv") if config.wedge.enabled else {}
    if config.wedge.enabled:
        mode_tag = "CONCURRENT with dip" if config.wedge.concurrent_with_dip else "replacing dip"
        console.print(f"[bold magenta]Wedge strategy ACTIVE[/] [bold]({mode_tag})[/] [dim](sides={config.wedge.sides}, persistence={config.wedge.persistence_mode}, "
                      f"max_borrow_fee={config.wedge.max_borrow_fee_pct}%, {len(borrow_fees)} borrow rows)[/]")
        if config.wedge.concurrent_with_dip:
            lev = config.risk.max_leverage
            lev_tag = "unlimited parallel funding" if lev is None else (f"{lev:g}x leverage cap" if lev > 1.0 else "cash-only funding")
            console.print(f"[dim]  concurrent capital model: {lev_tag} (risk.max_leverage={lev})[/]")
    # Resume: carry a previous run's completed results and only process what it
    # didn't finish (ERROR rows retry; unreached candidates run fresh). Validated
    # BEFORE the run folder is created so a config mismatch fails clean.
    carried: dict[tuple[str, str], list[TradeResult]] = {}
    pending = candidates
    if resume_from is not None:
        resume_from = Path(resume_from)
        keys = {(c.ticker.upper(), c.day.isoformat()) for c in candidates}
        carried, mismatches = load_resume_results(resume_from, config_to_dict(config), keys)
        if mismatches:
            raise ValueError(
                f"Cannot resume from {resume_from.name}: result-affecting settings differ "
                f"(previous -> current):\n  " + "\n  ".join(mismatches)
                + "\nMatch the config, or run without --resume."
            )
        pending = [c for c in candidates if (c.ticker.upper(), c.day.isoformat()) not in carried]
        console.print(
            f"[bold]Resuming from {resume_from.name}:[/bold] carrying "
            f"{sum(len(v) for v in carried.values())} results for {len(carried)} candidates "
            f"{_DOT} {len(pending)} left to process"
        )
    data = IBKRHistoricalData(config)
    results: list[TradeResult] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.data.output_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Incremental checkpoint: rows land here candidate by candidate, so a run
    # killed/crashed/aborted at ANY point leaves a --resume-able record. The
    # final trades CSV (written at run end) replaces it and it is deleted.
    # Seeded with any carried resume rows so resumes CHAIN: if this run dies
    # too, its checkpoint still holds everything accumulated so far.
    partial_path = run_dir / f"backtest_trades_partial_{stamp}.csv"
    if carried:
        _append_partial_rows(partial_path, [r for rows in carried.values() for r in rows])
    charts_dir = run_dir / TRADE_CHARTS_DIRNAME
    charts_link = None
    if config.reporting.write_trade_folders and config.reporting.write_charts:
        # Created up front (even though charts land in it as trades are found)
        # so the link printed here resolves immediately -- no need to wait for
        # the run to finish before you can start tailing/opening the folder.
        charts_dir.mkdir(parents=True, exist_ok=True)
        charts_link = f"[link=file:///{charts_dir.resolve().as_posix()}]{charts_dir}[/link]"
        console.print(f"Trade charts folder (fills in as trades are found): [green]{charts_link}[/green]")
    # Record the exact resolved config this run executed with (config.yaml plus any
    # env overrides and fallback defaults), and echo the strategy knobs up front --
    # so "did my config change actually apply?" is answerable from the run itself.
    config_snapshot = config_to_dict(config)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    console.print("[bold]Strategy config for this run[/bold] (full copy: config_used.yaml in the run folder):")
    console.print("  " + ", ".join(f"{k}={v}" for k, v in config_snapshot["strategy"].items()))
    # Tick data is rate-limited by IBKR, so it is only fetched for bars that break
    # a trigger/stop/target level, and only when the config opts in. Every window
    # the strategy requests corresponds to an existing bar -- trades demonstrably
    # printed there -- so expect_trades tells the data layer to distrust empty
    # responses (bypass poisoned caches, retry once, never cache the contradiction)
    # instead of silently degrading that window to bar-based fills forever.
    if config.strategy.intrabar_policy == "ticks":
        def tick_fetcher(ticker, start, end):
            return data.get_ticks(ticker, start, end, expect_trades=True)
    else:
        tick_fetcher = None
    # Bid/ask spread requires a live IBKR quote request per trigger attempt, so it's
    # only fetched when max_spread_cents is actually configured.
    bid_ask_fetcher = data.get_bid_ask_spread if config.strategy.max_spread_cents is not None else None
    # RVOL qualification always needs this ticker's own daily-volume baseline; it's
    # a cheap, cached daily-bar request, fetched lazily (only after the cheap local
    # data/price-filter checks pass) rather than up front for every candidate.
    def daily_volume_fetcher(ticker: str, day) -> float | None:
        return data.get_daily_volume_baseline(
            ticker, day, lookback_days=config.strategy.rvol_lookback_days, min_history_days=config.strategy.rvol_min_history_days,
        )
    # Share float is only needed when a float filter is configured; then it's a
    # single cached IBKR fundamental-data request per ticker (see get_float).
    float_gate_on = config.strategy.min_float is not None or config.strategy.max_float is not None
    float_fetcher = data.get_float_YF if float_gate_on else None
    # ib_insync logs every IBKR error event straight to the console -- error 162
    # ("HMDS query returned no data") for each tick-gap window and retry, error 200
    # for unknown/delisted tickers (warrants especially), pacing notices... These
    # are EXPECTED, handled conditions, but they flood the terminal with red and
    # mangle the progress display. Capture them in a per-run file instead, and
    # report a one-line count at the end.
    ib_logger = logging.getLogger("ib_insync")
    ib_log_path = run_dir / "ibkr_log.txt"
    ib_handler = logging.FileHandler(ib_log_path, encoding="utf-8")
    ib_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    ib_handler.setLevel(logging.WARNING)
    ib_log_count = {"records": 0}
    ib_handler.addFilter(lambda record: ib_log_count.__setitem__("records", ib_log_count["records"] + 1) or True)
    prev_propagate = ib_logger.propagate
    ib_logger.addHandler(ib_handler)
    ib_logger.propagate = False

    wins = losses = 0
    # Live account readout for the progress bar. Sized in candidate-processing
    # order (not chronological), so it's an indicative preview that grows/shrinks
    # as trades land; the summary's ending_equity is the chronological source of
    # truth. On a resume it previews only the freshly-run trades.
    live_equity = config.risk.starting_equity
    live_months: set[str] = set()
    acct_str = _acct_field(live_equity, config.risk.starting_equity)
    try:
        console.print(f"[bold cyan]{_BAR}[/][bold] connecting to IBKR{_ELLIPSIS}[/]")
        data.connect()
        with _progress() as progress:
            task = progress.add_task("starting", total=len(pending), status="", wins=0, losses=0, acct=acct_str)
            # Everything the run waits on (paced tick pages, bar fetches, RVOL
            # baselines) reports itself here -- no more silent stretches that
            # look like a hang while IBKR pacing does its thing.
            data.on_status = lambda msg: progress.update(task, status=msg)
            # A soft-throttled history farm (or a nightly-restarted TWS) burns the
            # whole RequestTimeout on most requests. Rather than abort, the run
            # SELF-HEALS: a cluster of recent timeouts (or a dropped connection)
            # triggers _await_farm_recovery -- reconnect + escalating cooldown +
            # a liveness probe -- and then RETRIES the same candidate on the healed
            # farm, so an interrupted overnight run still finishes a complete
            # dataset. A single candidate that keeps failing across recoveries is
            # finally recorded ERROR (still --resume-able) so it can't loop forever.
            timeout_window: deque[bool] = deque(maxlen=15)
            aborted = False
            for c in pending:
                progress.update(task, description=f"{c.ticker.upper()} {c.day}", status="reading cache")
                premarket = None
                trade_results: list[TradeResult] = []
                # Trading blackout: skip the day entirely (no fetch, no scan) when its
                # date falls in a configured blackout window. Recorded as a result so
                # it still shows in the funnel; the `while blackout is None` guard
                # below leaves trade_results as this single skip and never fetches.
                blackout = blackout_reason(c.day, config.blackout)
                if blackout is not None:
                    progress.update(task, status=f"blackout {_ARROW} {blackout}")
                    trade_results = [TradeResult(
                        ticker=c.ticker.upper(), date=c.day.isoformat(),
                        result=Result.SKIPPED_BLACKOUT_DATE.value,
                        reason=f"Trading blackout ({blackout}); the bot skips this day -- no scan, no trade.",
                    )]
                recovery_rounds = 0
                transient_retries = 0
                while blackout is None:
                    try:
                        bars, premarket, prev_rth = get_bars_with_prev_session(data, c.ticker, c.day, config)
                        progress.update(task, status="replaying session")
                        # WHICH engines run comes from routing.active_strategies --
                        # the one truth table the live bot shares via signal_core, so
                        # the two programs can never disagree about which strategies
                        # were in play (that disagreement would be a Class-0 parity
                        # divergence). Dip rows come first when both run; the
                        # position-sizing walk later arbitrates the shared capital
                        # across time-overlapping entries via risk.max_leverage.
                        engines = active_strategies(config.wedge)
                        trade_results = []
                        if "dip" in engines:
                            trade_results += backtest_all_trades(
                                c.ticker, c.day, bars, premarket, prev_rth, config,
                                float_shares=c.float_shares, tick_fetcher=tick_fetcher, bid_ask_fetcher=bid_ask_fetcher,
                                daily_volume_fetcher=daily_volume_fetcher, float_fetcher=float_fetcher,
                            )
                        if "wedge" in engines:
                            # The symmetrical-pennant engine (long OR short by the
                            # break), with the per-ticker borrow fee for its short gate.
                            trade_results += backtest_wedge(
                                c.ticker, c.day, bars, premarket, prev_rth, config,
                                float_shares=c.float_shares, tick_fetcher=tick_fetcher, bid_ask_fetcher=bid_ask_fetcher,
                                daily_volume_fetcher=daily_volume_fetcher, float_fetcher=float_fetcher,
                                borrow_fee=borrow_fees.get(c.ticker.upper()),
                            )
                        timeout_window.append(False)
                        break
                    except (ConnectionError, TimeoutError) as e:
                        # The two transient failures worth waiting out: a dropped/
                        # restarted TWS (ConnectionError) or a request the throttled
                        # farm let time out (TimeoutError -- raised by the fetch
                        # layer's timeout discriminator, never mislabeled "no data").
                        # NOTHING here is ERROR'd for a throttle without first fully
                        # recovering -- that's what makes a complete dataset in ONE
                        # run: a quick blip gets brief cooldown-retries; anything
                        # that persists (or a disconnect, or a cluster of timeouts)
                        # escalates to a full farm recovery and retries the SAME
                        # candidate on the healed farm.
                        timeout_window.append(isinstance(e, TimeoutError))
                        disconnected = isinstance(e, ConnectionError) or not data.ib.isConnected()
                        throttled = sum(timeout_window) >= _THROTTLE_TRIP
                        if not disconnected and not throttled and transient_retries < _MAX_TRANSIENT_RETRIES:
                            transient_retries += 1
                            progress.update(task, status="request timed out -- 30s cooldown, retrying")
                            data.ib.sleep(30)
                            continue
                        # Escalate to full recovery (disconnect, throttle cluster, or
                        # a blip that survived the quick retries).
                        recovery_rounds += 1
                        if recovery_rounds > _MAX_RECOVERY_ROUNDS_PER_CANDIDATE:
                            trade_results = [TradeResult(
                                ticker=c.ticker.upper(), date=c.day.isoformat(), result=Result.ERROR.value,
                                reason=f"candidate still failing after {recovery_rounds - 1} farm recoveries: {e!r}")]
                            break
                        progress.console.print(
                            f"[bold yellow]{_BAR}[/] IBKR data farm "
                            f"{'unreachable' if disconnected else 'throttling us'} at {c.ticker.upper()} {c.day} "
                            f"{_ARROW} pausing to let it recover; the run continues automatically."
                        )
                        recovered = _await_farm_recovery(data, config, lambda m: progress.update(task, status=m))
                        timeout_window.clear()
                        transient_retries = 0  # fresh quick-retry budget after a recovery
                        if not recovered:
                            hrs = config.data.recovery_max_total_wait_s / 3600.0
                            progress.console.print(
                                f"[bold red]{_BAR}[/] IBKR did not recover within {hrs:.0f}h {_ARROW} stopping with "
                                f"partial results saved. Re-run [bold]backtest --resume[/bold] once TWS/the farm is back."
                            )
                            trade_results = [TradeResult(
                                ticker=c.ticker.upper(), date=c.day.isoformat(), result=Result.ERROR.value,
                                reason=f"farm did not recover within cap: {e!r}")]
                            aborted = True
                            break
                        continue  # retry THIS candidate on the healed farm
                    except Exception as e:
                        # A genuine, non-transient failure (malformed data, a bug):
                        # record ERROR and move on -- retrying wouldn't help.
                        trade_results = [TradeResult(
                            ticker=c.ticker.upper(), date=c.day.isoformat(),
                            result=Result.ERROR.value, reason=f"candidate failed: {e!r}")]
                        break
                results.extend(trade_results)
                _append_partial_rows(partial_path, trade_results)
                wins += sum(1 for tr in trade_results if tr.result == "WIN")
                losses += sum(1 for tr in trade_results if tr.result == "LOSS")
                # Walk the live account forward over each filled trade (WIN/LOSS/
                # EXIT_AT_OPEN) so the bottom line shows the account value updating,
                # and (when verbose) print each trade's line in the same pass -- so a
                # fill's line can carry its own live sizing (% of account, risk %,
                # account change %), computed from the SAME caps apply_position_sizing
                # uses. This walk is in candidate-processing order, so it's an
                # indicative preview; the summary's chronological ending_equity stays
                # the source of truth.
                printed_verdicts: set[str] = set()   # a shared non-trade verdict prints once
                for tr in trade_results:
                    live = None
                    if tr.entry_price is not None and tr.result in ("WIN", "LOSS", "EXIT_AT_OPEN"):
                        if config.risk.monthly_fees and tr.date[:7] not in live_months:
                            live_months.add(tr.date[:7])
                            live_equity = round(live_equity - config.risk.monthly_fees, 2)
                        equity_before = live_equity
                        _is_wedge = tr.entry_mode == "wedge_break"
                        live_vol_pct = config.wedge.max_pct_of_volume_at_entry if _is_wedge else None
                        live_lb_pct = config.wedge.max_entry_pct_of_lookback_volume if _is_wedge else None
                        _basis, shares = _capped_shares(equity_before, tr.entry_price, tr.stop_price, tr.volume_at_entry, config.risk,
                                                        vol_pct=live_vol_pct, lookback_volume=tr.lookback_volume, lookback_pct=live_lb_pct)
                        pnl = round(tr.pnl_per_share * shares, 2)
                        live_equity = round(live_equity + pnl, 2)
                        acct_str = _acct_field(live_equity, config.risk.starting_equity)
                        live = (equity_before, shares, pnl)
                    if config.reporting.verbose:
                        # Both engines run per candidate, so a SHARED non-trade
                        # verdict (data/price/float/qualify) is reported by each --
                        # print it once so the terminal isn't doubled. Real trades,
                        # and distinct verdicts (e.g. dip NO_TRIGGER vs wedge
                        # NO_WEDGE), always print. Every result is still recorded in
                        # the CSV/SQLite regardless -- this only de-dupes the console.
                        is_trade = live is not None
                        if is_trade or tr.result not in printed_verdicts:
                            progress.console.print(_result_line(tr, live=live))
                            if not is_trade:
                                printed_verdicts.add(tr.result)
                progress.update(task, advance=1, wins=wins, losses=losses, status="", acct=acct_str)
                if config.reporting.write_trade_folders and premarket is not None:
                    for tr in trade_results:
                        # Folder artifacts (json/bars/chart) only for the result
                        # classes trade_folders_for asks for -- the CSV/SQLite row
                        # is every candidate's permanent record either way.
                        if wants_trade_folder(tr, config.reporting.trade_folders_for):
                            if tr.entry_mode == "wedge_break":
                                # Wedge trades get the dedicated dark 2-panel chart,
                                # filed under results/<run>/wedge/{wins,losses}/.
                                write_wedge_trade_folder(tr, premarket, run_dir, write_chart=config.reporting.write_charts)
                            else:
                                write_trade_folder(tr, premarket, run_dir, write_chart=config.reporting.write_charts)
                if aborted:
                    break
            data.on_status = None
    finally:
        data.disconnect()
        ib_logger.removeHandler(ib_handler)
        ib_handler.close()
        ib_logger.propagate = prev_propagate

    if carried:
        # Merge carried + fresh results back into input order so the report reads
        # as one continuous run, and bring the carried candidates' folder
        # artifacts (charts/json/bars) over so nothing needs re-rendering.
        fresh: dict[tuple[str, str], list[TradeResult]] = {}
        for r in results:
            fresh.setdefault((r.ticker.upper(), r.date), []).append(r)
        merged: list[TradeResult] = []
        seen: set[tuple[str, str]] = set()
        for c in candidates:
            key = (c.ticker.upper(), c.day.isoformat())
            if key not in seen:
                seen.add(key)
                merged.extend(carried.get(key, []) + fresh.get(key, []))
        results = merged
        _copy_carried_trade_folders(carried, resume_from, run_dir, config)

    # Position sizing needs the full chronological trade sequence, so it runs after
    # the loop -- then the already-written per-trade JSONs are refreshed with their
    # final shares/pnl_total/equity_after. On a resume this re-sizes CARRIED trades
    # together with the new ones, so the equity curve is one coherent compounding
    # walk across the whole merged dataset.
    monthly_fees_total = apply_position_sizing(results, config.risk, wedge_vol_pct=config.wedge.max_pct_of_volume_at_entry,
                                               wedge_lookback_pct=config.wedge.max_entry_pct_of_lookback_volume)
    if config.reporting.write_trade_folders:
        for r in results:
            if r.entry_price is None:
                continue
            trade_json = run_dir / "trade_folders" / trade_folder_name(r.ticker, r.date, r.setup_index, r.entry_mode) / "trade.json"
            if trade_json.parent.exists():
                trade_json.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")

    trades_path = run_dir / f"backtest_trades_{stamp}.csv"
    pd.DataFrame(jsonify_cells([r.to_dict() for r in results])).to_csv(trades_path, index=False)
    # The final CSV supersedes the incremental checkpoint (and, unlike it,
    # includes carried resume rows and final position sizing).
    partial_path.unlink(missing_ok=True)

    summary = summarize(
        results, config.risk.starting_equity, config.strategy.stop_offset, config.strategy.target_offset,
        candidate_count=len(candidates), offset_mode=config.strategy.offset_mode,
        target_offset_pct=config.strategy.target_offset_pct, stop_offset_pct=config.strategy.stop_offset_pct,
        entry_mode=config.strategy.entry_mode, stop_mode=config.strategy.stop_mode,
        target_mode=config.strategy.target_mode, target_r_multiple=config.strategy.target_r_multiple,
        monthly_fees_total=monthly_fees_total, exit_mode=config.strategy.exit_mode,
    )
    # Monte Carlo: resample the realized trades many times to show the RANGE of
    # equity outcomes the same edge could produce (the single backtest run is just
    # one draw). Computed before summary JSON is written so its metrics ride along.
    monte_carlo = None
    if config.monte_carlo.enabled:
        monte_carlo = run_monte_carlo(results, config.risk, config.monte_carlo, wedge_vol_pct=config.wedge.max_pct_of_volume_at_entry,
                                      wedge_lookback_pct=config.wedge.max_entry_pct_of_lookback_volume)
        if monte_carlo is not None:
            summary["monte_carlo"] = monte_carlo.metrics
        else:
            console.print(
                f"[yellow]Monte Carlo skipped: needs at least {MC_MIN_TRADES} executed trades.[/]"
            )
    summary_path = None
    if config.reporting.write_summary_json:
        summary_path = run_dir / f"summary_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html_path = None
    if config.reporting.write_sqlite:
        write_sqlite(results, run_dir / f"backtest_{stamp}.sqlite")
    if config.reporting.write_html_report:
        html_path = write_html_report(results, summary, run_dir, stamp, config_snapshot=config_snapshot, monte_carlo=monte_carlo)

    net = summary.get("net_pnl") or 0.0
    net_pct = summary.get("net_pnl_pct")
    net_style = "bold green" if net >= 0 else "bold red"
    console.print()
    console.print(
        f"[bold cyan]{_BAR} RUN COMPLETE[/] "
        f"[bold green]{summary['wins']}{_UP}[/][dim]/[/][bold red]{summary['losses']}{_DOWN}[/] of {summary['trades_taken']} trades"
        f" {_DOT} net [{net_style}]{net:+,.2f}[/]" + (f" [dim]({net_pct:+.2f}%)[/]" if net_pct is not None else "")
    )
    # Monthly platform fees: shown only when actually charged. These are deducted
    # from equity AS IT COMPOUNDS during the chronological walk (each month's fee
    # shrinks that month's -- and every later month's -- position size), not lumped
    # at the end, so net-after-fees and ending_equity already reflect the drag.
    fees_total = summary.get("monthly_fees_total") or 0.0
    if fees_total > 0:
        months = int(round(fees_total / config.risk.monthly_fees)) if config.risk.monthly_fees else 0
        after = summary.get("net_pnl_after_fees") or 0.0
        after_style = "bold green" if after >= 0 else "bold red"
        console.print(
            f"  [bold red]BILLS[/] [red]-${fees_total:,.2f}[/] over {months} month(s)"
            f" [dim](${config.risk.monthly_fees:,.0f}/mo platform fee, deducted from equity as it compounds)[/]"
            f" {_DOT} net after bills [{after_style}]{after:+,.2f}[/]"
            f" {_DOT} ending equity [bold]${summary.get('ending_equity', 0.0):,.2f}[/]"
        )
    # Position sizing: total/avg shares bought and which cap bound them. Makes the
    # risk_pct_of_equity regime visible -- "risk" means the risk cap held the size,
    # "notional" means the account-% ceiling won (a tight stop), "volume" the
    # liquidity cap. Shown whenever any trade was sized.
    basis = summary.get("size_basis_counts") or {}
    if summary.get("trades_taken"):
        basis_str = ", ".join(f"{k} {v}" for k, v in sorted(basis.items(), key=lambda kv: -kv[1]))
        console.print(
            f"[dim]  shares bought:[/] total [bold]{summary.get('total_shares_traded', 0):,}[/]"
            f" {_DOT} avg [bold]{summary.get('avg_shares_per_trade', 0):,}[/]/trade"
            + (f" {_DOT} [dim]sized by:[/] {basis_str}" if basis_str else "")
        )
    # Run identity: which variant actually produced these numbers.
    console.print(
        f"[dim]  strategy:[/] [bold]{summary.get('entry_mode')}[/] entry"
        f" {_DOT} stops [bold]{summary.get('offset_mode')}[/] {summary.get('risk_reward_configured')}"
        f" {_DOT} win rate [bold]{summary.get('win_rate_pct')}%[/]"
        f" {_DOT} PF {summary.get('profit_factor')}"
    )
    # The price/offset-mode interaction, formatted (a raw dict dump below would bury it).
    pb = summary.get("win_rate_by_price_bucket") or {}
    if pb:
        parts = " ".join(f"[dim]{k}[/] {d['win_rate_pct']:.0f}%[dim]({d['count']})[/]" for k, d in pb.items())
        console.print(f"[dim]  win rate by entry price:[/] {parts}")
    # candle_trail_1min: how the runner split played out.
    tr_stats = summary.get("trail")
    if tr_stats:
        rn, bh = tr_stats["runners"], tr_stats["base_hits"]
        def _fmt(c):
            return f"{c['count']} @ {c['win_rate_pct']}% win, {c['avg_r']:+}R" if c["count"] else "0"
        console.print(
            f"[dim]  trailing exit:[/] runners [bold]{_fmt(rn)}[/] {_DOT} base hits (1:1) [bold]{_fmt(bh)}[/]"
        )
    if ib_log_count["records"]:
        console.print(
            f"[dim]{ib_log_count['records']} IBKR warnings/errors captured (tick-gap windows, unknown/delisted"
            f" tickers, pacing) {_ARROW} {ib_log_path.name} in the run folder[/]"
        )
    console.print("\n[bold]Summary[/bold]")
    # Nested dicts get their own formatted blocks (Monte Carlo below; the win-rate
    # breakdowns just above) -- dumping them raw here is unreadable.
    _skip_raw = {"monte_carlo", "win_rate_by_hour", "win_rate_by_price_bucket", "trail", "size_basis_counts"}
    for k, v in summary.items():
        if k in _skip_raw:
            continue
        console.print(f"[dim]{k}:[/] {v}")
    if monte_carlo is not None:
        m = monte_carlo.metrics
        console.print(
            f"\n[bold cyan]{_BAR} MONTE CARLO[/] [dim]({m['iterations']:,} sims, {m['mode']}, {m['trades_per_sim']} trades each)[/]"
        )
        console.print(
            f"  Final equity: median [bold]${m['median_final_equity']:,.0f}[/]"
            f" [dim]| 5% ${m['p5_final_equity']:,.0f} · 95% ${m['p95_final_equity']:,.0f}[/]"
            f" [dim](realized ${m['actual_final_equity']:,.0f})[/]"
        )
        prof_style = "green" if m["prob_profit_pct"] >= 50 else "red"
        console.print(
            f"  Prob. of profit: [{prof_style}]{m['prob_profit_pct']:.1f}%[/]"
            f" [dim]· median max DD {m['median_max_drawdown_pct']:.1f}% · 95% max DD {m['p95_max_drawdown_pct']:.1f}%[/]"
            f" · risk of ruin (<{m['ruin_threshold_pct']:.0f}%): [red]{m['risk_of_ruin_pct']:.1f}%[/]"
        )
    console.print(f"\nRun folder: [green]{run_dir}[/green]")
    console.print(f"Trades CSV: [green]{trades_path}[/green]")
    if summary_path:
        console.print(f"Summary JSON: [green]{summary_path}[/green]")
    if html_path:
        console.print(f"HTML report: [green]{html_path}[/green]")
    if charts_link:
        console.print(f"Trade charts folder: [green]{charts_link}[/green]")
    return trades_path, summary_path
