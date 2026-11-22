"""Quant metrics for the report -- pure functions over a list of TradeResults.

Everything here answers a question the old summary either got WRONG or never
asked. Kept separate from engine.summarize() (which stays the run-level ledger)
so each metric is independently testable, and so the live bot can reuse the same
definitions for its live-vs-backtest comparison (PORTING_TO_LIVE.md §11.4/11.5)
rather than inventing a second set.

Three corrections to headline numbers are deliberate and documented at their
call sites:
  1. WIN RATE is by REALIZED PROFIT (pnl_per_share > 0), not by the Result label.
     The label bucket excludes EXIT_AT_OPEN, but those trades are mostly
     PROFITABLE (they were the strategy's best cohort in the 2026-07-23 run:
     +1.18R mean over 17 trades) -- counting them as non-wins understated the
     win rate by ~9 points and poisoned every derived statistic.
  2. EXPECTANCY is realized (mean R), per strategy. The old expected_value_r was
     a parametric w*R_target - (1-w)*1R model that assumed every win pays the
     configured target and every loss pays a full stop -- both false under a
     trailing exit, and meaningless when the wedge uses measured-move targets.
  3. STRATEGY SPLIT is mandatory. Under concurrent_with_dip the run is two
     different edges (dip 68% win/+0.52R vs wedge 30% win/+0.28R); a blended
     figure describes neither.
"""
from __future__ import annotations

import math
from statistics import median

import pandas as pd

from .models import TradeResult

# Trades-per-year assumption for annualizing. This strategy trades premarket
# only, ~1-2 setups/day, so the trade-count clock IS the trading-day clock;
# 252 is the conventional US equity year.
_ANN = 252.0
STRATEGY_LABELS = {"dip_red_break": "Dip", "wedge_break": "Wedge"}


def _executed(results: list[TradeResult]) -> list[TradeResult]:
    return [r for r in results if r.entry_price is not None and r.pnl_per_share is not None]


def _rs(trades: list[TradeResult]) -> list[float]:
    return [r.r_multiple for r in trades if r.r_multiple is not None]


def _hold_minutes(r: TradeResult) -> float | None:
    """Wall-clock minutes held. None when either timestamp is missing/unparseable
    (hand-built test rows), so callers can just drop them."""
    if not r.entry_time or not r.exit_time:
        return None
    try:
        a, b = pd.Timestamp(r.entry_time), pd.Timestamp(r.exit_time)
    except Exception:
        return None
    if pd.isna(a) or pd.isna(b):
        return None
    mins = (b - a).total_seconds() / 60.0
    return mins if mins >= 0 else None


def cohort_stats(trades: list[TradeResult]) -> dict:
    """The full stat block for one cohort (a strategy, or everything).

    win_rate_pct here is BY PROFIT -- see module docstring correction (1). The
    label-based count is reported alongside as wins_labeled so the difference is
    visible rather than silently swapped.
    """
    n = len(trades)
    if not n:
        return {"n": 0}
    rs = _rs(trades)
    pnl_sh = [r.pnl_per_share for r in trades if r.pnl_per_share is not None]
    pnl_tot = [r.pnl_total for r in trades if r.pnl_total is not None]
    winners = [r for r in trades if (r.pnl_per_share or 0) > 0]
    losers = [r for r in trades if (r.pnl_per_share or 0) < 0]
    win_rs = [r.r_multiple for r in winners if r.r_multiple is not None]
    loss_rs = [r.r_multiple for r in losers if r.r_multiple is not None]
    holds = [h for h in (_hold_minutes(r) for r in trades) if h is not None]
    eao = [r for r in trades if r.result == "EXIT_AT_OPEN"]
    eao_rs = [r.r_multiple for r in eao if r.r_multiple is not None]
    gross_win = sum(p for p in pnl_tot if p > 0)
    gross_loss = abs(sum(p for p in pnl_tot if p < 0))
    out = {
        "n": n,
        "wins": len(winners),
        "losses": len(losers),
        "wins_labeled": sum(1 for r in trades if r.result == "WIN"),
        "win_rate_pct": round(len(winners) / n * 100.0, 2),
        "mean_r": round(sum(rs) / len(rs), 3) if rs else None,
        "median_r": round(median(rs), 3) if rs else None,
        "sum_r": round(sum(rs), 2) if rs else 0.0,
        "avg_win_r": round(sum(win_rs) / len(win_rs), 3) if win_rs else None,
        "avg_loss_r": round(sum(loss_rs) / len(loss_rs), 3) if loss_rs else None,
        "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None,
        "expectancy_per_share": round(sum(pnl_sh) / len(pnl_sh), 4) if pnl_sh else None,
        "net_pnl": round(sum(pnl_tot), 2) if pnl_tot else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "best_r": round(max(rs), 2) if rs else None,
        "worst_r": round(min(rs), 2) if rs else None,
        # --- hold time (never previously computed) ---
        "avg_hold_min": round(sum(holds) / len(holds), 1) if holds else None,
        "median_hold_min": round(median(holds), 1) if holds else None,
        "max_hold_min": round(max(holds), 1) if holds else None,
        # --- exit-at-open cohort (tracked, not hidden) ---
        "exit_at_open": len(eao),
        "exit_at_open_pct": round(len(eao) / n * 100.0, 1),
        "exit_at_open_mean_r": round(sum(eao_rs) / len(eao_rs), 3) if eao_rs else None,
        "exit_at_open_sum_r": round(sum(eao_rs), 2) if eao_rs else 0.0,
        "exit_at_open_wins": sum(1 for r in eao if (r.pnl_per_share or 0) > 0),
    }
    # Longest-held trade, identified (a max with no name is not actionable).
    if holds:
        longest = max((r for r in trades if _hold_minutes(r) is not None),
                      key=lambda r: _hold_minutes(r))
        out["longest_hold_trade"] = f"{longest.ticker} {longest.date}"
        out["longest_hold_r"] = longest.r_multiple
    return out


def significance(trades: list[TradeResult]) -> dict:
    """Is the mean R distinguishable from zero, and how much of it is one trade?
    t-stat and tail concentration were both absent; for a trailing-exit strategy
    whose edge lives in rare runners, they are the two numbers that decide
    whether the headline is trustworthy."""
    rs = _rs(trades)
    n = len(rs)
    if n < 2:
        return {"n": n}
    mean = sum(rs) / n
    var = sum((x - mean) ** 2 for x in rs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else None
    srt = sorted(rs, reverse=True)
    total = sum(rs)
    out = {
        "n": n, "mean_r": round(mean, 3), "sd_r": round(sd, 3),
        "se_r": round(se, 4) if se else None,
        "t_stat": round(mean / se, 2) if se else None,
        "ci95_low": round(mean - 1.96 * se, 3) if se else None,
        "ci95_high": round(mean + 1.96 * se, 3) if se else None,
    }
    for k in (1, 3, 5):
        if n > k and total:
            out[f"top{k}_pct_of_sum_r"] = round(sum(srt[:k]) / total * 100.0, 1)
            out[f"mean_r_excl_top{k}"] = round((total - sum(srt[:k])) / (n - k), 3)
    return out


def equity_risk(trades: list[TradeResult], starting_equity: float) -> dict:
    """REALIZED equity-curve risk. The report previously showed only Monte-Carlo
    SIMULATED drawdown -- never what the actual run did."""
    chron = sorted(trades, key=lambda r: (r.date, r.entry_time or ""))
    curve = [float(starting_equity)]
    for r in chron:
        if r.equity_after is not None:
            curve.append(float(r.equity_after))
    if len(curve) < 2:
        return {}
    peak = curve[0]
    max_dd = 0.0
    max_dd_abs = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
            max_dd_abs = max(max_dd_abs, peak - v)
    pnls = [r.pnl_total for r in chron if r.pnl_total is not None]
    out = {
        "starting_equity": round(float(starting_equity), 2),
        "ending_equity": round(curve[-1], 2),
        "return_pct": round((curve[-1] / curve[0] - 1) * 100.0, 2) if curve[0] else None,
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "max_drawdown_abs": round(max_dd_abs, 2),
        "equity_curve": [round(v, 2) for v in curve],
    }
    if len(pnls) > 1:
        m = sum(pnls) / len(pnls)
        sd = math.sqrt(sum((x - m) ** 2 for x in pnls) / (len(pnls) - 1))
        downside = [x for x in pnls if x < 0]
        dsd = math.sqrt(sum(x * x for x in downside) / len(downside)) if downside else 0.0
        out["sharpe"] = round(m / sd * math.sqrt(_ANN), 2) if sd else None
        out["sortino"] = round(m / dsd * math.sqrt(_ANN), 2) if dsd else None
        if max_dd > 0:
            out["return_over_max_dd"] = round((out["return_pct"] or 0) / out["max_drawdown_pct"], 2)
    return out


def leverage_profile(trades: list[TradeResult]) -> dict:
    """Concurrency + margin: how much capital the run actually demanded, and how
    often two positions were open at once. All of this was computed by the sizing
    walk and stamped per trade, then never displayed anywhere."""
    lev = [r.leverage_at_entry for r in trades if r.leverage_at_entry is not None]
    marg = [r.margin_used for r in trades if r.margin_used is not None]
    conc = [r.open_positions_at_entry or 0 for r in trades]
    basis: dict[str, int] = {}
    for r in trades:
        if r.size_basis:
            basis[r.size_basis] = basis.get(r.size_basis, 0) + 1
    out = {
        "peak_leverage": round(max(lev), 2) if lev else None,
        "median_leverage": round(median(lev), 2) if lev else None,
        "avg_leverage": round(sum(lev) / len(lev), 2) if lev else None,
        "trades_above_1x": sum(1 for x in lev if x > 1.0),
        "concurrent_entries": sum(1 for c in conc if c > 0),
        "max_concurrent_positions": (max(conc) + 1) if conc else 0,
        "margin_funded_entries": sum(1 for m in marg if m > 0),
        "margin_used_total": round(sum(marg), 2) if marg else 0.0,
        "margin_used_max": round(max(marg), 2) if marg else 0.0,
        "size_basis_counts": basis,
    }
    return out


def exit_breakdown(trades: list[TradeResult]) -> dict[str, int]:
    """Every trade's exit path, across BOTH strategies. The old
    exit_trigger_counts only covered rows that set exit_trigger (the wedge), so
    it summed to 88 of 129 and read as if it described the whole run."""
    out: dict[str, int] = {}
    for r in trades:
        key = r.exit_trigger or ("session_end" if r.result == "EXIT_AT_OPEN" else
                                 ("target" if (r.pnl_per_share or 0) > 0 else "stop"))
        out[key] = out.get(key, 0) + 1
    return out


def strategy_correlation(trades: list[TradeResult]) -> dict:
    """Daily-R correlation between the concurrently-running strategies -- the
    number that says whether running both actually diversifies. Requires >=3
    overlapping days to mean anything."""
    rows = [(r.date, r.entry_mode, r.r_multiple) for r in trades
            if r.r_multiple is not None and r.entry_mode]
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["date", "mode", "r"])
    modes = sorted(df["mode"].unique())
    if len(modes) < 2:
        return {"strategies": modes}
    piv = df.pivot_table(index="date", columns="mode", values="r", aggfunc="sum")
    both = piv.dropna()
    out = {"strategies": modes, "days_total": int(piv.shape[0]), "days_both": int(both.shape[0])}
    if both.shape[0] >= 3:
        out["daily_r_correlation"] = round(float(both.corr().iloc[0, 1]), 3)
    daily = df.groupby("date")["r"].sum()
    out.update({
        "daily_r_mean": round(float(daily.mean()), 3),
        "daily_r_sd": round(float(daily.std()), 3) if len(daily) > 1 else None,
        "worst_day_r": round(float(daily.min()), 2),
        "best_day_r": round(float(daily.max()), 2),
        "trading_days": int(daily.shape[0]),
        "max_trades_in_a_day": int(df.groupby("date").size().max()),
    })
    return out


def build(results: list[TradeResult], starting_equity: float) -> dict:
    """The whole quant block for one run: per-strategy cohorts, significance,
    realized risk, leverage/concurrency, exits, and strategy interaction."""
    trades = _executed(results)
    if not trades:
        return {"trades": 0}
    by_mode: dict[str, list[TradeResult]] = {}
    for r in trades:
        by_mode.setdefault(r.entry_mode or "unknown", []).append(r)
    per_strategy = {m: cohort_stats(g) for m, g in sorted(by_mode.items())}
    for m, g in by_mode.items():
        per_strategy[m]["significance"] = significance(g)
        per_strategy[m]["exits"] = exit_breakdown(g)
        per_strategy[m]["label"] = STRATEGY_LABELS.get(m, m)
        longs = [r for r in g if r.direction != "short"]
        shorts = [r for r in g if r.direction == "short"]
        if shorts:
            per_strategy[m]["by_direction"] = {
                "long": cohort_stats(longs) if longs else {"n": 0},
                "short": cohort_stats(shorts),
            }
    return {
        "trades": len(trades),
        "combined": cohort_stats(trades),
        "per_strategy": per_strategy,
        "significance": significance(trades),
        "risk": equity_risk(trades, starting_equity),
        "leverage": leverage_profile(trades),
        "exits": exit_breakdown(trades),
        "interaction": strategy_correlation(trades),
    }
