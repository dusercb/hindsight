from __future__ import annotations

import html
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable

import matplotlib
# Headless backend: charts are only saved to PNG, and the default Tk backend is
# broken on some Windows Python installs (TclError: can't find init.tcl).
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .analytics import build_analytics
from .models import Result, TradeResult
from .report_build import build_page
from .strategy import add_indicators, build_1m_bars


def jsonify_cells(rows: list[dict]) -> list[dict]:
    """SQLite/CSV cells must be scalars: nested structures (e.g. exit_legs, a list
    of per-leg dicts) are stored as JSON text -- sqlite3 can't bind a Python list
    at all, and pandas' CSV writer would emit a non-parseable repr."""
    return [
        {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        for row in rows
    ]


def write_sqlite(results: Iterable[TradeResult], path: Path) -> None:
    rows = jsonify_cells([r.to_dict() for r in results])
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        pd.DataFrame(rows).to_sql("trades", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker_date ON trades(ticker, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_float ON trades(float_shares)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_price ON trades(stock_price)")


TRADE_CHARTS_DIRNAME = "trade_charts"


def trade_folder_name(ticker: str, date: str, setup_index: int = 1, entry_mode: str | None = None) -> str:
    """Folder-name slug for a trade's chart/json/bars.csv. Disambiguated by
    setup_index only when > 1 (the 2nd/3rd/... move of one ticker/day), so a
    single-trade day keeps the plain TICKER_DATE name every existing run used.
    Wedge trades (entry_mode "wedge_break") get a "_wedge" suffix: under
    wedge.concurrent_with_dip BOTH strategies can trade the same ticker/day, and
    without the suffix their folders would collide."""
    base = f"{ticker}_{date}".replace("/", "-")
    if entry_mode == "wedge_break":
        base = f"{base}_wedge"
    return base if setup_index <= 1 else f"{base}_move{setup_index}"


# Result classes for reporting.trade_folders_for: which candidates earn a trade
# folder (json + bars + chart). Every candidate keeps its full CSV/SQLite row
# regardless -- this only gates the expensive per-candidate artifacts (a measured
# run spent ~24 minutes rendering 1,613 charts for 24 actual trades).
_FOLDER_TRADES = {Result.WIN.value, Result.LOSS.value, Result.EXIT_AT_OPEN.value}
_FOLDER_SETUPS = _FOLDER_TRADES | {Result.STALE_HOD.value, Result.INVALID_RETRACE.value, Result.NO_TRIGGER.value}


def wants_trade_folder(result: TradeResult, mode: str) -> bool:
    """Should this candidate get a trade folder under reporting.trade_folders_for
    = `mode`? "trades" = filled entries only; "setups" adds formed-but-untraded
    setups (the charts that diagnose WHY a setup was rejected); "all" = everyone.
    A filled entry is always included whatever its result label says."""
    if mode == "all":
        return True
    if result.entry_price is not None:
        return True
    allowed = _FOLDER_TRADES if mode == "trades" else _FOLDER_SETUPS
    return result.result in allowed


def write_trade_folder(result: TradeResult, bars: pd.DataFrame, root: Path, write_chart: bool = True) -> None:
    safe = trade_folder_name(result.ticker, result.date, result.setup_index, result.entry_mode)
    d = root / "trade_folders" / safe
    d.mkdir(parents=True, exist_ok=True)
    (d / "trade.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if bars is not None and not bars.empty:
        bars.to_csv(d / "bars.csv", index=False)
    if write_chart and bars is not None and not bars.empty:
        try:
            chart_path = d / "chart.png"
            _write_chart(result, bars, chart_path)
            # Per-trade folders nest one chart per ticker/date under its own
            # subfolder, which is great for the HTML report's links but tedious
            # to flip through in a file browser. Mirror every chart into one flat
            # folder (unique names via trade_folder_name) so a whole session's
            # trades can be scrubbed as a single thumbnail strip.
            charts_dir = root / TRADE_CHARTS_DIRNAME
            charts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chart_path, charts_dir / f"{safe}.png")
            # WIN/LOSS are the only two outcomes worth flipping through visually
            # after the fact (everything else never reached a trade) -- split
            # those into their own subfolders so "show me the losers" doesn't
            # require cross-referencing the CSV first.
            if result.result in (Result.WIN.value, Result.LOSS.value):
                subdir = charts_dir / ("wins" if result.result == Result.WIN.value else "losses")
                subdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(chart_path, subdir / f"{safe}.png")
        except Exception as e:
            (d / "chart_error.txt").write_text(repr(e), encoding="utf-8")


def write_html_report(results: list[TradeResult], summary: dict, run_dir: Path, stamp: str, config_snapshot: dict | None = None, monte_carlo=None) -> Path:
    """Create a self-contained-ish HTML report plus chart assets.

    The report links to per-trade folders generated by write_trade_folder().
    It is intentionally dependency-light: matplotlib creates PNG charts, and the
    page is plain HTML/CSS so it opens locally in any browser. When
    `config_snapshot` is provided (a config_to_dict() dump), the report includes
    the exact strategy/risk values the run executed with, and the disqualification
    charts draw the configured thresholds as reference lines. When `monte_carlo`
    (a montecarlo.MonteCarloResult) is provided, a Monte Carlo section with its
    own stat cards and fan/histogram chart is rendered.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    assets = run_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in results])
    strategy_cfg = (config_snapshot or {}).get("strategy", {})
    chart_files = _write_report_charts(df, assets, strategy_cfg)
    mc_chart_rel = None
    if monte_carlo is not None:
        try:
            mc_chart_rel = _write_monte_carlo_chart(monte_carlo.artifacts, assets)
        except Exception as e:
            (assets / "monte_carlo_error.txt").write_text(repr(e), encoding="utf-8")
    # In-sample optimisation/edge analytics over the executed trades (offset grid,
    # tightening sweeps, temporal edges). Best-effort: a failure must never lose
    # the whole report.
    analytics = None
    try:
        pct = strategy_cfg.get("offset_mode") == "percent"
        cur_t = strategy_cfg.get("target_offset_pct" if pct else "target_offset", 0.0)
        cur_s = strategy_cfg.get("stop_offset_pct" if pct else "stop_offset", 0.0)
        analytics = build_analytics([r.to_dict() for r in results], strategy_cfg.get("offset_mode", "cents"),
                                    float(cur_t or 0.0), float(cur_s or 0.0),
                                    exit_mode=strategy_cfg.get("exit_mode", "standard"),
                                    stop_mode=strategy_cfg.get("stop_mode", "offset"),
                                    target_mode=strategy_cfg.get("target_mode", "offset"))
    except Exception as e:
        (assets / "analytics_error.txt").write_text(repr(e), encoding="utf-8")
    # The MAIN report (rebuilt 2026-07-23): interactive, self-contained, built
    # off summary["quant"] -- per-strategy cohorts, capital/leverage, realized
    # risk, hold times, exit-at-open tracking. See report_build.py.
    html_path = run_dir / f"report_{stamp}.html"
    html_path.write_text(build_page(summary, df, config_snapshot, stamp, monte_carlo), encoding="utf-8")
    # The previous matplotlib report is still written alongside it: it carries
    # the offset-grid / gate-sweep / temporal analytics and Monte-Carlo fan that
    # the new page doesn't reproduce yet, and it costs nothing to keep.
    try:
        legacy = run_dir / f"report_{stamp}_legacy.html"
        legacy.write_text(_render_html(df, summary, chart_files, run_dir, config_snapshot,
                                       monte_carlo, mc_chart_rel, analytics), encoding="utf-8")
    except Exception as e:
        (assets / "legacy_report_error.txt").write_text(repr(e), encoding="utf-8")
    return html_path


def _result_detail(row) -> str:
    """Granular result label: splits DID_NOT_QUALIFY by which gate rejected the
    candidate, and INSUFFICIENT_DATA by which data was missing -- one opaque
    bucket per result code hides exactly the things worth tuning."""
    result = row.get("result") or "UNKNOWN"
    if result == "DID_NOT_QUALIFY":
        return {
            "premarket_volume": "DNQ: premarket volume below min",
            "rvol": "DNQ: RVOL never reached min",
            "gain_pct": "DNQ: gain% below min",
        }.get(row.get("disqualify_reason"), "DNQ: other")
    if result == "INSUFFICIENT_DATA":
        return "NO DATA: RVOL baseline" if "RVOL baseline" in (row.get("reason") or "") else "NO DATA: bars"
    return result


def _write_report_charts(df: pd.DataFrame, assets: Path, strategy_cfg: dict | None = None) -> dict[str, str]:
    charts: dict[str, str] = {}
    strategy_cfg = strategy_cfg or {}
    if df.empty:
        return charts
    try:
        import matplotlib.pyplot as plt

        counts = df.apply(_result_detail, axis=1).value_counts()
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        colors = ["#dc2626" if str(lbl).startswith("DNQ") else "#0891b2" for lbl in counts.index]
        counts.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Result counts (disqualifications split by cause)")
        ax.set_xlabel("Result")
        ax.set_ylabel("Count")
        fig.tight_layout()
        out = assets / "result_counts.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        charts["Result counts"] = "report_assets/result_counts.png"

        # Near-miss views: how close did each disqualified candidate get to the
        # threshold that rejected it? The mass just under the red line is what a
        # small config loosening would unlock -- the actionable part of DNQ.
        def _near_miss_hist(values, threshold, title, xlabel, fname, chart_key):
            values = values.dropna()
            if values.empty:
                return
            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            values.plot(kind="hist", bins=25, ax=ax, color="#9ca3af", edgecolor="white")
            if threshold is not None:
                ax.axvline(float(threshold), color="#dc2626", linestyle="--", linewidth=1.5,
                           label=f"configured min ({threshold})")
                ax.legend()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            fig.tight_layout()
            fig.savefig(assets / fname, dpi=140)
            plt.close(fig)
            charts[chart_key] = f"report_assets/{fname}"

        if "disqualify_reason" in df.columns:
            dnq_vol = df[df["disqualify_reason"] == "premarket_volume"]
            _near_miss_hist(
                dnq_vol.get("premarket_volume", pd.Series(dtype=float)),
                strategy_cfg.get("min_premarket_volume"),
                f"Disqualified by premarket volume ({len(dnq_vol)}): each one's actual session volume",
                "Premarket session volume, shares",
                "dnq_premarket_volume_near_miss.png",
                "DNQ near-misses: premarket volume",
            )
            dnq_rvol = df[df["disqualify_reason"] == "rvol"]
            _near_miss_hist(
                dnq_rvol.get("max_rvol_seen", pd.Series(dtype=float)),
                strategy_cfg.get("min_rvol"),
                f"Disqualified by RVOL ({len(dnq_rvol)}): the peak RVOL each one DID reach",
                "Peak session RVOL (x baseline daily volume)",
                "dnq_rvol_near_miss.png",
                "DNQ near-misses: RVOL",
            )
            dnq_gain = df[df["disqualify_reason"] == "gain_pct"]
            _near_miss_hist(
                dnq_gain.get("premarket_gain_pct", pd.Series(dtype=float)),
                strategy_cfg.get("min_premarket_gain_pct"),
                f"Disqualified by gain% ({len(dnq_gain)}): the max gain each one DID reach",
                "Max premarket gain from open (%)",
                "dnq_gain_near_miss.png",
                "DNQ near-misses: gain%",
            )

        trades = df[df["entry_price"].notna()].copy()
        if not trades.empty:
            trades["cum_r"] = trades["r_multiple"].fillna(0).cumsum()
            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            ax.plot(range(1, len(trades) + 1), trades["cum_r"], linewidth=1.5)
            ax.set_title("Cumulative R")
            ax.set_xlabel("Trade number")
            ax.set_ylabel("Cumulative R")
            fig.tight_layout()
            out = assets / "cumulative_r.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            charts["Cumulative R"] = "report_assets/cumulative_r.png"

            if trades["pullback_retrace_pct"].notna().any():
                fig = plt.figure(figsize=(9, 5))
                ax = fig.add_subplot(111)
                trades["pullback_retrace_pct"].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Pullback retracement distribution")
                ax.set_xlabel("Pullback retracement %")
                fig.tight_layout()
                out = assets / "pullback_distribution.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                charts["Pullback distribution"] = "report_assets/pullback_distribution.png"

            if trades["stock_price"].notna().any():
                fig = plt.figure(figsize=(9, 5))
                ax = fig.add_subplot(111)
                trades["stock_price"].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Stock price distribution")
                ax.set_xlabel("Stock price / previous close")
                fig.tight_layout()
                out = assets / "price_distribution.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                charts["Price distribution"] = "report_assets/price_distribution.png"

            if trades["float_shares"].notna().any():
                fig = plt.figure(figsize=(9, 5))
                ax = fig.add_subplot(111)
                (trades["float_shares"].dropna() / 1_000_000).plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Float distribution")
                ax.set_xlabel("Float, millions of shares")
                fig.tight_layout()
                out = assets / "float_distribution.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                charts["Float distribution"] = "report_assets/float_distribution.png"

            # Did the scale-out ladder actually help, or just add legs to trades
            # that were already winning? Win rate and REALIZED R-multiple
            # (weighted-exit P&L / stop_offset -- the actualized risk:reward after
            # every leg, not the configured target), scaled vs non-scaled.
            scaled = trades[trades["scale_outs"] > 0] if "scale_outs" in trades.columns else trades.iloc[0:0]
            non_scaled = trades[trades["scale_outs"] == 0] if "scale_outs" in trades.columns else trades.iloc[0:0]
            cohorts = [(label, c) for label, c in (("Scaled", scaled), ("Non-scaled", non_scaled)) if not c.empty]
            if len(cohorts) >= 1 and not scaled.empty:
                labels = [label for label, _ in cohorts]
                colors = ["#0891b2", "#9ca3af"][: len(cohorts)]
                win_rates = [(c["result"] == "WIN").mean() * 100.0 for _, c in cohorts]
                counts = [len(c) for _, c in cohorts]
                avg_r = [c["r_multiple"].dropna().mean() if c["r_multiple"].notna().any() else 0.0 for _, c in cohorts]

                fig, (axw, axr) = plt.subplots(1, 2, figsize=(11, 4.8))
                axw.bar(labels, win_rates, color=colors)
                for x, (wr, n) in enumerate(zip(win_rates, counts)):
                    axw.annotate(f"n={n}", (x, wr), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color="#374151")
                axw.set_ylabel("Win rate %")
                axw.set_title("Win rate: scaled vs non-scaled")
                axw.set_ylim(0, 115)  # win rate is bounded at 100%; fixed headroom for the n= label

                axr.bar(labels, avg_r, color=colors)
                axr.axhline(0, color="#374151", linewidth=0.6)
                target_offset, stop_offset = strategy_cfg.get("target_offset"), strategy_cfg.get("stop_offset")
                if target_offset and stop_offset:
                    configured_r = target_offset / stop_offset
                    axr.axhline(configured_r, color="#dc2626", linestyle="--", linewidth=1.3, label=f"configured R ({configured_r:.2f})")
                    axr.legend(fontsize=8)
                axr.set_ylabel("Avg realized R-multiple")
                axr.set_title("Actualized risk:reward: scaled vs non-scaled")

                fig.suptitle("Scale-out ladder performance", fontsize=11)
                fig.tight_layout(rect=[0, 0, 1, 0.94])
                out = assets / "scale_out_performance.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                charts["Scale-out ladder performance"] = "report_assets/scale_out_performance.png"

            # Win rate by entry hour (ET). A coarse session-timing view -- small-N
            # hours are labeled with their own count so a single lucky/unlucky
            # trade in a thin hour doesn't read as a strong signal.
            if "entry_time" in trades.columns:
                entry_hour = pd.to_datetime(trades["entry_time"], errors="coerce", utc=False).apply(
                    lambda t: t.hour if pd.notna(t) else None
                )
                hourly_df = trades.assign(_hour=entry_hour).dropna(subset=["_hour"])
                if not hourly_df.empty:
                    grouped = hourly_df.groupby("_hour").agg(
                        n=("result", "size"), win_rate=("result", lambda s: (s == "WIN").mean() * 100.0),
                    ).sort_index()
                    fig, (axh1, axh2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
                    bar_colors = ["#16a34a" if wr >= 50 else "#dc2626" for wr in grouped["win_rate"]]
                    x_labels = [f"{int(h):02d}:00" for h in grouped.index]
                    axh1.bar(x_labels, grouped["win_rate"], color=bar_colors)
                    axh1.axhline(50, color="#9ca3af", linestyle=":", linewidth=1)
                    for x, (wr, n) in enumerate(zip(grouped["win_rate"], grouped["n"])):
                        axh1.annotate(f"n={int(n)}", (x, wr), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color="#374151")
                    axh1.set_ylabel("Win rate %")
                    axh1.set_title("Win rate by entry hour (ET)")
                    axh2.bar(x_labels, grouped["n"], color="#9ca3af")
                    axh2.set_ylabel("Trades")
                    axh2.set_xlabel("Entry hour (ET)")
                    fig.tight_layout()
                    out = assets / "hourly_win_rate.png"
                    fig.savefig(out, dpi=140)
                    plt.close(fig)
                    charts["Win rate by entry hour"] = "report_assets/hourly_win_rate.png"
    except Exception as e:
        (assets / "chart_error.txt").write_text(repr(e), encoding="utf-8")
    return charts


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if pd.isna(v):
            return "—"
        return f"{v:,.4g}"
    return html.escape(str(v))


_METRIC_LABELS = {
    "candidates": "Candidates",
    "qualified_or_traded": "Qualified Or Traded",
    "trades_taken": "Trades Taken",
    "win_rate_pct": "Win Rate",
    "risk_reward_configured": "Configured Risk:Reward",
    "realized_avg_reward_to_risk": "Realized Avg Reward:Risk",
    "avg_r_multiple": "Avg R-Multiple (Realized)",
    "expected_value_r": "Expected Value (R, Theoretical)",
    "expected_value_per_share": "Expected Value ($/Share, Theoretical)",
    "profit_factor": "Profit Factor",
    "net_pnl": "Net P&L ($)",
    "net_pnl_pct": "Net P&L (% of Equity)",
    "avg_pnl_per_trade": "Avg P&L / Trade ($)",
    "expectancy_per_share": "Expectancy ($/Share, Realized)",
    "expectancy_pct": "Expectancy (%/Trade, Realized)",
    "path_ambiguous_bars": "Path-Ambiguous Bars",
    "ambiguous_resolved_by_ticks": "Ambiguous Resolved By Ticks",
    "largest_win_streak": "Largest Win Streak",
    "largest_losing_streak": "Largest Losing Streak",
    "trades_with_scale_outs": "Trades That Scaled Out",
    "scaled_trades_win_rate_pct": "Scaled Trades: Win Rate",
    "scaled_trades_avg_r_multiple": "Scaled Trades: Avg R-Multiple",
    "non_scaled_trades_win_rate_pct": "Non-Scaled Trades: Win Rate",
    "non_scaled_trades_avg_r_multiple": "Non-Scaled Trades: Avg R-Multiple",
    "insufficient_rvol_history": "Insufficient Daily-Volume History",
    "did_not_qualify_premarket_volume": "Disqualified: Premarket Volume Below Min",
    "did_not_qualify_rvol": "Disqualified: RVOL Never Reached Min",
    "did_not_qualify_gain_pct": "Disqualified: Gain% Below Min",
    "trigger_touches_total": "Trigger Price Touches",
    "skipped_no_macd_bar_total": "Skipped: No MACD Bar",
    "skipped_rvol_not_yet_confirmed_total": "Skipped: RVOL Not Yet Confirmed",
    "skipped_premarket_volume_not_yet_confirmed_total": "Skipped: Premarket Volume Not Yet Confirmed",
    "skipped_red_volume_dominant_total": "Skipped: Red Volume Dominant",
    "skipped_macd_not_bullish_total": "Skipped: MACD Not Bullish",
    "skipped_fill_exceeds_limit_total": "Skipped: Fill Exceeds Limit",
    "skipped_spread_too_wide_total": "Skipped: Spread Too Wide",
    "skipped_outside_entry_window_total": "Skipped: Outside Entry Window",
    "skipped_retrace_below_ema9_total": "Skipped: Retrace Below EMA9",
    "skipped_retrace_below_ema20_total": "Skipped: Retrace Below EMA20",
    "skipped_retrace_below_vwap_total": "Skipped: Retrace Below VWAP",
    "skipped_buy_below_ema9_total": "Skipped: Buy Below EMA9",
    "skipped_buy_below_ema20_total": "Skipped: Buy Below EMA20",
    "skipped_buy_below_vwap_total": "Skipped: Buy Below VWAP",
    "skipped_risk_too_wide_total": "Skipped: Risk Too Wide",
    "skipped_continuation_vwap_total": "Skipped: Continuation VWAP Not Extended",
    "skipped_no_red_dip_candle_total": "Skipped: No Red Dip Candle",
    "invalid_retrace_attempts_total": "Invalid Retrace Attempts",
    "stale_hod_breaks_total": "Stale HOD Breaks",
}

_METRIC_TOOLTIPS = {
    "trades_with_scale_outs": "Trades that banked at least one scale-out sell: part sold at the planned exit, stop ratcheted to scale_out_stop_offset BELOW that exit, target raised by scale_out_step -- for at most scale_out_max_sells sells, the last one liquidating the remainder. A scaled trade's WIN/LOSS is the sign of its net P&L across all legs.",
    "scaled_trades_win_rate_pct": "Win rate among trades that banked at least one scale-out leg -- compare to Non-Scaled Trades: Win Rate to see whether the ladder correlates with better outcomes or just rides trades that were already winning.",
    "scaled_trades_avg_r_multiple": "Average REALIZED R-multiple (weighted-exit P&L ÷ stop_offset) among scaled trades -- the actualized risk:reward AFTER scaling, not the configured target. Compare to Configured Risk:Reward and to Non-Scaled Trades: Avg R-Multiple.",
    "non_scaled_trades_win_rate_pct": "Win rate among trades that exited in one piece (stopped, hit target once, or exited at the open) without ever scaling out.",
    "non_scaled_trades_avg_r_multiple": "Average realized R-multiple among trades that never scaled out -- the single-exit baseline to compare the ladder against.",
    "candidates": "Total ticker/date rows loaded from the input CSV.",
    "qualified_or_traded": "Candidates that weren't rejected outright by data availability, the price filter, or an error -- i.e. reached RVOL/gain%/pullback evaluation.",
    "trades_taken": "Candidates where an entry actually filled (WIN, LOSS, or exited at the regular open).",
    "win_rate_pct": "Wins ÷ trades taken. Does not exclude trades that exited at the regular open without hitting target or stop.",
    "risk_reward_configured": "The configured per-share reward (target_offset) and risk (stop_offset) from config.yaml, before fees or slippage.",
    "realized_avg_reward_to_risk": "Average winning trade's $/share ÷ average losing trade's $/share, from what actually happened (fills, fees, ambiguous-bar resolution) -- compare to Configured Risk:Reward to see how much real execution differs from the ideal.",
    "avg_r_multiple": "Average of each trade's actual P&L ÷ stop_offset. The realized, empirical counterpart to Expected Value (R) below.",
    "expected_value_r": "Theoretical expectancy in R-multiples: win rate × (target_offset ÷ stop_offset) − loss rate × 1, assuming every non-win is a full stop-out. Forward-looking -- uses the configured risk:reward and the win rate observed here, not what actually happened to each trade.",
    "expected_value_per_share": "Same theoretical expectancy as Expected Value (R), expressed in $/share: win rate × target_offset − loss rate × stop_offset. Compare to Expectancy ($/Share, Realized) to see the gap between theory and actual fills.",
    "profit_factor": "Gross profit ÷ gross loss (dollar totals across all trades). Above 1.0 means winners outweighed losers in aggregate.",
    "net_pnl": "Sum of P&L across all trades taken, in dollars, at the configured share size.",
    "net_pnl_pct": "Net P&L as a percentage of starting equity.",
    "avg_pnl_per_trade": "Net P&L ÷ trades taken, in dollars.",
    "expectancy_per_share": "Average actual P&L per share across all trades taken -- what really happened, including fees and imperfect fills.",
    "expectancy_pct": "Average actual per-trade return as a percentage of entry price.",
    "path_ambiguous_bars": "Trades where a single bar touched both stop and target, so the exit needed the intrabar_policy (ticks/heuristic/conservative) to resolve.",
    "ambiguous_resolved_by_ticks": "Of the path-ambiguous bars above, how many were resolved using real IBKR tick data rather than a bar-based fallback.",
    "largest_win_streak": "Longest consecutive run of WIN results, in candidate order.",
    "largest_losing_streak": "Longest consecutive run of LOSS results, in candidate order.",
    "insufficient_rvol_history": "Candidates with too little daily-volume history to compute an RVOL baseline at all (e.g. a recent IPO) -- not disqualified, just unresolvable.",
    "did_not_qualify_premarket_volume": "Candidates whose own premarket session volume never reached min_premarket_volume even by session close -- an absolute floor, independent of RVOL/the daily-average baseline.",
    "did_not_qualify_rvol": "Candidates where cumulative premarket volume never reached min_rvol × this ticker's baseline daily volume (rvol_baseline_stat over the trailing lookback days), anywhere in the session.",
    "did_not_qualify_gain_pct": "Candidates that cleared RVOL but whose premarket high never reached min_premarket_gain_pct above the premarket open (independent of RVOL timing).",
    "trigger_touches_total": "Total bars, across all candidates, where price reached the trigger price at all.",
    "skipped_no_macd_bar_total": "Of those trigger touches, how many were skipped because no completed 1-minute MACD bar existed for the prior minute.",
    "skipped_rvol_not_yet_confirmed_total": "Of those trigger touches, how many were skipped because cumulative volume hadn't ACTUALLY reached min_rvol as of that bar yet -- RVOL crossing later that session doesn't retroactively validate an earlier entry (no lookahead).",
    "skipped_premarket_volume_not_yet_confirmed_total": "Of those trigger touches, how many were skipped because cumulative premarket volume hadn't ACTUALLY reached min_premarket_volume as of that bar yet (same causal, no-lookahead check as RVOL, against an absolute floor instead of a relative one).",
    "skipped_red_volume_dominant_total": "Of those trigger touches, how many were skipped because the largest red candle's volume met or exceeded the largest green candle's, in this breakout's bars.",
    "skipped_macd_not_bullish_total": "Of those trigger touches, how many were skipped because MACD wasn't both above zero and above its signal line.",
    "skipped_fill_exceeds_limit_total": "Of those trigger touches, how many were skipped because the fill was further above the trigger than entry_limit_slippage_cents ($/share) allows.",
    "skipped_spread_too_wide_total": "Of those trigger touches, how many were skipped because the live bid/ask spread exceeded max_spread_cents.",
    "skipped_outside_entry_window_total": "Of those trigger touches, how many fell outside [earliest_entry_time, latest_entry_time] -- a later touch of the same setup inside the window can still fire.",
    "skipped_retrace_below_ema9_total": "Of those trigger touches, how many had a retrace (HOD through that touch) that traded below the 1-minute EMA9 at some point. Only counted when block_retrace_below_ema9 is enabled.",
    "skipped_retrace_below_ema20_total": "Of those trigger touches, how many had a retrace that traded below the 1-minute EMA20 at some point. Only counted when block_retrace_below_ema20 is enabled.",
    "skipped_retrace_below_vwap_total": "Of those trigger touches, how many had a retrace that traded below session VWAP at some point. Only counted when block_retrace_below_vwap is enabled.",
    "skipped_buy_below_ema9_total": "Of those trigger touches, how many had the trigger price ITSELF below the 1-minute EMA9 (as of the prior completed minute), regardless of the retrace path. Only counted when block_buy_below_ema9 is enabled.",
    "skipped_buy_below_ema20_total": "Of those trigger touches, how many had the trigger price itself below the 1-minute EMA20. Only counted when block_buy_below_ema20 is enabled.",
    "skipped_buy_below_vwap_total": "Of those trigger touches, how many had the trigger price itself below session VWAP. Only counted when block_buy_below_vwap is enabled.",
    "skipped_risk_too_wide_total": "Of those trigger touches, how many were rejected because the entry-to-stop distance (fill - stop, as a % of the fill) exceeded max_risk_pct -- the structural stop sat too far below the entry. Only counted when max_risk_pct is set.",
    "skipped_continuation_vwap_total": "Win-continuation breaks (trade_winners_qty) rejected because the session VWAP hadn't lifted continuation_vwap_above_entry_pct above the prior winning trade's entry yet -- the volume-confirmation gate for pressing winners. VWAP is volume-weighted, so it only climbs that far when heavy volume keeps printing at higher prices.",
    "skipped_no_red_dip_candle_total": "Of those trigger touches, how many had a dip (HOD through the pullback low) with no red 10-second candle. Only counted when require_red_dip_candle is enabled.",
    "invalid_retrace_attempts_total": "Total times a pullback breached max_pullback_retrace_pct. Under first_setup_only: true, the first one ends the candidate; under false, the scan keeps looking for a fresh setup.",
    "stale_hod_breaks_total": "Total times the armed HOD was broken by a trigger touch that failed an entry gate (RVOL/volume/red/MACD/indicator/window/spread/fill/red-dip). Under first_setup_only: true each ends the candidate as STALE_HOD; under false the scan resets to hunt a fresh setup. This is the count of first-pullback breakouts that fired without a clean entry.",
}

_MONEY_KEYS = {"net_pnl", "ending_equity", "avg_pnl_per_trade"}
_PER_SHARE_MONEY_KEYS = {"expectancy_per_share", "expected_value_per_share", "scaled_trades_avg_pnl_per_share", "non_scaled_trades_avg_pnl_per_share"}
_R_MULTIPLE_KEYS = {"avg_r_multiple", "expected_value_r", "scaled_trades_avg_r_multiple", "non_scaled_trades_avg_r_multiple"}


def _fmt_metric(key: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if key == "realized_avg_reward_to_risk" and isinstance(v, (int, float)):
        return f"{v:.2f} : 1"
    if key in _R_MULTIPLE_KEYS and isinstance(v, (int, float)):
        return f"{v:+.3f}R"
    if key in _PER_SHARE_MONEY_KEYS and isinstance(v, (int, float)):
        return f"{v:+,.4f}/sh"
    if isinstance(v, (int, float)) and key.endswith("_pct"):
        return f"{v:+,.2f}%" if "pnl" in key or "expectancy" in key else f"{v:,.2f}%"
    if isinstance(v, (int, float)) and key in _MONEY_KEYS:
        return f"${v:,.2f}"
    return _fmt(v)


# Metric cards grouped into meaningful sections instead of one flat wall of 45+
# tiles -- each group answers a different question (did it make money? is the edge
# real? why didn't candidates trade?), which is far easier to scan.
_METRIC_SECTIONS: list[tuple[str, list[str]]] = [
    ("Headline", [
        "candidates", "qualified_or_traded", "trades_taken", "wins", "losses",
        "win_rate_pct", "net_pnl", "net_pnl_pct", "profit_factor",
    ]),
    ("Edge & Expectancy", [
        "risk_reward_configured", "realized_avg_reward_to_risk", "avg_r_multiple",
        "expected_value_r", "expected_value_per_share", "expectancy_per_share",
        "expectancy_pct", "avg_pnl_per_trade", "largest_win_streak", "largest_losing_streak",
    ]),
    ("Scale-Outs", [
        "trades_with_scale_outs", "scaled_trades_win_rate_pct", "scaled_trades_avg_r_multiple",
        "non_scaled_trades_win_rate_pct", "non_scaled_trades_avg_r_multiple",
    ]),
    ("Execution Quality", ["path_ambiguous_bars", "ambiguous_resolved_by_ticks"]),
    ("Funnel — Why Candidates Didn't Trade", [
        "insufficient_rvol_history", "did_not_qualify_premarket_volume",
        "did_not_qualify_rvol", "did_not_qualify_gain_pct",
    ]),
    ("Gate Diagnostics — Trigger Touches", [
        "trigger_touches_total", "skipped_no_macd_bar_total", "skipped_rvol_not_yet_confirmed_total",
        "skipped_premarket_volume_not_yet_confirmed_total", "skipped_red_volume_dominant_total",
        "skipped_no_red_dip_candle_total", "skipped_macd_not_bullish_total", "skipped_fill_exceeds_limit_total",
        "skipped_spread_too_wide_total", "skipped_outside_entry_window_total",
        "skipped_retrace_below_ema9_total", "skipped_retrace_below_ema20_total", "skipped_retrace_below_vwap_total",
        "skipped_buy_below_ema9_total", "skipped_buy_below_ema20_total", "skipped_buy_below_vwap_total",
        "skipped_risk_too_wide_total", "skipped_continuation_vwap_total",
        "invalid_retrace_attempts_total", "stale_hod_breaks_total",
    ]),
]

# Keys whose card should be tinted by the sign of the value (green positive, red
# negative) -- the P&L / expectancy / R figures where sign is the headline.
_SIGNED_KEYS = {
    "net_pnl", "net_pnl_pct", "avg_pnl_per_trade", "avg_r_multiple", "expected_value_r",
    "expected_value_per_share", "expectancy_per_share", "expectancy_pct",
}


def _metric_card(k: str, summary: dict) -> str:
    label = _METRIC_LABELS.get(k, k.replace("_", " ").title())
    tooltip = _METRIC_TOOLTIPS.get(k, "")
    title_attr = f" title='{html.escape(tooltip)}'" if tooltip else ""
    v = summary.get(k)
    sign_cls = ""
    if k in _SIGNED_KEYS and isinstance(v, (int, float)) and not pd.isna(v):
        sign_cls = " pos" if v > 0 else (" neg" if v < 0 else "")
    return (
        f"<div class='card{sign_cls}'{title_attr}><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{_fmt_metric(k, v)}</div></div>"
    )


def _metric_cards(summary: dict) -> str:
    blocks = []
    for title, keys in _METRIC_SECTIONS:
        cards = "\n".join(_metric_card(k, summary) for k in keys)
        blocks.append(f"<h3 class='sec'>{html.escape(title)}</h3>\n<div class='grid'>\n{cards}\n</div>")
    return "\n".join(blocks)


def _fingerprint(summary: dict, config_snapshot: dict | None) -> str:
    """A compact 'what did this run actually test' band of chips at the top of the
    report -- the handful of settings that most define the run (entry mode, how
    stops are measured, the price/float/gain/RVOL bands, entry window, and which
    optional quality gates are on). Saves scrolling into Configuration Used to
    answer 'wait, which variant produced these numbers?'."""
    s = (config_snapshot or {}).get("strategy", {}) or {}
    def g(k, default=None):
        return s.get(k, default)

    chips: list[tuple[str, str]] = []
    chips.append(("Entry", str(summary.get("entry_mode") or g("entry_mode", "hod_break"))))
    percent = (summary.get("offset_mode") or g("offset_mode")) == "percent"
    if (summary.get("target_mode") or g("target_mode")) == "r_multiple":
        tgt = f"+{g('target_r_multiple')}R"
    else:
        tgt = f"+{g('target_offset_pct')}%" if percent else f"+{g('target_offset')}"
    if (summary.get("stop_mode") or g("stop_mode")) == "pullback_low":
        stp = "pullback low"
    else:
        stp = f"-{g('stop_offset_pct')}%" if percent else f"-{g('stop_offset')}"
    chips.append(("Stops", f"{tgt} / {stp}"))
    exit_mode = summary.get("exit_mode") or g("exit_mode", "standard")
    if exit_mode == "candle_trail_1min":
        chips.append(("Exit", f"1m trail (fat≥{g('fatgreen_body_min')})"))
    tf = g("chart_timeframe", "10s")
    if tf not in ("10s", None, "null"):
        chips.append(("Chart", str(tf)))
    lo, hi = g("min_price"), g("max_price")
    chips.append(("Price", f"{'∅' if lo in (None, 'null') else lo}–{'∅' if hi in (None, 'null') else hi}"))
    if g("max_float") not in (None, "null") or g("min_float") not in (None, "null"):
        chips.append(("Float", f"{g('min_float') or '∅'}–{g('max_float') or '∅'}"))
    chips.append(("Gain ≥", f"{g('min_premarket_gain_pct')}%"))
    chips.append(("RVOL ≥", f"{g('min_rvol')}x"))
    if g("min_premarket_volume") not in (None, "null"):
        chips.append(("PM vol ≥", f"{int(float(g('min_premarket_volume'))):,}"))
    ew = g("earliest_entry_time"); lw = g("latest_entry_time")
    if ew not in (None, "null") or lw not in (None, "null"):
        chips.append(("Window", f"{ew or '∅'}–{lw or '∅'}"))
    chips.append(("First setup only", "yes" if g("first_setup_only") else "no"))

    # Optional quality gates that are ON -- shown only when enabled, so the band
    # stays about what's actually active this run.
    gates = []
    if g("require_red_dip_candle"): gates.append("red-dip")
    if g("block_buy_below_vwap"): gates.append("buy≥VWAP")
    if g("block_buy_below_ema9"): gates.append("buy≥EMA9")
    if g("block_buy_below_ema20"): gates.append("buy≥EMA20")
    if g("block_retrace_below_vwap"): gates.append("retrace≥VWAP")
    if g("block_retrace_below_ema9"): gates.append("retrace≥EMA9")
    if g("block_retrace_below_ema20"): gates.append("retrace≥EMA20")
    if g("red_volume_check"): gates.append(f"red-vol:{g('red_volume_check')}")

    chip_html = "".join(
        f"<span class='chip'><span class='ck'>{html.escape(k)}</span><span class='cv'>{html.escape(str(v))}</span></span>"
        for k, v in chips
    )
    gate_html = ("<span class='chip gates'><span class='ck'>Gates on</span><span class='cv'>"
                 + (html.escape(" · ".join(gates)) if gates else "none") + "</span></span>")
    return f"<div class='fingerprint'>{chip_html}{gate_html}</div>"


def _breakdown_tables(summary: dict) -> str:
    """Win-rate splits that were computed into the summary JSON but never surfaced
    in the HTML before: by entry-price band (the price/offset-mode interaction) and
    by entry hour (session timing)."""
    def table(title: str, data: dict, key_label: str) -> str:
        if not data:
            return ""
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{d.get('count', 0)}</td>"
            f"<td>{d.get('win_rate_pct', 0):.1f}%</td></tr>"
            for k, d in data.items()
        )
        return (
            f"<div class='bd'><h3>{html.escape(title)}</h3>"
            f"<table class='mini'><thead><tr><th>{html.escape(key_label)}</th><th>Trades</th><th>Win rate</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    price = table("Win Rate by Entry Price", summary.get("win_rate_by_price_bucket", {}), "Price band")
    hour = table("Win Rate by Entry Hour (ET)", {f"{int(h):02d}:00": d for h, d in (summary.get("win_rate_by_hour", {}) or {}).items()}, "Hour")
    if not (price or hour):
        return ""
    return f"<h2>Breakdowns</h2><p class='muted'>Small-N bands are noisy — read alongside the trade count.</p><div class='bd-row'>{price}{hour}</div>"


def _trail_section(summary: dict) -> str:
    """exit_mode 'candle_trail_1min' breakdown: how many trades became RUNNERS (a
    fat-green 1-minute candle dropped the 1:1 target and the trail took over) vs
    stayed BASE HITS on the live 1:1 target, and how each cohort performed."""
    tr = summary.get("trail")
    if not tr:
        return ""
    rn, bh = tr["runners"], tr["base_hits"]
    total = rn["count"] + bh["count"]
    if total == 0:
        return ""

    def _row(name, c):
        wr = "&mdash;" if c["win_rate_pct"] is None else f"{c['win_rate_pct']:g}%"
        ar = "&mdash;" if c["avg_r"] is None else f"{c['avg_r']:+g}R"
        pps = "&mdash;" if c["avg_pnl_per_share"] is None else f"${c['avg_pnl_per_share']:+.4f}"
        return (
            f"<tr><td>{name}</td><td class='num'>{c['count']}</td>"
            f"<td class='num'>{wr}</td><td class='num'>{ar}</td><td class='num'>{pps}</td></tr>"
        )

    runner_pct = rn["count"] / total * 100
    return (
        "<h2>Trailing Exit (candle_trail_1min)</h2>"
        f"<p class='muted'>{runner_pct:.0f}% of trades became <b>runners</b> — a fat-green 1-minute candle "
        "dropped the fixed 1:1 target and the trade rode the trailing stop; the rest were <b>base hits</b> "
        "on the live 1:1. Compare the two cohorts to see whether letting winners run is paying for the give-backs.</p>"
        "<div class='bd'><table class='mini'><thead><tr><th>Cohort</th><th>Trades</th><th>Win rate</th>"
        "<th>Avg R</th><th>Avg $/share</th></tr></thead><tbody>"
        f"{_row('Runners (target dropped)', rn)}{_row('Base hits (1:1 target)', bh)}"
        "</tbody></table></div>"
    )


def _config_section(config_snapshot: dict | None) -> str:
    """The strategy/risk values this run actually executed with. Rendered from the
    resolved config (config.yaml + env overrides + fallback defaults), not from the
    yaml text, so what appears here is what the code truly saw."""
    if not config_snapshot:
        return ""

    def rows(section: dict) -> str:
        return "".join(
            f"<tr><td><code>{html.escape(str(k))}</code></td><td>{'null' if v is None else html.escape(str(v))}</td></tr>"
            for k, v in section.items()
        )

    blocks = []
    for name, open_attr in (("strategy", " open"), ("risk", ""), ("session", "")):
        section = config_snapshot.get(name)
        if section:
            blocks.append(
                f"<details{open_attr}><summary>{html.escape(name)}</summary>"
                f"<table class='cfg'><tbody>{rows(section)}</tbody></table></details>"
            )
    return (
        "<h2>Configuration Used</h2>"
        "<p class='muted'>The exact resolved values this run executed with. Settings live in <code>config.yaml</code>"
        " (config.py only defines the schema/fallbacks); a full copy including data settings is saved as"
        " <code>config_used.yaml</code> in this run folder.</p>"
        + "\n".join(blocks)
    )


def _charts_html(chart_files: dict[str, str]) -> str:
    if not chart_files:
        return "<p>No chart assets were generated.</p>"
    return "\n".join(
        f"<section class='chart'><h3>{html.escape(title)}</h3><img src='{html.escape(src)}' alt='{html.escape(title)}'></section>"
        for title, src in chart_files.items()
    )


def _write_monte_carlo_chart(artifacts: dict, assets: Path) -> str:
    """Two-panel Monte Carlo figure: (left) a fan chart of equity paths -- the
    p5-p95 and p25-p75 percentile bands, the p50 line, a thin spaghetti sample of
    individual simulated paths, and the realized (chronological) path overlaid for
    reference; (right) the distribution of final equity, with starting equity and
    the p5/median/p95 marked. Returns the report-relative image path."""
    import matplotlib.pyplot as plt

    band_pcts = artifacts["band_pcts"]
    band = artifacts["band_curves"]          # (5, steps+1)
    sample = artifacts["sample_curves"]      # (S, steps+1)
    final = artifacts["final_equities"]
    actual = artifacts["actual_curve"]
    start = artifacts["starting_equity"]
    x = range(band.shape[1])
    p5, p25, p50, p75, p95 = (band[band_pcts.index(p)] for p in (5, 25, 50, 75, 95))

    fig, (ax_fan, ax_hist) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.7, 1]})

    # Thin spaghetti behind the bands for texture (cap the count so it stays light).
    for row in sample[: min(60, len(sample))]:
        ax_fan.plot(x, row, color="#9ca3af", linewidth=0.3, alpha=0.25, zorder=1)
    ax_fan.fill_between(x, p5, p95, color="#3b82f6", alpha=0.15, zorder=2, label="5–95%")
    ax_fan.fill_between(x, p25, p75, color="#3b82f6", alpha=0.30, zorder=3, label="25–75%")
    ax_fan.plot(x, p50, color="#1d4ed8", linewidth=1.8, zorder=4, label="Median")
    if actual is not None:
        ax_fan.plot(range(len(actual)), actual, color="#dc2626", linewidth=1.8, zorder=5, label="Realized run")
    ax_fan.axhline(start, color="#6b7280", linestyle=":", linewidth=1, zorder=2)
    ax_fan.set_title("Simulated equity paths")
    ax_fan.set_xlabel("Trade number")
    ax_fan.set_ylabel("Equity ($)")
    ax_fan.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_fan.grid(True, axis="y", linewidth=0.4, alpha=0.3)

    ax_hist.hist(final, bins=60, color="#93c5fd", edgecolor="white", linewidth=0.3)
    med = float(np.percentile(final, 50))
    lo = float(np.percentile(final, 5))
    hi = float(np.percentile(final, 95))
    ax_hist.axvline(start, color="#6b7280", linestyle=":", linewidth=1.2, label=f"Start ${start:,.0f}")
    ax_hist.axvline(med, color="#1d4ed8", linewidth=1.5, label=f"Median ${med:,.0f}")
    ax_hist.axvline(lo, color="#dc2626", linestyle="--", linewidth=1.2, label=f"5% ${lo:,.0f}")
    ax_hist.axvline(hi, color="#15803d", linestyle="--", linewidth=1.2, label=f"95% ${hi:,.0f}")
    ax_hist.set_title("Final equity distribution")
    ax_hist.set_xlabel("Final equity ($)")
    ax_hist.set_ylabel("Simulations")
    ax_hist.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = assets / "monte_carlo.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return f"report_assets/{out.name}"


# (label, key, formatter) triples for the Monte Carlo stat cards. Kept local to
# the section rather than in the global _METRIC_LABELS/_TOOLTIPS maps so the whole
# feature stays self-contained and easy to remove.
def _mc_money(v) -> str:
    return "—" if v is None else f"${v:,.0f}"


def _mc_pct(v) -> str:
    return "—" if v is None else f"{v:+.1f}%"


def _mc_pct_plain(v) -> str:
    return "—" if v is None else f"{v:.1f}%"


_MC_CARDS = [
    ("Iterations", "iterations", lambda v: f"{v:,}", "Number of simulated equity paths drawn."),
    ("Trades / Sim", "trades_per_sim", lambda v: f"{v:,}", "Trades used per simulation (= the number of realized trades)."),
    ("Realized Final Equity", "actual_final_equity", _mc_money, "Where the ACTUAL backtest run ended -- the single realized path, for reference against the simulated distribution."),
    ("Median Final Equity", "median_final_equity", _mc_money, "The 50th-percentile ending equity across all simulations."),
    ("Median Return", "median_return_pct", _mc_pct, "Median ending return on starting equity across simulations."),
    ("5% Final Equity", "p5_final_equity", _mc_money, "5th-percentile ending equity: only 1 in 20 simulated paths ended below this. A pessimistic-but-plausible floor."),
    ("95% Final Equity", "p95_final_equity", _mc_money, "95th-percentile ending equity: only 1 in 20 simulated paths ended above this."),
    ("Prob. of Profit", "prob_profit_pct", _mc_pct_plain, "Share of simulations that ended above starting equity."),
    ("Median Max Drawdown", "median_max_drawdown_pct", _mc_pct_plain, "Median of each simulation's worst peak-to-trough equity drop."),
    ("95% Max Drawdown", "p95_max_drawdown_pct", _mc_pct_plain, "1 in 20 simulated paths suffered a max drawdown at least this deep -- the drawdown a single lucky run can hide."),
    ("Risk Of Ruin", "risk_of_ruin_pct", _mc_pct_plain, "Share of simulations whose equity fell to/below the ruin threshold at any point."),
    ("Realized Max Drawdown", "actual_max_drawdown_pct", _mc_pct_plain, "The actual backtest run's worst peak-to-trough drop -- compare to the simulated drawdown percentiles."),
]


def _monte_carlo_section(monte_carlo, chart_rel: str | None) -> str:
    if monte_carlo is None:
        return ""
    m = monte_carlo.metrics
    mode_desc = {
        "resample": "bootstrap resampling (trades drawn WITH replacement -- a different order and mix each run)",
        "reshuffle": "reshuffling (the same trades in a random order each run -- isolates sequence risk)",
    }.get(m.get("mode"), str(m.get("mode")))
    cards = []
    for label, key, fmt, tip in _MC_CARDS:
        v = m.get(key)
        shown = fmt(v) if v is not None else "—"
        cards.append(
            f"<div class='card' title='{html.escape(tip)}'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(shown)}</div></div>"
        )
    img = (
        f"<section class='chart'><img src='{html.escape(chart_rel)}' alt='Monte Carlo simulation'></section>"
        if chart_rel else ""
    )
    intro = (
        f"{m.get('iterations', 0):,} simulations via {mode_desc}. "
        f"Ruin threshold: {m.get('ruin_threshold_pct')}% of starting equity. "
        "The realized run is one draw from this distribution -- these percentiles show the range the same edge could plausibly have produced."
    )
    return (
        "<h2>Monte Carlo Simulation</h2>"
        f"<p class='muted'>{html.escape(intro)}</p>"
        f"<div class='grid'>{''.join(cards)}</div>"
        + img
    )


def _fmt_cell(col: str, val) -> str:
    if col == "trail_activated":
        # Runner (fat-green dropped the target) vs base hit on the live 1:1 target.
        if val is True or str(val).lower() == "true":
            return "<span class='badge run'>runner</span>"
        if val is False or str(val).lower() == "false":
            return "<span class='badge base'>base</span>"
        return "—"
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if col == "pnl_pct":
        return f"{val:+.2f}%"
    if col in {"premarket_gain_pct", "pullback_retrace_pct"}:
        return f"{val:.1f}%"
    if col in {"rvol_at_spike", "max_rvol_seen"}:
        return f"{val:.2f}x"
    if col == "daily_avg_volume":
        return f"{val / 1_000:.0f}K"
    if col == "shares":
        return f"{int(val):,}"
    if col == "pnl_total":
        return f"{val:+,.2f}"
    if col == "equity_after":
        return f"${val:,.2f}"
    if col in {"stock_price", "premarket_high", "entry_price", "exit_price", "pnl_per_share",
               "anchor_price", "rip_low", "trigger_price", "pullback_low", "stop_price",
               "target_price", "trail_stop_final"}:
        return f"{val:.2f}"
    if col == "r_multiple":
        return f"{val:+.2f}"
    if col == "float_shares":
        return f"{val / 1_000_000:.1f}M"
    if col == "exit_legs":
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except ValueError:
                pass
        if isinstance(val, list):
            kind_tag = {"target": "sold", "stop": "stopped", "session_end": "closed"}
            return html.escape(" → ".join(
                f"{kind_tag.get(leg.get('kind'), leg.get('kind'))} {leg.get('frac', 0) * 100:g}% @ {leg.get('price'):.2f}"
                for leg in val
            ))
    return _fmt(val)


_DIP_MODES = {"dip_red_break", "dip_swing_high", "dip_trendline", "dip_ema_reclaim"}

# Friendly, compact headers for the trade table (raw field names are cryptic).
_COL_LABELS = {
    "setup_index": "#", "premarket_gain_pct": "gain%", "rvol_at_spike": "RVOL",
    "premarket_high": "HOD", "stock_price": "HOD", "trigger_price": "trigger",
    "pullback_low": "pullback low", "pullback_retrace_pct": "retrace%",
    "stop_price": "stop", "target_price": "target", "entry_price": "entry",
    "exit_price": "exit", "trail_activated": "runner?", "trail_stop_final": "trail stop",
    "scale_outs": "scales", "exit_legs": "legs", "pnl_per_share": "$/sh",
    "pnl_total": "P&L $", "pnl_pct": "P&L %", "r_multiple": "R", "equity_after": "equity",
    "float_shares": "float", "path_ambiguous": "ambig?", "ambiguity_resolution": "ambig by",
    "anchor_price": "pm open", "rip_low": "rip low",
}


def _table_columns(df: pd.DataFrame, entry_mode: str, exit_mode: str) -> list[str]:
    """Column order tailored to the run's config: a dip entry leads with its own
    defining levels (trigger/last-red, pullback low, retrace) and keeps the HOD only
    as context; a hod_break entry leads with the HOD. candle_trail adds the runner
    columns; scale_out adds the ladder columns. Only columns present in df survive."""
    is_dip = entry_mode in _DIP_MODES
    is_trail = exit_mode == "candle_trail_1min"
    has_scale = "scale_outs" in df.columns and pd.to_numeric(df["scale_outs"], errors="coerce").fillna(0).gt(0).any()
    cols = ["ticker", "date", "setup_index", "result", "reason", "premarket_gain_pct", "rvol_at_spike"]
    cols += (["trigger_price", "pullback_low", "pullback_retrace_pct", "premarket_high"]
             if is_dip else ["premarket_high", "trigger_price"])
    cols += ["stop_price", "target_price", "entry_price", "exit_price"]
    if is_trail:
        cols += ["trail_activated", "trail_stop_final"]
    if has_scale:
        cols += ["scale_outs", "exit_legs"]
    cols += ["shares", "pnl_per_share", "pnl_total", "pnl_pct", "r_multiple", "equity_after",
             "float_shares", "path_ambiguous", "ambiguity_resolution"]
    return [c for c in cols if c in df.columns]


def _trade_table(df: pd.DataFrame, run_dir: Path | None = None, limit: int = 250, summary: dict | None = None) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    summary = summary or {}
    em = summary.get("entry_mode")
    if not em and "entry_mode" in df.columns:
        vals = df["entry_mode"].dropna()
        em = str(vals.iloc[0]) if not vals.empty else "hod_break"
    em = em or "hod_break"
    cols = _table_columns(df, em, summary.get("exit_mode", "standard"))
    rows = []
    for _, row in df.head(limit).iterrows():
        setup_idx = row.get("setup_index")
        setup_idx = int(setup_idx) if pd.notna(setup_idx) else 1
        row_em = row.get("entry_mode")
        safe = trade_folder_name(row.get("ticker"), row.get("date"), setup_idx,
                                 row_em if isinstance(row_em, str) else None)
        folder = f"trade_folders/{safe}"
        has_chart = run_dir is None or (run_dir / "trade_folders" / safe / "chart.png").exists()
        result = str(row.get("result", ""))
        row_class = {"WIN": "win", "LOSS": "loss"}.get(result, "")
        cells = []
        for c in cols:
            val = row.get(c)
            if c == "ticker" and has_chart:
                cells.append(f"<td><a href='{html.escape(folder)}/chart.png' target='_blank'>{_fmt(val)}</a></td>")
            elif c == "result":
                cells.append(f"<td class='result-{html.escape(result)}'>{_fmt(val)}</td>")
            else:
                cells.append(f"<td>{_fmt_cell(c, val)}</td>")
        rows.append(f"<tr class='{row_class}'>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{html.escape(_COL_LABELS.get(c, c))}</th>" for c in cols)
    note = f"<p class='muted'>Showing first {min(limit, len(df)):,} of {len(df):,} rows. Full data is in the trades CSV and SQLite database.</p>"
    return note + "<div class='tablewrap'><table><thead><tr>" + head + "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>"


def _diverge_bg(v: float, maxabs: float) -> str:
    """A diverging cell fill anchored on the panel colour: green tint for positive,
    red for negative, scaled by magnitude. Uses the theme's own --pos/--neg/--panel
    so it re-derives correctly in dark mode instead of being flipped."""
    if maxabs <= 0 or not v:
        return "var(--panel)"
    pct = min(abs(v) / maxabs, 1.0) * 70  # cap the tint so cell text stays legible
    hue = "--pos" if v > 0 else "--neg"
    return f"color-mix(in srgb, var({hue}) {pct:.0f}%, var(--panel))"


def _dbar(v: float | None, maxabs: float) -> str:
    """A zero-anchored diverging bar: positive grows right from the midline in the
    positive hue, negative grows left in the negative hue -- polarity by position
    AND colour, so it reads without relying on colour alone."""
    if v is None:
        return "<div class='dbar'></div>"
    w = 0.0 if maxabs <= 0 else min(abs(v) / maxabs, 1.0) * 50.0
    hue = "--pos" if v >= 0 else "--neg"
    side = "left:50%" if v >= 0 else "right:50%"
    return f"<div class='dbar'><span class='dfill' style='{side};width:{w:.1f}%;background:var({hue})'></span></div>"


def _exit_description(g: dict) -> str:
    """A short human phrase for a non-fixed exit, from the config modes on `g`."""
    if g.get("exit_mode") == "candle_trail_1min":
        base = ("a 1-minute trailing stop that drops its fixed target on the first fat-body green close "
                "and rides the trail (banking a partial at 1R)")
    else:
        target = {"r_multiple": "an R-multiple target"}.get(g.get("target_mode"), "a fixed target")
        stops = {"pullback_low": "a structural stop at the pullback low",
                 "red_candle_low": "a structural stop at the last red candle's low"}
        stop = stops.get(g.get("stop_mode"), "a fixed stop")
        base = f"{target} with {stop}"
    return base


def _offset_heatmap(g: dict) -> str:
    # The fixed target x stop grid only describes a fixed-offset exit. Under a
    # trailing/runner exit or structural stop/R-multiple target, replace it with an
    # honest note (the analytics layer flags this via applicable=False) rather than
    # a grid marking an "optimal" fixed target the strategy never places.
    if g.get("applicable") is False:
        realized = g.get("realized_expectancy_per_share")
        realized_txt = "&mdash;" if realized is None else f"{realized:+.3f}/sh"
        return (
            "<h3 class='sec'>Optimal limit placement &mdash; n/a for this exit</h3>"
            "<p class='muted'>This grid replays a <b>fixed</b> target&thinsp;&times;&thinsp;stop against each "
            f"trade's excursion, but your config exits via {html.escape(_exit_description(g))} &mdash; not a fixed "
            "target/stop. Two reasons it doesn't apply: (1) there is no single fixed target to optimise (the target "
            "is dropped on strength and the stop trails); (2) MFE/MAE are recorded only up to each trade's <i>actual</i> "
            "exit, so they can't fairly model a further-out fixed target either. Your actual exit realised "
            f"<b>{realized_txt}</b> in-sample &mdash; see the trailing-exit / runner breakdown for how that split "
            "between 1R base-hits and runners.</p>"
        )
    unit = "%" if g["mode"] == "percent" else "$"
    cells = [c for row in g["grid"] for c in row["cells"]]
    maxabs = max((abs(c["expectancy"]) for c in cells), default=0.0)
    head = f"<th class='corner'>stop&thinsp;\\&thinsp;target</th>" + "".join(f"<th>{t:g}{unit}</th>" for t in g["targets"])
    body = []
    for row in g["grid"]:
        tds = [f"<th>{row['stop']:g}{unit}</th>"]
        for c in row["cells"]:
            marks = ("★" if (g["best"] and c is g["best"]) else "") + ("◻" if c["is_current"] else "")
            cls = "hm" + (" cur" if c["is_current"] else "") + (" best" if (g["best"] and c is g["best"]) else "")
            wr = f"{c['win_rate']:.0f}%" if c["win_rate"] is not None else "—"
            title = (f"target {c['target']:g}{unit} / stop {c['stop']:g}{unit}: expectancy {c['expectancy']:+.3f}/sh · "
                     f"{c['wins']}W {c['losses']}L {c['open']} open · win {wr}")
            tds.append(f"<td class='{cls}' style='background:{_diverge_bg(c['expectancy'], maxabs)}' "
                       f"title='{html.escape(title)}'>{c['expectancy']:+.2f}<sup>{marks}</sup></td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (
        "<h3 class='sec'>Optimal limit placement — target × stop expectancy ($/share)</h3>"
        "<p class='muted'>Each of your executed trades replayed against every offset pair using its recorded "
        "MFE/MAE (max favourable/adverse excursion). A bar that reached BOTH levels is scored a loss "
        "(the default conservative intrabar policy); one that reached NEITHER rode to session close, modelled "
        "as breakeven — so a cell with many 'open' trades (hover) is not a clean win. "
        "<b>◻</b> your current offsets · <b>★</b> in-sample best.</p>"
        "<div class='tablewrap'><table class='heat'><thead><tr>" + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def _fmt_thr(threshold: float, unit: str) -> str:
    return f"{threshold / 1e6:g}M shares" if unit == "M shares" else f"{threshold:g}{unit}"


def _sweeps_html(sweeps: list[dict]) -> str:
    blocks = []
    for s in sweeps:
        sw, label, unit = s["sweep"], s["label"], s["unit"]
        steps = sw["steps"]
        if not steps:
            continue
        maxabs = max((abs(st["expectancy"]) for st in steps if st["expectancy"] is not None), default=0.0)
        op = "&ge;" if sw["side"] == "min" else "&le;"
        rows = []
        for st in steps:
            hl = " hl" if sw["best"] is st else ""
            n_txt = str(st["n"]) + (f" <span class='muted'>&minus;{st['dropped']}</span>" if st["dropped"] else "")
            wr = "&mdash;" if st["win_rate"] is None else f"{st['win_rate']:.0f}%"
            exp = st["expectancy"]
            exp_txt = "&mdash;" if exp is None else f"{exp:+.3f}"
            rows.append(
                f"<tr class='{hl.strip()}'><td class='thr'>{op} {_fmt_thr(st['threshold'], unit)}</td>"
                f"<td>{n_txt}</td><td>{wr}</td><td class='num'>{exp_txt}</td>"
                f"<td class='barcell'>{_dbar(exp, maxabs)}</td></tr>"
            )
        note = ""
        if sw["best"] and sw["best"]["dropped"] > 0:
            note = (f" <span class='muted'>&mdash; sample-best {op} {_fmt_thr(sw['best']['threshold'], unit)} "
                    f"(drops {sw['best']['dropped']}, win {sw['best']['win_rate']:.0f}%, exp {sw['best']['expectancy']:+.3f})</span>")
        blocks.append(
            f"<div class='sweep'><h4 class='swh'>{html.escape(label)}{note}</h4>"
            "<table class='mini'><thead><tr><th>Threshold</th><th>Trades</th><th>Win%</th>"
            "<th>Exp/sh</th><th>&nbsp;</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    return ("<h3 class='sec' style='color:var(--ink);font-size:15px;margin-top:26px'>Gate sensitivity &mdash; effect of TIGHTENING each filter</h3>"
            "<p class='muted'>How your executed trades' stats shift as each gate is tightened (trades dropped). This can "
            "only show tightening &mdash; loosening would admit candidates rejected before they became trades, whose "
            "outcome needs a re-run. Highlighted row = best expectancy keeping &ge; 5 trades.</p>"
            "<div class='sweeps'>" + "".join(blocks) + "</div>")


def _temporal_bars(rows: list[dict], key_label: str) -> str:
    if not rows:
        return ""
    maxabs = max((abs(c["expectancy"]) for c in rows if c["expectancy"] is not None), default=0.0)
    body = []
    for c in rows:
        wr = "&mdash;" if c["win_rate"] is None else f"{c['win_rate']:.0f}%"
        exp = c["expectancy"]
        exp_txt = "&mdash;" if exp is None else f"{exp:+.3f}"
        pnl_cls = "pos" if c["total_pnl"] >= 0 else "neg"
        body.append(
            f"<tr><td>{html.escape(str(c['key']))}</td><td>{c['n']}</td><td>{wr}</td>"
            f"<td class='num'>{exp_txt}</td><td class='barcell'>{_dbar(exp, maxabs)}</td>"
            f"<td class='num {pnl_cls}'>{c['total_pnl']:+,.0f}</td></tr>"
        )
    return (f"<div class='bd'><h4 class='swh'>{html.escape(key_label)}</h4>"
            "<table class='mini'><thead><tr><th>&nbsp;</th><th>n</th><th>Win%</th><th>Exp/sh</th>"
            "<th>&nbsp;</th><th>P&amp;L</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def _temporal_html(t: dict) -> str:
    ppd = t.get("profit_per_trading_day")
    tile = ""
    if ppd is not None:
        cls = "pos" if ppd >= 0 else "neg"
        tile = (f"<div class='card {cls}' style='max-width:280px;margin-bottom:12px'>"
                f"<div class='label'>Profit / trading day</div><div class='value'>${ppd:,.2f}</div>"
                f"<div class='muted'>{t['trading_days']} days traded · ${t['total_pnl']:,.0f} total</div></div>")
    return ("<h3 class='sec' style='color:var(--ink);font-size:15px;margin-top:26px'>Temporal edges &mdash; when the edge actually lived</h3>"
            + tile + "<div class='bd-row'>"
            + _temporal_bars(t["by_weekday"], "By weekday") + _temporal_bars(t["by_hour"], "By entry hour (ET)")
            + _temporal_bars(t["by_month"], "By month") + "</div>")


def _analytics_section(a: dict | None) -> str:
    if not a:
        return ""
    parts = ["<h2>Optimisation &amp; Edge Analysis</h2>",
             "<p class='muted'>Everything below is <b>in-sample</b> — computed on the very trades you'd tune from — so a "
             "hindsight-best value is a hypothesis to test forward, not a setting to adopt blindly. Small n makes any "
             "single 'best' noisy; every result carries its trade count.</p>"]
    if a["takeaways"]:
        items = "".join(f"<li>{html.escape(x)}</li>" for x in a["takeaways"])
        parts.append(f"<div class='callout'><div class='callout-h'>Key takeaways</div><ul>{items}</ul></div>")
    parts.append(_offset_heatmap(a["target_stop_grid"]))
    parts.append(_sweeps_html(a["threshold_sweeps"]))
    parts.append(_temporal_html(a["temporal"]))
    return "\n".join(parts)


def _render_html(df: pd.DataFrame, summary: dict, chart_files: dict[str, str], run_dir: Path | None = None, config_snapshot: dict | None = None, monte_carlo=None, mc_chart_rel: str | None = None, analytics: dict | None = None) -> str:
    generated = html.escape(str(summary.get("generated_at", "")))
    # Granular counts (DNQ split by cause, missing data split by kind) straight
    # from the rows; the summary's coarse result_counts is only a fallback.
    if not df.empty:
        result_counts = df.apply(_result_detail, axis=1).value_counts().to_dict()
    else:
        result_counts = summary.get("result_counts", {})
    net = summary.get("net_pnl")
    net_cls = "pos" if isinstance(net, (int, float)) and net > 0 else ("neg" if isinstance(net, (int, float)) and net < 0 else "")
    result_pills = "".join(
        f"<span class='pill r-{html.escape(str(k))}'>{html.escape(str(k))} <b>{_fmt(v)}</b></span>"
        for k, v in result_counts.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IBKR Premarket Backtest Report</title>
  <style>
    :root {{
      --bg:#f4f6f9; --panel:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
      --line2:#eef2f7; --accent:#2563eb; --pos:#059669; --neg:#dc2626;
      --pos-bg:#ecfdf5; --neg-bg:#fef2f2; --chip:#f1f5f9; --shadow:0 1px 2px rgba(15,23,42,.05),0 1px 3px rgba(15,23,42,.04);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg:#0b1220; --panel:#111a2b; --ink:#e5edf7; --muted:#94a3b8; --line:#1e293b;
        --line2:#172033; --accent:#60a5fa; --pos:#34d399; --neg:#f87171;
        --pos-bg:#0e2a20; --neg-bg:#2a1414; --chip:#1a2334; --shadow:0 1px 2px rgba(0,0,0,.3);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin:0; padding:0 clamp(16px,4vw,44px) 60px; color:var(--ink); background:var(--bg); line-height:1.45; }}
    header.top {{ position:sticky; top:0; z-index:5; margin:0 clamp(-16px,-4vw,-44px) 8px; padding:18px clamp(16px,4vw,44px); background:color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }}
    h1 {{ font-size:20px; margin:0; letter-spacing:-.01em; }}
    h2 {{ font-size:17px; margin:34px 0 6px; letter-spacing:-.01em; }}
    h3.sec {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:22px 0 10px; font-weight:700; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    .headline {{ display:flex; flex-wrap:wrap; gap:14px 26px; align-items:baseline; margin-top:6px; }}
    .headline .big {{ font-size:26px; font-weight:800; letter-spacing:-.02em; }}
    .headline .big.pos {{ color:var(--pos); }} .headline .big.neg {{ color:var(--neg); }}
    .headline .sub {{ font-size:13px; color:var(--muted); }}
    .fingerprint {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px; }}
    .chip {{ display:inline-flex; align-items:stretch; border:1px solid var(--line); border-radius:8px; overflow:hidden; font-size:12px; background:var(--panel); box-shadow:var(--shadow); }}
    .chip .ck {{ background:var(--chip); color:var(--muted); padding:4px 8px; font-weight:600; }}
    .chip .cv {{ padding:4px 9px; font-weight:600; }}
    .chip.gates .cv {{ color:var(--accent); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(178px,1fr)); gap:10px; margin:0 0 8px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; box-shadow:var(--shadow); }}
    .card[title] {{ cursor:help; }}
    .card.pos {{ border-left:3px solid var(--pos); }} .card.neg {{ border-left:3px solid var(--neg); }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
    .card[title] .label {{ border-bottom:1px dotted var(--line); display:inline-block; }}
    .value {{ font-size:20px; font-weight:750; margin-top:5px; font-variant-numeric:tabular-nums; }}
    .card.pos .value {{ color:var(--pos); }} .card.neg .value {{ color:var(--neg); }}
    .pills {{ display:flex; flex-wrap:wrap; gap:7px; margin:6px 0; }}
    .pill {{ font-size:12px; padding:4px 10px; border-radius:999px; background:var(--chip); border:1px solid var(--line); color:var(--muted); }}
    .pill b {{ color:var(--ink); }}
    .pill.r-WIN {{ background:var(--pos-bg); color:var(--pos); }} .pill.r-LOSS {{ background:var(--neg-bg); color:var(--neg); }}
    .bd-row {{ display:flex; flex-wrap:wrap; gap:16px; }}
    .bd {{ flex:1 1 260px; }} .bd h3 {{ font-size:13px; margin:0 0 6px; }}
    .chart {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; margin:14px 0; box-shadow:var(--shadow); }}
    .chart img {{ max-width:100%; height:auto; display:block; border-radius:6px; }}
    .tablewrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; box-shadow:var(--shadow); }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line2); padding:7px 11px; text-align:left; font-size:12.5px; white-space:nowrap; }}
    thead th {{ position:sticky; top:0; background:var(--chip); color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.03em; }}
    tbody tr:hover {{ background:color-mix(in srgb, var(--accent) 6%, transparent); }}
    table.mini, table.cfg {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
    table.cfg {{ width:auto; min-width:380px; }}
    a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    code {{ background:var(--chip); padding:1px 5px; border-radius:5px; font-size:12px; }}
    tr.win td {{ background:var(--pos-bg); }} tr.loss td {{ background:var(--neg-bg); }}
    tr.win:hover td, tr.loss:hover td {{ filter:brightness(0.98); }}
    details {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px 14px; margin:10px 0; box-shadow:var(--shadow); }}
    details summary {{ cursor:pointer; font-weight:650; }}
    td.result-WIN {{ color:var(--pos); font-weight:700; }}
    td.result-LOSS {{ color:var(--neg); font-weight:700; }}
    .badge {{ display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px; font-weight:650; }}
    .badge.run {{ color:var(--pos); background:var(--pos-bg); }}
    .badge.base {{ color:var(--muted); background:var(--chip); }}
    /* --- analytics --- */
    .callout {{ background:color-mix(in srgb, var(--accent) 7%, var(--panel)); border:1px solid color-mix(in srgb, var(--accent) 30%, var(--line)); border-left:3px solid var(--accent); border-radius:12px; padding:12px 16px; margin:12px 0 18px; box-shadow:var(--shadow); }}
    .callout-h {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); font-weight:700; margin-bottom:6px; }}
    .callout ul {{ margin:0; padding-left:18px; }} .callout li {{ font-size:13px; margin:4px 0; }}
    table.heat {{ border-collapse:separate; border-spacing:3px; background:transparent; width:auto; }}
    table.heat th {{ background:transparent; position:static; color:var(--muted); font-size:11px; font-weight:600; text-transform:none; letter-spacing:0; text-align:center; padding:4px 8px; border:0; white-space:nowrap; }}
    table.heat th.corner {{ text-align:right; color:var(--muted); }}
    table.heat td.hm {{ text-align:center; font-size:12px; font-weight:650; font-variant-numeric:tabular-nums; border:1px solid var(--line); border-radius:7px; padding:8px 10px; min-width:52px; color:var(--ink); cursor:help; }}
    table.heat td.hm sup {{ font-size:9px; margin-left:1px; }}
    table.heat td.cur {{ outline:2px solid var(--accent); outline-offset:-2px; }}
    table.heat td.best {{ box-shadow:0 0 0 2px var(--pos) inset; }}
    .dbar {{ position:relative; height:13px; min-width:74px; background:var(--chip); border-radius:3px; }}
    .dbar::before {{ content:''; position:absolute; left:50%; top:-1px; bottom:-1px; width:1px; background:var(--line); }}
    .dfill {{ position:absolute; top:2px; bottom:2px; border-radius:2px; }}
    td.barcell {{ width:88px; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    td.num.pos {{ color:var(--pos); }} td.num.neg {{ color:var(--neg); }}
    .sweeps {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px,1fr)); gap:14px; }}
    .swh {{ font-size:12.5px; margin:0 0 6px; font-weight:650; }}
    table.mini tbody tr.hl td {{ background:color-mix(in srgb, var(--pos) 12%, var(--panel)); }}
    table.mini td.thr {{ font-weight:600; white-space:nowrap; }}
  </style>
</head>
<body>
  <header class="top">
    <h1>IBKR Premarket Backtest Report</h1>
    <div class="headline">
      <span class="big {net_cls}">{_fmt_metric('net_pnl', summary.get('net_pnl'))}</span>
      <span class="sub">{_fmt_metric('net_pnl_pct', summary.get('net_pnl_pct'))} · {summary.get('wins', 0)}W / {summary.get('losses', 0)}L · {_fmt_metric('win_rate_pct', summary.get('win_rate_pct'))} win · PF {_fmt(summary.get('profit_factor'))}</span>
    </div>
    {_fingerprint(summary, config_snapshot)}
  </header>

  <p class="muted">Generated {generated}. Paper/backtesting analysis only — this report places no orders. Hover any metric card for how it's computed.</p>

  <h2>Result Counts</h2>
  <div class="pills">{result_pills}</div>

  <h2>Summary Metrics</h2>
  {_metric_cards(summary)}

  {_breakdown_tables(summary)}

  {_trail_section(summary)}

  {_analytics_section(analytics)}

  {_config_section(config_snapshot)}

  <h2>Charts</h2>
  {_charts_html(chart_files)}

  {_monte_carlo_section(monte_carlo, mc_chart_rel)}

  {_scale_out_section(df, run_dir)}

  <h2>Trade Rows</h2>
  <p class="muted">Click a ticker to open that trade's generated chart image when available. Columns are tailored to this run's entry/exit mode.</p>
  {_trade_table(df, run_dir, summary=summary)}
</body>
</html>
"""


def _scale_out_section(df: pd.DataFrame, run_dir: Path | None = None) -> str:
    """A dedicated log of just the trades that scaled out -- the full trade table
    below buries these among every other result; this is the "did the ladder
    work" record: legs, weighted exit, and the ACTUALIZED R-multiple per trade."""
    if df.empty or "scale_outs" not in df.columns:
        return ""
    scaled = df[pd.to_numeric(df["scale_outs"], errors="coerce").fillna(0) > 0]
    if scaled.empty:
        return "<h2>Scale-Out Trades</h2><p class='muted'>No trades scaled out this run.</p>"
    return (
        "<h2>Scale-Out Trades</h2>"
        f"<p class='muted'>{len(scaled)} trade(s) banked at least one scale-out leg. "
        "exit_legs shows the ladder in order; r_multiple is the REALIZED risk:reward across all legs "
        "(weighted-exit P&L ÷ stop_offset), not the configured target -- compare it to Configured Risk:Reward above.</p>"
        + _trade_table(scaled, run_dir)
    )


def _parse_time(v) -> pd.Timestamp | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return pd.Timestamp(v)
    except (ValueError, TypeError):
        return None


def _prep_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Raw bars with a proper tz-aware dt_et column, sorted -- the NATIVE
    (10-second) resolution the strategy actually ran on. The detail panel draws
    these directly so the HOD, entry and exit land on the true sub-minute price
    action instead of a 1-minute reaggregation that hides exactly those extremes."""
    work = bars.copy()
    work["dt_et"] = pd.to_datetime(work["dt_et"] if "dt_et" in work else work["date"])
    return work.sort_values("dt_et").reset_index(drop=True)


def _resample_candles(bars: pd.DataFrame) -> pd.DataFrame:
    """1-minute OHLC candles -- used ONLY for the full-session OVERVIEW panel,
    where native 10-second bars would be thousands of indistinguishable slivers.
    The detail panel uses the raw bars (see _prep_bars) for full precision."""
    work = _prep_bars(bars)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in work.columns:
        agg["volume"] = "sum"
    return (
        work.set_index("dt_et")
        .resample("1min")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )


def _one_min_indicators(bars: pd.DataFrame) -> pd.DataFrame | None:
    """The exact 1-minute EMA9/EMA20/VWAP series the strategy evaluated (same
    build_1m_bars + add_indicators path it uses), so the chart's overlaid lines
    are what the gates actually saw -- independent of the candle resolution the
    panel happens to draw (10-second detail vs 1-minute overview)."""
    work = _prep_bars(bars)
    if "volume" not in work.columns:
        work["volume"] = 0.0
    cols = ["dt_et", "open", "high", "low", "close", "volume"]
    try:
        one_min = build_1m_bars(work[cols])
        if one_min.empty:
            return None
        return add_indicators(one_min)
    except Exception:
        return None


_CANDLE_UP = "#16a34a"
_CANDLE_DOWN = "#dc2626"


def _bar_width_days(candles: pd.DataFrame) -> float:
    """Median bar interval in matplotlib date units (days). Candle bodies are
    drawn a fraction of this, so the SAME drawing code renders crisp bodies at
    both native 10-second (detail) and 1-minute (overview) resolutions instead of
    a hard-coded 1-minute width that turns 10-second bars into thin overlapping
    slivers."""
    if len(candles) < 2:
        return 60.0 / 86400.0
    deltas = candles["dt_et"].diff().dropna().dt.total_seconds()
    sec = float(deltas.median()) if not deltas.empty else 60.0
    return max(sec, 1.0) / 86400.0


def _draw_candlesticks(ax, candles: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    body_half_width = _bar_width_days(candles) * 0.35
    for _, c in candles.iterrows():
        t = mdates.date2num(c["dt_et"])
        o, h, low, close = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        color = _CANDLE_UP if close >= o else _CANDLE_DOWN
        ax.add_line(Line2D([t, t], [low, h], color=color, linewidth=0.8, zorder=2, solid_capstyle="round"))
        body_bottom, body_height = min(o, close), max(abs(close - o), 1e-4)
        ax.add_patch(Rectangle(
            (t - body_half_width, body_bottom), 2 * body_half_width, body_height,
            facecolor=color, edgecolor=color, linewidth=0.3, zorder=3,
        ))
    # date2num normalizes tz-aware timestamps to UTC; without handing the axis
    # the session tz, the tick labels silently render as UTC (+4h) while the
    # axis is captioned "Time (ET)".
    ax.xaxis_date(tz=getattr(candles["dt_et"].dt, "tz", None))

    if "volume" in candles.columns and float(candles["volume"].max() or 0) > 0:
        # Volume bars along the bottom of the panel, colored by candle direction.
        # A twin axis draws above its host, so the host is lifted and its
        # background hidden to keep candles/markers in front of the bars.
        vax = ax.twinx()
        times = [mdates.date2num(c) for c in candles["dt_et"]]
        colors = [_CANDLE_UP if c >= o else _CANDLE_DOWN for o, c in zip(candles["open"], candles["close"])]
        vax.bar(times, candles["volume"], width=2 * body_half_width, color=colors, alpha=0.32, linewidth=0, zorder=1)
        vax.set_ylim(0, float(candles["volume"].max()) * 5.0)  # tallest bar = ~20% of panel height
        vax.set_yticks([])
        ax.set_zorder(vax.get_zorder() + 1)
        ax.patch.set_visible(False)


def _setup_time_axis(ax, tz, detail: bool) -> None:
    """Precise, readable time axis. The detail panel shows HH:MM:SS with dense
    ticks and both major+minor x-gridlines so an event's time can be read to the
    second; the overview shows HH:MM. Without this the default locator gives a
    handful of coarse ticks that make it impossible to pin an entry/exit in time.
    Call AFTER the panel's x-limits are set so the minor interval is chosen from
    the visible span (an AutoDateLocator for the minor ticks can't find a fitting
    interval on a short intraday window and warns)."""
    import matplotlib.dates as mdates

    # Loose minticks: a tight [min,max] band can be unsatisfiable for some spans
    # (e.g. a 5-hour overview has no interval giving 6-9 ticks), which makes
    # AutoDateLocator warn and fall back. A wide band always has a fitting interval.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=(14 if detail else 10), tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S" if detail else "%H:%M", tz=tz))
    if not detail:
        return
    lo, hi = ax.get_xlim()
    span_s = max((hi - lo) * 86400.0, 1.0)  # x units are days
    if span_s <= 20 * 60:
        minor = mdates.SecondLocator(interval=30, tz=tz)
    elif span_s <= 60 * 60:
        minor = mdates.MinuteLocator(interval=1, tz=tz)
    elif span_s <= 3 * 3600:
        minor = mdates.MinuteLocator(interval=5, tz=tz)
    else:
        minor = mdates.MinuteLocator(interval=15, tz=tz)
    ax.xaxis.set_minor_locator(minor)
    ax.grid(True, which="major", axis="x", linewidth=0.5, alpha=0.30, zorder=0)
    ax.grid(True, which="minor", axis="x", linewidth=0.3, alpha=0.12, zorder=0)


def _zoom_window(result: TradeResult, session_start: pd.Timestamp, session_end: pd.Timestamp, pad_minutes: int = 10):
    """A time window around the actual setup/trade, padded for context. The setup
    and trade (HOD -> pullback -> trigger -> entry -> exit) often span only a few
    minutes out of a multi-hour session, so zooming here is what actually makes the
    action visible -- style alone can't fix candles compressed to a few pixels.
    Returns None when there's nothing to zoom to (no events at all)."""
    times = [_parse_time(t) for t in (
        result.premarket_high_time, result.pullback_low_time, result.trigger_time,
        result.entry_time, result.exit_time,
    )]
    times = [t for t in times if t is not None]
    if not times:
        return None
    pad = pd.Timedelta(minutes=pad_minutes)
    lo = max(session_start, min(times) - pad)
    hi = min(session_end, max(times) + pad)
    return (lo, hi) if hi > lo else None


def _draw_trade_panel(
    ax, result: TradeResult, candles: pd.DataFrame, indicators: pd.DataFrame | None,
    session_start: pd.Timestamp, session_end: pd.Timestamp, show_legend: bool,
    *, detail: bool = False, actual_hod: float | None = None, actual_hod_time=None,
) -> None:
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator

    entry_t = _parse_time(result.entry_time)
    exit_t = _parse_time(result.exit_time)
    hod_t = _parse_time(result.premarket_high_time)
    pullback_t = _parse_time(result.pullback_low_time)

    _draw_candlesticks(ax, candles)
    legend_handles = []
    tick_fmt = "{:.3f}" if detail else "{:.2f}"  # sub-cent precision in the detail panel

    # 1-minute EMA9/EMA20/VWAP -- the exact series the strategy's gates evaluated
    # (from _one_min_indicators), overlaid regardless of the candle resolution
    # this panel draws. Plotted from the separate indicator frame so the detail
    # panel can show 10-second candles under faithful 1-minute indicator lines.
    def indicator_line(col, color, label):
        if indicators is None or col not in indicators.columns or indicators[col].isna().all():
            return
        ax.plot(indicators["dt_et"], indicators[col], color=color, linewidth=1.1, alpha=0.9, zorder=3)
        if show_legend:
            legend_handles.append(Line2D([0], [0], color=color, linewidth=1.8, label=label))

    indicator_line("ema9", "#6b7280", "EMA9")
    indicator_line("ema20", "#1e3a8a", "EMA20")
    indicator_line("vwap", "#eab308", "VWAP")

    def hline(y, xmin, xmax, color, style, lw, label):
        if y is None or xmin is None or xmax is None:
            return
        ax.hlines(float(y), mdates.date2num(xmin), mdates.date2num(xmax), colors=color, linestyles=style, linewidth=lw, zorder=4)
        if show_legend:
            legend_handles.append(Line2D([0], [0], color=color, linestyle=style, linewidth=lw, label=f"{label}: {tick_fmt.format(float(y))}"))

    def point_marker(t, y, color, marker, size, label):
        if t is None or y is None:
            return
        ax.scatter([mdates.date2num(t)], [float(y)], color=color, marker=marker, s=size, zorder=6, edgecolors="white", linewidths=0.7)
        if show_legend:
            # Exact price AND time in the legend -- the authoritative, overlap-proof
            # record of each event. On-chart callouts were tried but collide badly
            # when the HOD/pullback/entry/exit cluster within seconds (common on
            # fast premarket movers), so the precise values live here instead.
            lab = f"{label}: {tick_fmt.format(float(y))} @ {t.strftime('%H:%M:%S')}"
            legend_handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=lab))

    def price_tag(y, color, text):
        """Exact entry/exit level pinned at the right edge of the DETAIL panel, on
        the price axis -- trace the fill to the cent without hunting for the line's
        height against the y-ticks. Only entry and exit are tagged (the trade's two
        fills); every level's exact value is already in the legend. Suppressed on
        the overview panel (too small to read)."""
        if y is None or not detail:
            return
        ax.annotate(
            f"{text} {tick_fmt.format(float(y))}", xy=(0.998, float(y)),
            xycoords=("axes fraction", "data"), ha="right", va="center",
            fontsize=7, color="white", zorder=9, annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.22", fc=color, ec="none", alpha=0.95),
        )

    # Background context: previous close spans the whole chart, thin and muted.
    hline(result.prev_close, session_start, session_end, "#9ca3af", ":", 1.0, "Prev close")

    # The ACTUAL running high of day up to entry -- the diagnostic that exposes a
    # stale/armed setup HOD. The strategy freezes its setup HOD the moment a
    # pullback registers, so on fast movers price can run WELL above that frozen
    # level before pulling back and re-breaking it (e.g. TNMG 2025-11-13: armed
    # HOD 7.86 while price had already printed 8.51). Drawn only when it
    # meaningfully exceeds the marked setup HOD, so ordinary trades stay clean.
    setup_hod = result.premarket_high
    if (actual_hod is not None and actual_hod_time is not None and setup_hod is not None
            and actual_hod > float(setup_hod) + 0.011):
        x1 = entry_t or session_end
        ax.hlines(actual_hod, mdates.date2num(session_start), mdates.date2num(x1),
                  colors="#db2777", linestyles=(0, (5, 2)), linewidth=1.4, zorder=4)
        ax.scatter([mdates.date2num(actual_hod_time)], [actual_hod], color="#db2777", marker="*",
                   s=110, zorder=7, edgecolors="white", linewidths=0.7)
        if show_legend:
            legend_handles.append(Line2D([0], [0], color="#db2777", linestyle=(0, (5, 2)), linewidth=1.8,
                                          label=f"Actual HOD (pre-entry): {tick_fmt.format(actual_hod)} @ {actual_hod_time.strftime('%H:%M:%S')}"))

    # Setup HOD (the armed/frozen level) and pullback low -- single events, marked
    # at the point where each occurred, not a line across the session.
    point_marker(hod_t, setup_hod, "#7c3aed", "^", 80, "Setup HOD (armed)")
    point_marker(pullback_t, result.pullback_low, "#0d9488", "v", 80, "Pullback low")

    # Trigger only matters up to the moment of entry (or session end if never filled).
    hline(result.trigger_price, session_start, entry_t or session_end, "#ea580c", "--", 1.1, "Trigger")

    # Target/stop only exist once a position is open -- span entry to exit.
    if entry_t is not None:
        span_end = exit_t or session_end
        hline(result.target_price, entry_t, span_end, "#16a34a", "--", 1.3, "Target")
        # candle_trail_1min ratchets the stop up under the 1-minute lows, so the
        # initial stop is not where the trade actually exited. When it trailed,
        # label the original as "Initial stop" and draw the final trail level too
        # (a static line at trail_stop_final -- the path isn't recorded, but this is
        # the effective protective stop the exit rode).
        trailed = (result.trail_stop_final is not None and result.stop_price is not None
                   and abs(float(result.trail_stop_final) - float(result.stop_price)) > 0.011)
        hline(result.stop_price, entry_t, span_end, "#dc2626", "--", 1.3, "Initial stop" if trailed else "Stop")
        if trailed:
            hline(result.trail_stop_final, entry_t, span_end, "#f59e0b", (0, (4, 2)), 1.5, "Trail stop (final)")

        # Vertical guides + holding-period shade so entry/exit times read exactly.
        ax.axvline(mdates.date2num(entry_t), color="#15803d", linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
        if exit_t is not None:
            ax.axvline(mdates.date2num(exit_t), color="#b91c1c", linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
            shade = "#16a34a" if result.result == "WIN" else ("#dc2626" if result.result == "LOSS" else "#9ca3af")
            ax.axvspan(mdates.date2num(entry_t), mdates.date2num(exit_t), color=shade, alpha=0.07, zorder=1)

    # Entry/exit -- the actual fill, marked precisely and tagged on the price axis.
    point_marker(entry_t, result.entry_price, "#15803d", "^", 150, "Entry")
    price_tag(result.entry_price, "#15803d", "Entry")

    legs = result.exit_legs if isinstance(result.exit_legs, list) else None
    if legs and len(legs) > 1:
        # Scale-out ladder: every leg was its own fill; the single exit_price is a
        # share-weighted average nothing traded at, so mark each leg.
        leg_style = {"target": ("Sold", "#15803d", "*"), "stop": ("Stopped", "#b91c1c", "X"),
                     "session_end": ("Closed", "#6b7280", "D")}
        for leg in legs:
            verb, color, marker = leg_style.get(leg.get("kind"), ("Exit", "#6b7280", "D"))
            point_marker(_parse_time(leg.get("time")), leg.get("price"), color, marker, 150,
                         f"{verb} {leg.get('frac', 0) * 100:g}%")
    else:
        exit_color = {"WIN": "#15803d", "LOSS": "#b91c1c"}.get(result.result, "#6b7280")
        exit_marker = "*" if result.result == "WIN" else ("X" if result.result == "LOSS" else "D")
        point_marker(exit_t, result.exit_price, exit_color, exit_marker, 170, "Exit")
        price_tag(result.exit_price, exit_color, "Exit")

    ax.set_ylabel("Price ($)")
    ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.3, zorder=0)
    if detail:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, axis="y", which="minor", linewidth=0.3, alpha=0.12, zorder=0)
    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.92)


def _draw_volume_at_entry_box(ax, result: TradeResult) -> None:
    """Small top-left annotation: cumulative session volume as of the actual
    entry (not the daily-average baseline) -- absent for trades that never
    reached an entry, so it's silently skipped rather than showing N/A."""
    if result.volume_at_entry is None:
        return
    ax.text(
        0.01, 0.98, f"DAILY VOL @ ENTRY\n{result.volume_at_entry:,.0f}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left", zorder=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#9ca3af", alpha=0.9),
    )


def _actual_hod_up_to(raw: pd.DataFrame, entry_t) -> tuple[float | None, object]:
    """The true running high-of-day through the entry (or None if no entry) --
    the reference for the stale-HOD diagnostic drawn in _draw_trade_panel."""
    if entry_t is None or raw.empty or "high" not in raw.columns:
        return None, None
    upto = raw[raw["dt_et"] <= entry_t]
    if upto.empty:
        return None, None
    idx = upto["high"].idxmax()
    return float(upto.loc[idx, "high"]), upto.loc[idx, "dt_et"]


def _write_chart(result: TradeResult, bars: pd.DataFrame, out: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    raw = _prep_bars(bars)                       # native 10-second bars (detail panel)
    if raw.empty:
        raise ValueError("no bars to chart")
    overview_candles = _resample_candles(bars)   # 1-minute candles (overview only)
    if overview_candles.empty:
        raise ValueError("no bars to chart after resampling")
    # The exact 1-minute EMA9/EMA20/VWAP the strategy's gates evaluated, overlaid
    # on both panels regardless of the candle resolution each one draws.
    indicators = _one_min_indicators(bars)

    session_start, session_end = raw["dt_et"].iloc[0], raw["dt_et"].iloc[-1]
    tz = getattr(raw["dt_et"].dt, "tz", None)
    zoom = _zoom_window(result, session_start, session_end)

    entry_t = _parse_time(result.entry_time)
    actual_hod, actual_hod_time = _actual_hod_up_to(raw, entry_t)

    subtitle_bits = [result.result]
    if result.entry_price is not None and result.exit_price is not None:
        subtitle_bits.append(f"entry {result.entry_price:.2f} -> exit {result.exit_price:.2f}")
    if result.pnl_per_share:
        subtitle_bits.append(f"P&L/sh {result.pnl_per_share:+.2f}")
    if result.trail_stop_final is not None and result.entry_price is not None:  # candle_trail
        subtitle_bits.append("runner (target dropped)" if result.trail_activated else "base hit (1:1)")
    title = f"{result.ticker}  {result.date}\n{'  |  '.join(subtitle_bits)}"

    if zoom is None:
        # Nothing eventful to zoom to (e.g. disqualified before any HOD/pullback
        # formed): a single detailed full-session view is all there is. Use the
        # 1-minute candles here since the whole multi-hour session is on screen.
        fig, ax = plt.subplots(figsize=(15, 7.5))
        _draw_trade_panel(ax, result, overview_candles, indicators, session_start, session_end,
                          show_legend=True, detail=True, actual_hod=actual_hod, actual_hod_time=actual_hod_time)
        _draw_volume_at_entry_box(ax, result)
        _setup_time_axis(ax, tz, detail=True)
        ax.set_xlabel("Time (ET)")
        ax.set_title(title, fontsize=11)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return

    zoom_lo, zoom_hi = zoom
    fig, (ax_overview, ax_zoom) = plt.subplots(
        2, 1, figsize=(15, 10.5), gridspec_kw={"height_ratios": [1, 2.7]}, sharex=False,
    )

    _draw_trade_panel(ax_overview, result, overview_candles, indicators, session_start, session_end,
                      show_legend=False, detail=False, actual_hod=actual_hod, actual_hod_time=actual_hod_time)
    _draw_volume_at_entry_box(ax_overview, result)
    ax_overview.axvspan(mdates.date2num(zoom_lo), mdates.date2num(zoom_hi), color="#2563eb", alpha=0.10, zorder=5)
    _setup_time_axis(ax_overview, tz, detail=False)
    ax_overview.set_title("Full premarket session (shaded: detail view below)", fontsize=9, color="#6b7280")

    # DETAIL: native 10-second bars inside the zoom window -- full sub-minute
    # precision on the HOD/pullback/entry/exit instead of 1-minute reaggregation.
    zoom_raw = raw[(raw["dt_et"] >= zoom_lo) & (raw["dt_et"] <= zoom_hi)]
    detail_candles = zoom_raw if not zoom_raw.empty else overview_candles
    zoom_ind = None
    if indicators is not None:
        zoom_ind = indicators[(indicators["dt_et"] >= zoom_lo) & (indicators["dt_et"] <= zoom_hi)]
    _draw_trade_panel(ax_zoom, result, detail_candles, zoom_ind, session_start, session_end,
                      show_legend=True, detail=True, actual_hod=actual_hod, actual_hod_time=actual_hod_time)
    ax_zoom.set_xlim(mdates.date2num(zoom_lo), mdates.date2num(zoom_hi))
    _setup_time_axis(ax_zoom, tz, detail=True)
    ax_zoom.set_xlabel("Time (ET)")
    ax_zoom.set_title("Setup & trade detail — native 10-second bars", fontsize=9, color="#6b7280")

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=200)
    plt.close(fig)
