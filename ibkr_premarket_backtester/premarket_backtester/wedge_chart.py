"""Wedge-strategy trade chart: a purpose-built, dark (TWS-style) two-panel image
for symmetrical-pennant trades -- deliberately separate from reporting._write_chart
(the dip's chart, left untouched).

Layout: 1-MINUTE candles on top (context), the native 10-SECOND candles zoomed in
below (detail). Both panels carry the same overlays: the pennant's two converging
trendlines (descending resistance, ascending support), the entry / stop / take-
profit as dotted lines (white / red / green), and the 1-minute EMA9 (blue) / EMA20
(grey) / VWAP (yellow) the strategy evaluated -- drawn on both panels regardless of
candle resolution. Charts are filed under results/<run>/wedge/{wins,losses}/.

Reuses the validated bar/axis primitives from reporting (bar prep, resample,
1-minute indicators, the time-axis formatter) but does its own dark drawing.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .models import TradeResult
from .reporting import (
    _bar_width_days, _one_min_indicators, _parse_time, _prep_bars,
    _resample_candles, _setup_time_axis, trade_folder_name,
)

# Render resolution. High DPI + a large figure so the image stays crisp at full
# size / when zoomed (matplotlib scales fonts/line widths by points, so raising
# DPI only sharpens -- it doesn't change the layout).
_DPI = 220
_FIGSIZE = (15.0, 12.5)

# Dark TWS/TradingView-style palette.
_BG = "#0e1117"
_PANEL = "#131722"
_INK = "#d1d4dc"
_GRID = "#2a2e39"
_UP = "#26a69a"
_DOWN = "#ef5350"
_EMA9 = "#2962ff"      # blue
_EMA20 = "#b2b5be"     # grey
_VWAP = "#f5c518"      # yellow
_TREND = "#c678dd"     # pennant trendlines (purple)
_ENTRY = "#ffffff"     # entry (white dotted)
_STOP = "#ef5350"      # stop (red dotted)
_TARGET = "#26a69a"    # take-profit (green dotted)


def _wedge_zoom(result: TradeResult, session_start, session_end, pad_minutes: int = 3):
    """Time window around the pennant + trade (line start -> break -> entry -> exit),
    padded for context. None when there's nothing to zoom to."""
    times = [_parse_time(t) for t in (
        result.wedge_line_start_time, result.wedge_line_end_time,
        result.entry_time, result.exit_time,
    )]
    times = [t for t in times if t is not None]
    if not times:
        return None
    pad = pd.Timedelta(minutes=pad_minutes)
    lo = max(session_start, min(times) - pad)
    hi = min(session_end, max(times) + pad)
    return (lo, hi) if hi > lo else None


def _trade_zoom(result: TradeResult, session_start, session_end):
    """A TIGHT window around the execution itself (entry -> exit) for the detail
    panel, padded by ~half the hold (min 45s) so the break into entry and the exit
    both breathe. None when there's no entry to zoom to."""
    et, xt = _parse_time(result.entry_time), _parse_time(result.exit_time)
    if et is None:
        return None
    xt = xt if xt is not None else et
    pad = max((xt - et) * 0.5, pd.Timedelta(seconds=45))
    lo = max(session_start, et - pad)
    hi = min(session_end, xt + pad)
    return (lo, hi) if hi > lo else None


def _draw_candles(ax, candles: pd.DataFrame) -> None:
    """Dark candlesticks (no volume twin, to keep the theme clean). Width scales to
    the median bar interval so 1-minute and 10-second panels both render crisply."""
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    if candles.empty:
        return
    half = _bar_width_days(candles) * 0.35
    for _, c in candles.iterrows():
        t = mdates.date2num(c["dt_et"])
        o, h, lo, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        color = _UP if cl >= o else _DOWN
        ax.add_line(Line2D([t, t], [lo, h], color=color, linewidth=0.9, alpha=0.85, zorder=2, solid_capstyle="butt"))
        ax.add_patch(Rectangle((t - half, min(o, cl)), 2 * half, max(abs(cl - o), 1e-4),
                               facecolor=color, edgecolor=color, linewidth=0.5, zorder=3))
    ax.xaxis_date(tz=getattr(candles["dt_et"].dt, "tz", None))


def _fmt_vol(v: float) -> str:
    v = float(v)
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def _draw_volume(vax, candles, tz, detail: bool, xlim) -> None:
    """A slim volume sub-panel beneath a price panel: bars coloured by candle
    direction, a single compact max-label on the y-axis, and the shared time axis
    (the price panel above hides its x labels so time reads once, here)."""
    import matplotlib.dates as mdates
    vax.set_facecolor(_PANEL)
    if candles is not None and not candles.empty and "volume" in candles.columns and float(candles["volume"].max() or 0) > 0:
        half = _bar_width_days(candles) * 0.35
        x = mdates.date2num(candles["dt_et"])
        colors = [_UP if c >= o else _DOWN for o, c in zip(candles["open"], candles["close"])]
        vax.bar(x, candles["volume"].astype(float), width=2 * half, color=colors, alpha=0.65, linewidth=0, zorder=2)
        vmax = float(candles["volume"].max())
        vax.set_ylim(0, vmax * 1.15)
        vax.set_yticks([vmax])
        vax.set_yticklabels([_fmt_vol(vmax)])
    else:
        vax.set_yticks([])
    vax.set_xlim(mdates.date2num(xlim[0]), mdates.date2num(xlim[1]))
    _setup_time_axis(vax, tz, detail)
    vax.set_ylabel("Vol", color=_INK, fontsize=9.5)
    vax.tick_params(colors=_INK, labelsize=9)
    for spine in vax.spines.values():
        spine.set_color(_GRID)


def _draw_wedge_panel(ax, candles, indicators, result: TradeResult, tz, detail: bool, xlim, panel_title: str, wide: bool = True) -> None:
    import matplotlib.dates as mdates
    ax.set_facecolor(_PANEL)
    _draw_candles(ax, candles)

    # 1-minute EMA9/EMA20/VWAP overlaid on BOTH panels (their real 1-minute x).
    if indicators is not None and not indicators.empty:
        x = mdates.date2num(indicators["dt_et"])
        for col, color, lbl, lw in (("ema9", _EMA9, "EMA9", 1.6), ("ema20", _EMA20, "EMA20", 1.6), ("vwap", _VWAP, "VWAP", 2.0)):
            if col in indicators:
                ax.plot(x, indicators[col], color=color, linewidth=lw, label=lbl, zorder=4, alpha=0.95)

    # The pennant's two converging trendlines.
    t0, t1 = _parse_time(result.wedge_line_start_time), _parse_time(result.wedge_line_end_time)
    if t0 is not None and t1 is not None and result.wedge_res_start is not None:
        xs = [mdates.date2num(t0), mdates.date2num(t1)]
        ax.plot(xs, [result.wedge_res_start, result.wedge_res_end], color=_TREND, linestyle="--", linewidth=2.0, zorder=5, label="Resistance")
        ax.plot(xs, [result.wedge_sup_start, result.wedge_sup_end], color=_TREND, linestyle="--", linewidth=2.0, zorder=5, label="Support")

    # Entry / stop / take-profit levels drawn ONLY during the trade (entry -> exit),
    # not across the whole panel -- with vertical BARRIERS marking the entry and exit
    # times and a light holding-period shade, so each event's timing reads at a glance
    # (inspired by the dip chart). Right-edge tags keep the price levels readable.
    et, xt = _parse_time(result.entry_time), _parse_time(result.exit_time)
    if et is not None and result.entry_price is not None:
        won = (result.pnl_per_share or 0) >= 0
        exit_color = _TARGET if won else _STOP
        x0 = mdates.date2num(et)
        x1 = mdates.date2num(xt) if xt is not None else mdates.date2num(xlim[1])
        if xt is not None:
            ax.axvspan(x0, x1, color=exit_color, alpha=0.06, zorder=1)                      # holding period
            ax.axvline(x1, color=exit_color, linestyle=(0, (2, 2)), linewidth=1.3, alpha=0.7, zorder=5)   # exit barrier
        ax.axvline(x0, color=_ENTRY, linestyle=(0, (2, 2)), linewidth=1.3, alpha=0.65, zorder=5)          # entry barrier
        for y, color, lbl in ((result.stop_price, _STOP, "Stop"), (result.target_price, _TARGET, "TP"), (result.entry_price, _ENTRY, "Entry")):
            if y is not None:
                ax.hlines(float(y), x0, x1, colors=color, linestyles=":", linewidth=1.8, zorder=6, alpha=0.95)
                ax.annotate(f" {lbl} {float(y):.2f}", xy=(1.0, float(y)), xycoords=("axes fraction", "data"),
                            xytext=(3, 0), textcoords="offset points", va="center", ha="left",
                            fontsize=9, color=color, fontweight="bold", annotation_clip=False)
        ax.scatter([x0], [result.entry_price], marker=("^" if result.direction == "long" else "v"),
                   s=180, color=_ENTRY, edgecolors="black", linewidths=0.9, zorder=8)
        if xt is not None and result.exit_price is not None:
            ax.scatter([x1], [result.exit_price], marker="X", s=180, color=exit_color, edgecolors="black", linewidths=0.9, zorder=8)

    # Y-limits framed on the TRADE levels (and the pennant only on the wide panel),
    # padded for context -- NOT the raw candle high/low, so a single 10-second wick
    # spike (common on low-float premarket names) can't compress the trade zone into
    # a sliver. Extreme wicks simply clip, as on a price-zoomed TWS chart.
    levels = [v for v in (result.entry_price, result.stop_price, result.target_price, result.exit_price) if v is not None]
    if wide:
        levels += [v for v in (result.wedge_res_start, result.wedge_res_end, result.wedge_sup_start, result.wedge_sup_end) if v is not None]
    if levels:
        ylo, yhi = float(min(levels)), float(max(levels))
        span = (yhi - ylo) or max(ylo * 0.02, 0.05)
        ax.set_ylim(ylo - span * 0.32, yhi + span * 0.32)
    ax.set_xlim(mdates.date2num(xlim[0]), mdates.date2num(xlim[1]))
    _setup_time_axis(ax, tz, detail)
    ax.set_ylabel("Price ($)", color=_INK, fontsize=11)
    ax.tick_params(colors=_INK, labelsize=9.5)
    ax.tick_params(axis="x", labelbottom=False)   # time reads on the volume panel below
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_title(panel_title, color=_INK, fontsize=11, loc="left", pad=5)


def _write_wedge_chart(result: TradeResult, bars: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    raw = _prep_bars(bars)                       # native 10-second bars (detail panel)
    if raw.empty:
        raise ValueError("no bars to chart")
    one_min = _resample_candles(bars)            # 1-minute candles (overview panel)
    indicators = _one_min_indicators(bars)       # 1-minute EMA9/EMA20/VWAP
    session_start, session_end = raw["dt_et"].iloc[0], raw["dt_et"].iloc[-1]
    tz = getattr(raw["dt_et"].dt, "tz", None)
    # Two windows: the top (1-minute) shows the pennant + trade in context; the
    # bottom (10-second) zooms TIGHT on the execution (entry -> exit).
    wide = _wedge_zoom(result, session_start, session_end) or (session_start, session_end)
    tight = _trade_zoom(result, session_start, session_end) or wide
    margin = pd.Timedelta(minutes=1)             # let lines/indicators reach the panel edges

    def _win(df, window):
        if df is None or df.empty:
            return df
        return df[(df["dt_et"] >= window[0] - margin) & (df["dt_et"] <= window[1] + margin)]

    top, ind_top = _win(one_min, wide), _win(indicators, wide)
    bottom, ind_bot = _win(raw, tight), _win(indicators, tight)

    # Figure: two price panels, each with a slim volume sub-panel ATTACHED beneath
    # it (nested gridspecs -- small gap within a group, a wider gap between groups).
    # Margins set here (no tight_layout) so the right-edge price tags have room.
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI, facecolor=_BG)
    outer = fig.add_gridspec(2, 1, hspace=0.17, left=0.065, right=0.905, top=0.915, bottom=0.058)
    g_top = outer[0].subgridspec(2, 1, height_ratios=[3.2, 0.8], hspace=0.04)
    g_bot = outer[1].subgridspec(2, 1, height_ratios=[3.2, 0.8], hspace=0.04)
    ax1, vax1 = fig.add_subplot(g_top[0]), fig.add_subplot(g_top[1])
    ax2, vax2 = fig.add_subplot(g_bot[0]), fig.add_subplot(g_bot[1])

    direction = (result.direction or "long").upper()
    bits = [direction, result.result]
    if result.entry_price is not None and result.exit_price is not None:
        bits.append(f"{result.entry_price:.2f} → {result.exit_price:.2f}")
    if result.pnl_per_share:
        bits.append(f"{result.pnl_per_share:+.2f}/sh")
    if result.r_multiple is not None:
        bits.append(f"{result.r_multiple:+.2f}R")
    # Cumulative session volume as of the entry (the strategy's volume_at_entry).
    if result.volume_at_entry is not None:
        bits.append(f"vol@entry {_fmt_vol(result.volume_at_entry)}")
    fig.suptitle(f"{result.ticker}   {result.date}   · wedge\n" + "   |   ".join(bits),
                 color=_INK, fontsize=15.5, fontweight="bold")

    _draw_wedge_panel(ax1, top, ind_top, result, tz, detail=False, xlim=wide, panel_title="1-minute · pennant + trade", wide=True)
    _draw_volume(vax1, top, tz, detail=False, xlim=wide)
    _draw_wedge_panel(ax2, bottom, ind_bot, result, tz, detail=True, xlim=tight, panel_title="10-second · zoom on the trade", wide=False)
    _draw_volume(vax2, bottom, tz, detail=True, xlim=tight)
    vax2.set_xlabel("Time (ET)", color=_INK, fontsize=11)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        seen = dict(zip(labels, handles))       # de-dupe (both trendlines etc.)
        leg = ax1.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9.5,
                         facecolor=_PANEL, edgecolor=_GRID, labelcolor=_INK, framealpha=0.92)
        leg.set_zorder(20)

    fig.savefig(out, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


def write_wedge_trade_folder(result: TradeResult, bars: pd.DataFrame, run_dir: Path, write_chart: bool = True) -> None:
    """Wedge counterpart to reporting.write_trade_folder: trade.json + bars.csv in
    the usual trade_folders/<name>/ (so the HTML report still links a chart.png),
    plus the chart filed under the strategy's own results/<run>/wedge/{wins,losses}/."""
    safe = trade_folder_name(result.ticker, result.date, result.setup_index, result.entry_mode)
    d = run_dir / "trade_folders" / safe
    d.mkdir(parents=True, exist_ok=True)
    (d / "trade.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if bars is not None and not bars.empty:
        bars.to_csv(d / "bars.csv", index=False)
    if write_chart and bars is not None and not bars.empty:
        try:
            chart_path = d / "chart.png"
            _write_wedge_chart(result, bars, chart_path)
            if result.entry_price is not None:
                sub = "wins" if (result.pnl_per_share or 0) >= 0 else "losses"
                dest = run_dir / "wedge" / sub
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(chart_path, dest / f"{safe}.png")
        except Exception as e:
            (d / "chart_error.txt").write_text(repr(e), encoding="utf-8")
