"""The interactive HTML report (rebuild, 2026-07-23).

Self-contained by construction: inline SVG + CSS + a little vanilla JS, no CDN,
no external fonts, no matplotlib -- the file opens offline forever and can be
mailed as one artifact.

Chart/color method follows the dataviz skill. Categorical slots are used in
fixed order and never cycled: slot 1 blue = Dip, slot 2 orange = Wedge, slot 3
aqua = Combined. That 3-slot set was run through the palette validator in BOTH
modes on the all-pairs list (worst CVD dE 9.2 light / 9.4 dark, normal-vision
24.0 / 20.9 -- all PASS). Light-mode aqua sits below 3:1 on the light surface,
so the relief rule applies and is honored: every series is direct-labelled and
a table view exists for each chart's data. Status colors (good/critical) are
reserved for P&L sign and never reused as a series.

Design intent -- what changed and why (the audit that drove this):
  * the run is labelled by the strategies that ACTUALLY traded, never by a
    single config scalar (a 2-strategy run used to report as "dip_red_break");
  * every stat is available per strategy, because the two edges are inverted
    (dip ~68% win / small R vs wedge ~30-43% win / fat R) and a blended figure
    describes neither;
  * capital & leverage get a first-class section (peak/median leverage,
    concurrency, margin funded, which cap bound each trade) -- all of it was
    computed and none of it was displayed;
  * realized risk (max drawdown, Sharpe/Sortino, t-stat, tail concentration)
    replaces the previous "Monte-Carlo drawdown only" blind spot;
  * hold times and the exit-at-open cohort are tracked explicitly;
  * gate/scale-out cards that are structurally zero (feature disabled) are
    hidden rather than padding the page with noise.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

# --- palette (validated; see module docstring) --------------------------------
PALETTE = {
    "dip": ("#2a78d6", "#3987e5"),
    "wedge": ("#eb6834", "#d95926"),
    "combined": ("#1baf7a", "#199e70"),
}
STRAT_SLOT = {"dip_red_break": "dip", "wedge_break": "wedge"}


def _esc(v) -> str:
    return html.escape(str(v))


def _num(v, dec=2, suffix="", plus=False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "&mdash;"
    if isinstance(v, (int, float)):
        s = f"{v:+.{dec}f}" if plus else f"{v:,.{dec}f}"
        return f"{s}{suffix}"
    return _esc(v)


def _money(v) -> str:
    if v is None:
        return "&mdash;"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def _tile(label: str, value: str, sub: str = "", tone: str = "", title: str = "") -> str:
    t = f" title='{_esc(title)}'" if title else ""
    s = f"<div class='sub'>{sub}</div>" if sub else ""
    return (f"<div class='tile {tone}'{t}><div class='lab'>{_esc(label)}</div>"
            f"<div class='val'>{value}</div>{s}</div>")


def _tone(v, good_high=True) -> str:
    if v is None:
        return ""
    if v == 0:
        return ""
    pos = v > 0
    return "good" if (pos == good_high) else "bad"


# --- SVG chart primitives -----------------------------------------------------

def _svg_open(w: int, h: int, cls: str = "") -> str:
    return (f"<svg viewBox='0 0 {w} {h}' class='chart {cls}' role='img' "
            f"preserveAspectRatio='xMidYMid meet'>")


def equity_chart(curve: list[float], start: float, w=760, h=240) -> str:
    """Equity line with drawdown shading beneath the running peak. One series ->
    no legend box needed (the title names it); hover crosshair via JS."""
    if not curve or len(curve) < 2:
        return "<p class='muted'>Not enough trades for an equity curve.</p>"
    pad_l, pad_r, pad_t, pad_b = 56, 12, 14, 24
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    lo, hi = min(curve), max(curve)
    if hi == lo:
        hi = lo + 1
    rng = hi - lo
    lo -= rng * 0.08
    hi += rng * 0.08
    n = len(curve)

    def X(i):
        return pad_l + (i / (n - 1)) * iw

    def Y(v):
        return pad_t + (1 - (v - lo) / (hi - lo)) * ih

    peak, peaks = curve[0], []
    for v in curve:
        peak = max(peak, v)
        peaks.append(peak)
    line = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(curve))
    dd_top = " ".join(f"{X(i):.1f},{Y(p):.1f}" for i, p in enumerate(peaks))
    dd_bot = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in reversed(list(enumerate(curve))))
    grid = "".join(
        f"<line class='grid' x1='{pad_l}' x2='{w - pad_r}' y1='{Y(lo + (hi - lo) * f):.1f}' y2='{Y(lo + (hi - lo) * f):.1f}'/>"
        f"<text class='axl' x='{pad_l - 8}' y='{Y(lo + (hi - lo) * f) + 4:.1f}' text-anchor='end'>${lo + (hi - lo) * f:,.0f}</text>"
        for f in (0, 0.25, 0.5, 0.75, 1.0))
    pts = json.dumps([[round(X(i), 1), round(Y(v), 1), round(v, 2), i] for i, v in enumerate(curve)])
    return (
        f"{_svg_open(w, h, 'equity')}{grid}"
        f"<polygon class='ddfill' points='{dd_top} {dd_bot}'/>"
        f"<line class='base' x1='{pad_l}' x2='{w - pad_r}' y1='{Y(start):.1f}' y2='{Y(start):.1f}'/>"
        f"<text class='axl' x='{w - pad_r}' y='{Y(start) - 5:.1f}' text-anchor='end'>start ${start:,.0f}</text>"
        f"<polyline class='eqline' points='{line}'/>"
        f"<g class='hoverlayer' data-pts='{_esc(pts)}'><line class='xhair' style='display:none'/>"
        f"<circle class='dot' r='4' style='display:none'/></g>"
        f"</svg><div class='tip' hidden></div>")


def r_histogram(rs: list[float], w=380, h=200) -> str:
    """R distribution. Diverging by SIGN (loss/win), so status colors, gaps
    between bars, and a zero line -- not a rainbow."""
    if not rs:
        return "<p class='muted'>No R data.</p>"
    lo, hi = min(rs), max(rs)
    lo, hi = min(lo, -1.5), max(hi, 1.5)
    bins = 24
    step = (hi - lo) / bins
    counts = [0] * bins
    for r in rs:
        k = min(int((r - lo) / step), bins - 1)
        counts[k] += 1
    mx = max(counts) or 1
    pad_l, pad_b, pad_t = 32, 22, 8
    iw, ih = w - pad_l - 10, h - pad_b - pad_t
    bw = iw / bins
    bars = []
    for i, c in enumerate(counts):
        if not c:
            continue
        x = pad_l + i * bw
        bh = c / mx * ih
        mid = lo + (i + 0.5) * step
        cls = "bar-good" if mid > 0 else "bar-bad"
        bars.append(f"<rect class='{cls}' x='{x + 1:.1f}' y='{pad_t + ih - bh:.1f}' "
                    f"width='{max(bw - 2, 1):.1f}' height='{bh:.1f}' rx='2'>"
                    f"<title>{c} trades near {mid:+.2f}R</title></rect>")
    zx = pad_l + ((0 - lo) / (hi - lo)) * iw
    return (f"{_svg_open(w, h)}"
            f"<line class='grid' x1='{pad_l}' x2='{w - 10}' y1='{pad_t + ih}' y2='{pad_t + ih}'/>"
            f"{''.join(bars)}"
            f"<line class='zero' x1='{zx:.1f}' x2='{zx:.1f}' y1='{pad_t}' y2='{pad_t + ih}'/>"
            f"<text class='axl' x='{zx:.1f}' y='{h - 6}' text-anchor='middle'>0R</text>"
            f"<text class='axl' x='{pad_l}' y='{h - 6}'>{lo:.1f}R</text>"
            f"<text class='axl' x='{w - 10}' y='{h - 6}' text-anchor='end'>{hi:.1f}R</text>"
            f"</svg>")


def leverage_chart(trades: list[dict], cap: float | None, w=760, h=200) -> str:
    """Leverage at each entry, in chronological order, with the configured
    max_leverage cap as a reference line. Marks colored by STRATEGY (identity),
    direct-labelled via the legend beside it."""
    if not trades:
        return "<p class='muted'>No sized trades.</p>"
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 24
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    hi = max([t["lev"] for t in trades] + [cap or 1.0, 1.0]) * 1.12
    n = len(trades)

    def X(i):
        return pad_l + (i / max(n - 1, 1)) * iw

    def Y(v):
        return pad_t + (1 - v / hi) * ih

    grid = "".join(
        f"<line class='grid' x1='{pad_l}' x2='{w - pad_r}' y1='{Y(hi * f):.1f}' y2='{Y(hi * f):.1f}'/>"
        f"<text class='axl' x='{pad_l - 8}' y='{Y(hi * f) + 4:.1f}' text-anchor='end'>{hi * f:.1f}x</text>"
        for f in (0, 0.5, 1.0))
    dots = "".join(
        f"<circle class='lev-{t['slot']}' cx='{X(i):.1f}' cy='{Y(t['lev']):.1f}' r='4'>"
        f"<title>{_esc(t['ticker'])} {_esc(t['date'])} — {t['lev']:.2f}x"
        f"{' (margin)' if t['margin'] > 0 else ''} · {_esc(t['label'])}</title></circle>"
        for i, t in enumerate(trades))
    capline = ""
    if cap:
        capline = (f"<line class='cap' x1='{pad_l}' x2='{w - pad_r}' y1='{Y(cap):.1f}' y2='{Y(cap):.1f}'/>"
                   f"<text class='axl cap-lab' x='{w - pad_r}' y='{Y(cap) - 5:.1f}' text-anchor='end'>cap {cap:g}x</text>")
    one = (f"<line class='onex' x1='{pad_l}' x2='{w - pad_r}' y1='{Y(1.0):.1f}' y2='{Y(1.0):.1f}'/>"
           f"<text class='axl' x='{pad_l + 4}' y='{Y(1.0) - 4:.1f}'>1.0x — own cash</text>")
    return f"{_svg_open(w, h)}{grid}{one}{capline}{dots}</svg>"


def hbar_row(label: str, value: float, maxv: float, slot: str, right: str) -> str:
    pct = 0 if not maxv else max(min(value / maxv * 100.0, 100.0), 0.0)
    return (f"<div class='hb'><div class='hb-l'>{_esc(label)}</div>"
            f"<div class='hb-t'><div class='hb-f hb-{slot}' style='width:{pct:.1f}%'></div></div>"
            f"<div class='hb-r'>{right}</div></div>")


def tail_chart(sig: dict, w=380, h=200) -> str:
    """How much of net R survives removing the best N trades -- the single most
    important robustness read for a runner-driven strategy."""
    keys = [("all", sig.get("mean_r")), ("excl top 1", sig.get("mean_r_excl_top1")),
            ("excl top 3", sig.get("mean_r_excl_top3")), ("excl top 5", sig.get("mean_r_excl_top5"))]
    keys = [(k, v) for k, v in keys if v is not None]
    if not keys:
        return "<p class='muted'>Not enough trades.</p>"
    mx = max(abs(v) for _, v in keys) or 1
    rows = "".join(
        f"<div class='hb'><div class='hb-l'>{_esc(k)}</div>"
        f"<div class='hb-t'><div class='hb-f {"hb-good" if v > 0 else "hb-bad"}' "
        f"style='width:{abs(v) / mx * 100:.1f}%'></div></div>"
        f"<div class='hb-r'>{v:+.3f}R</div></div>" for k, v in keys)
    return f"<div class='hbars'>{rows}</div>"
