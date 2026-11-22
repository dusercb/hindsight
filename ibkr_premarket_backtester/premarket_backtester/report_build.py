"""Assembles the interactive report page from a summary dict + trade rows.

Section order follows what a quant reads first: verdict -> can I trust it ->
what did each strategy do -> what capital did it demand -> how did it exit ->
the rows. Anything structurally zero (a disabled gate, an unused ladder) is
omitted rather than rendered as a 0 that implies the feature ran.
"""
from __future__ import annotations

import html
import json

import pandas as pd

from .report_html import (
    PALETTE, STRAT_SLOT, _esc, _money, _num, _tile, _tone,
    equity_chart, hbar_row, leverage_chart, r_histogram, tail_chart,
)
from .report_style import CSS, JS


def _sig_callout(sig: dict, lev: dict) -> str:
    """The honesty box. A trailing-exit strategy lives on rare runners, so the
    t-stat and tail concentration belong ON the headline, not buried."""
    if not sig or sig.get("n", 0) < 5:
        return ""
    bits = []
    t = sig.get("t_stat")
    lo, hi = sig.get("ci95_low"), sig.get("ci95_high")
    if t is not None:
        verdict = ("<b>not statistically separable from zero</b> at 95%"
                   if (lo is not None and lo <= 0) else "<b>positive at 95% confidence</b>")
        bits.append(f"Mean R {sig['mean_r']:+.3f} on n={sig['n']} — t={t:.2f}, 95% CI "
                    f"[{lo:+.3f}, {hi:+.3f}] — {verdict}.")
    top1 = sig.get("top1_pct_of_sum_r")
    if top1 is not None:
        bits.append(f"The single best trade is <b>{top1:.0f}% of total R</b>; "
                    f"excluding the top 3 the mean falls to "
                    f"{sig.get('mean_r_excl_top3', float('nan')):+.3f}R — the edge is "
                    f"tail-driven, so live slippage on the few big winners matters most.")
    cls = "callout crit" if (lo is not None and lo <= 0) else "callout"
    return f"<div class='{cls}'>{' '.join(bits)}</div>"


def _headline(summary: dict, q: dict) -> str:
    risk = q.get("risk", {})
    comb = q.get("combined", {})
    sig = q.get("significance", {})
    net = summary.get("net_pnl_after_fees", summary.get("net_pnl"))
    tiles = [
        _tile("Net P&L", _money(net), f"{_num(risk.get('return_pct'), 1, '%', True)} on "
              f"{_money(risk.get('starting_equity'))}", _tone(net)),
        _tile("Ending equity", _money(risk.get("ending_equity"))),
        _tile("Trades", str(comb.get("n", 0)),
              f"{q.get('interaction', {}).get('trading_days', 0)} trading days"),
        _tile("Win rate", _num(comb.get("win_rate_pct"), 1, "%"),
              f"by realized profit ({comb.get('wins', 0)}W/{comb.get('losses', 0)}L)",
              title="Counts every trade that MADE MONEY, including EXIT_AT_OPEN "
                    "rows that carry no WIN label."),
        _tile("Expectancy", _num(comb.get("mean_r"), 3, "R", True), "realized mean R/trade",
              _tone(comb.get("mean_r"))),
        _tile("Max drawdown", _num(risk.get("max_drawdown_pct"), 1, "%"),
              f"realized · {_money(risk.get('max_drawdown_abs'))}", "bad"),
        _tile("Sharpe", _num(risk.get("sharpe"), 2), f"Sortino {_num(risk.get('sortino'), 2)}"),
        _tile("Profit factor", _num(comb.get("profit_factor"), 2)),
        _tile("t-stat", _num(sig.get("t_stat"), 2),
              "≥1.96 ⇒ 95% confident", "warn" if (sig.get("t_stat") or 0) < 1.96 else "good"),
        _tile("Return / MaxDD", _num(risk.get("return_over_max_dd"), 2)),
    ]
    return f"<div class='tiles'>{''.join(tiles)}</div>{_sig_callout(sig, q.get('leverage', {}))}"


def _strategy_section(q: dict) -> str:
    per = q.get("per_strategy", {})
    if not per:
        return ""
    inter = q.get("interaction", {})
    cols = list(per.items())
    rows = [
        ("Trades", lambda s: str(s.get("n", 0))),
        ("Win rate (profit)", lambda s: _num(s.get("win_rate_pct"), 1, "%")),
        ("Mean R", lambda s: _num(s.get("mean_r"), 3, "R", True)),
        ("Median R", lambda s: _num(s.get("median_r"), 3, "R", True)),
        ("Total R", lambda s: _num(s.get("sum_r"), 1, "R", True)),
        ("Avg win / avg loss", lambda s: f"{_num(s.get('avg_win_r'), 2, 'R', True)} / {_num(s.get('avg_loss_r'), 2, 'R', True)}"),
        ("Best / worst", lambda s: f"{_num(s.get('best_r'), 2, 'R', True)} / {_num(s.get('worst_r'), 2, 'R', True)}"),
        ("Net P&amp;L", lambda s: _money(s.get("net_pnl"))),
        ("Profit factor", lambda s: _num(s.get("profit_factor"), 2)),
        ("Avg hold", lambda s: _num(s.get("avg_hold_min"), 1, " min")),
        ("Median hold", lambda s: _num(s.get("median_hold_min"), 1, " min")),
        ("Longest hold", lambda s: (f"{_num(s.get('max_hold_min'), 1, ' min')}"
                                    f"<div class='muted sm'>{_esc(s.get('longest_hold_trade', ''))}</div>")),
        ("Held to open", lambda s: (f"{s.get('exit_at_open', 0)} ({_num(s.get('exit_at_open_pct'), 0, '%')})"
                                    + (f"<div class='muted sm'>{_num(s.get('exit_at_open_mean_r'), 2, 'R', True)} avg, "
                                       f"{s.get('exit_at_open_wins', 0)} green</div>" if s.get("exit_at_open") else ""))),
        ("t-stat", lambda s: _num((s.get("significance") or {}).get("t_stat"), 2)),
    ]
    head = "".join(
        f"<th><span class='sw' style='display:inline-block;width:8px;height:8px;border-radius:2px;"
        f"margin-right:5px;background:var(--{STRAT_SLOT.get(m, 'combined')})'></span>{_esc(s.get('label', m))}</th>"
        for m, s in cols)
    body = "".join(
        f"<tr><td>{lab}</td>" + "".join(f"<td>{fn(s)}</td>" for _, s in cols) + "</tr>"
        for lab, fn in rows)
    corr = ""
    if inter.get("daily_r_correlation") is not None:
        c = inter["daily_r_correlation"]
        read = ("genuinely diversifying — they lose on different days"
                if c < 0.2 else ("mildly correlated" if c < 0.5 else
                                 "<b>highly correlated</b> — much less diversification than the trade count suggests"))
        corr = (f"<div class='callout'>Daily-R correlation between strategies: <b>{c:+.3f}</b> "
                f"over {inter.get('days_both', 0)} days both traded — {read}. "
                f"Daily R: mean {inter.get('daily_r_mean', 0):+.2f}, sd {inter.get('daily_r_sd', 0) or 0:.2f}, "
                f"worst {inter.get('worst_day_r', 0):+.2f}, best {inter.get('best_day_r', 0):+.2f}.</div>")
    return (f"<h2>Per-strategy performance</h2>"
            f"<p class='note'>The two edges are structurally different — a blended figure describes neither.</p>"
            f"<div class='panel'><table><thead><tr><th>Metric</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>{corr}")


def _capital_section(q: dict, cfg: dict | None) -> str:
    lev = q.get("leverage", {})
    if not lev:
        return ""
    cap = None
    if cfg:
        cap = (cfg.get("risk") or {}).get("max_leverage")
    basis = lev.get("size_basis_counts", {})
    total_basis = sum(basis.values()) or 1
    basis_labels = {"risk": "Risk % cap", "notional": "Notional % cap",
                    "volume": "Volume participation", "funding": "Funding/leverage cap",
                    "shares": "Fixed share count"}
    bars = "".join(
        hbar_row(basis_labels.get(k, k), v, total_basis, "neutral",
                 f"{v} ({v / total_basis * 100:.0f}%)")
        for k, v in sorted(basis.items(), key=lambda kv: -kv[1]))
    tiles = [
        _tile("Peak leverage", _num(lev.get("peak_leverage"), 2, "x"),
              f"cap {cap:g}x" if cap else "no cap configured",
              "warn" if (cap and (lev.get("peak_leverage") or 0) > cap * 0.95) else ""),
        _tile("Median leverage", _num(lev.get("median_leverage"), 2, "x"),
              f"avg {_num(lev.get('avg_leverage'), 2, 'x')}"),
        _tile("Trades above 1.0x", str(lev.get("trades_above_1x", 0)), "used borrowed capital"),
        _tile("Max concurrent", str(lev.get("max_concurrent_positions", 0)),
              f"{lev.get('concurrent_entries', 0)} entries overlapped"),
        _tile("Margin-funded", str(lev.get("margin_funded_entries", 0)),
              f"peak {_money(lev.get('margin_used_max'))}"),
        _tile("Margin borrowed", _money(lev.get("margin_used_total")), "sum across entries"),
    ]
    return (f"<h2>Capital &amp; leverage</h2>"
            f"<p class='note'>What the account actually had to fund — concurrency is what makes "
            f"two strategies cost more than one.</p>"
            f"<div class='tiles'>{''.join(tiles)}</div>"
            f"<div class='row'><div class='panel'><h3>Leverage at each entry</h3>{_LEV_PLACEHOLDER}</div>"
            f"<div class='panel'><h3>Which cap set the position size</h3>"
            f"<div class='hbars'>{bars}</div>"
            f"<p class='note'>The binding constraint is where your sizing actually lives — "
            f"tuning any other cap changes nothing.</p></div></div>")


_LEV_PLACEHOLDER = "@@LEVCHART@@"


def _exits_section(q: dict) -> str:
    ex = q.get("exits", {})
    if not ex:
        return ""
    total = sum(ex.values()) or 1
    labels = {"target": "Target hit", "stop": "Stop hit", "trail_stop": "Trailing stop",
              "session_end": "Held to open", "time_stop": "Time stop",
              "flat_by_time": "Flat-by time", "stale_progress": "Stale progress"}
    bars = "".join(
        hbar_row(labels.get(k, k), v, total,
                 "good" if k == "target" else ("bad" if k in ("stop", "trail_stop") else "neutral"),
                 f"{v} ({v / total * 100:.0f}%)")
        for k, v in sorted(ex.items(), key=lambda kv: -kv[1]))
    comb = q.get("combined", {})
    eao = comb.get("exit_at_open", 0)
    note = ""
    if eao:
        note = (f"<div class='callout'>{eao} trades were <b>held into the 09:30 open</b> "
                f"({comb.get('exit_at_open_pct', 0):.0f}% of the run), averaging "
                f"<b>{comb.get('exit_at_open_mean_r', 0):+.2f}R</b> with "
                f"{comb.get('exit_at_open_wins', 0)} green. These carry no WIN/LOSS label — they are "
                f"counted as wins here only when they actually made money.</div>")
    return (f"<h2>How trades ended</h2>"
            f"<div class='panel'><div class='hbars'>{bars}</div></div>{note}")


def _dist_section(q: dict, rs: list[float]) -> str:
    sig = q.get("significance", {})
    return (f"<h2>Distribution &amp; robustness</h2>"
            f"<div class='row'>"
            f"<div class='panel'><h3>R-multiple distribution</h3>{r_histogram(rs)}"
            f"<p class='note'>Green = profitable trades, red = losing. Zero line marked.</p></div>"
            f"<div class='panel'><h3>Mean R after removing the best trades</h3>{tail_chart(sig)}"
            f"<p class='note'>How much of the edge survives without its luckiest outcomes.</p></div>"
            f"</div>")


def _funnel_section(summary: dict) -> str:
    """Funnel + gates, with every structurally-zero gate omitted: a disabled
    filter rendering '0' reads as 'ran and never fired', which is false."""
    funnel = [("Candidates scanned", "candidates"), ("Qualified", "qualified_or_traded"),
              ("Traded", "trades_taken")]
    fr = "".join(f"<tr><td>{lab}</td><td>{_num(summary.get(k), 0)}</td></tr>" for lab, k in funnel)
    dq = [("No RVOL history", "insufficient_rvol_history"),
          ("Failed RVOL", "did_not_qualify_rvol"),
          ("Failed gain %", "did_not_qualify_gain_pct"),
          ("Failed premarket volume", "did_not_qualify_premarket_volume")]
    dqr = "".join(f"<tr><td>{lab}</td><td>{_num(summary.get(k), 0)}</td></tr>"
                  for lab, k in dq if summary.get(k))
    gates = [(k, v) for k, v in summary.items()
             if k.startswith("skipped_") and k.endswith("_total") and v]
    gr = "".join(f"<tr><td>{_esc(k[8:-6].replace('_', ' ').title())}</td><td>{_num(v, 0)}</td></tr>"
                 for k, v in sorted(gates, key=lambda kv: -kv[1]))
    hidden = sum(1 for k, v in summary.items()
                 if k.startswith("skipped_") and k.endswith("_total") and not v)
    tail = (f"<p class='note'>{hidden} gate counters are zero (feature disabled or never fired) "
            f"and are hidden.</p>" if hidden else "")
    return (f"<h2>Funnel &amp; gates</h2><div class='row'>"
            f"<div class='panel'><h3>Candidate funnel</h3><table><tbody>{fr}{dqr}</tbody></table></div>"
            f"<div class='panel'><h3>Entry gates that fired</h3>"
            f"<table><tbody>{gr or '<tr><td class=muted>None fired</td><td></td></tr>'}</tbody></table>"
            f"{tail}</div></div>")


def _trade_table(df: pd.DataFrame) -> str:
    """Rows with the columns that were missing: leverage, size basis, hold,
    slippage, MFE/MAE, exit path. Filterable by strategy."""
    t = df[df["entry_price"].notna()].copy() if "entry_price" in df else pd.DataFrame()
    if t.empty:
        return ""
    for c in ("entry_time", "exit_time"):
        if c in t:
            t[c + "_p"] = pd.to_datetime(t[c], utc=True, errors="coerce")
    if "entry_time_p" in t and "exit_time_p" in t:
        t["hold_min"] = (t["exit_time_p"] - t["entry_time_p"]).dt.total_seconds() / 60
    t = t.sort_values(["date", "entry_time"], na_position="last")
    cols = [("date", "Date"), ("ticker", "Ticker"), ("entry_mode", "Strat"), ("direction", "Dir"),
            ("entry_price", "Entry"), ("stop_price", "Stop"), ("target_price", "Target"),
            ("exit_price", "Exit"), ("exit_trigger", "Exit via"), ("shares", "Shares"),
            ("size_basis", "Size cap"), ("leverage_at_entry", "Lev"), ("margin_used", "Margin"),
            ("open_positions_at_entry", "Conc"), ("hold_min", "Hold m"), ("slippage", "Slip"),
            ("mfe", "MFE"), ("mae", "MAE"), ("r_multiple", "R"), ("pnl_total", "P&L"),
            ("equity_after", "Equity")]
    cols = [(c, lab) for c, lab in cols if c in t.columns]
    head = "".join(f"<th>{_esc(lab)}</th>" for _, lab in cols)
    body = []
    for _, r in t.iterrows():
        mode = r.get("entry_mode", "")
        cells = []
        for c, _lab in cols:
            v = r.get(c)
            if c == "entry_mode":
                cls = "tagd" if mode == "dip_red_break" else "tagw"
                cells.append(f"<td class='{cls}'>{'Dip' if mode == 'dip_red_break' else 'Wedge'}</td>")
            elif c in ("r_multiple", "pnl_total"):
                cls = "pos" if (pd.notna(v) and v > 0) else ("neg" if (pd.notna(v) and v < 0) else "")
                dec = 2 if c == "r_multiple" else 0
                cells.append(f"<td class='{cls}'>{_num(v if pd.notna(v) else None, dec, '', True)}</td>")
            elif c in ("leverage_at_entry",):
                cells.append(f"<td>{_num(v if pd.notna(v) else None, 2, 'x')}</td>")
            elif c in ("margin_used", "equity_after"):
                cells.append(f"<td>{_money(v) if pd.notna(v) else '&mdash;'}</td>")
            elif c in ("hold_min",):
                cells.append(f"<td>{_num(v if pd.notna(v) else None, 1)}</td>")
            elif isinstance(v, float) and pd.notna(v):
                cells.append(f"<td>{_num(v, 2)}</td>")
            else:
                cells.append(f"<td>{_esc(v) if pd.notna(v) else '&mdash;'}</td>")
        body.append(f"<tr data-mode='{_esc(mode)}'>{''.join(cells)}</tr>")
    return (f"<h2>Trades <span class='muted sm'>({len(t)})</span></h2>"
            f"<div class='ctrls'>"
            f"<button class='f' data-filter='all' aria-pressed='true'>All</button>"
            f"<button class='f' data-filter='dip_red_break'>Dip</button>"
            f"<button class='f' data-filter='wedge_break'>Wedge</button></div>"
            f"<div class='scroll'><table id='tradeTable'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def build_page(summary: dict, df: pd.DataFrame, config_snapshot: dict | None,
               stamp: str, monte_carlo=None) -> str:
    q = summary.get("quant") or {}
    risk = q.get("risk", {})
    strategies = summary.get("strategies_traded") or []
    chips = [f"<span class='chip'><span class='sw' style='background:var(--{STRAT_SLOT.get(m, 'combined')})'></span>"
             f"{_esc((q.get('per_strategy', {}).get(m, {}) or {}).get('label', m))} "
             f"{(q.get('per_strategy', {}).get(m, {}) or {}).get('n', 0)}</span>" for m in strategies]
    for k in ("exit_mode", "stop_mode", "target_mode"):
        if summary.get(k):
            chips.append(f"<span class='chip'>{_esc(k.replace('_mode', ''))}: {_esc(summary[k])}</span>")
    rs = []
    if "r_multiple" in df:
        rs = [float(x) for x in pd.to_numeric(df["r_multiple"], errors="coerce").dropna()]
    lev_trades = []
    if "leverage_at_entry" in df:
        d = df[df["entry_price"].notna()].sort_values(["date", "entry_time"], na_position="last")
        for _, r in d.iterrows():
            if pd.notna(r.get("leverage_at_entry")):
                m = r.get("entry_mode", "")
                lev_trades.append({"lev": float(r["leverage_at_entry"]),
                                   "slot": STRAT_SLOT.get(m, "combined"),
                                   "label": "Dip" if m == "dip_red_break" else "Wedge",
                                   "ticker": r.get("ticker", ""), "date": r.get("date", ""),
                                   "margin": float(r.get("margin_used") or 0)})
    cap = ((config_snapshot or {}).get("risk") or {}).get("max_leverage")
    capital = _capital_section(q, config_snapshot).replace(
        _LEV_PLACEHOLDER,
        leverage_chart(lev_trades, cap) +
        "<p class='note'><span class='chip'><span class='sw' style='background:var(--dip)'></span>Dip</span> "
        "<span class='chip'><span class='sw' style='background:var(--wedge)'></span>Wedge</span> "
        "&nbsp;each dot is one entry, chronological.</p>")
    cfg_html = ""
    if config_snapshot:
        cfg_html = ("<details><summary>Configuration used</summary>"
                    f"<div class='panel'><pre class='sm' style='overflow:auto;margin:0'>"
                    f"{_esc(json.dumps(config_snapshot, indent=2, default=str))}</pre></div></details>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest {_esc(stamp)}</title><style>{CSS}</style></head><body><div class="wrap">
<div class="hdr"><div>
  <h1>Premarket backtest &mdash; {' + '.join(_esc((q.get('per_strategy', {}).get(m, {}) or {}).get('label', m)) for m in strategies) or 'no trades'}</h1>
  <div class="muted sm">Generated {_esc(summary.get('generated_at', stamp))}</div>
  <div class="chips">{''.join(chips)}</div>
</div><button class="theme" id="themeBtn">Dark mode</button></div>

{_headline(summary, q)}

<h2>Equity curve</h2>
<div class="panel"><div class="chartbox">{equity_chart(risk.get('equity_curve', []), risk.get('starting_equity', 0))}</div>
<p class="note">Red shading is drawdown from the running peak — max
{_num(risk.get('max_drawdown_pct'), 1, '%')} ({_money(risk.get('max_drawdown_abs'))}).
Hover for trade-by-trade equity.</p></div>

{_strategy_section(q)}
{capital}
{_exits_section(q)}
{_dist_section(q, rs)}
{_funnel_section(summary)}
{_trade_table(df)}
{cfg_html}
</div><script>{JS}</script></body></html>"""
