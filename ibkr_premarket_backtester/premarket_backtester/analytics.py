"""Post-hoc analytics over the EXECUTED trades of a run -- everything the report's
"where did the edge live / what would have been optimal" section is built from.

Two honesty rules run through all of it:

* Everything here is IN-SAMPLE and computed on the SAME trades you'd tune from, so
  a hindsight-optimal value is a HYPOTHESIS TO TEST FORWARD, never a setting to
  adopt blindly. Small samples make single "best" cells meaningless; every result
  carries its n.
* Threshold sensitivity can only ever show the effect of TIGHTENING a gate
  (dropping trades you took). LOOSENING would admit candidates that were rejected
  before they ever became trades -- those aren't in the executed set, so their
  outcome is unknowable without a re-run. The report labels this explicitly.

The one genuinely predictive-feeling analysis, the target/stop grid, leans on each
trade's recorded MFE (max favorable excursion, $/share) and MAE (max adverse
excursion, $/share) -- the true extremes over the whole hold -- to replay every
(target, stop) pair against the same price paths. Its one assumption (a bar that
touched both levels is scored a loss, matching the default conservative intrabar
policy) is stated wherever it's shown.
"""
from __future__ import annotations

from datetime import datetime
from statistics import mean, median
from typing import Any

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def executed_trades(rows: list[dict]) -> list[dict]:
    """Rows that actually filled an entry (WIN/LOSS/EXIT_AT_OPEN) -- the only rows
    with the MFE/MAE/entry-time context the analyses need."""
    return [r for r in rows if _f(r, "entry_price") is not None and str(r.get("result")) in ("WIN", "LOSS", "EXIT_AT_OPEN")]


def _cohort(trades: list[dict]) -> dict:
    """Headline stats for a set of trades: count, win rate, expectancy per share,
    total P&L, and average realized R -- the shared vocabulary every breakdown row
    and sweep cell reports."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "win_rate": None, "expectancy": None, "total_pnl": 0.0, "avg_r": None, "wins": 0, "losses": 0}
    wins = sum(1 for t in trades if t.get("result") == "WIN")
    losses = sum(1 for t in trades if t.get("result") == "LOSS")
    pnl_sh = [_f(t, "pnl_per_share") or 0.0 for t in trades]
    rs = [_f(t, "r_multiple") for t in trades]
    rs = [r for r in rs if r is not None]
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n * 100, 1),
        "expectancy": round(mean(pnl_sh), 4),
        "total_pnl": round(sum(_f(t, "pnl_total") or 0.0 for t in trades), 2),
        "avg_r": round(mean(rs), 3) if rs else None,
    }


# --------------------------------------------------------------------------- #
# 1. Target/stop optimisation from MFE/MAE                                     #
# --------------------------------------------------------------------------- #

def _resolve(fav: float, adv: float, target: float, stop: float) -> tuple[str, float]:
    """Replay one trade's excursions against a (target, stop) pair. A path that
    reached BOTH is scored a loss (conservative -- the default intrabar policy);
    reaching neither rode to session close, modelled as breakeven (0)."""
    hit_stop = adv >= stop
    hit_target = fav >= target
    if hit_stop:
        return "loss", -stop
    if hit_target:
        return "win", target
    return "open", 0.0


def target_stop_grid(rows: list[dict], offset_mode: str, cur_target: float, cur_stop: float,
                     targets: list[float] | None = None, stops: list[float] | None = None) -> dict:
    """Expectancy ($/share) across a grid of target x stop offsets, replayed over
    every executed trade's MFE/MAE. In cents mode the axes are $/share; in percent
    mode they are % of entry (a trade's fav%/adv% = MFE/MAE ÷ entry). The current
    configured offsets are always included so the report can mark 'you are here',
    and the max-expectancy cell (with n_resolved >= 40% of trades, to avoid a
    one-trade fluke) is flagged as the in-sample optimum."""
    trades = executed_trades(rows)
    # Per-trade favorable/adverse excursion in the grid's units.
    paths = []
    for t in trades:
        mfe, mae, entry = _f(t, "mfe"), _f(t, "mae"), _f(t, "entry_price")
        if mfe is None or mae is None or entry is None or entry <= 0:
            continue
        if offset_mode == "percent":
            paths.append((mfe / entry * 100.0, mae / entry * 100.0))
        else:
            paths.append((mfe, mae))
    if offset_mode == "percent":
        targets = targets or sorted({round(x, 1) for x in [4, 6, 8, 10, 12, 15, 20, 25, 30, cur_target]})
        stops = stops or sorted({round(x, 1) for x in [2, 3, 4, 5, 6, 8, 10, 12, 15, cur_stop]})
    else:
        targets = targets or sorted({round(x, 2) for x in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, cur_target]})
        stops = stops or sorted({round(x, 2) for x in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, cur_stop]})

    grid = []          # rows keyed by stop, cells keyed by target
    best = None
    n = len(paths)
    for s in stops:
        row = []
        for tgt in targets:
            outcomes = [_resolve(fav, adv, tgt, s) for fav, adv in paths]
            w = sum(1 for k, _ in outcomes if k == "win")
            loss = sum(1 for k, _ in outcomes if k == "loss")
            opn = sum(1 for k, _ in outcomes if k == "open")
            resolved = w + loss
            exp = (sum(p for _, p in outcomes) / n) if n else 0.0
            cell = {
                "target": tgt, "stop": s, "wins": w, "losses": loss, "open": opn,
                "expectancy": round(exp, 4),
                "win_rate": round(w / resolved * 100, 1) if resolved else None,
                "is_current": abs(tgt - cur_target) < 1e-9 and abs(s - cur_stop) < 1e-9,
            }
            row.append(cell)
            if resolved >= 0.4 * n and (best is None or exp > best["expectancy"]):
                best = cell
        grid.append({"stop": s, "cells": row})
    return {"mode": offset_mode, "targets": targets, "stops": stops, "grid": grid, "best": best,
            "current": {"target": cur_target, "stop": cur_stop}, "n": n}


# --------------------------------------------------------------------------- #
# 2. Threshold sensitivity (tightening only)                                   #
# --------------------------------------------------------------------------- #

def threshold_sweep(rows: list[dict], field: str, side: str, thresholds: list[float]) -> dict:
    """How the executed trades' stats move as one gate is TIGHTENED. side "min"
    keeps trades with field >= threshold (rvol, gain%); side "max" keeps field <=
    threshold (retrace%, price). The best row (max expectancy among thresholds
    that keep >= 5 trades) is flagged. Loosening is NOT modelled -- see module doc."""
    trades = [t for t in executed_trades(rows) if _f(t, field) is not None]
    steps = []
    best = None
    for th in thresholds:
        if side == "min":
            kept = [t for t in trades if _f(t, field) >= th]
        else:
            kept = [t for t in trades if _f(t, field) <= th]
        c = _cohort(kept)
        c["threshold"] = th
        c["dropped"] = len(trades) - c["n"]
        steps.append(c)
        if c["n"] >= 5 and c["expectancy"] is not None and (best is None or c["expectancy"] > best["expectancy"]):
            best = c
    return {"field": field, "side": side, "steps": steps, "best": best, "baseline_n": len(trades)}


def _with_risk_pct(rows: list[dict]) -> list[dict]:
    """Attach a derived risk_pct = (entry - stop) / entry * 100 to each row so the
    sweep can tighten on the entry-to-stop distance (the same quantity the live
    strategy.max_risk_pct gate screens on). Rows missing entry/stop pass through
    unchanged; the sweep drops them like any other missing-field row."""
    out = []
    for r in rows:
        entry, stop = _f(r, "entry_price"), _f(r, "stop_price")
        if entry is not None and entry > 0 and stop is not None:
            r = {**r, "risk_pct": (entry - stop) / entry * 100.0}
        out.append(r)
    return out


def all_threshold_sweeps(rows: list[dict]) -> list[dict]:
    """The standard battery of tightening sweeps the report shows."""
    return [
        {"label": "Min RVOL at spike", "unit": "x",
         "sweep": threshold_sweep(rows, "rvol_at_spike", "min", [1, 2, 3, 4, 5, 6, 7, 8, 10])},
        {"label": "Min premarket gain", "unit": "%",
         "sweep": threshold_sweep(rows, "premarket_gain_pct", "min", [25, 30, 35, 40, 50, 60, 75, 100])},
        {"label": "Max pullback retrace", "unit": "%",
         "sweep": threshold_sweep(rows, "pullback_retrace_pct", "max", [20, 25, 30, 35, 40, 45, 50])},
        {"label": "Max risk (entry-to-stop)", "unit": "%",
         "sweep": threshold_sweep(_with_risk_pct(rows), "risk_pct", "max", [3, 4, 5, 6, 7, 8, 10, 12, 15])},
        {"label": "Max entry price", "unit": "$",
         "sweep": threshold_sweep(rows, "entry_price", "max", [3, 4, 5, 6, 8, 10, 15, 20])},
        {"label": "Max float", "unit": "M shares",
         "sweep": threshold_sweep(rows, "float_shares", "max",
                                  [2e6, 5e6, 10e6, 20e6, 30e6, 50e6, 100e6])},
    ]


# --------------------------------------------------------------------------- #
# 3. Temporal edges: weekday / hour / month + profit per day                   #
# --------------------------------------------------------------------------- #

def _entry_dt(t: dict) -> datetime | None:
    s = t.get("entry_time")
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s))
    except ValueError:
        return None


def _grouped(trades: list[dict], keyfn, labels: list[str] | None = None) -> list[dict]:
    buckets: dict[Any, list[dict]] = {}
    for t in trades:
        k = keyfn(t)
        if k is None:
            continue
        buckets.setdefault(k, []).append(t)
    order = labels if labels is not None else sorted(buckets)
    out = []
    for k in order:
        if labels is not None and k not in buckets:
            continue
        c = _cohort(buckets[k])
        c["key"] = k
        out.append(c)
    # keep only present keys, in the given order
    present = {c["key"] for c in out}
    if labels is not None:
        out = [c for c in out if c["key"] in present]
    return out


def temporal_edges(rows: list[dict]) -> dict:
    """P&L / win-rate / expectancy sliced by weekday, entry hour (ET) and month,
    plus profit-per-trading-day. Answers 'which days/times/months actually paid'."""
    trades = [t for t in executed_trades(rows) if _entry_dt(t) is not None]
    by_weekday = _grouped(trades, lambda t: _WEEKDAYS[_entry_dt(t).weekday()], _WEEKDAYS)
    by_hour = _grouped(trades, lambda t: f"{_entry_dt(t).hour:02d}:00")
    by_month = _grouped(trades, lambda t: _MONTHS[_entry_dt(t).month - 1], _MONTHS)
    days = {_entry_dt(t).date() for t in trades}
    total = round(sum(_f(t, "pnl_total") or 0.0 for t in trades), 2)
    return {
        "by_weekday": by_weekday,
        "by_hour": by_hour,
        "by_month": by_month,
        "trading_days": len(days),
        "profit_per_trading_day": round(total / len(days), 2) if days else None,
        "total_pnl": total,
    }


# --------------------------------------------------------------------------- #
# Top-level bundle + headline takeaways                                        #
# --------------------------------------------------------------------------- #

def build_analytics(rows: list[dict], offset_mode: str, cur_target: float, cur_stop: float,
                    exit_mode: str = "standard", stop_mode: str = "offset", target_mode: str = "offset") -> dict | None:
    """The whole analytics payload for the report, or None when there aren't
    enough executed trades to say anything (< 5).

    The target x stop expectancy grid only describes trades that exit on a FIXED
    target and FIXED stop (exit_mode "standard", target_mode/stop_mode "offset").
    Under a structural stop (pullback_low/red_candle_low), an R-multiple target, or
    a trailing/runner exit (candle_trail_1min), there is no fixed target to
    optimise -- and MFE/MAE are recorded only up to each trade's actual exit, so
    they can't fairly model a further-out fixed target either. In those cases the
    grid is marked not-applicable (with the realized per-share expectancy of the
    ACTUAL exit for reference) instead of showing a misleading 'optimal' cell."""
    trades = executed_trades(rows)
    if len(trades) < 5:
        return None
    fixed_offset_exit = exit_mode == "standard" and target_mode == "offset" and stop_mode == "offset"
    if fixed_offset_exit:
        grid = target_stop_grid(rows, offset_mode, cur_target, cur_stop)
        grid["applicable"] = True
    else:
        realized = sum((_f(t, "pnl_per_share") or 0.0) for t in trades) / len(trades)
        grid = {
            "applicable": False,
            "exit_mode": exit_mode, "stop_mode": stop_mode, "target_mode": target_mode,
            "realized_expectancy_per_share": round(realized, 4),
            "n": len(trades),
        }
    sweeps = all_threshold_sweeps(rows)
    temporal = temporal_edges(rows)

    takeaways = []
    if fixed_offset_exit and grid["best"] and not grid["best"]["is_current"]:
        b, cur = grid["best"], grid["current"]
        unit = "%" if offset_mode == "percent" else "$"
        takeaways.append(
            f"In hindsight, a {b['target']}{unit} target / {b['stop']}{unit} stop maximised expectancy "
            f"({b['expectancy']:+.3f}/sh vs your {cur['target']}{unit}/{cur['stop']}{unit}) -- test forward, don't just adopt."
        )
    for s in sweeps:
        sw = s["sweep"]
        base = sw["baseline_n"]
        if sw["best"] and sw["best"]["dropped"] > 0 and sw["best"]["dropped"] < base:
            op = ">=" if sw["side"] == "min" else "<="
            takeaways.append(
                f"Tightening {s['label'].lower()} to {op} {sw['best']['threshold']:g}{s['unit']} would have dropped "
                f"{sw['best']['dropped']} trade(s) and moved win rate to {sw['best']['win_rate']}% "
                f"(expectancy {sw['best']['expectancy']:+.3f}/sh)."
            )
    wk = [c for c in temporal["by_weekday"] if c["n"] >= 3]
    if wk:
        best_wk = max(wk, key=lambda c: c["expectancy"])
        worst_wk = min(wk, key=lambda c: c["expectancy"])
        if best_wk["key"] != worst_wk["key"]:
            takeaways.append(
                f"By weekday, {best_wk['key']} paid best ({best_wk['expectancy']:+.3f}/sh, n={best_wk['n']}) and "
                f"{worst_wk['key']} worst ({worst_wk['expectancy']:+.3f}/sh, n={worst_wk['n']})."
            )

    return {
        "n_trades": len(trades),
        "target_stop_grid": grid,
        "threshold_sweeps": sweeps,
        "temporal": temporal,
        "takeaways": takeaways,
    }
