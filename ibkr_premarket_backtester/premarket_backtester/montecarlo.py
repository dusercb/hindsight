"""Monte Carlo resampling of a completed backtest's realized trades.

A single backtest run is ONE realized ordering of the trades the strategy took;
its headline net P&L is a single draw from a distribution. This module re-samples
that trade population many times and, for each sample, re-runs the EXACT position
sizing the engine uses (compounding position_pct_of_equity, whole shares, the
max_pct_of_volume_at_entry cap -- see engine.apply_position_sizing) to build an
equity path. The spread of those paths shows the range of outcomes the same edge
could plausibly produce, how much of the realized result was sequencing/luck, and
the risk of a deep drawdown or ruin that a single run can hide.

Deliberately faithful to THIS backtester's mechanics: it resamples the actual
(entry_price, pnl_per_share, volume_at_entry) tuples and re-sizes them, rather
than shuffling abstract R-multiples -- so whole-share flooring and the volume cap
behave exactly as they did live, and the distribution reflects real sizing.

One deliberate simplification: each simulated path sizes its trades SEQUENTIALLY
(one at a time), so the engine's concurrent-position funding model
(risk.max_leverage, margin_used) has nothing to bind against here -- a resampled
ordering has no meaningful overlap intervals to replay. The realized run's
funding effects are already baked into the per-trade tuples being resampled.

No matplotlib here (reporting.py draws the chart from the returned artifacts); no
mutation of the input TradeResults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MonteCarloConfig, RiskConfig
from .models import TradeResult

# Fewer than this many realized trades can't produce a meaningful distribution --
# the caller reports a skip instead.
MIN_TRADES = 5


@dataclass
class MonteCarloResult:
    # JSON-safe scalar metrics (goes into summary["monte_carlo"] and the console).
    metrics: dict
    # numpy arrays for the chart (percentile band curves, a spaghetti sample, the
    # final-equity distribution, the realized path). Never serialized to JSON.
    artifacts: dict


def _size_shares(equity: np.ndarray, entry_price: np.ndarray, risk_per_share: np.ndarray, volume_cap_shares: np.ndarray, risk: RiskConfig) -> np.ndarray:
    """Vectorized copy of engine.apply_position_sizing's per-trade sizing, applied
    across all simulations at once. `volume_cap_shares` is pre-computed as
    floor(volume_at_entry * max_pct/100) or +inf where the cap is off/unavailable,
    so the min() below is a no-op there. `risk_per_share` is entry-to-stop dollars
    (or nan where no stop exists); the risk cap is skipped for those, matching the
    engine's fallback to the notional size."""
    if risk.position_pct_of_equity is not None:
        shares = np.floor(equity * (risk.position_pct_of_equity / 100.0) / entry_price)
    else:
        shares = np.full(equity.shape, float(risk.shares))
    if risk.risk_pct_of_equity is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            risk_cap = np.floor(equity * (risk.risk_pct_of_equity / 100.0) / risk_per_share)
        # No usable stop (nan) or non-positive risk -> no cap (+inf leaves shares).
        risk_cap = np.where(risk_per_share > 0, risk_cap, np.inf)
        shares = np.minimum(shares, risk_cap)
    shares = np.minimum(shares, volume_cap_shares)
    return np.maximum(shares, 0.0)


def run_monte_carlo(results: list[TradeResult], risk: RiskConfig, mc: MonteCarloConfig, wedge_vol_pct: float | None = None, wedge_lookback_pct: float | None = None) -> MonteCarloResult | None:
    """Resample the executed trades `mc.iterations` times and return the outcome
    distribution, or None when there are too few trades (< MIN_TRADES) to bother.
    wedge_vol_pct: the wedge strategy's independent volume-participation cap --
    applied to trades with entry_mode "wedge_break" instead of the shared
    risk.max_pct_of_volume_at_entry, mirroring engine.apply_position_sizing."""
    trades = [r for r in results if r.entry_price and r.pnl_per_share is not None]
    n = len(trades)
    if n < MIN_TRADES:
        return None

    entry_price = np.array([float(t.entry_price) for t in trades])
    pnl_per_share = np.array([float(t.pnl_per_share) for t in trades])
    # Per-trade entry-to-stop risk for the fixed-fractional risk cap; nan where a
    # trade recorded no stop (the cap is then skipped for it, as in the engine).
    # abs(): a SHORT's stop sits ABOVE the entry -- the risk is the distance
    # either way, exactly as engine._capped_shares computes it.
    risk_per_share = np.array([
        abs(float(t.entry_price) - float(t.stop_price)) if t.stop_price is not None else np.nan
        for t in trades
    ])
    # Per-trade share ceiling from the volume cap; +inf disables it (cap off, or a
    # trade with no recorded volume_at_entry) so np.minimum leaves shares alone.
    # Wedge trades use their own participation cap when configured.
    def _vol_cap(t: TradeResult) -> float:
        is_wedge = t.entry_mode == "wedge_break"
        pct = wedge_vol_pct if (wedge_vol_pct is not None and is_wedge) else risk.max_pct_of_volume_at_entry
        caps = []
        if pct is not None and t.volume_at_entry:
            caps.append(float(np.floor(float(t.volume_at_entry) * pct / 100.0)))
        # Wedge lookback-window cap, stacking with the session cap (the min binds),
        # mirroring _capped_shares so a resampled equity path sizes consistently.
        if wedge_lookback_pct is not None and is_wedge and t.lookback_volume:
            caps.append(float(np.floor(float(t.lookback_volume) * wedge_lookback_pct / 100.0)))
        return min(caps) if caps else np.inf
    volume_cap_shares = np.array([_vol_cap(t) for t in trades])

    rng = np.random.default_rng(mc.random_seed)
    iters = int(mc.iterations)
    if mc.mode == "reshuffle":
        # Each row an independent permutation of 0..n-1 (same trades, new order).
        idx = np.argsort(rng.random((iters, n)), axis=1)
    else:  # "resample": bootstrap, draw n trades WITH replacement per simulation.
        idx = rng.integers(0, n, size=(iters, n))

    start = float(risk.starting_equity)
    ruin_level = start * (mc.ruin_threshold_pct / 100.0)
    equity = np.full(iters, start)
    peak = np.full(iters, start)
    max_dd = np.zeros(iters)
    ruin_hit = np.zeros(iters, dtype=bool)
    curves = np.empty((iters, n + 1))
    curves[:, 0] = start

    for step in range(n):
        sel = idx[:, step]
        shares = _size_shares(equity, entry_price[sel], risk_per_share[sel], volume_cap_shares[sel], risk)
        equity = equity + pnl_per_share[sel] * shares
        curves[:, step + 1] = equity
        peak = np.maximum(peak, equity)
        max_dd = np.maximum(max_dd, np.where(peak > 0, (peak - equity) / peak, 0.0))
        ruin_hit |= equity <= ruin_level

    final = curves[:, -1]
    final_return_pct = (final / start - 1.0) * 100.0

    # The realized (chronological) path, reconstructed from the equity_after the
    # engine's sizing pass already stamped on each trade -- the single line every
    # simulated band is spread around, for reference on the chart.
    chron = sorted(trades, key=lambda r: (r.date, r.entry_time or ""))
    actual_after = [float(r.equity_after) for r in chron if r.equity_after is not None]
    actual_curve = np.array([start] + actual_after) if actual_after else None
    if actual_curve is not None:
        actual_final = float(actual_curve[-1])
        actual_peak = np.maximum.accumulate(actual_curve)
        actual_max_dd = float(np.max(np.where(actual_peak > 0, (actual_peak - actual_curve) / actual_peak, 0.0)) * 100.0)
        actual_return_pct = (actual_final / start - 1.0) * 100.0
    else:
        actual_final = actual_max_dd = actual_return_pct = None

    pcts = [5, 25, 50, 75, 95]
    fe_pcts = np.percentile(final, pcts)
    ret_pcts = np.percentile(final_return_pct, pcts)
    band_curves = np.percentile(curves, pcts, axis=0)  # shape (5, n+1)
    n_sample = min(150, iters)
    sample_rows = rng.choice(iters, size=n_sample, replace=False)
    sample_curves = curves[sample_rows]

    def r2(x: float) -> float:
        return round(float(x), 2)

    metrics = {
        "mode": mc.mode,
        "iterations": iters,
        "trades_per_sim": n,
        "starting_equity": r2(start),
        "actual_final_equity": r2(actual_final) if actual_final is not None else None,
        "actual_return_pct": r2(actual_return_pct) if actual_return_pct is not None else None,
        "actual_max_drawdown_pct": r2(actual_max_dd) if actual_max_dd is not None else None,
        "median_final_equity": r2(fe_pcts[2]),
        "mean_final_equity": r2(final.mean()),
        "p5_final_equity": r2(fe_pcts[0]),
        "p25_final_equity": r2(fe_pcts[1]),
        "p75_final_equity": r2(fe_pcts[3]),
        "p95_final_equity": r2(fe_pcts[4]),
        "min_final_equity": r2(final.min()),
        "max_final_equity": r2(final.max()),
        "median_return_pct": r2(ret_pcts[2]),
        "p5_return_pct": r2(ret_pcts[0]),
        "p95_return_pct": r2(ret_pcts[4]),
        "prob_profit_pct": r2((final > start).mean() * 100.0),
        "prob_loss_pct": r2((final < start).mean() * 100.0),
        "median_max_drawdown_pct": r2(np.percentile(max_dd, 50) * 100.0),
        "p95_max_drawdown_pct": r2(np.percentile(max_dd, 95) * 100.0),
        "worst_max_drawdown_pct": r2(max_dd.max() * 100.0),
        "ruin_threshold_pct": r2(mc.ruin_threshold_pct),
        "risk_of_ruin_pct": r2(ruin_hit.mean() * 100.0),
    }
    artifacts = {
        "band_pcts": pcts,
        "band_curves": band_curves,
        "sample_curves": sample_curves,
        "final_equities": final,
        "actual_curve": actual_curve,
        "starting_equity": start,
        "ruin_level": ruin_level,
        "steps": n,
    }
    return MonteCarloResult(metrics=metrics, artifacts=artifacts)
