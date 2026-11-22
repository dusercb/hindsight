"""quant.py -- the report's metric layer, including the three corrections to
previously-wrong headline numbers (see quant.py's module docstring)."""
import pytest

from premarket_backtester import quant
from premarket_backtester.models import TradeResult


def _t(ticker="AAA", date="2026-01-05", mode="dip_red_break", r=1.0, pnl_sh=0.10,
       pnl_tot=100.0, result="WIN", entry="2026-01-05T08:00:00-05:00",
       exit_="2026-01-05T08:05:00-05:00", **kw):
    tr = TradeResult(ticker=ticker, date=date, result=result)
    tr.entry_price, tr.pnl_per_share, tr.pnl_total = 5.0, pnl_sh, pnl_tot
    tr.r_multiple, tr.entry_mode = r, mode
    tr.entry_time, tr.exit_time = entry, exit_
    for k, v in kw.items():
        setattr(tr, k, v)
    return tr


# --- correction 1: win rate by PROFIT, not by the Result label ----------------

def test_win_rate_counts_profitable_exit_at_open_as_wins():
    """The bug this fixes: EXIT_AT_OPEN rows are excluded from the WIN label but
    are mostly profitable, so the label-based rate understated the truth."""
    trades = [
        _t(result="WIN", pnl_sh=0.10, r=1.0),
        _t(result="LOSS", pnl_sh=-0.10, r=-1.0),
        _t(result="EXIT_AT_OPEN", pnl_sh=0.25, r=2.5),   # profitable, unlabelled
        _t(result="EXIT_AT_OPEN", pnl_sh=0.15, r=1.5),   # profitable, unlabelled
    ]
    s = quant.cohort_stats(trades)
    assert s["win_rate_pct"] == 75.0     # 3 of 4 made money
    assert s["wins_labeled"] == 1        # ...but only 1 carries the WIN label
    assert s["exit_at_open"] == 2 and s["exit_at_open_wins"] == 2
    assert s["exit_at_open_mean_r"] == pytest.approx(2.0)


# --- correction 3: per-strategy split ----------------------------------------

def test_build_splits_by_strategy_and_never_blends_them_away():
    trades = [_t(mode="dip_red_break", r=1.0) for _ in range(3)] + \
             [_t(mode="wedge_break", r=-0.5, pnl_sh=-0.05, pnl_tot=-50.0, result="LOSS") for _ in range(2)]
    q = quant.build(trades, 5000.0)
    assert set(q["per_strategy"]) == {"dip_red_break", "wedge_break"}
    assert q["per_strategy"]["dip_red_break"]["n"] == 3
    assert q["per_strategy"]["wedge_break"]["n"] == 2
    assert q["per_strategy"]["dip_red_break"]["mean_r"] == pytest.approx(1.0)
    assert q["per_strategy"]["wedge_break"]["mean_r"] == pytest.approx(-0.5)
    assert q["combined"]["n"] == 5


# --- hold times (never previously computed) ----------------------------------

def test_hold_times_avg_median_max_and_longest_is_identified():
    trades = [
        _t(exit_="2026-01-05T08:02:00-05:00"),                      # 2 min
        _t(exit_="2026-01-05T08:04:00-05:00"),                      # 4 min
        _t(ticker="ZZZ", exit_="2026-01-05T09:00:00-05:00", r=3.0),  # 60 min
    ]
    s = quant.cohort_stats(trades)
    assert s["avg_hold_min"] == pytest.approx(22.0)
    assert s["median_hold_min"] == pytest.approx(4.0)
    assert s["max_hold_min"] == pytest.approx(60.0)
    assert s["longest_hold_trade"] == "ZZZ 2026-01-05"   # named, not just a number
    assert s["longest_hold_r"] == 3.0


def test_hold_time_tolerates_missing_timestamps():
    s = quant.cohort_stats([_t(entry=None, exit_=None), _t()])
    assert s["avg_hold_min"] == pytest.approx(5.0)   # only the parseable one counts


# --- exits cover EVERY trade (the old counter missed 41 of 129) ---------------

def test_exit_breakdown_covers_all_trades_including_unlabelled():
    trades = [
        _t(exit_trigger="stop", result="LOSS", pnl_sh=-0.1),
        _t(exit_trigger="trail_stop"),
        _t(result="EXIT_AT_OPEN", exit_trigger=None, pnl_sh=0.2),   # no trigger set
        _t(result="WIN", exit_trigger=None, pnl_sh=0.2),            # no trigger set
    ]
    ex = quant.exit_breakdown(trades)
    assert sum(ex.values()) == len(trades)      # nothing silently dropped
    assert ex["session_end"] == 1               # EXIT_AT_OPEN inferred
    assert ex["target"] == 1 and ex["stop"] == 1 and ex["trail_stop"] == 1


# --- realized equity risk (only simulated MC drawdown existed before) --------

def test_equity_risk_max_drawdown_is_the_realized_curve():
    trades = [
        _t(pnl_tot=1000.0, equity_after=5000.0, date="2026-01-01"),
        _t(pnl_tot=-2000.0, equity_after=3000.0, date="2026-01-02", result="LOSS", pnl_sh=-0.2, r=-1.0),
        _t(pnl_tot=1500.0, equity_after=4500.0, date="2026-01-03"),
    ]
    risk = quant.equity_risk(trades, 5000.0)
    assert risk["ending_equity"] == 4500.0
    # peak 5000 -> trough 3000 = 40%
    assert risk["max_drawdown_pct"] == pytest.approx(40.0)
    assert risk["max_drawdown_abs"] == pytest.approx(2000.0)
    assert risk["equity_curve"][0] == 5000.0


# --- leverage / concurrency (computed for months, displayed never) -----------

def test_leverage_profile_surfaces_concurrency_and_margin():
    trades = [
        _t(leverage_at_entry=0.5, margin_used=0.0, open_positions_at_entry=0, size_basis="risk"),
        _t(leverage_at_entry=1.9, margin_used=800.0, open_positions_at_entry=1, size_basis="funding"),
        _t(leverage_at_entry=0.7, margin_used=0.0, open_positions_at_entry=0, size_basis="notional"),
    ]
    lev = quant.leverage_profile(trades)
    assert lev["peak_leverage"] == 1.9
    assert lev["trades_above_1x"] == 1
    assert lev["concurrent_entries"] == 1
    assert lev["max_concurrent_positions"] == 2      # 1 already open + this one
    assert lev["margin_funded_entries"] == 1 and lev["margin_used_total"] == 800.0
    assert lev["size_basis_counts"] == {"risk": 1, "funding": 1, "notional": 1}


# --- significance / tails ----------------------------------------------------

def test_significance_reports_tail_concentration():
    trades = [_t(r=0.1) for _ in range(9)] + [_t(r=10.0)]
    sig = quant.significance(trades)
    assert sig["n"] == 10
    assert sig["mean_r"] == pytest.approx(1.09, abs=1e-2)
    # one trade dominates: 10 of 10.9 total R
    assert sig["top1_pct_of_sum_r"] > 90
    assert sig["mean_r_excl_top1"] == pytest.approx(0.1, abs=1e-6)
    assert sig["t_stat"] is not None


def test_correlation_needs_overlapping_days():
    same_day = [_t(mode="dip_red_break", date=f"2026-01-{d:02d}", r=1.0) for d in range(1, 6)] + \
               [_t(mode="wedge_break", date=f"2026-01-{d:02d}", r=-1.0) for d in range(1, 6)]
    inter = quant.strategy_correlation(same_day)
    assert inter["days_both"] == 5
    assert inter["daily_r_correlation"] is not None
    # single-strategy run -> no correlation claim
    solo = quant.strategy_correlation([_t(mode="dip_red_break")])
    assert "daily_r_correlation" not in solo
