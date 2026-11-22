"""All-in compounding position sizing: each trade commits position_pct_of_equity %
of the CURRENT account value (whole shares at the actual entry fill), with equity
compounding through the trades in chronological order -- regardless of the
arbitrary order candidates appear in the input file."""

import pytest

from premarket_backtester.config import RiskConfig
from premarket_backtester.engine import apply_position_sizing
from premarket_backtester.models import TradeResult


def _trade(ticker, date, entry_time, entry_price, pnl_per_share, result="WIN", stop_price=None, volume_at_entry=None, lookback_volume=None):
    return TradeResult(
        ticker=ticker, date=date, result=result, entry_time=entry_time,
        entry_price=entry_price, pnl_per_share=pnl_per_share,
        stop_price=stop_price, volume_at_entry=volume_at_entry,
        lookback_volume=lookback_volume,
    )


def test_all_in_sizing_compounds_chronologically_despite_input_order():
    t1 = _trade("AAA", "2026-04-01", "2026-04-01T07:10:00-04:00", 2.00, +0.20)
    t2 = _trade("BBB", "2026-04-02", "2026-04-02T07:05:00-04:00", 4.00, -0.10, result="LOSS")
    t3 = _trade("CCC", "2026-04-02", "2026-04-02T08:00:00-04:00", 10.00, +0.20)
    skipped = TradeResult(ticker="DDD", date="2026-04-01", result="NO_TRIGGER")
    results = [t2, t3, skipped, t1]  # deliberately NOT in time order

    apply_position_sizing(results, RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0))

    # t1 first by time: 95% of 2000 = 1900 -> 950 whole shares at 2.00 -> +190.
    assert (t1.shares, t1.pnl_total, t1.equity_after) == (950, 190.0, 2190.0)
    # t2: 95% of 2190 = 2080.50 -> 520 shares at 4.00 -> -52.
    assert (t2.shares, t2.pnl_total, t2.equity_after) == (520, -52.0, 2138.0)
    # t3: 95% of 2138 = 2031.10 -> 203 shares at 10.00 -> +40.60.
    assert (t3.shares, t3.pnl_total, t3.equity_after) == (203, 40.6, 2178.6)
    # Non-trades are untouched.
    assert skipped.shares is None and skipped.equity_after is None


def test_null_position_pct_uses_fixed_shares():
    t1 = _trade("AAA", "2026-04-01", "2026-04-01T07:10:00-04:00", 2.00, +0.20)
    t2 = _trade("BBB", "2026-04-02", "2026-04-02T07:05:00-04:00", 4.00, -0.10, result="LOSS")

    apply_position_sizing([t1, t2], RiskConfig(position_pct_of_equity=None, shares=100, starting_equity=2000.0))

    assert t1.shares == 100 and t1.pnl_total == pytest.approx(20.0)
    assert t2.shares == 100 and t2.pnl_total == pytest.approx(-10.0)
    assert t2.equity_after == pytest.approx(2010.0)  # equity walk still recorded


def test_unaffordable_entry_buys_zero_shares_and_leaves_equity_unchanged():
    t = _trade("AAA", "2026-04-01", "2026-04-01T07:10:00-04:00", 5000.00, +0.20)

    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0))

    assert (t.shares, t.pnl_total, t.equity_after) == (0, 0.0, 2000.0)


def test_monthly_fees_charge_once_per_month_across_the_span():
    # Fixed 100 shares to isolate the fee effect. Trades in Apr and Jun (May is a
    # gap month with no trades but is still charged).
    t_apr = _trade("AAA", "2026-04-10", "2026-04-10T07:10:00-04:00", 2.00, +0.50)
    t_jun = _trade("BBB", "2026-06-05", "2026-06-05T07:05:00-04:00", 2.00, +0.50)
    risk = RiskConfig(position_pct_of_equity=None, shares=100, starting_equity=2000.0, monthly_fees=50.0)

    total_fees = apply_position_sizing([t_jun, t_apr], risk)  # out of order on purpose

    # Apr charged before its trade: 2000 - 50 = 1950, +50 pnl -> 2000.
    assert t_apr.equity_after == pytest.approx(2000.0)
    # May + Jun both charged crossing into June (2 months x 50 = 100): 2000 - 100
    # = 1900, +50 -> 1950.
    assert t_jun.equity_after == pytest.approx(1950.0)
    assert total_fees == pytest.approx(150.0)   # Apr + May + Jun


def test_monthly_fees_default_null_is_a_no_op():
    t = _trade("AAA", "2026-04-01", "2026-04-01T07:10:00-04:00", 2.00, +0.20)
    total_fees = apply_position_sizing([t], RiskConfig(position_pct_of_equity=None, shares=100, starting_equity=2000.0))
    assert total_fees == 0.0 and t.equity_after == pytest.approx(2020.0)


# --- concurrent (time-overlapping) positions & the funding model ----------------

def _timed(ticker, date, entry_hms, exit_hms, entry_price, pnl_per_share, result="WIN", stop_price=None):
    return TradeResult(
        ticker=ticker, date=date, result=result,
        entry_time=f"{date}T{entry_hms}-04:00", exit_time=f"{date}T{exit_hms}-04:00",
        entry_price=entry_price, pnl_per_share=pnl_per_share, stop_price=stop_price,
    )


def _overlap_pair():
    """AAA holds 07:00-08:00; BBB enters 07:30 while AAA still owns the capital."""
    t1 = _timed("AAA", "2026-04-01", "07:00:00", "08:00:00", 2.00, +0.20)
    t2 = _timed("BBB", "2026-04-01", "07:30:00", "07:50:00", 2.00, +0.10)
    return t1, t2


def test_cash_only_funding_caps_an_overlapping_second_entry():
    t1, t2 = _overlap_pair()
    apply_position_sizing([t1, t2], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0, max_leverage=1.0))
    # AAA takes 95% of 2000 = 950 shares; BBB wants the same but only $100 of
    # cash is uncommitted -> 50 shares, bound by "funding", zero margin.
    assert (t1.shares, t1.size_basis) == (950, "notional")
    assert (t2.shares, t2.size_basis) == (50, "funding")
    assert t2.open_positions_at_entry == 1 and t2.margin_used == pytest.approx(0.0)
    # P&L realizes at each EXIT: BBB (07:50) first, then AAA (08:00).
    assert t2.equity_after == pytest.approx(2005.0)   # 2000 + 50 * 0.10
    assert t1.equity_after == pytest.approx(2195.0)   # 2005 + 950 * 0.20


def test_leverage_funds_the_full_second_entry_on_margin():
    t1, t2 = _overlap_pair()
    apply_position_sizing([t1, t2], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0, max_leverage=2.0))
    # With 2x funding the overlapping entry keeps its FULL risk-derived size --
    # sized off equity exactly as if alone -- funded on margin: notional $1900
    # against $100 free cash = $1800 borrowed, gross leverage 1.9x.
    assert (t2.shares, t2.size_basis) == (950, "notional")
    assert t2.margin_used == pytest.approx(1800.0)
    assert t2.leverage_at_entry == pytest.approx(1.9)
    assert t2.open_positions_at_entry == 1
    assert t1.equity_after == pytest.approx(2285.0)   # 2095 (BBB's +95) + 950 * 0.20


def test_overlap_entry_never_sizes_against_unrealized_pnl():
    # AAA's +$190 is still open when BBB enters at 07:30 -- BBB must size off the
    # STARTING 2000, not 2190 (compounding only at exits). With no funding cap
    # (max_leverage null, the legacy model) BBB still gets its full size, and the
    # implicit borrowing is recorded rather than capped.
    t1, t2 = _overlap_pair()
    apply_position_sizing([t1, t2], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0, max_leverage=None))
    assert t2.shares == 950                            # floor(0.95 * 2000 / 2.00), not of 2190
    assert t2.size_basis == "notional"
    assert t2.margin_used == pytest.approx(1800.0)     # telemetry still stamped
    assert t2.leverage_at_entry == pytest.approx(1.9)


def test_size_to_risk_funds_the_full_risk_target_on_a_tight_stop():
    # entry 2.00, stop 1.98 -> $0.02 risk. Full 2% of $2000 = $40 = 2000 shares,
    # whose $4000 notional is 2x equity -- reachable only on margin. size_to_risk
    # lifts the cash cap to the leverage ceiling so the RISK cap governs.
    t = _trade("AAA", "2026-01-05", "2026-01-05T08:00:00-05:00", 2.00, 0.10, stop_price=1.98)
    apply_position_sizing([t], RiskConfig(starting_equity=2000.0, risk_pct_of_equity=2.0,
                                          max_leverage=2.0, size_to_risk=True))
    # ~2000 shares (float floor of $0.02 risk gives 1999): the RISK cap governs,
    # NOT the 95% cash cap that would clip this tight stop to 950.
    assert t.shares >= 1990
    assert t.size_basis == "risk"                      # risk target reached, not cash-capped
    assert t.shares * abs(2.00 - 1.98) == pytest.approx(40.0, abs=0.5)   # ~2% of $2000
    assert t.leverage_at_entry == pytest.approx(2.0, abs=0.01)           # funded to ~2x


def test_without_size_to_risk_the_cash_cap_shortchanges_a_tight_stop():
    # Same trade, flag OFF: the 95% cash cap clips it to 950 shares (~0.95% risk),
    # exactly the shortfall size_to_risk exists to fix.
    t = _trade("AAA", "2026-01-05", "2026-01-05T08:00:00-05:00", 2.00, 0.10, stop_price=1.98)
    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0,
                                          risk_pct_of_equity=2.0, max_leverage=2.0, size_to_risk=False))
    assert t.shares == 950
    assert t.size_basis == "notional"


def test_risk_cap_still_binds_on_a_margin_funded_entry():
    # "Leverage maintains the capital, not the risk": the concurrent entry's
    # risk cap (2% of equity / its own entry-to-stop) is computed off FULL
    # equity, margin only funds it. 2% of 2000 = $40 at $0.10/share risk = ~400
    # shares (399 after float-representation flooring, the same arithmetic
    # _capped_shares always used), well under the notional 950 -- so "risk"
    # binds, and margin covers what cash can't.
    t1, t2 = _overlap_pair()
    t2.stop_price = 1.90
    apply_position_sizing([t1, t2], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0,
                                               risk_pct_of_equity=2.0, max_leverage=2.0))
    assert (t2.shares, t2.size_basis) == (399, "risk")
    assert t2.margin_used == pytest.approx(698.0)      # 399 * 2.00 - 100 free cash


def test_non_overlapping_trades_are_unchanged_by_the_event_walk():
    # Sequential trades (BBB enters after AAA exits) behave exactly as the old
    # entry-ordered walk: full size each, compounding in order, no margin.
    t1 = _timed("AAA", "2026-04-01", "07:00:00", "07:20:00", 2.00, +0.20)
    t2 = _timed("BBB", "2026-04-01", "08:00:00", "08:30:00", 2.00, +0.10)
    apply_position_sizing([t2, t1], RiskConfig(position_pct_of_equity=95.0, starting_equity=2000.0, max_leverage=1.0))
    assert (t1.shares, t1.equity_after) == (950, 2190.0)
    assert t2.shares == 1040                           # floor(0.95 * 2190 / 2.00)
    assert t2.open_positions_at_entry == 0 and t2.margin_used == pytest.approx(0.0)


def test_risk_cap_binds_when_stop_is_wide():
    # Wide stop: entry 5.00, stop 4.00 -> $1.00 risk/share. Risk 2% of $2000 = $40
    # budget -> 40 shares. Notional 95% of 2000 = 1900 -> 380 shares. Risk cap wins.
    t = _trade("AAA", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +2.00, stop_price=4.00)
    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, risk_pct_of_equity=2.0, starting_equity=2000.0))
    assert (t.shares, t.size_basis) == (40, "risk")
    assert t.pnl_total == pytest.approx(80.0)  # 40 shares x +2.00


def test_notional_wins_when_stop_is_tight_risk_not_the_limiting_factor():
    # Tight stop: entry 5.00, stop 4.95 -> $0.05 risk/share. Risk 2% of 2000 = $40
    # budget / 0.05 = 800 shares, but notional 95% of 2000 = 1900 -> 380 shares.
    # "Default to the % of account when risk isn't the limiting factor."
    t = _trade("AAA", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +0.10, stop_price=4.95)
    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, risk_pct_of_equity=2.0, starting_equity=2000.0))
    assert (t.shares, t.size_basis) == (380, "notional")


def test_volume_cap_can_still_win_over_risk_and_notional():
    # Risk cap 40, notional 380, but volume_at_entry 500 @ 3% = 15 shares -> volume.
    t = _trade("AAA", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +2.00, stop_price=4.00, volume_at_entry=500)
    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, risk_pct_of_equity=2.0, max_pct_of_volume_at_entry=3.0, starting_equity=2000.0))
    assert (t.shares, t.size_basis) == (15, "volume")


def test_risk_sizing_falls_back_to_notional_when_no_stop_recorded():
    # No stop_price -> risk cap can't be computed -> notional size, basis "notional".
    t = _trade("AAA", "2026-04-01", "2026-04-01T08:00:00-04:00", 2.00, +0.20, stop_price=None)
    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, risk_pct_of_equity=2.0, starting_equity=2000.0))
    assert (t.shares, t.size_basis) == (950, "notional")


def test_risk_pct_null_leaves_notional_sizing_and_basis_unchanged():
    t = _trade("AAA", "2026-04-01", "2026-04-01T08:00:00-04:00", 2.00, +0.20, stop_price=1.90)
    apply_position_sizing([t], RiskConfig(position_pct_of_equity=95.0, risk_pct_of_equity=None, starting_equity=2000.0))
    assert (t.shares, t.size_basis) == (950, "notional")


def test_wedge_trades_use_their_own_volume_cap():
    """wedge.max_pct_of_volume_at_entry overrides the shared participation cap for
    entry_mode "wedge_break" ONLY: same inputs, the wedge trade caps at 1% of
    volume_at_entry while the dip trade keeps the shared 3%."""
    risk = RiskConfig(position_pct_of_equity=None, shares=5_000, max_pct_of_volume_at_entry=3.0, starting_equity=100_000.0)
    wedge = _trade("WDG", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +1.00, volume_at_entry=10_000)
    wedge.entry_mode = "wedge_break"
    dip = _trade("DIP", "2026-04-02", "2026-04-02T08:00:00-04:00", 5.00, +1.00, volume_at_entry=10_000)
    apply_position_sizing([wedge, dip], risk, wedge_vol_pct=1.0)
    assert (wedge.shares, wedge.size_basis) == (100, "volume")   # 1% of 10,000
    assert (dip.shares, dip.size_basis) == (300, "volume")       # shared 3% untouched


def test_wedge_volume_cap_null_inherits_the_shared_cap():
    risk = RiskConfig(position_pct_of_equity=None, shares=5_000, max_pct_of_volume_at_entry=3.0, starting_equity=100_000.0)
    w = _trade("WDG", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +1.00, volume_at_entry=10_000)
    w.entry_mode = "wedge_break"
    apply_position_sizing([w], risk, wedge_vol_pct=None)
    assert (w.shares, w.size_basis) == (300, "volume")           # falls back to risk's 3%


def test_wedge_lookback_volume_cap_binds_when_tighter():
    """max_entry_pct_of_lookback_volume caps at a % of the LOOKBACK-window volume,
    a separate base from volume_at_entry. Here the lookback cap (10% of 8,000 =
    800) is looser than the session cap (1% of 100,000 = 1,000)? No -- 800 < 1000,
    so the lookback cap binds, and size_basis records it."""
    risk = RiskConfig(position_pct_of_equity=None, shares=5_000, max_pct_of_volume_at_entry=1.0, starting_equity=100_000.0)
    w = _trade("WDG", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +1.00,
               volume_at_entry=100_000, lookback_volume=8_000)
    w.entry_mode = "wedge_break"
    apply_position_sizing([w], risk, wedge_vol_pct=1.0, wedge_lookback_pct=10.0)
    assert (w.shares, w.size_basis) == (800, "lookback_vol")      # 10% of 8,000 < 1% of 100,000


def test_wedge_lookback_cap_yields_to_the_session_cap_when_looser():
    risk = RiskConfig(position_pct_of_equity=None, shares=5_000, max_pct_of_volume_at_entry=1.0, starting_equity=100_000.0)
    w = _trade("WDG", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +1.00,
               volume_at_entry=10_000, lookback_volume=50_000)  # 10% of 50k=5000 > 1% of 10k=100
    w.entry_mode = "wedge_break"
    apply_position_sizing([w], risk, wedge_vol_pct=1.0, wedge_lookback_pct=10.0)
    assert (w.shares, w.size_basis) == (100, "volume")           # session cap still tighter


def test_wedge_lookback_cap_is_wedge_only_and_off_by_default():
    risk = RiskConfig(position_pct_of_equity=None, shares=5_000, max_pct_of_volume_at_entry=None, starting_equity=100_000.0)
    dip = _trade("DIP", "2026-04-01", "2026-04-01T08:00:00-04:00", 5.00, +1.00,
                 volume_at_entry=100_000, lookback_volume=1_000)
    apply_position_sizing([dip], risk, wedge_lookback_pct=10.0)  # dip trade -> cap ignored
    assert dip.shares == 5_000 and dip.size_basis == "shares"    # not clipped to 100
