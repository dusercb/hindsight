"""The rebuilt interactive report: the properties that make it trustworthy.

These lock in the audit findings that drove the rebuild -- a regression here
means the report has gone back to hiding or mislabelling something.
"""
import pandas as pd
import pytest

from premarket_backtester.engine import summarize
from premarket_backtester.models import TradeResult
from premarket_backtester.report_build import build_page


def _t(ticker, date, mode, r, pnl_sh, pnl_tot, result="WIN", **kw):
    tr = TradeResult(ticker=ticker, date=date, result=result)
    tr.entry_price, tr.stop_price, tr.target_price, tr.exit_price = 5.0, 4.5, 6.0, 5.5
    tr.pnl_per_share, tr.pnl_total, tr.r_multiple, tr.entry_mode = pnl_sh, pnl_tot, r, mode
    tr.entry_time = f"{date}T08:00:00-05:00"
    tr.exit_time = f"{date}T08:07:00-05:00"
    tr.shares, tr.equity_after, tr.size_basis = 100, 4100.0, "risk"
    tr.leverage_at_entry, tr.margin_used, tr.open_positions_at_entry = 0.8, 0.0, 0
    for k, v in kw.items():
        setattr(tr, k, v)
    return tr


@pytest.fixture
def two_strategy_run():
    results = [
        _t("AAA", "2026-01-05", "dip_red_break", 1.2, 0.12, 120.0),
        _t("BBB", "2026-01-06", "dip_red_break", -1.0, -0.10, -100.0, result="LOSS"),
        _t("CCC", "2026-01-07", "wedge_break", 3.5, 0.35, 350.0, direction="short",
           leverage_at_entry=1.6, margin_used=900.0, open_positions_at_entry=1, size_basis="funding"),
        # A profitable EXIT_AT_OPEN -- carries no WIN label but MADE MONEY.
        _t("DDD", "2026-01-08", "wedge_break", 2.0, 0.20, 200.0, result="EXIT_AT_OPEN",
           exit_time="2026-01-08T09:29:00-05:00"),
    ]
    summary = summarize(results, 5000.0, 0.25, 0.50, entry_mode="dip_red_break",
                        exit_mode="candle_trail_1min", target_mode="r_multiple",
                        target_r_multiple=1.0)
    df = pd.DataFrame([r.to_dict() for r in results])
    return results, summary, df


def test_page_is_fully_self_contained(two_strategy_run):
    """No CDN, no external font, no <img>: the file must open offline forever."""
    _, summary, df = two_strategy_run
    page = build_page(summary, df, None, "test")
    assert "http://" not in page and "https://" not in page
    assert "<img" not in page
    assert page.startswith("<!doctype html>")


def test_header_names_both_strategies_not_one_config_scalar(two_strategy_run):
    """The old report labelled a 2-strategy run 'dip_red_break' from config."""
    _, summary, df = two_strategy_run
    page = build_page(summary, df, None, "test")
    assert "Dip + Wedge" in page
    assert summary["strategies_traded"] == ["dip_red_break", "wedge_break"]


def test_win_rate_counts_profitable_exit_at_open(two_strategy_run):
    """3 of 4 trades made money (incl. the unlabelled EXIT_AT_OPEN) -> 75%,
    not the 50% the WIN-label bucket would report."""
    _, summary, _ = two_strategy_run
    assert summary["win_rate_pct"] == 75.0
    assert summary["win_rate_labeled_pct"] == 50.0


def test_leverage_and_concurrency_are_rendered(two_strategy_run):
    _, summary, df = two_strategy_run
    page = build_page(summary, df, {"risk": {"max_leverage": 2.0}}, "test")
    assert "Capital &amp; leverage" in page or "Capital & leverage" in page
    assert "Peak leverage" in page and "1.60x" in page
    assert "Max concurrent" in page
    assert "Margin borrowed" in page
    assert "Which cap set the position size" in page   # size_basis breakdown
    assert "cap 2x" in page                            # configured max_leverage reference


def test_hold_times_and_exit_at_open_are_reported(two_strategy_run):
    _, summary, df = two_strategy_run
    page = build_page(summary, df, None, "test")
    assert "Avg hold" in page and "Longest hold" in page
    assert "Held to open" in page
    q = summary["quant"]
    assert q["per_strategy"]["wedge_break"]["exit_at_open"] == 1
    assert q["per_strategy"]["wedge_break"]["max_hold_min"] is not None


def test_realized_risk_replaces_simulated_only_drawdown(two_strategy_run):
    _, summary, df = two_strategy_run
    page = build_page(summary, df, None, "test")
    assert "Max drawdown" in page and "realized" in page
    assert "Sharpe" in page
    assert summary["quant"]["risk"]["max_drawdown_pct"] is not None


def test_significance_is_shown_not_buried(two_strategy_run):
    _, summary, df = two_strategy_run
    page = build_page(summary, df, None, "test")
    assert "t-stat" in page
    assert "tail-driven" in page or "best trade" in page


def test_zero_gates_are_hidden_not_padded_as_zeroes(two_strategy_run):
    """A disabled filter rendering '0' reads as 'ran and never fired'."""
    _, summary, df = two_strategy_run
    summary["skipped_buy_below_vwap_total"] = 0
    summary["skipped_macd_not_bullish_total"] = 7
    page = build_page(summary, df, None, "test")
    assert "Macd Not Bullish" in page          # fired -> shown
    assert "Buy Below Vwap" not in page        # zero -> hidden
    assert "counters are zero" in page         # ...and the omission is disclosed


def test_trade_table_exposes_the_previously_missing_columns(two_strategy_run):
    _, summary, df = two_strategy_run
    page = build_page(summary, df, None, "test")
    for col in ("Lev", "Size cap", "Margin", "Conc", "Hold m", "MFE", "MAE", "Exit via"):
        assert f">{col}<" in page, f"trade table missing {col!r}"


def test_empty_run_does_not_crash():
    summary = summarize([], 5000.0, 0.25, 0.50)
    page = build_page(summary, pd.DataFrame(), None, "test")
    assert "<!doctype html>" in page
