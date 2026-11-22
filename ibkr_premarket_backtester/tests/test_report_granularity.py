"""DID_NOT_QUALIFY is not one opaque bucket: the report splits it by cause (RVOL
never confirmed vs. gain% too small), lists the split in Result Counts, and draws
near-miss histograms with the configured threshold as a reference line."""

import json
import sqlite3

import pytest

from premarket_backtester.config import load_config
from premarket_backtester.engine import _result_line, summarize
from premarket_backtester.models import TradeResult
from premarket_backtester.reporting import _result_detail, _scale_out_section, write_html_report, write_sqlite


def _results():
    return [
        TradeResult(ticker="AAA", date="2026-04-01", result="DID_NOT_QUALIFY", disqualify_reason="rvol",
                    reason="Premarket cumulative volume never reached 4.0x this ticker's average daily volume (peak 2.17x).",
                    max_rvol_seen=2.17),
        TradeResult(ticker="BBB", date="2026-04-02", result="DID_NOT_QUALIFY", disqualify_reason="gain_pct",
                    reason="Max premarket gain (from premarket open) was 21.0%, below 25.0%.",
                    premarket_gain_pct=21.0, rvol_at_spike=4.0),
        TradeResult(ticker="CCC", date="2026-04-03", result="INSUFFICIENT_DATA",
                    reason="Not enough daily-volume history to compute an RVOL baseline (need >= 0 trading days)."),
        TradeResult(ticker="DDD", date="2026-04-04", result="WIN", entry_price=2.00, exit_price=2.20,
                    entry_time="2026-04-04T07:10:00-04:00", pnl_per_share=0.20, pnl_total=20.0,
                    scale_outs=1,
                    exit_legs=[{"kind": "target", "frac": 0.5, "price": 2.20, "time": "2026-04-04T07:12:00-04:00"},
                               {"kind": "stop", "frac": 0.5, "price": 2.20, "time": "2026-04-04T07:13:00-04:00"}]),
    ]


def test_result_detail_splits_dnq_and_missing_data():
    rows = [r.to_dict() for r in _results()]
    labels = [_result_detail(row) for row in rows]
    assert labels == ["DNQ: RVOL never reached min", "DNQ: gain% below min", "NO DATA: RVOL baseline", "WIN"]


def test_summarize_dnq_split_uses_explicit_reason():
    s = summarize(_results(), starting_equity=2000.0, stop_offset=0.10, target_offset=0.20)
    assert s["did_not_qualify_rvol"] == 1
    assert s["did_not_qualify_gain_pct"] == 1


def test_sqlite_write_survives_nested_exit_legs(tmp_path):
    """sqlite3 can't bind a Python list (pandas.errors.DatabaseError: Execution
    failed) -- exit_legs and any other nested cell must land as JSON text."""
    db = tmp_path / "trades.sqlite"
    write_sqlite(_results(), db)

    with sqlite3.connect(db) as conn:
        stored = conn.execute("SELECT exit_legs FROM trades WHERE ticker = 'DDD'").fetchone()[0]
    legs = json.loads(stored)
    assert [leg["kind"] for leg in legs] == ["target", "stop"]


def test_html_report_writes_granular_counts_and_near_miss_charts(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("strategy:\n  min_rvol: 4.0\n  min_premarket_gain_pct: 25.0\n", encoding="utf-8")
    from premarket_backtester.config import config_to_dict
    snapshot = config_to_dict(load_config(yaml_path))

    results = _results()
    summary = summarize(results, starting_equity=2000.0, stop_offset=0.10, target_offset=0.20)
    html_path = write_html_report(results, summary, tmp_path / "run", "test", config_snapshot=snapshot)

    # These assertions cover the LEGACY matplotlib report (granular DNQ charts,
    # scale-out section), which write_html_report still emits alongside the new
    # interactive page as report_<stamp>_legacy.html.
    html_text = html_path.with_name(html_path.stem + "_legacy.html").read_text(encoding="utf-8")
    assert "DNQ: RVOL never reached min" in html_text
    assert "DNQ: gain% below min" in html_text
    # The scale-out legs are spelled out per trade, not hidden behind a count.
    assert "sold 50% @ 2.20" in html_text
    assert "stopped 50% @ 2.20" in html_text
    assert (tmp_path / "run" / "report_assets" / "dnq_rvol_near_miss.png").exists()
    assert (tmp_path / "run" / "report_assets" / "dnq_gain_near_miss.png").exists()


# --- scale-out performance + hourly win rate ---

def _mixed_scale_and_hour_results():
    """3 scaled trades (2 WIN @ hour 8, 1 LOSS @ hour 4) and 3 non-scaled trades
    (1 WIN @ hour 8, 2 LOSS @ hour 4) -- hand-computable cohort and hourly stats:
    scaled: 2/3 win (66.67%), avg_r (2.0+1.5-0.5)/3 = 1.0, avg pnl/sh 0.10.
    non-scaled: 1/3 win (33.33%), avg_r (2.0-1.0-1.0)/3 = 0.0, avg pnl/sh 0.0.
    hour 4: 0/3 win (all three losses); hour 8: 3/3 win (all three wins)."""
    def trade(ticker, result, r_multiple, pnl_per_share, scale_outs, hour):
        return TradeResult(
            ticker=ticker, date="2026-05-01", result=result,
            entry_time=f"2026-05-01T{hour:02d}:00:00-04:00", entry_price=5.0, exit_price=5.0 + pnl_per_share,
            pnl_per_share=pnl_per_share, pnl_total=pnl_per_share * 100, r_multiple=r_multiple, scale_outs=scale_outs,
            exit_legs=[{"kind": "target", "frac": 1.0, "price": 5.0 + pnl_per_share, "time": f"2026-05-01T{hour:02d}:05:00-04:00"}],
        )

    return [
        trade("SC1", "WIN", 2.0, 0.20, 1, 8),
        trade("SC2", "WIN", 1.5, 0.15, 2, 8),
        trade("SC3", "LOSS", -0.5, -0.05, 1, 4),
        trade("NS1", "WIN", 2.0, 0.20, 0, 8),
        trade("NS2", "LOSS", -1.0, -0.10, 0, 4),
        trade("NS3", "LOSS", -1.0, -0.10, 0, 4),
    ]


def test_summarize_scaled_vs_non_scaled_cohort_stats():
    s = summarize(_mixed_scale_and_hour_results(), starting_equity=2000.0, stop_offset=0.10, target_offset=0.20)
    assert s["scaled_trades_count"] == 3
    assert s["scaled_trades_win_rate_pct"] == pytest.approx(66.67, abs=0.01)
    assert s["scaled_trades_avg_r_multiple"] == pytest.approx(1.0)
    assert s["scaled_trades_avg_pnl_per_share"] == pytest.approx(0.10)
    assert s["non_scaled_trades_count"] == 3
    assert s["non_scaled_trades_win_rate_pct"] == pytest.approx(33.33, abs=0.01)
    assert s["non_scaled_trades_avg_r_multiple"] == pytest.approx(0.0)
    assert s["non_scaled_trades_avg_pnl_per_share"] == pytest.approx(0.0)


def test_summarize_win_rate_by_hour():
    s = summarize(_mixed_scale_and_hour_results(), starting_equity=2000.0, stop_offset=0.10, target_offset=0.20)
    by_hour = s["win_rate_by_hour"]
    assert by_hour[4] == {"count": 3, "win_rate_pct": 0.0}
    assert by_hour[8] == {"count": 3, "win_rate_pct": 100.0}


def test_scale_out_section_lists_scaled_trades_with_realized_r():
    import pandas as pd

    results = _mixed_scale_and_hour_results()
    df = pd.DataFrame([r.to_dict() for r in results])
    section = _scale_out_section(df, run_dir=None)
    assert "Scale-Out Trades" in section
    assert "3 trade(s)" in section
    assert "SC1" in section and "SC2" in section and "SC3" in section
    assert "NS1" not in section  # non-scaled trades excluded


def test_scale_out_section_reports_none_when_no_trade_scaled():
    import pandas as pd

    results = [r for r in _mixed_scale_and_hour_results() if r.scale_outs == 0]
    df = pd.DataFrame([r.to_dict() for r in results])
    section = _scale_out_section(df, run_dir=None)
    assert "No trades scaled out" in section


def test_html_report_includes_scale_out_and_hourly_charts(tmp_path):
    results = _mixed_scale_and_hour_results()
    summary = summarize(results, starting_equity=2000.0, stop_offset=0.10, target_offset=0.20)
    html_path = write_html_report(results, summary, tmp_path / "run", "test")

    # These assertions cover the LEGACY matplotlib report (granular DNQ charts,
    # scale-out section), which write_html_report still emits alongside the new
    # interactive page as report_<stamp>_legacy.html.
    html_text = html_path.with_name(html_path.stem + "_legacy.html").read_text(encoding="utf-8")
    assert "Scale-Out Trades" in html_text
    assert (tmp_path / "run" / "report_assets" / "scale_out_performance.png").exists()
    assert (tmp_path / "run" / "report_assets" / "hourly_win_rate.png").exists()


def test_result_line_shows_r_multiple_and_scale_out_tag():
    scaled = TradeResult(ticker="SC1", date="2026-05-01", result="WIN", entry_price=5.0, exit_price=5.20,
                          pnl_per_share=0.20, r_multiple=2.0, scale_outs=2)
    line = _result_line(scaled)
    assert "+2.00R" in line
    assert "2x scale" in line

    plain = TradeResult(ticker="NS1", date="2026-05-01", result="LOSS", entry_price=5.0, exit_price=4.90,
                         pnl_per_share=-0.10, r_multiple=-1.0, scale_outs=0)
    line2 = _result_line(plain)
    assert "-1.00R" in line2
    assert "scale" not in line2
