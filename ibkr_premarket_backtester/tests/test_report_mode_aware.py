"""The HTML report tailors itself to the run's config: a dip entry leads with its
own levels (trigger / pullback low / retrace) and keeps the HOD only as context; a
hod_break entry leads with the HOD; candle_trail adds the runner columns."""
import pandas as pd

from premarket_backtester.reporting import _table_columns

_COLS = {
    "ticker", "date", "result", "trigger_price", "pullback_low", "pullback_retrace_pct",
    "premarket_high", "stock_price", "stop_price", "target_price", "entry_price", "exit_price",
    "trail_activated", "trail_stop_final", "scale_outs", "r_multiple", "setup_index",
    "reason", "premarket_gain_pct", "rvol_at_spike", "shares", "pnl_per_share", "pnl_total",
    "pnl_pct", "equity_after", "float_shares", "path_ambiguous", "ambiguity_resolution",
}


def _df():
    return pd.DataFrame([{c: 1 for c in _COLS}])


def test_dip_mode_leads_with_dip_levels_not_hod():
    cols = _table_columns(_df(), "dip_red_break", "standard")
    # dip levels appear, and BEFORE the HOD (which is only context here).
    assert cols.index("trigger_price") < cols.index("premarket_high")
    assert cols.index("pullback_low") < cols.index("premarket_high")
    assert "pullback_retrace_pct" in cols
    # no trail columns under the standard exit.
    assert "trail_activated" not in cols and "trail_stop_final" not in cols


def test_hod_mode_leads_with_the_hod():
    cols = _table_columns(_df(), "hod_break", "standard")
    assert cols.index("premarket_high") < cols.index("trigger_price")
    assert "pullback_low" not in cols          # not a dip level for hod_break


def test_candle_trail_adds_runner_columns():
    cols = _table_columns(_df(), "dip_red_break", "candle_trail_1min")
    assert "trail_activated" in cols and "trail_stop_final" in cols
    # runner columns sit after the fill columns.
    assert cols.index("trail_activated") > cols.index("entry_price")


def test_scale_out_columns_only_appear_when_a_trade_actually_scaled():
    df = _df(); df["scale_outs"] = [0]
    assert "exit_legs" not in _table_columns(df, "dip_red_break", "standard")
    df["scale_outs"] = [2]; df["exit_legs"] = [[]]
    assert "scale_outs" in _table_columns(df, "dip_red_break", "standard")
