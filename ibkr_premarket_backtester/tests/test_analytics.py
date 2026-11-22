"""Post-hoc analytics: the MFE/MAE target-stop grid, tightening sweeps, and
temporal edges the report's optimisation section is built from. Every number is
in-sample (a hypothesis to test forward, never a setting to adopt) and sweeps only
model TIGHTENING -- both asserted here via the shapes the report relies on."""
from premarket_backtester.analytics import (
    _resolve, _with_risk_pct, all_threshold_sweeps, build_analytics, executed_trades,
    target_stop_grid, temporal_edges, threshold_sweep,
)


def _t(result, entry, mfe, mae, pnl_sh, r, when="2026-04-09T08:00:00-04:00", **extra):
    d = {"result": result, "entry_price": entry, "mfe": mfe, "mae": mae,
         "pnl_per_share": pnl_sh, "pnl_total": pnl_sh * 100, "r_multiple": r, "entry_time": when}
    d.update(extra)
    return d


def test_executed_trades_filters_to_filled_entries():
    rows = [
        _t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0),
        {"result": "NO_TRIGGER", "entry_price": ""},          # never filled
        {"result": "STALE_HOD", "entry_price": None},
        _t("EXIT_AT_OPEN", 4.0, 0.1, 0.1, 0.02, 0.1),
    ]
    assert len(executed_trades(rows)) == 2


def test_resolve_replays_one_path_against_offsets():
    assert _resolve(0.6, 0.20, 0.5, 0.25) == ("win", 0.5)     # target only
    assert _resolve(0.3, 0.30, 0.5, 0.25) == ("loss", -0.25)  # stop only
    assert _resolve(0.6, 0.30, 0.5, 0.25) == ("loss", -0.25)  # both -> conservative loss
    assert _resolve(0.3, 0.10, 0.5, 0.25) == ("open", 0.0)    # neither -> breakeven


def test_target_stop_grid_finds_the_expectancy_optimum_and_marks_current():
    # Three trades whose MFE/MAE make a 0.50/0.25 stop lose two of them to noise
    # while a 0.50/0.50 stop keeps them -> wider stop is the in-sample optimum,
    # exactly the tight-stop lesson.
    rows = [
        _t("LOSS", 4.0, 0.55, 0.30, -0.25, -1.0),  # 0.25 stop: loss; 0.50 stop: target
        _t("LOSS", 4.0, 0.60, 0.28, -0.25, -1.0),
        _t("WIN", 4.0, 0.70, 0.10, 0.50, 2.0),
    ]
    g = target_stop_grid(rows, "cents", cur_target=0.50, cur_stop=0.25,
                         targets=[0.50], stops=[0.25, 0.50])
    by_stop = {r["stop"]: r["cells"][0] for r in g["grid"]}
    assert by_stop[0.25]["wins"] == 1 and by_stop[0.25]["losses"] == 2   # tight stop noise-stops two
    assert by_stop[0.50]["wins"] == 3 and by_stop[0.50]["losses"] == 0   # wide stop keeps them
    assert g["best"]["stop"] == 0.50
    assert by_stop[0.25]["is_current"] is True and by_stop[0.50]["is_current"] is False


def test_percent_mode_grid_uses_excursion_as_pct_of_entry():
    # MFE 0.80 on a $8 entry = 10%; MAE 0.16 = 2%. A 10%/5% offset -> target hit.
    rows = [_t("WIN", 8.0, 0.80, 0.16, 0.80, 2.0)]
    g = target_stop_grid(rows, "percent", cur_target=10.0, cur_stop=5.0, targets=[10.0], stops=[5.0])
    cell = g["grid"][0]["cells"][0]
    assert cell["wins"] == 1 and cell["losses"] == 0 and cell["is_current"] is True


def test_threshold_sweep_tightens_only_and_flags_best():
    rows = [
        _t("LOSS", 4.0, 0.1, 0.3, -0.25, -1.0, rvol_at_spike=3.0),
        _t("LOSS", 4.0, 0.1, 0.3, -0.25, -1.0, rvol_at_spike=4.0),
        _t("WIN", 4.0, 0.6, 0.1, 0.50, 2.0, rvol_at_spike=8.0),
        _t("WIN", 4.0, 0.6, 0.1, 0.50, 2.0, rvol_at_spike=9.0),
        _t("WIN", 4.0, 0.6, 0.1, 0.50, 2.0, rvol_at_spike=10.0),
        _t("WIN", 4.0, 0.6, 0.1, 0.50, 2.0, rvol_at_spike=7.0),
    ]
    sw = threshold_sweep(rows, "rvol_at_spike", "min", [3, 5, 7])
    steps = {s["threshold"]: s for s in sw["steps"]}
    assert steps[3]["n"] == 6 and steps[3]["dropped"] == 0
    assert steps[7]["n"] == 4 and steps[7]["dropped"] == 2 and steps[7]["win_rate"] == 100.0
    # Only thresholds that keep >= 5 trades qualify as "best" (no one-trade flukes):
    # 3 keeps 6, 5 and 7 keep < 5, so the flagged best is 3.
    assert sw["best"]["threshold"] == 3


def test_risk_pct_sweep_tightens_on_entry_to_stop_distance():
    # Five tight-risk winners (3-5% stop distance) and three wide-risk losers
    # (12.5-15%). Tightening the max entry-to-stop cut drops the wide losers.
    rows = [
        _t("LOSS", 4.0, 0.1, 0.5, -0.5, -1.0, stop_price=3.52),   # 12%
        _t("LOSS", 4.0, 0.1, 0.5, -0.5, -1.0, stop_price=3.56),   # 11%
        _t("LOSS", 4.0, 0.1, 0.5, -0.5, -1.0, stop_price=3.60),   # 10%
        _t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0, stop_price=3.84),      # 4%
        _t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0, stop_price=3.86),      # 3.5%
        _t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0, stop_price=3.88),      # 3%
        _t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0, stop_price=3.90),      # 2.5%
        _t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0, stop_price=3.92),      # 2%
    ]
    sw = threshold_sweep(_with_risk_pct(rows), "risk_pct", "max", [3, 5, 8, 20])
    steps = {s["threshold"]: s for s in sw["steps"]}
    assert steps[20]["n"] == 8 and steps[20]["dropped"] == 0
    assert steps[8]["n"] == 5 and steps[8]["dropped"] == 3 and steps[8]["win_rate"] == 100.0
    assert steps[5]["n"] == 5 and steps[5]["dropped"] == 3
    assert steps[3]["n"] < 5                                  # too-tight also drops good winners
    # best = tightest cut that still keeps >= 5 trades and maximises expectancy = 5%
    assert sw["best"]["threshold"] == 5

    # and it is part of the standard battery the report renders
    labels = [s["label"] for s in all_threshold_sweeps(rows)]
    assert "Max risk (entry-to-stop)" in labels


def test_temporal_edges_group_by_weekday_and_profit_per_day():
    rows = [
        _t("WIN", 4.0, 0.6, 0.1, 0.50, 2.0, when="2026-04-06T08:00:00-04:00"),   # Mon
        _t("LOSS", 4.0, 0.1, 0.3, -0.25, -1.0, when="2026-04-06T09:00:00-04:00"),  # Mon (same day)
        _t("WIN", 4.0, 0.6, 0.1, 0.50, 2.0, when="2026-04-09T08:00:00-04:00"),   # Thu
    ]
    te = temporal_edges(rows)
    wk = {c["key"]: c for c in te["by_weekday"]}
    assert wk["Mon"]["n"] == 2 and wk["Thu"]["n"] == 1
    assert te["trading_days"] == 2                        # two distinct dates
    assert te["total_pnl"] == 75.0                        # (50 -25 +50) * 100 = 7500? per-share*100
    assert te["profit_per_trading_day"] == 37.5           # 75 / 2


def test_build_analytics_needs_at_least_five_trades():
    few = [_t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0) for _ in range(4)]
    assert build_analytics(few, "cents", 0.5, 0.25) is None
    enough = [_t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0) for _ in range(5)]
    a = build_analytics(enough, "cents", 0.5, 0.25)
    assert a is not None and a["n_trades"] == 5
    assert "target_stop_grid" in a and "threshold_sweeps" in a and "temporal" in a


def test_fixed_offset_exit_builds_the_grid():
    rows = [_t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0) for _ in range(6)]
    a = build_analytics(rows, "cents", 0.5, 0.25, exit_mode="standard", stop_mode="offset", target_mode="offset")
    g = a["target_stop_grid"]
    assert g["applicable"] is True and "grid" in g and "best" in g


def test_trailing_exit_marks_grid_not_applicable_with_realized_expectancy():
    # A trailing/runner exit has no fixed target to optimise -> grid n/a, and the
    # realized per-share expectancy of the ACTUAL exit is reported instead.
    rows = [_t("WIN", 4.0, 0.6, 0.1, 0.4, 2.0) for _ in range(4)] + [_t("LOSS", 4.0, 0.1, 0.3, -0.2, -1.0)]
    a = build_analytics(rows, "cents", 0.5, 0.25,
                        exit_mode="candle_trail_1min", stop_mode="pullback_low", target_mode="r_multiple")
    g = a["target_stop_grid"]
    assert g["applicable"] is False
    assert g["exit_mode"] == "candle_trail_1min"
    # mean pnl/sh = (0.4*4 - 0.2) / 5 = 0.28
    assert g["realized_expectancy_per_share"] == 0.28
    # No fixed-offset "in hindsight T/S" takeaway when the exit isn't fixed-offset.
    assert not any("target /" in t and "stop maximised" in t for t in a["takeaways"])


def test_structural_stop_alone_also_marks_grid_not_applicable():
    rows = [_t("WIN", 4.0, 0.6, 0.1, 0.5, 2.0) for _ in range(6)]
    a = build_analytics(rows, "cents", 0.5, 0.25, exit_mode="standard", stop_mode="pullback_low", target_mode="offset")
    assert a["target_stop_grid"]["applicable"] is False
