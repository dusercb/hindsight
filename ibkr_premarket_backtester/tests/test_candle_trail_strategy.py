"""The 'red-break, tight stop, 1-minute trailing runner' variant, and the base
chart-timeframe toggle. All three are additive toggles (defaults leave the original
behavior untouched):
  - stop_mode "red_candle_low": stop at the last red candle's OWN low (tighter than
    the whole dip's low).
  - exit_mode "candle_trail_1min": trail the stop to the prior completed 1-minute
    candle's low; drop the fixed target on the first fat-green 1-minute close.
  - chart_timeframe "1m": detect the whole setup on 1-minute candles.
"""
from datetime import date

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig,
)
from premarket_backtester.engine import summarize
from premarket_backtester.reporting import _trail_section
from premarket_backtester.strategy import backtest_one, build_1m_bars

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DV = lambda t, d: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    d = dict(min_rvol=2.5, min_premarket_gain_pct=25.0, offset_mode="cents", entry_mode="dip_red_break",
             stop_mode="pullback_low", target_mode="r_multiple", target_r_multiple=1.0, min_pullback_cents=0.08,
             intrabar_policy="conservative", scale_out=False, earliest_entry_time=None, latest_entry_time=None,
             min_premarket_volume=None, first_setup_only=True)
    d.update(overrides)
    return AppConfig(ibkr=IBKRConfig(), data=DataConfig(input_csv="u", reference_csv=None, cache_dir="u", output_dir="u"),
                     session=SessionConfig(), strategy=StrategyConfig(**d), risk=RiskConfig(), reporting=ReportingConfig())


def _bar(t, o, h, l, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _prev():
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 2.5, 2.5, 2.5, 2.5, 1000)])


def _rip(start):
    rows = []
    for k in range(40):
        o = 3.00 + 0.05 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.05, o - 0.01, o + 0.05, 8000))
    return rows


def _sess_redlow():
    """Dip low 4.20 (early flush), but the LAST red candle's own low is 4.45 (high
    4.60 -> trigger 4.61). So pullback_low stops at 4.20; red_candle_low stops at 4.45."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = _rip(start)
    t = start + pd.Timedelta(seconds=400)
    for (o, h, l, c) in [(4.95, 4.96, 4.20, 4.25), (4.55, 4.60, 4.45, 4.50)]:
        rows.append(_bar(t, o, h, l, c, 3000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.50, 4.58, 4.48, 4.55, 2000)); t += pd.Timedelta(seconds=10)   # green pause below 4.61
    rows.append(_bar(t, 4.55, 4.75, 4.54, 4.72, 9000)); t += pd.Timedelta(seconds=10)   # breaker
    rows.append(_bar(t, 4.72, 4.90, 4.70, 4.88, 6000)); t += pd.Timedelta(seconds=10)
    for _ in range(4):
        rows.append(_bar(t, 4.85, 4.90, 4.80, 4.85, 2000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_red_candle_low_is_tighter_than_pullback_low():
    pm = _sess_redlow()
    pl = backtest_one("TST", DAY, pm, pm, _prev(), _config(stop_mode="pullback_low"), daily_volume_fetcher=_DV)
    rc = backtest_one("TST", DAY, pm, pm, _prev(), _config(stop_mode="red_candle_low"), daily_volume_fetcher=_DV)
    assert pl.stop_price == pytest.approx(4.20)          # the whole dip's low
    assert rc.stop_price == pytest.approx(4.45)          # the last red candle's OWN low -> tighter
    assert rc.entry_price == pl.entry_price == pytest.approx(4.61)
    # Tighter stop -> a closer 1:1 target (4.77) that this tape reaches; the wide
    # pullback_low target (5.02) does not.
    assert rc.target_price == pytest.approx(4.77) and rc.result == "WIN"
    assert pl.target_price == pytest.approx(5.02)


def _sess_trail():
    """Enter ~4.61, dip low 4.20, 1:1 target ~5.02. Then: a fat-green 1-min candle
    (08:08) closes ~4.98 BELOW the target; 08:09 runs to 5.40 (past the target);
    08:10 pulls back. Standard takes 5.02; candle_trail drops the target on the fat
    green and rides the trail out at the 08:09 low (4.97)."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = _rip(start)
    t = start + pd.Timedelta(seconds=400)
    for (o, h, l, c) in [(4.95, 4.96, 4.75, 4.78), (4.78, 4.80, 4.55, 4.58), (4.58, 4.60, 4.20, 4.25)]:
        rows.append(_bar(t, o, h, l, c, 3000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.25, 4.35, 4.24, 4.33, 2000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.55, 4.70, 4.54, 4.66, 9000)); t += pd.Timedelta(seconds=10)   # breaker -> entry ~4.61
    while t < start + pd.Timedelta(seconds=480):
        rows.append(_bar(t, 4.66, 4.72, 4.62, 4.70, 3000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):                                                                  # 08:08 fat green, closes 4.98 < 5.02
        rows.append(_bar(t, 4.70, 4.99, 4.69, 4.98, 4000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):                                                                  # 08:09 runs to 5.40 (past target)
        rows.append(_bar(t, 4.98, 5.40, 4.97, 5.38, 5000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):                                                                  # 08:10 pulls back to 4.96 (hits trail 4.97)
        rows.append(_bar(t, 5.38, 5.39, 4.96, 5.00, 4000)); t += pd.Timedelta(seconds=10)
    for _ in range(3):
        rows.append(_bar(t, 5.00, 5.02, 4.98, 5.00, 2000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_candle_trail_drops_target_on_fat_green_and_trails_out():
    pm = _sess_trail()
    std = backtest_one("TST", DAY, pm, pm, _prev(), _config(exit_mode="standard"), daily_volume_fetcher=_DV)
    trl = backtest_one("TST", DAY, pm, pm, _prev(), _config(exit_mode="candle_trail_1min"), daily_volume_fetcher=_DV)
    # Standard banks the fixed 1:1 target the moment price reaches it.
    assert std.result == "WIN" and std.exit_price == pytest.approx(5.02)
    # candle_trail: the fat-green minute drops the target, so it does NOT take 5.02;
    # it holds past the target and exits later on the trailing stop (the 08:09 low).
    assert trl.result == "WIN"
    assert trl.exit_price != pytest.approx(5.02)
    assert trl.exit_price == pytest.approx(4.97)                     # prior 1-min candle low = trailing stop
    assert pd.Timestamp(trl.exit_time) > pd.Timestamp(std.exit_time)  # held past the fixed-target exit


def test_chart_timeframe_1m_detects_the_setup_on_minute_candles():
    pm = _sess_trail()
    ten = backtest_one("TST", DAY, pm, pm, _prev(), _config(chart_timeframe="10s"), daily_volume_fetcher=_DV)
    one = backtest_one("TST", DAY, pm, pm, _prev(), _config(chart_timeframe="1m"), daily_volume_fetcher=_DV)
    # Both find a trade, but off different candles -> different entries.
    assert ten.entry_price is not None and one.entry_price is not None
    assert one.entry_price != ten.entry_price


def test_candle_trail_records_runner_telemetry():
    pm = _sess_trail()
    trl = backtest_one("TST", DAY, pm, pm, _prev(), _config(exit_mode="candle_trail_1min"), daily_volume_fetcher=_DV)
    assert trl.trail_activated is True                       # the fat-green minute dropped the target
    assert trl.trail_stop_final == pytest.approx(4.97)       # the final ratcheted trail level
    assert "Runner" in trl.reason and "fat-green" in trl.reason
    # standard-mode trades never set the trail telemetry.
    std = backtest_one("TST", DAY, pm, pm, _prev(), _config(exit_mode="standard"), daily_volume_fetcher=_DV)
    assert std.trail_activated is False and std.trail_stop_final is None


def test_summary_and_html_expose_the_runner_split():
    pm = _sess_trail()
    trl = backtest_one("TST", DAY, pm, pm, _prev(), _config(exit_mode="candle_trail_1min"), daily_volume_fetcher=_DV)
    s = summarize([trl], 2000.0, 0.10, 0.20, exit_mode="candle_trail_1min")
    assert s["exit_mode"] == "candle_trail_1min"
    assert s["trail"]["runners"]["count"] == 1 and s["trail"]["base_hits"]["count"] == 0
    assert "Trailing Exit" in _trail_section(s)
    # standard runs carry no trail block (nothing to render).
    s2 = summarize([trl], 2000.0, 0.10, 0.20, exit_mode="standard")
    assert s2["trail"] is None and _trail_section(s2) == ""


def _sess_fast_to_target():
    """Enter ~4.61 (1:1 target 5.02), then rip straight to 5.05 WITHIN the entry
    minute -- the target is hit before any fat-green 1-minute candle completes, so
    the partial-runner branch fires (not the fat-green branch). Then it runs to 5.40
    and flushes."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = _rip(start)
    t = start + pd.Timedelta(seconds=400)
    for (o, h, l, c) in [(4.95, 4.96, 4.75, 4.78), (4.78, 4.80, 4.55, 4.58), (4.58, 4.60, 4.20, 4.25)]:
        rows.append(_bar(t, o, h, l, c, 3000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.25, 4.35, 4.24, 4.33, 2000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.55, 4.70, 4.54, 4.66, 9000)); t += pd.Timedelta(seconds=10)   # breaker -> entry ~4.61
    for c in (4.80, 5.05):                                                              # rip to 5.05 (tags 5.02) in the entry minute
        rows.append(_bar(t, c - 0.05, c + 0.03, c - 0.06, c, 6000)); t += pd.Timedelta(seconds=10)
    while t < start + pd.Timedelta(seconds=480):
        rows.append(_bar(t, 5.05, 5.08, 5.02, 5.06, 3000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):
        rows.append(_bar(t, 5.06, 5.40, 5.05, 5.38, 5000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):
        rows.append(_bar(t, 5.38, 5.39, 5.00, 5.02, 4000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):
        rows.append(_bar(t, 5.02, 5.03, 4.55, 4.60, 4000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_trail_partial_banks_at_target_then_runs_the_rest():
    pm = _sess_fast_to_target()
    whole = backtest_one("TST", DAY, pm, pm, _prev(),
                         _config(exit_mode="candle_trail_1min", trail_partial_at_target=1.0), daily_volume_fetcher=_DV)
    part = backtest_one("TST", DAY, pm, pm, _prev(),
                        _config(exit_mode="candle_trail_1min", trail_partial_at_target=0.5), daily_volume_fetcher=_DV)
    # 1.0 = the original all-out exit at the 1:1; no runner engaged.
    assert whole.exit_price == pytest.approx(5.02) and whole.trail_activated is False
    assert len(whole.exit_legs) == 1 and whole.exit_legs[0]["kind"] == "target"
    # 0.5 = bank half at the target, run the rest on the trail -> two legs, a runner,
    # and a realized R at least as good as the flat 1:1.
    assert part.trail_activated is True
    kinds = [(leg["kind"], round(leg["frac"], 2)) for leg in part.exit_legs]
    assert ("target", 0.5) in kinds and ("stop", 0.5) in kinds
    assert part.r_multiple >= whole.r_multiple


def test_chart_timeframe_2m_resamples_and_runs_without_error():
    # 2-minute resampling produces genuine 2-minute candles from 10-second bars...
    pm = _sess_trail()
    two = build_1m_bars(pm, "2min")
    gaps = two["dt_et"].diff().dropna().dt.total_seconds().unique()
    assert set(gaps) <= {120.0}                              # bars are 2 minutes apart
    # ...and the strategy runs on them end-to-end (a valid result, never an ERROR).
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(chart_timeframe="2m"), daily_volume_fetcher=_DV)
    assert tr.result != "ERROR"
