"""strategy.earliest_entry_time / latest_entry_time: a trigger touch outside the
configured [earliest, latest] time-of-day window (inclusive, "HH:MM" 24-hour) is
skipped like any other gate. Added after backtest data showed premarket entries
before 07:00 ET losing far more often than later ones (thin premarket liquidity).

A HOD that formed BEFORE earliest_entry_time is additionally discarded outright
once the clock reaches the window (not just the touch skipped): PBM 2026-04-20
armed its HOD at 04:00, got touched 591 times over the next 4+ hours (590
rejected purely for landing before the window), and the eventual "entry" was
just whichever stale touch happened to land first after 07:00 -- nothing about
volume or momentum was actually fresh at that point. So unlike a same-window
touch, a pre-window HOD does NOT get a later chance once the window opens; the
strategy instead hunts for a genuinely new HOD/pullback formed at or after the
window."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig,
    DataConfig,
    IBKRConfig,
    ReportingConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
    load_config,
)
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 4, 10)
_DAILY_VOLUME_FETCHER = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0,
        min_premarket_gain_pct=10.0,
        target_offset=0.20,
        stop_offset=0.10,
        intrabar_policy="conservative",
        scale_out=False,
        max_pullback_retrace_pct=55.0,
        earliest_entry_time="07:00",
    )
    defaults.update(overrides)
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("unused.csv"), reference_csv=None, cache_dir=Path("unused"), output_dir=Path("unused")),
        session=SessionConfig(),
        strategy=StrategyConfig(**defaults),
        risk=RiskConfig(),
        reporting=ReportingConfig(),
    )


def _bar(t, o, h, low, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": low, "close": c, "volume": v}


def _premarket_with_early_and_late_touch():
    """HOD 4.00 at 06:58:10, valid pullback to 3.55, an EARLY trigger touch at
    06:58:30 (before 07:00), then a gentle drift (keeps MACD bullish without
    re-crossing the trigger) up to a LATE touch exactly at 07:00:00, which
    completes a clean single-bar WIN (entry 4.01 -> exit 4.21) via the bar
    fallback."""
    start = pd.Timestamp(f"{DAY} 06:58:00", tz=TZ)
    rows = [
        _bar(start, 3.00, 3.05, 2.99, 3.02, 5000),
        _bar(start + pd.Timedelta(seconds=10), 3.02, 4.00, 3.01, 3.90, 5000),
        _bar(start + pd.Timedelta(seconds=20), 3.90, 3.91, 3.55, 3.60, 1000),
        _bar(start + pd.Timedelta(seconds=30), 3.90, 4.05, 3.89, 3.95, 3000),
    ]
    t = start + pd.Timedelta(seconds=40)
    o = 3.91
    for _ in range(8):
        rows.append(_bar(t, o, o + 0.01, o - 0.01, o + 0.01, 1000))
        o += 0.01
        t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, o, 4.45, o - 0.01, 4.30, 6000))
    return pd.DataFrame(rows)


def _prev_rth():
    return pd.DataFrame([_bar(pd.Timestamp("2026-04-09 15:59:00", tz=TZ), 2.0, 2.0, 2.0, 2.0, 1000)])


def test_pre_window_hod_break_invalidates_as_stale():
    """The HOD (4.00 @ 06:58:10) is broken by a trigger touch at 06:58:30, before
    the 07:00 window opens. That break is a first-pullback breakout that can't be
    traded on time, so under first_setup_only the candidate goes STALE_HOD right
    then -- it is NOT held open to re-break the same (now stale) level later."""
    premarket = _premarket_with_early_and_late_touch()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(earliest_entry_time="07:00"),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.result == "STALE_HOD"
    assert tr.entry_price is None
    assert tr.skipped_outside_entry_window == 1  # the 06:58:30 pre-window break
    assert tr.stale_hod_breaks == 1


def test_no_window_configured_lets_the_early_touch_fire():
    premarket = _premarket_with_early_and_late_touch()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(earliest_entry_time=None),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.result == "WIN"
    assert tr.skipped_outside_entry_window == 0


def test_latest_entry_time_blocks_a_too_late_touch():
    # The only valid trigger touch (07:00:00) lands after latest_entry_time 06:59:
    # the HOD breaks with no in-window entry, so the setup goes stale.
    premarket = _premarket_with_early_and_late_touch()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                       _config(earliest_entry_time=None, latest_entry_time="06:59"),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"
    assert tr.skipped_outside_entry_window == 1  # the 07:00:00 touch, now too late
    assert tr.stale_hod_breaks == 1


def test_latest_entry_time_boundary_is_inclusive():
    """A touch exactly AT latest_entry_time (07:00:00) must be allowed, not
    skipped -- the window is inclusive on both ends. (earliest_entry_time is
    left null here so the new pre-window-HOD staleness check, which only
    engages when earliest_entry_time is set, doesn't confound this.)"""
    premarket = _premarket_with_early_and_late_touch()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                       _config(earliest_entry_time=None, latest_entry_time="07:00"),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.result == "WIN"
    assert tr.entry_time == "2026-04-10T07:00:00-04:00"


def _premarket_with_stale_hod_then_fresh_post_window_setup():
    """A HOD/pullback/early-touch entirely BEFORE 07:00 (same shape as
    _premarket_with_early_and_late_touch, minus its late touch) -- discarded
    once the window opens. Then an independent, genuinely fresh HOD/pullback/
    trigger sequence forms entirely AFTER 07:00, which should trade normally."""
    start = pd.Timestamp(f"{DAY} 06:58:00", tz=TZ)
    rows = [
        _bar(start, 3.00, 3.05, 2.99, 3.02, 5000),
        _bar(start + pd.Timedelta(seconds=10), 3.02, 4.00, 3.01, 3.90, 5000),  # stale HOD 4.00 (06:58:10)
        _bar(start + pd.Timedelta(seconds=20), 3.90, 3.91, 3.55, 3.60, 1000),  # stale pullback low 3.55
        _bar(start + pd.Timedelta(seconds=30), 3.90, 4.05, 3.89, 3.95, 3000),  # early touch, pre-window, rejected
    ]
    # Flat filler staying above the stale pullback low (3.55) so it doesn't
    # additionally invalidate the stale setup via the (unrelated) retrace-depth
    # check -- only the new staleness reset, once the clock reaches 07:00,
    # should discard it. Ends with a bar exactly at 07:00:00: that bar is
    # consumed by the reset itself (mirroring the existing invalid-retrace
    # reset's same one-bar lag), so the fresh HOD forms on the bar right after.
    t = start + pd.Timedelta(seconds=40)
    for _ in range(9):
        rows.append(_bar(t, 3.65, 3.66, 3.60, 3.65, 500))
        t += pd.Timedelta(seconds=10)
    rows += [
        _bar(t, 3.65, 5.00, 3.60, 4.95, 5000),                             # fresh HOD 5.00 (07:00:10)
        _bar(t + pd.Timedelta(seconds=10), 4.95, 4.96, 4.80, 4.85, 1000),  # fresh pullback to 4.80
    ]
    o = 4.85
    t2 = t + pd.Timedelta(seconds=20)
    for _ in range(8):
        rows.append(_bar(t2, o, o + 0.01, o - 0.01, o + 0.01, 1000))
        o += 0.01
        t2 += pd.Timedelta(seconds=10)
    rows.append(_bar(t2, o, 5.50, o - 0.01, 5.40, 6000))  # fresh trigger touch, fires
    return pd.DataFrame(rows)


def test_fresh_hod_after_stale_one_still_fires_when_second_setup_allowed():
    """The pre-window HOD break makes the first setup stale. With first_setup_only
    FALSE, that break RESETS the scan instead of terminating it, so a genuinely
    fresh HOD forming after the window is a valid SECOND setup and trades. (Under
    first_setup_only True the candidate would end at the stale break -- the fresh
    HOD is a second pullback, which that mode does not take.)"""
    premarket = _premarket_with_stale_hod_then_fresh_post_window_setup()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                       _config(earliest_entry_time="07:00", first_setup_only=False),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.result == "WIN"
    assert tr.premarket_high_time == "2026-04-10T07:00:10-04:00"  # the FRESH HOD, not the stale 06:58:10 one
    assert tr.entry_price == 5.01
    assert tr.skipped_outside_entry_window == 1  # the one early, pre-window touch of the stale HOD
    assert tr.stale_hod_breaks == 1


def test_earliest_and_latest_entry_time_flow_from_yaml_and_validate(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  earliest_entry_time: \"07:00\"\n  latest_entry_time: \"09:00\"\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.strategy.earliest_entry_time == "07:00"
    assert cfg.strategy.latest_entry_time == "09:00"
    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    cfg2 = load_config(p2)
    assert cfg2.strategy.earliest_entry_time is None  # default: no filter
    assert cfg2.strategy.latest_entry_time is None
    p3 = tmp_path / "config3.yaml"
    p3.write_text("strategy:\n  earliest_entry_time: \"07:00:00\"\n", encoding="utf-8")  # not "HH:MM"
    with pytest.raises(ValueError, match="earliest_entry_time"):
        load_config(p3)
    p4 = tmp_path / "config4.yaml"
    p4.write_text("strategy:\n  latest_entry_time: \"25:00\"\n", encoding="utf-8")  # out of range
    with pytest.raises(ValueError, match="latest_entry_time"):
        load_config(p4)
