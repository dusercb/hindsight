"""Trading more than the first move of the day: strategy.max_trades_per_day caps
how many completed trades one ticker/date can produce. After a trade completes,
if the session isn't over, the hunt resumes right where it exited, looking for a
brand-new HOD/pullback/trigger -- the 2nd move, 3rd, etc. -- judged by the exact
same gates (gain% still anchored to the ORIGINAL premarket open). backtest_one
stays a thin single-result wrapper returning only the first move, unchanged."""
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
from premarket_backtester.reporting import trade_folder_name
from premarket_backtester.strategy import backtest_all_trades, backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DAILY_VOLUME_FETCHER = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0,
        min_premarket_gain_pct=10.0,
        target_offset=0.20,
        stop_offset=0.10,
        intrabar_policy="conservative",  # single-bar bar-fallback fills, no tick fetcher needed
        scale_out=False,
        max_pullback_retrace_pct=55.0,
        max_trades_per_day=1,
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


def _prev_rth():
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 2.0, 2.0, 2.0, 2.0, 1000)])


def _two_move_premarket():
    """Move 1: climb 3.00 -> 4.00, pullback to 3.55, trigger+exit (WIN, entry 4.02
    -> exit 4.22) all resolved on one bar via the bar fallback. Move 2: a fresh
    climb 4.49 -> 5.00, a valid pullback, trigger+exit (WIN, entry 5.02 -> exit
    5.22), also one bar. Gain% for move 2 is measured vs the SAME premarket open
    as move 1 (66.7% vs 33.3%), not reset to move 2's own local base."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    t = start
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(t, o, o + 0.01, o - 0.01, o + 0.01, 5000))
        t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 3.98, 3.99, 3.55, 3.60, 1000))
    t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.02, 4.45, 3.98, 4.30, 6000))
    t += pd.Timedelta(seconds=10)
    for k in range(50):
        o = 4.50 + 0.01 * k
        rows.append(_bar(t, o, o + 0.01, o - 0.01, o + 0.01, 5000))
        t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.98, 4.99, 4.80, 4.85, 1000))
    t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 5.02, 5.45, 4.98, 5.30, 6000))
    return pd.DataFrame(rows)


def _premarket_with_false_start():
    """Move 1 (same as above), then a false start (a small climb to 4.60 that
    immediately retraces past max_pullback_retrace_pct -- never triggers), then a
    genuine move 2 (climb to 5.00, valid pullback, trigger+exit)."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    t = start
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(t, o, o + 0.01, o - 0.01, o + 0.01, 5000))
        t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 3.98, 3.99, 3.55, 3.60, 1000))
    t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.02, 4.45, 3.98, 4.30, 6000))
    t += pd.Timedelta(seconds=10)
    for k in range(30):
        o = 4.30 + 0.01 * k
        rows.append(_bar(t, o, o + 0.01, o - 0.01, o + 0.01, 5000))
        t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.55, 4.60, 4.00, 4.05, 1000))  # deep retrace: invalidates the false start
    t += pd.Timedelta(seconds=10)
    for k in range(50):
        o = 4.50 + 0.01 * k
        rows.append(_bar(t, o, o + 0.01, o - 0.01, o + 0.01, 5000))
        t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.98, 4.99, 4.80, 4.85, 1000))
    t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 5.02, 5.45, 4.98, 5.30, 6000))
    return pd.DataFrame(rows)


def test_default_max_trades_per_day_is_one_move_only():
    premarket = _two_move_premarket()
    trades = backtest_all_trades("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                                  daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert len(trades) == 1
    assert trades[0].setup_index == 1
    assert trades[0].entry_price == 4.02


def test_max_trades_per_day_two_takes_both_moves():
    premarket = _two_move_premarket()
    trades = backtest_all_trades("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_trades_per_day=2),
                                  daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert len(trades) == 2
    assert [t.setup_index for t in trades] == [1, 2]
    assert [t.result for t in trades] == ["WIN", "WIN"]
    assert trades[0].entry_price == 4.02
    assert trades[1].entry_price == 5.02
    # Move 2's gain% is measured vs the SAME premarket open as move 1 (3.00),
    # not reset to move 2's own local base.
    assert trades[0].premarket_gain_pct == pytest.approx(33.33, abs=0.01)
    assert trades[1].premarket_gain_pct == pytest.approx(66.67, abs=0.01)
    # Move 2 starts strictly after move 1's exit.
    assert trades[1].entry_time > trades[0].exit_time


def test_max_trades_per_day_does_not_pad_with_synthetic_rows():
    """Only 2 real moves exist in the fixture; asking for up to 3 must not
    manufacture a fake 3rd row -- the day just ends with the 2 real trades."""
    premarket = _two_move_premarket()
    trades = backtest_all_trades("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_trades_per_day=3),
                                  daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert len(trades) == 2


def test_backtest_one_still_returns_only_the_first_move():
    """backtest_one's contract is unchanged: even with multi-trade enabled, it
    returns exactly the first move, matching backtest_all_trades(...)[0]."""
    premarket = _two_move_premarket()
    cfg = _config(max_trades_per_day=3)
    one = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), cfg, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    all_trades = backtest_all_trades("TEST", DAY, premarket, premarket, _prev_rth(), cfg, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert one.setup_index == 1
    assert one.entry_price == all_trades[0].entry_price == 4.02


def test_first_setup_only_true_stops_the_whole_day_on_a_later_invalid_retrace():
    """A false start during move 2's OWN hunt breaches max_pullback_retrace_pct
    before ever triggering. With first_setup_only=True (give up entirely on any
    invalid retrace), no further moves are attempted even though a real move 2
    setup follows later in the same session -- move 1 is the whole day."""
    premarket = _premarket_with_false_start()
    trades = backtest_all_trades("TEST", DAY, premarket, premarket, _prev_rth(),
                                  _config(max_trades_per_day=3, first_setup_only=True),
                                  daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert len(trades) == 1
    assert trades[0].setup_index == 1


def test_first_setup_only_false_survives_a_false_start_within_a_moves_own_hunt():
    """Same false-start fixture, but first_setup_only=False: the false start is
    abandoned (not the whole day) and the SAME move-2 hunt keeps scanning,
    finding the genuine move-2 setup afterward -- still 2 trades total."""
    premarket = _premarket_with_false_start()
    trades = backtest_all_trades("TEST", DAY, premarket, premarket, _prev_rth(),
                                  _config(max_trades_per_day=3, first_setup_only=False),
                                  daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert len(trades) == 2
    assert trades[1].entry_price == 5.02
    assert trades[1].invalid_retrace_attempts == 1  # the false start it survived


def test_trade_folder_name_disambiguates_only_when_not_the_first_move():
    assert trade_folder_name("ABC", "2026-05-05", 1) == "ABC_2026-05-05"
    assert trade_folder_name("ABC", "2026-05-05", 2) == "ABC_2026-05-05_move2"
    assert trade_folder_name("ABC", "2026-05-05") == "ABC_2026-05-05"  # default


def test_max_trades_per_day_flows_from_yaml_and_validates(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  max_trades_per_day: 3\n", encoding="utf-8")
    assert load_config(p).strategy.max_trades_per_day == 3
    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    assert load_config(p2).strategy.max_trades_per_day == 1  # default: today's behavior
    p3 = tmp_path / "config3.yaml"
    p3.write_text("strategy:\n  max_trades_per_day: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_trades_per_day"):
        load_config(p3)
    p4 = tmp_path / "config4.yaml"
    p4.write_text("strategy:\n  max_trades_per_day: 2.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_trades_per_day"):
        load_config(p4)
