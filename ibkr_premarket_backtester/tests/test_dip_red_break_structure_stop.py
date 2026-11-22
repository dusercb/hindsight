"""The 'buy the dip reversal' triple:

1. entry_mode "dip_red_break" -- the dip is built of red candles; breaking back
   above the LAST red candle's high is the first proof the sellers lost control.
   Each newer red candle trails the trigger down; a dip with no red candle yet
   produces no trigger at all.
2. stop_mode "pullback_low" -- the dip's own low is the thesis: price back below
   it means the reversal failed, so that low is the maximum loss wherever the
   entry filled (works with ANY entry mode).
3. target_mode "r_multiple" -- entry and max loss are both known at fill time,
   so the sell is placed target_r_multiple x that risk above the entry
   (target = fill + R * (fill - stop)): a constant reward:risk per trade even
   though the structural dollar risk varies.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig, load_config,
)
from premarket_backtester.engine import summarize
from premarket_backtester.models import TradeResult
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DV = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=2.5, min_premarket_gain_pct=25.0, offset_mode="cents",
        target_offset=0.20, stop_offset=0.10, min_pullback_cents=0.08,
        intrabar_policy="conservative", scale_out=False, earliest_entry_time=None,
        latest_entry_time=None, min_premarket_volume=None,
    )
    defaults.update(overrides)
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("u"), reference_csv=None, cache_dir=Path("u"), output_dir=Path("u")),
        session=SessionConfig(), strategy=StrategyConfig(**defaults), risk=RiskConfig(), reporting=ReportingConfig(),
    )


def _bar(t, o, h, l, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _prev():
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 2.5, 2.5, 2.5, 2.5, 1000)])


def _dip_session(red_dip: bool = True, wins: bool = True) -> pd.DataFrame:
    """Climb 3.00 -> 5.00 HOD, then a three-candle dip to 4.20 whose candles are
    RED (closes below opens; last red high = 4.60 -> trigger 4.61) or GREEN
    (same descending lows, closes above opens -> no red to break), a green pause
    bar, a breaker bar through 4.61, and either a run to 5.50 (wins) or a flush
    through the 4.20 dip low (loses)."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    for k in range(40):
        o = 3.00 + 0.05 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.05, o - 0.01, o + 0.05, 8000))
    t = start + pd.Timedelta(seconds=400)
    if red_dip:
        dip = [(4.95, 4.96, 4.75, 4.78), (4.78, 4.80, 4.55, 4.58), (4.58, 4.60, 4.20, 4.25)]
    else:  # same lows, but every candle closes UP -- no red candle to break
        dip = [(4.95, 4.96, 4.75, 4.96), (4.76, 4.78, 4.55, 4.77), (4.56, 4.58, 4.20, 4.57)]
    for (o, h, l, c) in dip:
        rows.append(_bar(t, o, h, l, c, 3000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.25, 4.35, 4.24, 4.33, 2000)); t += pd.Timedelta(seconds=10)   # green pause, below trigger
    rows.append(_bar(t, 4.55, 4.70, 4.54, 4.65, 9000)); t += pd.Timedelta(seconds=10)   # breaker: crosses 4.61
    if wins:
        rows.append(_bar(t, 4.70, 5.00, 4.65, 4.95, 6000)); t += pd.Timedelta(seconds=10)
        rows.append(_bar(t, 4.95, 5.50, 4.90, 5.45, 7000)); t += pd.Timedelta(seconds=10)
        for _ in range(3):
            rows.append(_bar(t, 5.40, 5.45, 5.35, 5.40, 2000)); t += pd.Timedelta(seconds=10)
    else:
        rows.append(_bar(t, 4.50, 4.52, 4.15, 4.18, 6000)); t += pd.Timedelta(seconds=10)
        for _ in range(3):
            rows.append(_bar(t, 4.16, 4.17, 4.10, 4.16, 2000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_dip_red_break_enters_on_the_break_of_the_last_red_candle():
    pm = _dip_session(red_dip=True, wins=True)
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(entry_mode="dip_red_break"), daily_volume_fetcher=_DV)
    # Last red candle's high is 4.60 (the THIRD red -- the trigger trailed down
    # from 4.96 past 4.80 to 4.60), so entry is its break at 4.61, below the HOD.
    assert tr.trigger_price == pytest.approx(4.61)
    assert tr.entry_price == pytest.approx(4.61)
    assert tr.entry_price < 5.00
    assert tr.entry_mode == "dip_red_break"


def test_a_dip_with_no_red_candle_never_triggers():
    pm = _dip_session(red_dip=False, wins=True)
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(entry_mode="dip_red_break"), daily_volume_fetcher=_DV)
    assert tr.entry_price is None
    assert tr.trigger_touches == 0
    assert tr.result == "NO_TRIGGER"


def test_pullback_low_stop_with_r_multiple_target_wins_at_2r():
    pm = _dip_session(red_dip=True, wins=True)
    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(entry_mode="dip_red_break", stop_mode="pullback_low",
                              target_mode="r_multiple", target_r_multiple=2.0),
                      daily_volume_fetcher=_DV)
    assert tr.entry_price == pytest.approx(4.61)
    assert tr.stop_price == pytest.approx(4.20)                     # the dip low, not an offset
    assert tr.target_price == pytest.approx(4.61 + 2 * 0.41)        # entry + 2 x known risk = 5.43
    assert tr.result == "WIN"
    assert tr.exit_price == pytest.approx(5.43)
    assert tr.r_multiple == pytest.approx(2.0, abs=0.05)


def test_pullback_low_stop_exits_at_the_dip_low_for_minus_one_r():
    pm = _dip_session(red_dip=True, wins=False)
    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(entry_mode="dip_red_break", stop_mode="pullback_low",
                              target_mode="r_multiple", target_r_multiple=2.0),
                      daily_volume_fetcher=_DV)
    assert tr.result == "LOSS"
    assert tr.exit_price == pytest.approx(4.20)   # the maximum loss the user defined
    assert tr.r_multiple == pytest.approx(-1.0, abs=0.05)


def test_structure_stop_and_r_target_also_work_with_hod_break():
    """stop_mode / target_mode are orthogonal knobs -- with the classic hod_break
    entry they give 'enter the HOD break, risk the whole dip, sell at 2R'."""
    pm = _dip_session(red_dip=True, wins=True)
    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(entry_mode="hod_break", stop_mode="pullback_low",
                              target_mode="r_multiple", target_r_multiple=2.0),
                      daily_volume_fetcher=_DV)
    assert tr.entry_price == pytest.approx(5.01)                    # HOD 5.00 + offset
    assert tr.stop_price == pytest.approx(4.20)
    assert tr.target_price == pytest.approx(5.01 + 2 * 0.81)        # 6.63, beyond this tape
    assert tr.result == "EXIT_AT_OPEN"                              # neither level hit


def test_yaml_flow_and_validation(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "strategy:\n  entry_mode: dip_red_break\n  stop_mode: pullback_low\n"
        "  target_mode: r_multiple\n  target_r_multiple: 3.0\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.strategy.entry_mode == "dip_red_break"
    assert cfg.strategy.stop_mode == "pullback_low"
    assert cfg.strategy.target_mode == "r_multiple"
    assert cfg.strategy.target_r_multiple == 3.0
    for body, key in (
        ("strategy:\n  stop_mode: structure\n", "stop_mode"),
        ("strategy:\n  target_mode: ratio\n", "target_mode"),
        ("strategy:\n  target_r_multiple: 0\n", "target_r_multiple"),
    ):
        bad = tmp_path / f"bad_{key}.yaml"
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            load_config(bad)


def test_summarize_reports_the_configured_r_ratio():
    def mk(res):
        tr = TradeResult(ticker="T", date="2026-05-05", result=res)
        tr.entry_price = 4.61
        tr.pnl_per_share = 0.82 if res == "WIN" else -0.41
        tr.pnl_total = tr.pnl_per_share * 100
        tr.r_multiple = 2.0 if res == "WIN" else -1.0
        return tr
    s = summarize([mk("WIN"), mk("LOSS")], 2000.0, 0.10, 0.20,
                  target_mode="r_multiple", target_r_multiple=2.0, stop_mode="pullback_low")
    assert s["risk_reward_configured"] == "+2R/-1R (stop @ pullback low)"
    assert s["expected_value_per_share"] is None      # no single cents figure exists
    assert s["expected_value_r"] == pytest.approx(0.5)  # 0.5*2R - 0.5*1R
    assert s["stop_mode"] == "pullback_low" and s["target_mode"] == "r_multiple"


def _red_break_session_with_bar_times():
    """Climb to a 4.00 HOD, one RED pullback candle (last red high 3.92 -> trigger
    3.93, BELOW the HOD), then a breaker bar crossing 3.93, then a clean target
    bar. Returns (premarket, breaker_at, exit_at) so the tick fetcher can target
    each bar precisely."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    for k in range(40):
        o = 3.00 + 0.025 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.025, o - 0.005, o + 0.025, 8000))
    t = start + pd.Timedelta(seconds=400)
    rows.append(_bar(t, 3.90, 3.92, 3.65, 3.70, 3000)); t += pd.Timedelta(seconds=10)   # RED dip candle, high 3.92
    breaker_at = t
    # crosses trigger 3.93; low stays above the eventual stop so the entry bar
    # doesn't wick itself out (fill 3.96, stop 3.86).
    rows.append(_bar(t, 3.90, 4.05, 3.89, 4.02, 9000)); t += pd.Timedelta(seconds=10)
    exit_at = t
    rows.append(_bar(t, 4.05, 4.30, 4.04, 4.25, 6000)); t += pd.Timedelta(seconds=10)   # hits target
    for _ in range(3):
        rows.append(_bar(t, 4.20, 4.24, 4.16, 4.20, 2000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows), breaker_at, exit_at


def test_dip_red_break_resolves_entry_and_exit_from_ticks_like_hod_break():
    """The tick fill machinery keys off the computed trigger, never off entry_mode,
    so dip_red_break's entry (break of the last red candle's high, BELOW the HOD)
    and its exit both resolve from REAL ticks -- exactly as hod_break does. Guards
    against ever gating tick fills to a single entry mode."""
    pm, breaker_at, exit_at = _red_break_session_with_bar_times()

    def fetcher(ticker, start, end):
        start = pd.Timestamp(start)
        if start == breaker_at:
            # Trigger is 3.93; the tape crossed at 3.96 (not the bar open 3.68 the
            # bar-fallback would guess) -- so a "ticks" fill proves ticks were used.
            return pd.DataFrame({"time": [breaker_at + pd.Timedelta(seconds=2)], "price": [3.96], "size": [100]})
        if start == exit_at:
            # Target is 4.16 (fill 3.96 + 0.20); the tape overshoots to 4.18.
            return pd.DataFrame({"time": [exit_at + pd.Timedelta(seconds=2)], "price": [4.18], "size": [100]})
        return pd.DataFrame(columns=["time", "price", "size"])

    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(entry_mode="dip_red_break", intrabar_policy="ticks"),
                      tick_fetcher=fetcher, daily_volume_fetcher=_DV)

    assert tr.entry_mode == "dip_red_break"
    assert tr.entry_price == pytest.approx(3.96)      # the real crossing print, not 3.93/bar-open
    assert tr.entry_fill_source == "ticks"            # <-- the whole point
    assert tr.trigger_price == pytest.approx(3.93)    # last red high 3.92 + entry_offset
    assert tr.entry_price < 4.00                       # entered BELOW the HOD (a real dip break)
    assert tr.result == "WIN"
    assert tr.exit_fill_source == "ticks"
    assert tr.exit_price == pytest.approx(4.18)        # real overshoot print, not the idealized target


def test_last_red_high_ceiling_excludes_run_up_candles():
    """A red candle that printed ABOVE the armed HOD is a later run-up bar, not a
    dip candle -- the ceiling filters it out so the dip trigger can't float up."""
    import pandas as pd
    from premarket_backtester.strategy import _last_red_high
    bars = pd.DataFrame([
        {"open": 4.0, "high": 4.0, "low": 3.9, "close": 3.95},   # HOD bar (idx 0), green-ish
        {"open": 3.9, "high": 3.92, "low": 3.6, "close": 3.65},  # RED, high 3.92 (in the dip)
        {"open": 6.0, "high": 6.5, "low": 5.8, "close": 5.9},    # RED, high 6.5 (run-up, above HOD)
    ])
    # No ceiling: the last red anywhere wins (the 6.5 run-up candle) -- the old bug.
    assert _last_red_high(bars, 0, 3) == 6.5
    # With the HOD as ceiling: only the sub-HOD dip candle counts.
    assert _last_red_high(bars, 0, 3, ceiling=4.0) == 3.92


def _rocket_above_hod_session():
    """HIND 2025-09-17 in miniature: climb to a 4.00 HOD, a SHALLOW green pullback
    (wick to 3.70, closes up -> no red dip candle, survives the retrace gate), then
    a violent GREEN rocket to ~7.0 far above the HOD, and finally a RED candle up at
    7.25. Pre-fix, dip_red_break chased that 7.25 red-candle break and entered ~7.26
    -- 80% above its own 4.00 HOD."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    for k in range(40):
        o = 3.00 + 0.025 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.025, o - 0.005, o + 0.025, 8000))
    t = start + pd.Timedelta(seconds=400)
    rows.append(_bar(t, 3.98, 4.00, 3.70, 3.92, 3000)); t += pd.Timedelta(seconds=10)   # green wick down, HOD stays 4.00
    for hi in (4.60, 5.40, 6.20, 7.00):                                                 # green rocket, all above HOD
        rows.append(_bar(t, hi - 0.30, hi, hi - 0.35, hi - 0.05, 9000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 7.20, 7.25, 6.80, 6.85, 9000)); t += pd.Timedelta(seconds=10)   # RED at the top (high 7.25)
    rows.append(_bar(t, 6.90, 7.60, 6.88, 7.55, 9000)); t += pd.Timedelta(seconds=10)   # would break 7.26 pre-fix
    for _ in range(3):
        rows.append(_bar(t, 7.50, 7.55, 7.40, 7.50, 3000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_dip_red_break_never_enters_above_the_hod_on_a_post_pullback_rocket():
    pm = _rocket_above_hod_session()
    tr = backtest_one("RKT", DAY, pm, pm, _prev(), _config(entry_mode="dip_red_break"), daily_volume_fetcher=_DV)
    # The fix: the dip trigger is capped at the HOD and above-HOD red candles are
    # ignored, so there is no valid dip entry here -- certainly not a ~7.26 chase.
    assert tr.entry_price is None or tr.entry_price <= 4.05
    assert tr.result not in ("WIN", "LOSS") or tr.entry_price <= 4.05
