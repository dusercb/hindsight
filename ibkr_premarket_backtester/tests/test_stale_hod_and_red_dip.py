"""Two related first-pullback correctness rules:

1. STALE HOD -- once the armed HOD is broken by a trigger touch that fails ANY
   entry gate, no valid entry filled at the break, so the first-pullback setup is
   spent. A re-break of a level price already blew past is no longer a FIRST
   pullback. Under first_setup_only (the default) the candidate ends as STALE_HOD;
   under first_setup_only=False the scan resets and hunts a fresh (second) setup.

2. require_red_dip_candle -- the dip after the HOD must contain at least one red
   10-second candle (a genuine down-tick), not just a green-staircase drift or a
   single-print wobble. Evaluated at the trigger touch; a dip with no red candle
   rejects the entry like any other gate (and thus goes stale).
"""
from datetime import date
from pathlib import Path

import pandas as pd

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig, load_config,
)
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0, min_premarket_gain_pct=25.0, target_offset=0.20, stop_offset=0.10,
        min_pullback_pct=10.0, intrabar_policy="conservative", scale_out=False,
    )
    defaults.update(overrides)
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("unused.csv"), reference_csv=None, cache_dir=Path("unused"), output_dir=Path("unused")),
        session=SessionConfig(), strategy=StrategyConfig(**defaults), risk=RiskConfig(), reporting=ReportingConfig(),
    )


def _bar(t, price, volume, high=None, low=None, close=None, open_=None):
    return {
        "dt_et": t, "open": price if open_ is None else open_, "high": price if high is None else high,
        "low": price if low is None else low, "close": price if close is None else close, "volume": volume,
    }


def _prev_rth():
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 50.0, 1000)])


def _daily_volume(baseline):
    return lambda ticker, day: baseline


def _climb_dip_rebreak(dip_red: bool) -> pd.DataFrame:
    """A clean climb to a 4.00 HOD, a single-bar dip to 3.55, then a re-break to
    4.40. The dip bar is RED (close 3.60 < open 3.98) or GREEN (a gap-down open
    3.55 that closes UP at 3.65) depending on `dip_red` -- the only difference
    between the two datasets is the color of that one dip candle."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    pullback_at = start + pd.Timedelta(seconds=1000)
    if dip_red:
        rows.append(_bar(pullback_at, 3.98, 500, high=3.99, low=3.55, close=3.60, open_=3.98))  # red
    else:
        rows.append(_bar(pullback_at, 3.55, 500, high=3.66, low=3.55, close=3.65, open_=3.55))  # green
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 3.66, 6000, high=4.40, low=3.65, close=4.30, open_=3.66))
    return pd.DataFrame(rows)


# --- require_red_dip_candle ---

def test_require_red_dip_candle_off_by_default_lets_a_green_only_dip_trade():
    premarket = _climb_dip_rebreak(dip_red=False)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is not None
    assert tr.result in ("WIN", "LOSS")
    assert tr.skipped_no_red_dip_candle == 0


def test_require_red_dip_candle_rejects_a_green_only_dip():
    premarket = _climb_dip_rebreak(dip_red=False)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                      _config(require_red_dip_candle=True), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"  # the break failed the gate -> setup spent
    assert tr.skipped_no_red_dip_candle == 1
    assert tr.stale_hod_breaks == 1


def test_require_red_dip_candle_allows_a_dip_that_has_a_red_candle():
    premarket = _climb_dip_rebreak(dip_red=True)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                      _config(require_red_dip_candle=True), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is not None
    assert tr.result in ("WIN", "LOSS")
    assert tr.skipped_no_red_dip_candle == 0


def test_require_red_dip_candle_flows_from_yaml_and_defaults_false(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  require_red_dip_candle: true\n", encoding="utf-8")
    assert load_config(p).strategy.require_red_dip_candle is True
    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    assert load_config(p2).strategy.require_red_dip_candle is False


# --- stale-HOD terminate vs. reset ---

def test_stale_hod_break_terminates_under_first_setup_only():
    """A green-only dip fails the red-dip gate at the trigger touch. Under
    first_setup_only (default) that break ends the candidate as STALE_HOD."""
    premarket = _climb_dip_rebreak(dip_red=False)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                      _config(require_red_dip_candle=True, first_setup_only=True),
                      daily_volume_fetcher=_daily_volume(100.0))
    assert tr.result == "STALE_HOD"
    assert tr.stale_hod_breaks == 1
    assert tr.entry_price is None


def test_stale_hod_break_resets_and_rescans_when_second_setup_allowed():
    """The SAME rejected break, with first_setup_only=False, resets the scan to
    hunt a fresh setup instead of terminating -- so the result is not STALE_HOD
    (here there's no later setup, so it ends NO_TRIGGER), and the break is still
    counted (stale_hod_breaks == 1)."""
    premarket = _climb_dip_rebreak(dip_red=False)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(),
                      _config(require_red_dip_candle=True, first_setup_only=False),
                      daily_volume_fetcher=_daily_volume(100.0))
    assert tr.result != "STALE_HOD"
    assert tr.stale_hod_breaks == 1
    assert tr.entry_price is None
