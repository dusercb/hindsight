"""Tests for tick-based fill resolution beyond the ambiguous-both-touched case:
every candle that breaks through the trigger, stop, or target gets resolved from
its real trade ticks when available -- including a clean (non-ambiguous) entry bar
and a clean single-sided exit bar, plus simulated order latency."""
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
)
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 4, 10)
_DAILY_VOLUME_FETCHER = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_premarket_gain_pct=25.0,
        target_offset=0.20,
        stop_offset=0.10,
        min_pullback_pct=10.0,
        intrabar_policy="ticks",
        order_latency_seconds=0.0,
        scale_out=False,  # these tests pin the single all-out exit behavior
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


def _clean_entry_then_separate_exit_premarket():
    """Climb 3.00 -> 4.00, valid pullback to 3.55, then a CLEAN trigger bar (high
    4.05, doesn't touch stop 3.91 or target 4.21 -- no ambiguity), followed by a
    later bar where only the target is touched. Returns (premarket, prev_rth,
    trigger_at, exit_bar_at) so tests can target ticks at each bar independently.
    """
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append({"dt_et": start + pd.Timedelta(seconds=10 * k), "open": o, "high": o + 0.01,
                     "low": o - 0.01, "close": o + 0.01, "volume": 5000})
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append({"dt_et": pullback_at, "open": 3.98, "high": 3.99, "low": 3.55, "close": 3.60, "volume": 1000})
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append({"dt_et": trigger_at, "open": 3.98, "high": 4.05, "low": 4.00, "close": 4.02, "volume": 6000})
    # A few quiet bars, then the exit bar (target-only touch).
    for k in range(1, 4):
        rows.append({"dt_et": trigger_at + pd.Timedelta(seconds=10 * k), "open": 4.03, "high": 4.06,
                     "low": 4.01, "close": 4.04, "volume": 2000})
    exit_bar_at = trigger_at + pd.Timedelta(seconds=40)
    rows.append({"dt_et": exit_bar_at, "open": 4.10, "high": 4.30, "low": 4.08, "close": 4.25, "volume": 5000})
    premarket = pd.DataFrame(rows)
    prev_rth = pd.DataFrame(
        [{"dt_et": pd.Timestamp("2026-04-09 15:59:00", tz=TZ), "open": 2.0, "high": 2.0,
          "low": 2.0, "close": 2.0, "volume": 1000}]
    )
    return premarket, prev_rth, trigger_at, exit_bar_at


def test_clean_entry_bar_resolved_from_ticks_not_bar_open():
    """A non-ambiguous entry bar still gets its fill from real ticks when
    available, not the bar's open -- the crossing print may differ from both."""
    premarket, prev_rth, trigger_at, exit_bar_at = _clean_entry_then_separate_exit_premarket()

    def fetcher(ticker, start, end):
        if start == trigger_at:
            # Trigger is 4.01; the real tape shows it crossed at 4.015, not
            # exactly the bar's open (3.98) as the bar-based fallback would guess.
            return pd.DataFrame({
                "time": [trigger_at + pd.Timedelta(seconds=2), trigger_at + pd.Timedelta(seconds=4)],
                "price": [4.00, 4.017],
                "size": [100, 100],
            })
        return pd.DataFrame({
            "time": [exit_bar_at + pd.Timedelta(seconds=2)],
            "price": [4.21],
            "size": [100],
        })

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.path_ambiguous is False
    assert tr.entry_fill_source == "ticks"
    assert tr.entry_price == 4.02


def test_clean_single_sided_exit_resolved_from_ticks_shows_real_overshoot():
    """A bar that only touches the target (no ambiguity) still gets tick
    resolution, and the exit price is the real print -- an overshoot past the
    idealized target level -- not an assumed exact fill at target."""
    premarket, prev_rth, trigger_at, exit_bar_at = _clean_entry_then_separate_exit_premarket()

    def fetcher(ticker, start, end):
        if start == trigger_at:
            return pd.DataFrame({"time": [trigger_at + pd.Timedelta(seconds=2)], "price": [4.02], "size": [100]})
        # Target is 4.22 (fill 4.02 + 0.20); real tape gaps through it to 4.28.
        return pd.DataFrame({
            "time": [exit_bar_at + pd.Timedelta(seconds=1), exit_bar_at + pd.Timedelta(seconds=3)],
            "price": [4.15, 4.28],
            "size": [100, 100],
        })

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "WIN"
    assert tr.path_ambiguous is False
    assert tr.ambiguity_resolution is None  # only set for genuinely ambiguous bars
    assert tr.exit_fill_source == "ticks"
    assert tr.exit_price == 4.28  # real print, not the idealized 4.22 target


def test_ticks_unavailable_falls_back_to_idealized_bar_fill():
    """When the tick fetcher raises/returns nothing useful, both entry and exit
    fall back to the existing bar-based idealized-fill behavior."""
    premarket, prev_rth, trigger_at, exit_bar_at = _clean_entry_then_separate_exit_premarket()

    def empty_fetcher(ticker, start, end):
        return pd.DataFrame(columns=["time", "price", "size"])

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=empty_fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.entry_fill_source == "bar"
    assert tr.exit_fill_source == "bar"
    assert tr.entry_price == 4.01  # max(trigger, bar open) = max(4.01, 3.98)
    assert tr.exit_price == tr.target_price  # idealized exact fill, no tick data to say otherwise


def test_tick_fill_beyond_entry_limit_does_not_enter():
    """A marketable limit at trigger + entry_limit_slippage_cents can never fill
    above itself. When the tape gaps straight through the trigger (halt resume,
    spike print), the tick-resolved fill must be rejected like the bar fallback
    already was -- unenforced, a 4.51 trigger once 'filled' at 7.94 and instantly
    stopped out, a phantom break-even trade."""
    premarket, prev_rth, trigger_at, exit_bar_at = _clean_entry_then_separate_exit_premarket()

    def gapped_fetcher(ticker, start, end):
        # Trigger is 4.01, limit 4.21 (5%); the tape jumped to 7.90 and stayed.
        return pd.DataFrame({
            "time": [pd.Timestamp(start) + pd.Timedelta(seconds=s) for s in (1, 3)],
            "price": [7.90, 7.94],
            "size": [100, 100],
        })

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=gapped_fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    # Price gapped through the trigger too fast to fill: the HOD broke with no
    # valid entry, so the first-pullback setup goes stale.
    assert tr.skipped_fill_exceeds_limit >= 1
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"
    assert tr.stale_hod_breaks >= 1


def test_order_latency_delays_entry_fill_to_a_later_tick():
    premarket, prev_rth, trigger_at, exit_bar_at = _clean_entry_then_separate_exit_premarket()

    def fetcher(ticker, start, end):
        if start == trigger_at:
            # Dense ticks so a short latency lands within the same bar, not on a
            # sparse, artificially-distant next print. Prices chosen to stay
            # clearly distinct after cent-rounding, so the fill value itself
            # (not just the timestamp) proves the delayed tick was used.
            return pd.DataFrame({
                "time": [trigger_at + pd.Timedelta(milliseconds=ms) for ms in (1900, 2000, 2100, 2200)],
                "price": [4.00, 4.01, 4.03, 4.05],
                "size": [100] * 4,
            })
        return pd.DataFrame({"time": [exit_bar_at + pd.Timedelta(seconds=1)], "price": [4.22], "size": [100]})

    tr = backtest_one(
        "TEST", DAY, premarket, premarket, prev_rth,
        _config(order_latency_seconds=0.1), tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER,
    )

    assert tr.entry_fill_source == "ticks"
    # Crossing tick is at +2.000s (price 4.01); with 0.1s latency the fill should
    # be the next tick at/after +2.100s (price 4.03), not the crossing print itself.
    assert tr.entry_price == 4.03
    assert tr.entry_time == (trigger_at + pd.Timedelta(milliseconds=2100)).isoformat()


def _clean_entry_that_never_touches_through_session_end():
    """Climb 3.00 -> 4.00, valid pullback, a clean trigger bar (stop 3.91, target
    4.21), then a couple of quiet bars that stay inside [stop, target] all the way
    to the last bar of the session -- so the position rides to session end without
    ever hitting stop or target (the plain EXIT_AT_OPEN case)."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append({"dt_et": start + pd.Timedelta(seconds=10 * k), "open": o, "high": o + 0.01,
                     "low": o - 0.01, "close": o + 0.01, "volume": 5000})
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append({"dt_et": pullback_at, "open": 3.98, "high": 3.99, "low": 3.55, "close": 3.60, "volume": 1000})
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append({"dt_et": trigger_at, "open": 3.98, "high": 4.05, "low": 4.00, "close": 4.02, "volume": 6000})
    for k in range(1, 3):
        rows.append({"dt_et": trigger_at + pd.Timedelta(seconds=10 * k), "open": 4.03, "high": 4.06,
                     "low": 4.01, "close": 4.04, "volume": 2000})
    closing_bar_at = trigger_at + pd.Timedelta(seconds=20)  # the last bar of the session
    premarket = pd.DataFrame(rows)
    prev_rth = pd.DataFrame(
        [{"dt_et": pd.Timestamp("2026-04-09 15:59:00", tz=TZ), "open": 2.0, "high": 2.0,
          "low": 2.0, "close": 2.0, "volume": 1000}]
    )
    return premarket, prev_rth, trigger_at, closing_bar_at


def test_plain_exit_at_open_resolves_from_ticks_not_the_closing_bar_ohlc():
    """Nothing ever hits stop or target -- the position rides to session end.
    That exit is still a real fill and gets its own tick fetch for the closing
    bar, using the actual last print there (4.06) rather than the closing bar's
    OHLC close (4.04)."""
    premarket, prev_rth, trigger_at, closing_bar_at = _clean_entry_that_never_touches_through_session_end()

    def fetcher(ticker, start, end):
        if start == trigger_at:
            return pd.DataFrame({"time": [trigger_at + pd.Timedelta(seconds=2)], "price": [4.02], "size": [100]})
        if start == closing_bar_at:
            return pd.DataFrame({
                "time": [closing_bar_at + pd.Timedelta(seconds=3), closing_bar_at + pd.Timedelta(seconds=7)],
                "price": [4.03, 4.06],
                "size": [100, 100],
            })
        return pd.DataFrame(columns=["time", "price", "size"])

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(exit_at_regular_open=True),
                      tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "EXIT_AT_OPEN"
    assert tr.exit_legs[-1]["kind"] == "session_end"
    assert tr.exit_price == pytest.approx(4.06)  # the real last print, not the bar's 4.04 close
    assert tr.exit_fill_source == "ticks"


def test_plain_exit_at_open_falls_back_to_bar_close_without_ticks():
    """When the closing bar's ticks are unavailable, the session-end exit falls
    back to the bar's aggregate close exactly as before this behavior existed."""
    premarket, prev_rth, trigger_at, closing_bar_at = _clean_entry_that_never_touches_through_session_end()

    def fetcher(ticker, start, end):
        if start == trigger_at:
            return pd.DataFrame({"time": [trigger_at + pd.Timedelta(seconds=2)], "price": [4.02], "size": [100]})
        return pd.DataFrame(columns=["time", "price", "size"])

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(exit_at_regular_open=True),
                      tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "EXIT_AT_OPEN"
    assert tr.exit_price == pytest.approx(4.04)  # bar-fallback close
    assert tr.exit_fill_source == "bar"
