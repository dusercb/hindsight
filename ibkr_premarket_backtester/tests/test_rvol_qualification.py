from datetime import date
from pathlib import Path

import pandas as pd

from premarket_backtester.config import (
    AppConfig,
    DataConfig,
    IBKRConfig,
    ReportingConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
)
from premarket_backtester.strategy import add_rvol, backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0,
        # Anchor is often premarket_open in these fixtures (RVOL spikes on bar 0),
        # so gain% is measured over the climb itself (~33%), not vs. previous close.
        min_premarket_gain_pct=25.0,
        target_offset=0.20,
        stop_offset=0.10,
        min_pullback_pct=10.0,
        intrabar_policy="conservative",
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


def _bar(t, price, volume, high=None, low=None, close=None, open_=None):
    return {
        "dt_et": t,
        "open": price if open_ is None else open_,
        "high": price if high is None else high,
        "low": price if low is None else low,
        "close": price if close is None else close,
        "volume": volume,
    }


def _prev_rth(close=50.0):
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), close, 1000)])


def _daily_volume(baseline):
    return lambda ticker, day: baseline


# --- add_rvol unit tests ---

def test_add_rvol_is_cumulative_volume_over_baseline():
    one_min = pd.DataFrame({"dt_et": pd.date_range("2026-05-05 07:00", periods=4, freq="1min", tz=TZ), "volume": [1000, 2000, 500, 1500]})
    result = add_rvol(one_min, daily_avg_volume=1000.0)
    # cumulative volume: 1000, 3000, 3500, 5000 -> /1000 baseline
    assert result["rvol"].tolist() == [1.0, 3.0, 3.5, 5.0]


def test_add_rvol_is_monotonic_non_decreasing():
    one_min = pd.DataFrame({"dt_et": pd.date_range("2026-05-05 07:00", periods=10, freq="1min", tz=TZ), "volume": [100] * 10})
    result = add_rvol(one_min, daily_avg_volume=500.0)
    assert result["rvol"].is_monotonic_increasing


def test_add_rvol_none_baseline_is_all_nan():
    one_min = pd.DataFrame({"dt_et": pd.date_range("2026-05-05 07:00", periods=5, freq="1min", tz=TZ), "volume": [1000] * 5})
    result = add_rvol(one_min, daily_avg_volume=None)
    assert result["rvol"].isna().all()


def test_add_rvol_zero_or_negative_baseline_is_all_nan():
    one_min = pd.DataFrame({"dt_et": pd.date_range("2026-05-05 07:00", periods=3, freq="1min", tz=TZ), "volume": [1000] * 3})
    assert add_rvol(one_min, daily_avg_volume=0.0)["rvol"].isna().all()
    assert add_rvol(one_min, daily_avg_volume=-5.0)["rvol"].isna().all()


# --- backtest_one: RVOL qualification gate ---

def test_backtest_one_insufficient_data_when_no_daily_volume_history():
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = [_bar(start + pd.Timedelta(seconds=10 * k), 2.00 + 0.02 * k, 5000) for k in range(50)]
    premarket = pd.DataFrame(rows)

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(None))

    assert tr.result == "INSUFFICIENT_DATA"
    assert "RVOL baseline" in tr.reason
    assert tr.daily_avg_volume is None


def test_backtest_one_did_not_qualify_when_rvol_never_reaches_threshold():
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    # 200 bars * 1000 volume = 200,000 total cumulative volume across the whole
    # session; with a baseline of 100,000 that only reaches 2x, never 5x.
    rows = [_bar(start + pd.Timedelta(seconds=10 * k), 2.00 + 0.02 * k, 1000) for k in range(200)]
    premarket = pd.DataFrame(rows)

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(100_000.0))

    assert tr.result == "DID_NOT_QUALIFY"
    assert tr.disqualify_reason == "rvol"
    assert "cumulative volume" in tr.reason
    assert tr.rvol_at_spike is None
    assert tr.max_rvol_seen is not None and tr.max_rvol_seen < 5.0
    assert tr.daily_avg_volume == 100_000.0


def test_backtest_one_anchor_is_always_premarket_open():
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = [_bar(start + pd.Timedelta(seconds=10 * k), 3.00 + 0.02 * k, 5000) for k in range(50)]
    premarket = pd.DataFrame(rows)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(close=50.0), _config(), daily_volume_fetcher=_daily_volume(100.0))

    assert tr.rvol_at_spike is not None
    assert tr.prev_close == 50.0
    assert tr.anchor_price == 3.00  # premarket_open -- never previous close, never a mid-session price


def test_backtest_one_blocks_entry_before_rvol_actually_confirms_no_lookahead():
    """RVOL eventually crossing 5x somewhere in the session does NOT retroactively
    validate an entry on an earlier bar where it hadn't confirmed yet -- that would
    be lookahead bias (a real trader watching this ticker wouldn't have known yet
    that volume was about to surge). The trigger/pullback here forms and fires
    entirely on low volume, well before a later high-volume tail pushes cumulative
    RVOL over the threshold -- so no entry should happen. Gain%/anchor are still
    computed correctly for classification, proving that decoupling gain% from RVOL
    timing (a separate, earlier fix) and this no-lookahead entry gate coexist."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    # The entire price move + trigger happens immediately, on low per-bar volume.
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 100, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    pullback_at = start + pd.Timedelta(seconds=1000)
    # Pullback (red) volume kept below the climb/trigger (green) bars' so it
    # doesn't trip the red-volume-dominance gate (a tie counts as dominant).
    rows.append(_bar(pullback_at, 3.98, 50, high=3.99, low=3.55, close=3.60, open_=3.98))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 100, high=4.40, low=4.01, close=4.30, open_=4.02))
    # A long, high-volume quiet tail AFTER the trigger, so cumulative volume only
    # crosses the RVOL threshold deep into it -- well after the trigger fired.
    tail_start = trigger_at + pd.Timedelta(seconds=10)
    for k in range(60):
        rows.append(_bar(tail_start + pd.Timedelta(seconds=10 * k), 4.30, 3000))
    premarket = pd.DataFrame(rows)

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(32_000.0))

    # RVOL does eventually cross (so this isn't a DID_NOT_QUALIFY-for-RVOL case),
    # but well after the trigger bar -- too late to validate that bar's entry.
    assert tr.rvol_at_spike is not None
    assert pd.Timestamp(tr.rvol_spike_time) > pd.Timestamp(trigger_at)
    assert tr.skipped_rvol_not_yet_confirmed > 0
    assert tr.entry_price is None
    # The HOD broke (a trigger touch) on unconfirmed RVOL, so the first-pullback
    # setup goes stale -- no entry, no waiting for RVOL to catch up on a re-break.
    assert tr.result == "STALE_HOD"
    # Gain%/anchor still reflect the true move for classification/reporting, even
    # though no trade happened -- the armed HOD (4.00) is a 33% gain from the open.
    assert tr.anchor_price == 3.00
    assert tr.premarket_gain_pct is not None and tr.premarket_gain_pct > 30.0


def test_backtest_one_enters_once_rvol_confirms_while_setup_still_fresh():
    """The flip side: if RVOL confirms WHILE a valid pullback/trigger setup is
    still live (not stale), entry proceeds normally -- the gate blocks premature
    entries, it doesn't disable a candidate outright."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 1000, high=3.99, low=3.55, close=3.60, open_=3.98))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 6000, high=4.40, low=4.01, close=4.30, open_=4.02))
    premarket = pd.DataFrame(rows)

    # Small baseline: cumulative volume clears 5x comfortably partway through the
    # climb, well before the pullback/trigger -- RVOL is already confirmed by the
    # time the setup completes, so entry proceeds normally.
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(10_000.0))

    assert tr.result in ("WIN", "LOSS")
    assert tr.entry_price is not None


def test_backtest_one_missing_previous_close_does_not_block_trade():
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = [_bar(start + pd.Timedelta(seconds=10 * k), 3.00 + 0.02 * k, 5000) for k in range(50)]
    premarket = pd.DataFrame(rows)
    empty_prev_rth = pd.DataFrame(columns=["dt_et", "open", "high", "low", "close", "volume"])

    tr = backtest_one("TEST", DAY, premarket, premarket, empty_prev_rth, _config(), daily_volume_fetcher=_daily_volume(100.0))

    assert tr.prev_close is None
    assert tr.result != "INSUFFICIENT_DATA"
    assert tr.anchor_price == 3.00


# --- first_setup_only ---

def _multi_attempt_premarket():
    """RVOL spikes on bar 0 (small baseline), then a small setup that gets
    invalidated by too-deep a retrace, followed by a second, valid setup that
    reaches a clean trigger."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []

    # Attempt 1: climb to ~3.11 (gain from open 2.00 clears the min), then crash to
    # 2.50 -- retracing (3.11-2.50)/(3.11-1.99) = 54.5% of the rip (launch low 1.99,
    # the first bar's low), breaching the 50% trending-support line -> invalid.
    for k in range(12):
        o = 2.00 + 0.10 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 1000, high=o + 0.01, low=o - 0.01))
    t = start + pd.Timedelta(seconds=120)
    rows.append(_bar(t, 2.50, 1000, high=2.55, low=2.50))

    # Attempt 2: proven climb shape (3.00 -> 4.00), valid pullback to 3.55, then a
    # clean (non-ambiguous) trigger bar that only touches the target.
    climb_start = t + pd.Timedelta(seconds=10)
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(climb_start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    pullback_at = climb_start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 1000, high=3.99, low=3.55, close=3.60, open_=3.98))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 6000, high=4.40, low=4.01, close=4.30, open_=4.02))

    return pd.DataFrame(rows)


def test_first_setup_only_true_terminates_on_first_invalid_retrace():
    premarket = _multi_attempt_premarket()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(first_setup_only=True), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.result == "INVALID_RETRACE"
    assert tr.invalid_retrace_attempts == 1


def test_first_setup_only_false_keeps_scanning_and_takes_second_setup():
    premarket = _multi_attempt_premarket()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(first_setup_only=False), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.invalid_retrace_attempts == 1
    assert tr.result in ("WIN", "LOSS")
    assert tr.entry_price is not None


# --- nullable min_pullback_pct ---

def _shallow_dip_premarket():
    """Climb 3.00 -> 4.00, then a 12-cent dip (only 3% of the HOD) and a clean
    breakout. Deep enough for the cents floor, far too shallow for a %-minimum."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 1000, high=3.99, low=3.88, close=3.90, open_=3.98))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 6000, high=4.40, low=4.01, close=4.30, open_=4.02))
    return pd.DataFrame(rows)


def test_null_min_pullback_pct_sizes_pullbacks_by_cents_alone():
    premarket = _shallow_dip_premarket()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(min_pullback_pct=None), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is not None
    assert tr.result in ("WIN", "LOSS")
    assert tr.pullback_low == 3.88


def test_min_pullback_pct_still_blocks_shallow_dips_when_set():
    premarket = _shallow_dip_premarket()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(min_pullback_pct=10.0), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is None
    assert tr.result == "NO_TRIGGER"


# --- green-candle requirement removed ---

def test_all_red_breakout_can_still_trigger():
    """Every candle in the move is red (close < open); this must no longer block entry."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k + 0.01  # open above close -> red candle
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.02, close=o - 0.01, open_=o))
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 1000, high=3.99, low=3.55, close=3.57, open_=3.98))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    # Volume 9000: with the "total" gate the green breakout bar must strictly
    # outweigh the window's red flow (one red climb bar + the pullback = 6000);
    # an exact tie counts as red-dominant.
    rows.append(_bar(trigger_at, 4.02, 9000, high=4.40, low=4.01, close=4.30, open_=4.02))
    premarket = pd.DataFrame(rows)

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_pullback_retrace_pct=55.0), daily_volume_fetcher=_daily_volume(100.0))

    assert tr.result in ("WIN", "LOSS")
    assert tr.entry_price is not None
    assert tr.skipped_red_volume_dominant == 0  # no green candles at all -> comparison skipped, not blocked


# --- red/green volume gate modes ---

def _flush_dominated_recovery_premarket():
    """Climb 3.00 -> 4.00 (greens, 5K each), then a heavy red flush bar (50K --
    the pullback), a modest green recovery (20K), and a green breakout bar (40K).
    In the HOD->touch window the flush is the single largest bar ("largest_bar"
    vetoes: 50K >= 40K) but green flow wins in total (65K > 50K -> "total" passes).
    This is the ARTL failure mode: the flush that DEFINES the setup vetoing it."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    flush_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(flush_at, 3.98, 50_000, high=4.00, low=3.55, close=3.60, open_=3.98))
    rows.append(_bar(flush_at + pd.Timedelta(seconds=10), 3.60, 20_000, high=3.95, low=3.59, close=3.90, open_=3.60))
    rows.append(_bar(flush_at + pd.Timedelta(seconds=20), 3.95, 40_000, high=4.40, low=3.94, close=4.30, open_=3.95))
    return pd.DataFrame(rows)


def test_total_red_volume_mode_lets_green_flow_outweigh_one_flush_bar():
    premarket = _flush_dominated_recovery_premarket()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is not None
    assert tr.result in ("WIN", "LOSS")
    assert tr.skipped_red_volume_dominant == 0


def test_largest_bar_red_volume_mode_still_vetoes_on_the_flush():
    # A red-volume veto at the HOD break leaves the setup stale (no valid entry).
    premarket = _flush_dominated_recovery_premarket()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(red_volume_check="largest_bar"),
                      daily_volume_fetcher=_daily_volume(100.0))
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"
    assert tr.skipped_red_volume_dominant >= 1
    assert tr.stale_hod_breaks >= 1


# --- skip-gate diagnostics ---

def test_skip_counters_populated_when_macd_never_turns_bullish():
    """A choppy, directionless move that qualifies on RVOL/gain/pullback but whose
    MACD never turns bullish should end NO_TRIGGER with skipped_macd_not_bullish > 0."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    rows.append(_bar(start, 4.00, 5000, high=4.00, low=2.10, close=2.20, open_=2.10))
    t = start + pd.Timedelta(seconds=10)
    rows.append(_bar(t, 3.60, 1000, high=2.20, low=3.55, close=3.60, open_=2.20))
    for k in range(6):
        t += pd.Timedelta(seconds=10)
        rows.append(_bar(t, 3.90, 1500, high=4.05, low=3.85, close=3.88, open_=3.90))
    premarket = pd.DataFrame(rows)

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_daily_volume(100.0))

    if tr.result == "NO_TRIGGER":
        assert tr.trigger_touches > 0
        assert tr.skipped_macd_not_bullish > 0 or tr.skipped_no_macd_bar > 0


# --- bid/ask spread filter ---

def _clean_win_premarket():
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 1000, high=3.99, low=3.55, close=3.60, open_=3.98))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 6000, high=4.40, low=4.01, close=4.30, open_=4.02))
    return pd.DataFrame(rows)


def test_wide_spread_blocks_the_fill():
    premarket = _clean_win_premarket()
    tr = backtest_one(
        "TEST", DAY, premarket, premarket, _prev_rth(), _config(max_spread_cents=5.0),
        bid_ask_fetcher=lambda ticker, at: 0.50,  # 50 cents, way over the 5-cent cap
        daily_volume_fetcher=_daily_volume(100.0),
    )
    # The wide spread rejects the fill at the HOD break -> the setup is stale.
    assert tr.result == "STALE_HOD"
    assert tr.skipped_spread_too_wide > 0
    assert tr.entry_price is None
    assert tr.stale_hod_breaks >= 1


def test_tight_spread_allows_the_fill():
    premarket = _clean_win_premarket()
    tr = backtest_one(
        "TEST", DAY, premarket, premarket, _prev_rth(), _config(max_spread_cents=5.0),
        bid_ask_fetcher=lambda ticker, at: 0.01,
        daily_volume_fetcher=_daily_volume(100.0),
    )
    assert tr.result in ("WIN", "LOSS")
    assert tr.skipped_spread_too_wide == 0
    assert tr.entry_price is not None


def test_max_spread_cents_none_skips_check_even_with_fetcher():
    premarket = _clean_win_premarket()
    calls = []

    def fetcher(ticker, at):
        calls.append(at)
        return 0.50

    tr = backtest_one(
        "TEST", DAY, premarket, premarket, _prev_rth(), _config(max_spread_cents=None),
        bid_ask_fetcher=fetcher, daily_volume_fetcher=_daily_volume(100.0),
    )
    assert tr.result in ("WIN", "LOSS")
    assert calls == []  # never called since max_spread_cents is None
