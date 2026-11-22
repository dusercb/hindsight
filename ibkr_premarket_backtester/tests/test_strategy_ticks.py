from dataclasses import replace
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
from premarket_backtester.strategy import _bar_seconds, backtest_one, first_touch_from_ticks, resolve_intrabar

STOP = 4.90
TARGET = 5.10
TRIGGER = 5.01


def ticks(*rows):
    """rows of (second, price) inside one bar -> tick DataFrame."""
    return pd.DataFrame(
        [
            {
                "time": pd.Timestamp(f"2026-04-10 07:12:{sec:02d}", tz="America/New_York"),
                "price": price,
                "size": 100,
            }
            for sec, price in rows
        ]
    )


def test_target_traded_first_is_win():
    t = ticks((0, 5.00), (1, 5.05), (2, 5.10), (4, 4.90))
    outcome, when, price = first_touch_from_ticks(t, STOP, TARGET)
    assert outcome == "WIN"
    assert when == pd.Timestamp("2026-04-10 07:12:02", tz="America/New_York")
    assert price == 5.10


def test_stop_traded_first_is_loss():
    t = ticks((0, 5.00), (1, 4.89), (2, 5.12))
    outcome, when, price = first_touch_from_ticks(t, STOP, TARGET)
    assert outcome == "LOSS"
    assert when == pd.Timestamp("2026-04-10 07:12:01", tz="America/New_York")
    assert price == 4.89


def test_crossing_beyond_level_counts():
    # A tick gapping through the level (not exactly touching it) still decides,
    # and the reported price is the real gapped print, not the level itself.
    t = ticks((0, 5.00), (3, 5.25))
    outcome, _, price = first_touch_from_ticks(t, STOP, TARGET)
    assert outcome == "WIN"
    assert price == 5.25


def test_entry_bar_ignores_ticks_before_trigger():
    # Dip below the stop level happens BEFORE the trigger crossed; position
    # did not exist yet, so it must not count as a stop-out.
    t = ticks((0, 4.85), (2, 4.95), (4, 5.01), (6, 5.10))
    outcome, when, price = first_touch_from_ticks(t, STOP, TARGET, entry_cross_price=TRIGGER)
    assert outcome == "WIN"
    assert when == pd.Timestamp("2026-04-10 07:12:06", tz="America/New_York")
    assert price == 5.10


def test_entry_bar_stop_after_fill_is_loss():
    t = ticks((0, 4.95), (2, 5.01), (5, 4.90), (7, 5.11))
    outcome, _, _ = first_touch_from_ticks(t, STOP, TARGET, entry_cross_price=TRIGGER)
    assert outcome == "LOSS"


def test_entry_bar_fill_tick_itself_does_not_exit():
    # First tick at/above the trigger is the fill; even if it is beyond the
    # target it cannot simultaneously be the exit.
    t = ticks((0, 4.95), (2, 5.15), (4, 4.90))
    outcome, _, _ = first_touch_from_ticks(t, STOP, TARGET, entry_cross_price=TRIGGER)
    assert outcome == "LOSS"


def test_entry_bar_without_trigger_cross_is_unresolved():
    # Bar OHLC said the trigger traded but the tick stream disagrees.
    t = ticks((0, 4.95), (2, 4.99))
    assert first_touch_from_ticks(t, STOP, TARGET, entry_cross_price=TRIGGER) is None


def test_no_level_traded_is_unresolved():
    t = ticks((0, 5.00), (5, 5.05))
    assert first_touch_from_ticks(t, STOP, TARGET) is None


def test_empty_or_missing_ticks_are_unresolved():
    assert first_touch_from_ticks(None, STOP, TARGET) is None
    assert first_touch_from_ticks(ticks(), STOP, TARGET) is None


def test_out_of_order_ticks_are_sorted_before_scanning():
    t = ticks((4, 4.90), (1, 5.10))
    outcome, _, _ = first_touch_from_ticks(t, STOP, TARGET)
    assert outcome == "WIN"


def test_latency_delays_stop_fill_to_a_later_tick_but_not_the_outcome():
    # Stop crosses at :02 (4.89); with 2s latency the fill is the next tick
    # at/after :04 -- a triggered stop still has to race to the market.
    # Outcome stays LOSS either way; the crossing decides it.
    t = ticks((0, 5.00), (2, 4.89), (3, 4.87), (4, 4.80), (6, 4.95))
    outcome, when, price = first_touch_from_ticks(t, STOP, TARGET, latency=pd.Timedelta(seconds=2))
    assert outcome == "LOSS"
    assert when == pd.Timestamp("2026-04-10 07:12:04", tz="America/New_York")
    assert price == 4.80


def test_latency_does_not_delay_target_fill():
    # The target is a limit order already resting at the exchange: it fills at
    # the crossing print itself (5.10 at :02), never at where the tape sat
    # `latency` later (5.05 at :04) -- delayed target fills were turning
    # spike-through-target sells into break-even phantoms.
    t = ticks((0, 5.00), (2, 5.10), (4, 5.05))
    outcome, when, price = first_touch_from_ticks(t, STOP, TARGET, latency=pd.Timedelta(seconds=2))
    assert outcome == "WIN"
    assert when == pd.Timestamp("2026-04-10 07:12:02", tz="America/New_York")
    assert price == 5.10


def test_stop_latency_past_end_of_ticks_falls_back_to_last_tick():
    t = ticks((0, 5.00), (2, 4.89), (3, 4.91))
    outcome, when, price = first_touch_from_ticks(t, STOP, TARGET, latency=pd.Timedelta(seconds=30))
    assert outcome == "LOSS"
    assert when == pd.Timestamp("2026-04-10 07:12:03", tz="America/New_York")
    assert price == 4.91


def test_bar_seconds_parsing():
    assert _bar_seconds("10 secs") == 10
    assert _bar_seconds("30 secs") == 30
    assert _bar_seconds("1 min") == 60
    assert _bar_seconds("5 mins") == 300
    assert _bar_seconds("1 hour") == 3600


# --- backtest_one integration: ambiguous entry bar resolved via ticks ---

TZ = "America/New_York"
DAY = date(2026, 4, 10)


def _config() -> AppConfig:
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(
            input_csv=Path("unused.csv"),
            reference_csv=None,
            cache_dir=Path("unused_cache"),
            output_dir=Path("unused_out"),
        ),
        session=SessionConfig(),
        strategy=StrategyConfig(
            # anchor_price is always premarket_open (3.00), so the eventual HOD
            # (~4.00) is only a ~33% gain -- keep the bar low enough to clear that,
            # unrelated to the production default.
            min_premarket_gain_pct=25.0,
            target_offset=0.20,
            stop_offset=0.10,
            min_pullback_pct=10.0,
            intrabar_policy="ticks",
            # These fixtures use synthetic ticks spaced ~2s apart for readability;
            # order_latency_seconds has its own dedicated tests with tick spacing
            # designed for it, so disable it here to keep this file focused on
            # sequencing/ambiguity resolution.
            order_latency_seconds=0.0,
            scale_out=False,  # these tests pin the single all-out exit behavior
        ),
        risk=RiskConfig(),
        reporting=ReportingConfig(),
    )


# A small daily-volume baseline so the very first premarket bar's volume (5000)
# already clears min_rvol(5.0) x this baseline -- RVOL qualifies immediately,
# and these tests can focus purely on the pullback/trigger/tick-resolution mechanics.
_DAILY_VOLUME_FETCHER = lambda ticker, day: 100.0  # noqa: E731


def _ambiguous_entry_bar_premarket() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Premarket where the trade triggers and the entry bar touches both levels.

    A steady 1-cent-per-bar rise 3.00 -> 4.00 (HOD), a valid pullback to 3.55,
    then a trigger bar (trigger 4.01) spanning 3.60-4.25: with target 4.21 and
    stop 3.91 the bar is path-ambiguous.

    Returns (premarket, prev_rth, trigger_at) -- trigger_at anchors the tick-window
    assertions in the tests below without hardcoding absolute clock times.
    """
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(
            {"dt_et": start + pd.Timedelta(seconds=10 * k), "open": o, "high": o + 0.01,
             "low": o - 0.01, "close": o + 0.01, "volume": 5000}
        )
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(
        {"dt_et": pullback_at, "open": 3.98, "high": 3.99, "low": 3.55, "close": 3.60, "volume": 1000}
    )
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(
        {"dt_et": trigger_at, "open": 3.98, "high": 4.25, "low": 3.60, "close": 4.10, "volume": 6000}
    )
    premarket = pd.DataFrame(rows)
    prev_rth = pd.DataFrame(
        [{"dt_et": pd.Timestamp("2026-04-09 15:59:00", tz=TZ), "open": 2.0, "high": 2.0,
          "low": 2.0, "close": 2.0, "volume": 1000}]
    )
    return premarket, prev_rth, trigger_at


def test_backtest_one_resolves_ambiguous_entry_bar_with_ticks():
    premarket, prev_rth, trigger_at = _ambiguous_entry_bar_premarket()
    calls = []

    def fake_fetcher(ticker, start, end):
        calls.append((ticker, start, end))
        # Pre-trigger dip to 3.60 (below stop) must be ignored; target trades first after the fill.
        return pd.DataFrame(
            {
                "time": [
                    trigger_at + pd.Timedelta(seconds=1),
                    trigger_at + pd.Timedelta(seconds=3),
                    trigger_at + pd.Timedelta(seconds=5),
                ],
                "price": [3.60, 4.01, 4.25],
                "size": [100, 100, 100],
            }
        )

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=fake_fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "WIN"
    assert tr.path_ambiguous is True
    assert tr.ambiguity_resolution == "ticks"
    assert tr.entry_fill_source == "ticks"
    assert tr.exit_fill_source == "ticks"
    assert tr.entry_price == 4.01
    # Exit price is the real tick print (4.25), not the idealized target level
    # (4.21) -- this bar's ticks show a real overshoot past the target.
    assert tr.exit_price == 4.25
    assert tr.exit_time == (trigger_at + pd.Timedelta(seconds=5)).isoformat()
    assert len(calls) == 1  # entry and exit resolution reuse the same fetched ticks
    ticker, win_start, win_end = calls[0]
    assert ticker == "TEST"
    assert pd.Timestamp(win_start) == trigger_at
    assert pd.Timestamp(win_end) == trigger_at + pd.Timedelta(seconds=10)


def test_backtest_one_falls_back_to_heuristic_when_no_ticks():
    """With intrabar_policy \"ticks\" but no ticks available, an ambiguous bar is
    resolved by the candle-path guess -- NOT an automatic conservative loss. This
    fixture's ambiguous bar is bullish (open 3.98 -> close 4.10), so the assumed
    path prints its low (3.60, through the stop) before its high: still a LOSS,
    but a reasoned one, labeled heuristic."""
    premarket, prev_rth, _ = _ambiguous_entry_bar_premarket()

    def empty_fetcher(ticker, start, end):
        return pd.DataFrame(columns=["time", "price", "size"])

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=empty_fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "LOSS"
    assert tr.path_ambiguous is True
    assert tr.ambiguity_resolution == "heuristic"
    assert tr.exit_price == 3.91
    assert "Ambiguous bar" in tr.reason and "heuristic" in tr.reason


def test_backtest_one_without_fetcher_uses_heuristic_not_conservative():
    premarket, prev_rth, _ = _ambiguous_entry_bar_premarket()

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, _config(), tick_fetcher=None, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "LOSS"
    assert tr.path_ambiguous is True
    assert tr.ambiguity_resolution == "heuristic"


def test_backtest_one_explicit_conservative_policy_still_counts_stop_first():
    """\"conservative\" remains available, but only as an explicit opt-in."""
    premarket, prev_rth, _ = _ambiguous_entry_bar_premarket()
    cfg = _config()
    cfg = replace(cfg, strategy=replace(cfg.strategy, intrabar_policy="conservative"))

    tr = backtest_one("TEST", DAY, premarket, premarket, prev_rth, cfg, tick_fetcher=None, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.result == "LOSS"
    assert tr.ambiguity_resolution == "conservative"


def test_resolve_intrabar_bullish_bar_prints_low_early():
    # Bullish (close >= open): assumed path open -> low -> high -> close; the low
    # goes through the stop before the high reaches the target.
    assert resolve_intrabar(open_=9.75, high=10.29, low=9.70, close=10.27, stop=9.74, target=10.04) == "LOSS"


def test_resolve_intrabar_bearish_bar_prints_high_early():
    # Bearish (close < open): assumed path open -> high -> low -> close; the high
    # tags the target before the slide through the stop.
    assert resolve_intrabar(open_=10.20, high=10.29, low=9.70, close=9.75, stop=9.74, target=10.28) == "WIN"


def test_resolve_intrabar_open_beyond_level_decides_immediately():
    # Opening at/beyond a level decides before any path assumption matters.
    assert resolve_intrabar(open_=10.30, high=10.35, low=9.70, close=9.80, stop=9.74, target=10.28) == "WIN"
    assert resolve_intrabar(open_=9.70, high=10.35, low=9.65, close=10.30, stop=9.74, target=10.28) == "LOSS"
