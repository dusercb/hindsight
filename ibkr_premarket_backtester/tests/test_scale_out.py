"""Scale-out ladder: when the planned exit (target) is hit, sell scale_out_sell_pct
of the REMAINING shares at it, ratchet the stop to scale_out_stop_offset BELOW that
exit (a stop placed AT it would fill on the very next print -- price is already
trading right there), and raise the target by scale_out_step. The ladder completes
in at most scale_out_max_sells sells, the final one liquidating the remainder
(defaults: 50% -> 25% -> final 25%); a ratcheted stop or session end may cut it
short. A scaled trade's WIN/LOSS is the sign of its NET P&L across all legs."""

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
        scale_out=True,
        scale_out_sell_pct=50.0,
        scale_out_step=0.10,
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


def _premarket(trigger_bar, extra_bars=()):
    """Climb 3.00 -> 4.00 (HOD 4.00, so trigger 4.01), pullback to 3.55, then the
    given trigger bar dict and any extra bars after it."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append({"dt_et": start + pd.Timedelta(seconds=10 * k), "open": o, "high": o + 0.01,
                     "low": o - 0.01, "close": o + 0.01, "volume": 5000})
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append({"dt_et": pullback_at, "open": 3.98, "high": 3.99, "low": 3.55, "close": 3.60, "volume": 1000})
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append({"dt_et": trigger_at, **trigger_bar})
    for n, b in enumerate(extra_bars, start=1):
        rows.append({"dt_et": trigger_at + pd.Timedelta(seconds=10 * n), **b})
    return pd.DataFrame(rows), trigger_at


def _prev_rth():
    return pd.DataFrame([{"dt_et": pd.Timestamp("2026-04-09 15:59:00", tz=TZ), "open": 2.0, "high": 2.0,
                          "low": 2.0, "close": 2.0, "volume": 1000}])


def _tick_feed(trigger_at, *sec_price):
    return pd.DataFrame({
        "time": [trigger_at + pd.Timedelta(seconds=s) for s, _ in sec_price],
        "price": [p for _, p in sec_price],
        "size": [100] * len(sec_price),
    })


def test_tick_ladder_scales_out_twice_then_stops_remainder():
    """Entry 4.01 -> leg1 50% at 4.22 (target 4.21 crossed), stop ratchets to 4.20
    (0.01 below the hit goal), target to 4.31 -> leg2 25% at 4.34 (4.31 crossed),
    stop 4.30, target 4.41 -> remainder 25% stopped at 4.30 (<= 4.30). All within
    one bar's tick stream."""
    premarket, trigger_at = _premarket({"open": 4.02, "high": 4.45, "low": 3.98, "close": 4.21, "volume": 6000})
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.34), (4, 4.30))

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                      tick_fetcher=lambda t, s, e: feed, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.entry_price == 4.01
    assert tr.scale_outs == 2
    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "target", "stop"]
    assert [leg["frac"] for leg in tr.exit_legs] == [0.5, 0.25, 0.25]
    assert [leg["price"] for leg in tr.exit_legs] == [4.22, 4.34, 4.30]
    # Weighted exit: 0.5*4.22 + 0.25*4.34 + 0.25*4.30 = 4.27
    assert tr.exit_price == pytest.approx(4.27)
    assert tr.pnl_per_share == pytest.approx(0.26)
    assert tr.result == "WIN"
    assert tr.exit_fill_source == "ticks"
    assert "remainder stopped at 4.30" in tr.reason


def test_retest_of_the_hit_goal_does_not_stop_the_ladder():
    """The ratcheted stop sits 0.01 BELOW the hit goal. A print back AT the goal
    (4.21) must not stop the remainder -- with the old stop-at-goal rule the very
    next tick always filled it, collapsing every ladder into a single-price exit.
    Only trading THROUGH 4.20 takes the remainder out."""
    premarket, trigger_at = _premarket({"open": 4.02, "high": 4.45, "low": 3.98, "close": 4.21, "volume": 6000})
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.21), (4, 4.20))

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                      tick_fetcher=lambda t, s, e: feed, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.scale_outs == 1
    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "stop"]
    assert [leg["price"] for leg in tr.exit_legs] == [4.22, 4.20]  # 4.21 retest survived
    assert tr.pnl_per_share == pytest.approx(0.20)
    assert tr.result == "WIN"


def test_ladder_completes_in_max_sells_with_final_sell_liquidating():
    """Sell #scale_out_max_sells closes the WHOLE remainder: 50% -> 25% -> final
    25% (not another halving), and the ladder ends there -- no fourth sell even
    though the tape keeps climbing, and no session_end tail."""
    premarket, trigger_at = _premarket({"open": 4.02, "high": 4.60, "low": 3.98, "close": 4.55, "volume": 6000})
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.31), (4, 4.41), (5, 4.60))

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                      tick_fetcher=lambda t, s, e: feed, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.scale_outs == 3
    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "target", "target"]
    assert [leg["frac"] for leg in tr.exit_legs] == [0.5, 0.25, 0.25]
    assert [leg["price"] for leg in tr.exit_legs] == [4.22, 4.31, 4.41]
    assert tr.exit_price == pytest.approx(4.29)
    assert tr.pnl_per_share == pytest.approx(0.28)
    assert tr.result == "WIN"
    assert "ladder completed at 4.41" in tr.reason


def test_max_sells_is_configurable():
    premarket, trigger_at = _premarket({"open": 4.02, "high": 4.60, "low": 3.98, "close": 4.55, "volume": 6000})
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.31), (4, 4.41), (5, 4.60))

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(scale_out_max_sells=2),
                      tick_fetcher=lambda t, s, e: feed, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.scale_outs == 2
    assert [leg["frac"] for leg in tr.exit_legs] == [0.5, 0.5]  # second sell liquidates
    assert [leg["price"] for leg in tr.exit_legs] == [4.22, 4.31]
    assert tr.result == "WIN"


def test_scaled_remainder_exits_at_session_end():
    """The session-end leg is a real fill too, not just a mark -- it gets its own
    tick fetch for the closing bar and uses the actual last print there (4.27),
    NOT the closing bar's aggregate OHLC close (4.26) and NOT the earlier target
    leg's tick feed (a distinct window, so a fetcher that ignored `start`/`end`
    would leak the wrong value here)."""
    premarket, trigger_at = _premarket(
        {"open": 4.02, "high": 4.30, "low": 3.98, "close": 4.25, "volume": 6000},
        extra_bars=({"open": 4.25, "high": 4.28, "low": 4.23, "close": 4.26, "volume": 2000},),
    )
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.25))  # leg1 banks, then no more crossings
    closing_bar_at = trigger_at + pd.Timedelta(seconds=10)
    closing_feed = _tick_feed(closing_bar_at, (2, 4.27))  # the closing bar's own real last print

    def fetcher(ticker, start, end):
        return closing_feed if pd.Timestamp(start) == closing_bar_at else feed

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                      tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.scale_outs == 1
    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "session_end"]
    assert tr.exit_legs[1]["price"] == pytest.approx(4.27)
    assert tr.exit_price == pytest.approx(4.25)  # weighted 0.5*4.22 + 0.5*4.27, cent-rounded
    assert tr.exit_fill_source == "ticks"
    assert tr.result == "WIN"
    assert "Scaled out 1x" in tr.reason


def test_scaled_remainder_session_end_falls_back_to_bar_close_without_ticks():
    """When the closing bar's ticks are unavailable, the session-end leg falls
    back to the bar's aggregate close exactly as before."""
    premarket, trigger_at = _premarket(
        {"open": 4.02, "high": 4.30, "low": 3.98, "close": 4.25, "volume": 6000},
        extra_bars=({"open": 4.25, "high": 4.28, "low": 4.23, "close": 4.26, "volume": 2000},),
    )
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.25))
    closing_bar_at = trigger_at + pd.Timedelta(seconds=10)

    def fetcher(ticker, start, end):
        if pd.Timestamp(start) == closing_bar_at:
            return pd.DataFrame(columns=["time", "price", "size"])
        return feed

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                      tick_fetcher=fetcher, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "session_end"]
    assert tr.exit_legs[1]["price"] == pytest.approx(4.26)  # bar-fallback close
    assert tr.exit_fill_source == "bar"


def test_bar_fallback_ladder_ratchets_stop_pessimistically():
    """No ticks: leg1 banks 50% at the idealized target 4.22 and the stop ratchets
    to 4.21 (0.01 below). The SAME bar is then re-checked against the raised
    levels: its low (4.01) breaches the new stop AND its high (4.34) reaches the
    new target (4.32) -- OHLC can't order the prints, so the explicitly
    conservative policy assumes stop-first at 4.21. Net +0.195/share (lands on
    0.19 or 0.20 after cent-rounding), one scale-out, still a WIN."""
    premarket, _ = _premarket({"open": 4.02, "high": 4.34, "low": 4.01, "close": 4.33, "volume": 6000})

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(intrabar_policy="conservative"),
                      tick_fetcher=None, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.entry_price == 4.02  # bar fallback: max(trigger, bar open)
    assert tr.scale_outs == 1
    assert [leg["kind"] for leg in tr.exit_legs] == ["target", "stop"]
    assert [leg["price"] for leg in tr.exit_legs] == [4.22, 4.21]
    assert tr.pnl_per_share == pytest.approx(0.195, abs=0.006)
    assert tr.result == "WIN"
    assert tr.exit_fill_source == "bar"


def test_scale_out_off_single_exit_records_one_full_leg():
    premarket, trigger_at = _premarket({"open": 4.02, "high": 4.45, "low": 3.98, "close": 4.21, "volume": 6000})
    feed = _tick_feed(trigger_at, (1, 4.01), (2, 4.22), (3, 4.34), (4, 4.30))

    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(scale_out=False),
                      tick_fetcher=lambda t, s, e: feed, daily_volume_fetcher=_DAILY_VOLUME_FETCHER)

    assert tr.scale_outs == 0
    assert [leg["frac"] for leg in tr.exit_legs] == [1.0]
    assert tr.exit_price == pytest.approx(4.22)  # the real print at the first target crossing
    assert tr.result == "WIN"
