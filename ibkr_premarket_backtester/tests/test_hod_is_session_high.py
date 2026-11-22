"""A HOD is, by definition, the HIGHEST price traded on the day so far -- so no
setup may ever arm a 'HOD' below a level price has already printed this session.

Before this was enforced, any reset that hunts a fresh setup (a stale break or an
invalid retrace under first_setup_only=False, or the pre-window HOD discard) rewound
the HOD tracker to -inf and let the scan re-arm a LOWER local peak as a new HOD and
trade its break -- e.g. AGPU 2026-04-22 peaked at 14.47 @ 08:05, faded, then armed a
12.20 bounce @ 08:10 and traded it. The scan now carries the true running session
high across every reset and move; a fresh HOD must exceed it.
"""
from datetime import date
from pathlib import Path

import pandas as pd

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig,
)
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DV = lambda ticker, day: 500.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=1.0, min_premarket_gain_pct=25.0, min_pullback_cents=0.08,
        max_pullback_retrace_pct=55.0, intrabar_policy="conservative", scale_out=False,
        first_setup_only=False, earliest_entry_time=None, latest_entry_time=None,
        block_buy_below_vwap=False, block_buy_below_ema9=False, block_buy_below_ema20=False,
        min_premarket_volume=None, require_red_dip_candle=False,
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
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 3.0, 3.0, 3.0, 3.0, 1000)])


def _climb(rows, t, start, end, step=0.20, dt=10):
    """Append a green up-staircase from `start` to `end`, one bar per `dt` sec."""
    p = start
    while p < end - 1e-9:
        rows.append(_bar(t, p, p + step, p - 0.02, p + step, 8000))
        p += step
        t += pd.Timedelta(seconds=dt)
    return t


def _peak_flush_then(second_peak: float) -> pd.DataFrame:
    """Climb 5.00 -> 10.00 (the true HOD), FLUSH to 6.00 (an >50% retrace of the
    rip that invalidates the first setup -> reset), then climb toward `second_peak`,
    dip, and re-break it. With second_peak below 10.00 that re-break is a LOWER-HOD
    trap that must NOT trade; above 10.00 it's a genuine new HOD that should."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    t = _climb(rows, start, 5.00, 10.00)          # -> true HOD 10.00 (last high ~10.00)
    # deep flush to 6.00: retrace (10-6)/(10-5)=80% of the rip -> invalid, resets
    rows.append(_bar(t, 9.80, 9.80, 6.00, 6.10, 6000)); t += pd.Timedelta(seconds=10)
    # recover toward second_peak
    t = _climb(rows, t, 6.10, second_peak)
    hi = rows[-1]["high"]
    # a dip (~10%) then a clean re-break of `hi`
    rows.append(_bar(t, hi, hi, hi - 0.60, hi - 0.55, 4000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, hi - 0.55, hi - 0.50, hi - 0.58, hi - 0.52, 3000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, hi - 0.50, hi + 0.80, hi - 0.51, hi + 0.75, 12000)); t += pd.Timedelta(seconds=10)
    for _ in range(5):
        rows.append(_bar(t, hi + 0.75, hi + 0.85, hi + 0.70, hi + 0.80, 5000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_lower_local_peak_after_a_reset_is_not_a_hod_and_does_not_trade():
    """The second peak (8.00) is BELOW the true HOD (10.00) that already printed, so
    even though it forms a clean pullback+break, no entry may arm there."""
    pm = _peak_flush_then(second_peak=8.00)
    tr = backtest_one("TRAP", DAY, pm, pm, _prev(), _config(), daily_volume_fetcher=_DV)
    assert tr.entry_price is None, f"entered a lower-HOD trap at {tr.entry_price}"
    assert tr.result != "WIN" and tr.result != "LOSS"
    # the recognized HOD is the true session high, not the 8.00 bounce
    assert tr.premarket_high >= 10.0


def test_a_genuine_new_high_after_a_reset_still_arms_and_trades():
    """When the second peak (12.00) EXCEEDS the prior HOD (10.00), it's a real new
    high of day -- the scan must still arm it and trade its break."""
    pm = _peak_flush_then(second_peak=12.00)
    tr = backtest_one("REAL", DAY, pm, pm, _prev(), _config(), daily_volume_fetcher=_DV)
    assert tr.entry_price is not None
    assert tr.result in ("WIN", "LOSS")
    assert tr.premarket_high >= 12.0           # armed the genuine new HOD
    assert tr.entry_price > 10.0               # entered above the old high, not below it


def test_first_setup_only_true_also_never_arms_below_the_session_high():
    """The invariant holds regardless of first_setup_only: a lower second peak can
    never become a HOD. (Under first_setup_only the first invalidation terminates
    anyway, but this pins that no lower-HOD entry sneaks through either mode.)"""
    pm = _peak_flush_then(second_peak=8.00)
    tr = backtest_one("TRAP", DAY, pm, pm, _prev(), _config(first_setup_only=True), daily_volume_fetcher=_DV)
    assert tr.entry_price is None
