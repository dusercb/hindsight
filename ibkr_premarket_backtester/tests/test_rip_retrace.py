"""The dip/retrace check is measured against the rip itself -- the move from its
launch low (the lowest low printed while its HOD was still ratcheting) to that HOD,
treated as the rip's trending support line -- NOT against the move-from-open used
by the gain% gate. The distinction matters exactly when the rip launches from
below the open: a flush first, then the real move (the ELAB 2026-03-30 pattern)."""

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
DAY = date(2026, 5, 5)


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0,
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


def _flush_then_rip_premarket(dip_low: float):
    """Open 3.00, flush to 2.00, rip 2.00 -> ~4.01, dip to `dip_low`, then a clean
    breakout bar over the rip's HOD. The rip's launch low (2.00) sits well below
    the premarket open (3.00), so the rip-based and open-based retrace calcs
    disagree hard: a dip to 3.30 is 35% of the rip but 70% of the move-from-open."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    t = 0
    # A few flat opening bars at 3.00.
    for _ in range(3):
        rows.append(_bar(start + pd.Timedelta(seconds=10 * t), 3.00, 5000))
        t += 1
    # Flush: 3.00 down to 2.00. Gain-from-open is ~0% here, so the gain gate keeps
    # these bars invisible -- the flush must NOT register as a pullback or trip the
    # retrace rule; it simply becomes the launch low of the rip that follows.
    for k in range(1, 21):
        rows.append(_bar(start + pd.Timedelta(seconds=10 * t), 3.00 - 0.05 * k, 2000))
        t += 1
    # The rip: 2.00 -> 4.00 on green, heavy-volume bars (HOD ratchets to 4.01).
    for k in range(101):
        o = 2.00 + 0.02 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * t), o, 5000, high=o + 0.01, low=o - 0.01, close=o + 0.01, open_=o))
        t += 1
    # The dip (red, lighter volume than the rip's green bars).
    rows.append(_bar(start + pd.Timedelta(seconds=10 * t), 3.98, 1000, high=3.99, low=dip_low, close=dip_low + 0.05, open_=3.98))
    t += 1
    # Breakout bar through the rip's HOD; touches target only (clean win, no ambiguity).
    rows.append(_bar(start + pd.Timedelta(seconds=10 * t), 4.03, 6000, high=4.40, low=4.02, close=4.30, open_=4.03))
    return pd.DataFrame(rows)


def test_dip_holding_rip_support_is_valid_even_when_deep_vs_open(tmp_path):
    """Dip to 3.30 = 35% of the rip (2.00 -> 4.01): holds the trending support,
    still bullish, trades. Measured the old way (vs. the move-from-open, 3.00 ->
    4.01) the same dip is 70% and the candidate would have been discarded as
    INVALID_RETRACE without ever trading."""
    premarket = _flush_then_rip_premarket(dip_low=3.30)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=lambda t, d: 100.0)

    assert tr.invalid_retrace_attempts == 0
    assert tr.entry_price is not None
    assert tr.result in ("WIN", "LOSS")
    assert tr.rip_low == pytest.approx(1.99)  # first rip bar's low (o - 0.01), a cent under the flush bottom
    assert tr.anchor_price == pytest.approx(3.00)  # gain% anchor: still the premarket open
    assert tr.pullback_retrace_pct == pytest.approx(35.1, abs=1.0)


def test_dip_breaching_rip_support_is_still_invalid(tmp_path):
    """Dip to 2.90 = 55% of the rip: breaks the trending support line, formation
    dead -- the rip-based measurement loosens nothing for genuinely broken setups."""
    premarket = _flush_then_rip_premarket(dip_low=2.90)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(first_setup_only=True), daily_volume_fetcher=lambda t, d: 100.0)

    assert tr.result == "INVALID_RETRACE"
    assert "of the rip" in tr.reason
    assert tr.rip_low == pytest.approx(1.99)
    assert tr.entry_price is None


def test_gain_gate_still_measured_from_premarket_open(tmp_path):
    """The rip is huge (2.00 -> 4.01 = +100%) but the session never gets 50% above
    its OPEN (3.00; session high 4.40 = +46.7%) -- with min_premarket_gain_pct at
    50 the candidate must not trade: the gain% check stays anchored to the
    premarket open, unchanged by the rip-based retrace fix."""
    premarket = _flush_then_rip_premarket(dip_low=3.30)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(min_premarket_gain_pct=50.0), daily_volume_fetcher=lambda t, d: 100.0)

    assert tr.entry_price is None
    assert tr.result == "DID_NOT_QUALIFY"
    assert tr.disqualify_reason == "gain_pct"
    assert "premarket open" in tr.reason
