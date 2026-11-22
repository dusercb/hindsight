"""A bar-fetch window that straddles a DST change carries BOTH UTC offsets (e.g.
-04:00 for the prior EDT session and -05:00 for the EST day). pandas 3.0 refuses to
parse a mixed-offset column without utc=True, which used to blow up EVERY candidate
on the Mon after the Nov fall-back / the Mon of the Mar spring-forward with
'Mixed timezones detected'. split_sessions must now tolerate it."""
from datetime import date

import pandas as pd

from premarket_backtester.ibkr_data import _parse_bar_dates, split_sessions


def _mixed_offset_frame():
    # Fri 2025-10-31 is EDT (-04:00); Mon 2025-11-03 is EST (-05:00) -- DST ended Nov 2.
    return pd.DataFrame([
        {"date": "2025-10-31 09:30:00-04:00", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 10},
        {"date": "2025-10-31 15:59:00-04:00", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 10},
        {"date": "2025-11-03 07:00:00-05:00", "open": 2.0, "high": 2.2, "low": 1.9, "close": 2.1, "volume": 20},
        {"date": "2025-11-03 09:29:50-05:00", "open": 3.0, "high": 3.2, "low": 2.9, "close": 3.1, "volume": 30},
    ])


def test_plain_to_datetime_would_raise_on_mixed_offsets():
    # Guards the premise: the naive parse is exactly what broke.
    import pytest
    with pytest.raises(ValueError):
        pd.to_datetime(_mixed_offset_frame()["date"])


def test_parse_bar_dates_normalizes_mixed_offsets():
    d = _parse_bar_dates(_mixed_offset_frame()["date"])
    assert str(d.dt.tz) == "UTC"          # mixed offsets fall back to UTC normalization
    assert len(d) == 4


def test_split_sessions_survives_the_dst_fallback_window():
    pm, prev = split_sessions(_mixed_offset_frame(), date(2025, 11, 3), "America/New_York", "04:00", "09:30")
    # The Nov-3 EST premarket rows land in premarket; the Oct-31 EDT rows are the prior session.
    assert len(pm) == 2 and len(prev) == 2
    assert str(pm["dt_et"].dt.tz) == "America/New_York"
    assert pm["dt_et"].dt.strftime("%H:%M").tolist() == ["07:00", "09:29"]


def test_split_sessions_via_written_then_read_cache(tmp_path):
    # The engine round-trips bars through a CSV cache; a raw-string mixed-offset
    # read (read_csv gives str dtype, no raise) must still split cleanly.
    raw = tmp_path / "bars.csv"
    _mixed_offset_frame().to_csv(raw, index=False)
    rc = pd.read_csv(raw, parse_dates=["date"])
    pm, prev = split_sessions(rc, date(2025, 11, 3), "America/New_York", "04:00", "09:30")
    assert len(pm) == 2 and len(prev) == 2
