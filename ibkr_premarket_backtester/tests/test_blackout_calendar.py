"""Trading-blackout calendar: per-year Thanksgiving resolution, New-Year-wrapping
recurring ranges, weekends, explicit dates, and config validation."""

from datetime import date

import pytest

from premarket_backtester.calendar_blackout import (
    blackout_reason,
    parse_recurring_range,
    thanksgiving,
    thanksgiving_week_bounds,
)
from premarket_backtester.config import BlackoutConfig, load_config


def test_thanksgiving_is_fourth_thursday_of_november():
    # Known US Thanksgiving dates.
    assert thanksgiving(2024) == date(2024, 11, 28)
    assert thanksgiving(2025) == date(2025, 11, 27)
    assert thanksgiving(2026) == date(2026, 11, 26)


def test_thanksgiving_week_is_monday_through_friday():
    mon, fri = thanksgiving_week_bounds(2025)
    assert (mon, fri) == (date(2025, 11, 24), date(2025, 11, 28))


def test_parse_recurring_range_and_wrap_detection():
    assert parse_recurring_range("12-20..01-01") == ((12, 20), (1, 1))
    assert parse_recurring_range(" 03-01 .. 03-05 ") == ((3, 1), (3, 5))
    assert parse_recurring_range("02-29..03-01")[0] == (2, 29)  # leap day allowed
    for bad in ("12-20", "13-01..01-01", "12-32..01-01", "12/20..01/01", "a-b..c-d"):
        with pytest.raises(ValueError):
            parse_recurring_range(bad)


def _cfg(**kw):
    return BlackoutConfig(enabled=True, **kw)


def test_year_end_range_wraps_new_year():
    cfg = _cfg(recurring_ranges=["12-20..01-01"])
    assert blackout_reason(date(2025, 12, 20), cfg)      # inclusive start
    assert blackout_reason(date(2025, 12, 31), cfg)      # before year end
    assert blackout_reason(date(2026, 1, 1), cfg)        # inclusive end, next year
    assert blackout_reason(date(2026, 1, 2), cfg) is None # just past the window
    assert blackout_reason(date(2025, 12, 19), cfg) is None # just before it


def test_thanksgiving_week_blackout():
    cfg = _cfg(thanksgiving_week=True)
    assert blackout_reason(date(2025, 11, 27), cfg)      # Thanksgiving itself
    assert blackout_reason(date(2025, 11, 24), cfg)      # Monday of that week
    assert blackout_reason(date(2025, 11, 28), cfg)      # Friday after
    assert blackout_reason(date(2025, 11, 21), cfg) is None  # prior Friday, clear


def test_weekend_and_explicit_date_and_disabled():
    assert blackout_reason(date(2025, 7, 12), _cfg(skip_weekends=True))   # Saturday
    assert blackout_reason(date(2025, 7, 12), _cfg(skip_weekends=False)) is None
    assert blackout_reason(date(2025, 7, 4), _cfg(dates=["2025-07-04"]))
    assert blackout_reason(date(2025, 7, 4), _cfg(dates=["2025-07-05"])) is None
    # Disabled calendar never blacks anything out, whatever the rules say.
    off = BlackoutConfig(enabled=False, thanksgiving_week=True, recurring_ranges=["12-20..01-01"])
    assert blackout_reason(date(2025, 12, 25), off) is None


def test_config_loads_blackout_section(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "blackout:\n"
        "  enabled: true\n"
        "  thanksgiving_week: true\n"
        "  recurring_ranges:\n"
        "    - \"12-20..01-01\"\n"
        "  dates: [\"2025-07-04\"]\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.blackout.enabled and cfg.blackout.thanksgiving_week
    assert cfg.blackout.recurring_ranges == ["12-20..01-01"]
    assert blackout_reason(date(2025, 12, 25), cfg.blackout)


def test_config_rejects_malformed_range_and_date(tmp_path):
    bad_range = tmp_path / "r.yaml"
    bad_range.write_text('blackout:\n  recurring_ranges: ["13-40..01-01"]\n')
    with pytest.raises(ValueError, match="recurring_ranges"):
        load_config(bad_range)
    bad_date = tmp_path / "d.yaml"
    bad_date.write_text('blackout:\n  dates: ["2025-13-99"]\n')
    with pytest.raises(ValueError, match="blackout.dates"):
        load_config(bad_date)
