"""Trading-blackout calendar: which dates the bot skips entirely (no scan, no
trade). The rules RESOLVE per year rather than being a static list -- US
Thanksgiving is the 4th Thursday of November and the Dec-20..Jan-1 holiday window
wraps the New Year, so both move every year and can't be hardcoded.

Pure date logic, no config/pandas imports, so it stays trivially unit-testable and
the config layer can reuse parse_recurring_range for validation without a cycle.
"""
from __future__ import annotations

from datetime import date, timedelta

# Mon=0 .. Sun=6 (date.weekday()); Thursday is 3, Saturday is 5.
_THURSDAY = 3


def thanksgiving(year: int) -> date:
    """US Thanksgiving Day: the 4th Thursday of November of `year`."""
    first = date(year, 11, 1)
    days_to_first_thursday = (_THURSDAY - first.weekday()) % 7
    return first + timedelta(days=days_to_first_thursday + 21)  # +3 weeks = 4th


def thanksgiving_week_bounds(year: int) -> tuple[date, date]:
    """The Monday..Friday of the trading week that contains Thanksgiving."""
    tg = thanksgiving(year)
    monday = tg - timedelta(days=tg.weekday())
    return monday, monday + timedelta(days=4)


def parse_recurring_range(spec: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse a recurring month/day range "MM-DD..MM-DD" into ((sm, sd), (em, ed)).
    Year-agnostic (it recurs every year); a start that falls AFTER its end wraps the
    New Year (e.g. "12-20..01-01"). Raises ValueError on any malformed piece."""
    if not isinstance(spec, str) or spec.count("..") != 1:
        raise ValueError(f"recurring range must look like 'MM-DD..MM-DD', got {spec!r}")
    start_s, end_s = (part.strip() for part in spec.split(".."))

    def _md(piece: str) -> tuple[int, int]:
        bits = piece.split("-")
        if len(bits) != 2:
            raise ValueError(f"range endpoint must be 'MM-DD', got {piece!r}")
        month, day = int(bits[0]), int(bits[1])
        # Validate against a leap year so 02-29 is accepted as a legal recurring day.
        date(2024, month, day)
        return month, day

    return _md(start_s), _md(end_s)


def _in_recurring_range(day: date, start_md: tuple[int, int], end_md: tuple[int, int]) -> bool:
    start = date(day.year, *start_md)
    end = date(day.year, *end_md)
    if start <= end:
        return start <= day <= end
    # Wraps the New Year: [start .. Dec 31] U [Jan 1 .. end], both anchored to day.year.
    return day >= start or day <= end


def blackout_reason(day: date, cfg) -> str | None:
    """A human-readable reason `day` is a trading blackout, or None if it's tradeable.
    `cfg` is duck-typed (a BlackoutConfig): .enabled, .skip_weekends, .thanksgiving_week,
    .recurring_ranges, .dates. Cheapest checks first."""
    if not getattr(cfg, "enabled", False):
        return None
    if getattr(cfg, "skip_weekends", False) and day.weekday() >= 5:
        return "weekend"
    if getattr(cfg, "thanksgiving_week", False):
        mon, fri = thanksgiving_week_bounds(day.year)
        if mon <= day <= fri:
            return f"Thanksgiving week (Thanksgiving {thanksgiving(day.year).isoformat()})"
    for spec in getattr(cfg, "recurring_ranges", ()) or ():
        start_md, end_md = parse_recurring_range(spec)
        if _in_recurring_range(day, start_md, end_md):
            return f"recurring blackout {spec}"
    if day.isoformat() in (getattr(cfg, "dates", ()) or ()):
        return f"blackout date {day.isoformat()}"
    return None
