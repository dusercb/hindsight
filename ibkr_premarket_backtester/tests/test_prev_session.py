from datetime import date, datetime
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
from premarket_backtester.engine import get_bars_with_prev_session
from premarket_backtester.ibkr_data import split_sessions

TZ = "America/New_York"


def _day_bars(day_str: str) -> pd.DataFrame:
    """One premarket bar (07:00 ET) and one RTH bar (09:35 ET) for a session, so
    the fixture works whether it plays the role of "day" (premarket half used)
    or "previous day" (RTH half used). Naive timestamps, as IBKR returns them."""
    times = [f"{day_str} 07:00:00", f"{day_str} 09:35:00"]
    n = len(times)
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(t) for t in times],
            "open": [10.0] * n,
            "high": [10.1] * n,
            "low": [9.9] * n,
            "close": [10.05] * n,
            "volume": [1000] * n,
        }
    )


def test_split_sessions_monday_finds_friday_close():
    # Monday 2026-04-06's previous session is Friday 2026-04-03, two calendar
    # days back -- not "day - 1" (Sunday), which is what the old code assumed.
    bars = pd.concat([_day_bars("2026-04-03"), _day_bars("2026-04-06")], ignore_index=True)
    premarket, prev_rth = split_sessions(bars, date(2026, 4, 6), TZ, "04:00", "09:30")
    assert not prev_rth.empty
    assert prev_rth["dt_et"].dt.date.iloc[0] == date(2026, 4, 3)


def test_split_sessions_midweek_holiday_gap():
    # 2026-04-09 (Thu) previous session is 2026-04-07 (Tue); 2026-04-08 (Wed) is
    # a holiday with no bars at all, so "day - 1" would also miss this one.
    bars = pd.concat([_day_bars("2026-04-07"), _day_bars("2026-04-09")], ignore_index=True)
    premarket, prev_rth = split_sessions(bars, date(2026, 4, 9), TZ, "04:00", "09:30")
    assert not prev_rth.empty
    assert prev_rth["dt_et"].dt.date.iloc[0] == date(2026, 4, 7)


def test_split_sessions_no_prior_data_returns_empty():
    bars = _day_bars("2026-04-06")
    premarket, prev_rth = split_sessions(bars, date(2026, 4, 6), TZ, "04:00", "09:30")
    assert prev_rth.empty
    assert not premarket.empty


def _config(cache_dir: Path) -> AppConfig:
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("unused.csv"), reference_csv=None, cache_dir=cache_dir, output_dir=cache_dir),
        session=SessionConfig(),
        strategy=StrategyConfig(),
        risk=RiskConfig(),
        reporting=ReportingConfig(),
    )


class FakeData:
    """Simulates the default 2-day window missing a holiday-adjacent-to-weekend
    gap, and a wider window finding it."""

    def __init__(self, narrow: pd.DataFrame, wide: pd.DataFrame):
        self.narrow = narrow
        self.wide = wide
        self.calls: list[str] = []

    def get_day_bars(self, ticker, day, duration="2 D"):
        self.calls.append(duration)
        return self.wide if duration != "2 D" else self.narrow


def test_get_bars_with_prev_session_retries_wider_on_long_gap(tmp_path):
    # Requested day only; the previous session is 4 calendar days back (holiday
    # sitting next to a weekend), so the narrow fetch alone has no prior session.
    narrow = _day_bars("2026-04-13")
    wide = pd.concat([_day_bars("2026-04-09"), _day_bars("2026-04-13")], ignore_index=True)
    data = FakeData(narrow, wide)
    cfg = _config(tmp_path)

    bars, premarket, prev_rth = get_bars_with_prev_session(data, "TEST", date(2026, 4, 13), cfg)

    assert data.calls == ["2 D", cfg.data.wide_duration]
    assert not prev_rth.empty
    assert prev_rth["dt_et"].dt.date.iloc[0] == date(2026, 4, 9)


def test_get_bars_with_prev_session_no_retry_when_narrow_already_has_it(tmp_path):
    bars_df = pd.concat([_day_bars("2026-04-03"), _day_bars("2026-04-06")], ignore_index=True)
    data = FakeData(bars_df, bars_df)
    cfg = _config(tmp_path)

    bars, premarket, prev_rth = get_bars_with_prev_session(data, "TEST", date(2026, 4, 6), cfg)

    assert data.calls == ["2 D"]  # no widened retry needed
    assert not prev_rth.empty


def test_get_bars_with_prev_session_gives_up_when_still_missing(tmp_path):
    # Even the widened fetch has no prior session -- e.g. the ticker's actual
    # first trading day. Falls back to the narrow result unchanged.
    narrow = _day_bars("2026-04-13")
    data = FakeData(narrow, narrow)
    cfg = _config(tmp_path)

    bars, premarket, prev_rth = get_bars_with_prev_session(data, "TEST", date(2026, 4, 13), cfg)

    assert data.calls == ["2 D", cfg.data.wide_duration]
    assert prev_rth.empty
    assert not premarket.empty
