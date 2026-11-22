from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytz
from ib_insync.objects import BarData

from premarket_backtester.config import (
    AppConfig,
    DataConfig,
    IBKRConfig,
    ReportingConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
)
from premarket_backtester.ibkr_data import IBKRHistoricalData


class FakeIB:
    """Serves a fixed tick feed the way reqHistoricalTicks does: ascending from
    startDateTime, at most numberOfTicks per call. Also serves a fixed set of daily
    bars for reqHistoricalData, ignoring durationStr/endDateTime (tests control
    exactly what's "returned" by choosing the fixture's bars directly)."""

    def __init__(self, ticks=(), daily_bars=(), fundamental_xml=None):
        self.ticks = ticks
        self.daily_bars = daily_bars
        self.fundamental_xml = fundamental_xml
        self.calls = 0
        self.daily_calls = 0
        self.fundamental_calls = 0

    def reqHistoricalTicks(self, contract, startDateTime, endDateTime, numberOfTicks, whatToShow, useRth, ignoreSize=False):
        self.calls += 1
        sel = [t for t in self.ticks if t.time >= startDateTime]
        return sel[:numberOfTicks]

    def reqHistoricalData(self, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate=1, keepUpToDate=False):
        self.daily_calls += 1
        return list(self.daily_bars)

    def reqFundamentalData(self, contract, reportType, fundamentalDataOptions=None):
        self.fundamental_calls += 1
        return self.fundamental_xml

    def sleep(self, seconds):
        pass


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(
            input_csv=Path("unused.csv"),
            reference_csv=None,
            cache_dir=tmp_path,
            output_dir=tmp_path,
            tick_request_pause_s=0.0,
        ),
        session=SessionConfig(),
        strategy=StrategyConfig(),
        risk=RiskConfig(),
        reporting=ReportingConfig(),
    )


def test_get_ticks_paginates_dedupes_filters_and_caches(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    base = datetime(2026, 4, 10, 11, 16, 50, tzinfo=timezone.utc)  # 07:16:50 ET
    # 300 ticks per second (1-second timestamp resolution, like IBKR) for 12
    # seconds; the request window only covers the first 10. Forces pagination
    # across page boundaries that split a second.
    feed = [
        SimpleNamespace(time=base + timedelta(seconds=sec), price=4.0 + sec * 0.01 + k * 1e-6, size=1, exchange="X")
        for sec in range(12)
        for k in range(300)
    ]
    fake = FakeIB(feed)
    data.ib = fake
    data._contracts["TEST"] = object()  # skip contract qualification

    et = pytz.timezone("America/New_York")
    start = base.astimezone(et)
    end = (base + timedelta(seconds=10)).astimezone(et)

    df = data.get_ticks("TEST", start, end)
    assert len(df) == 3000              # only ticks inside [start, end)
    assert df["time"].max() < end
    assert df["price"].is_unique        # boundary-second ticks not double-counted
    assert fake.calls >= 3              # 3600-tick feed needed multiple pages

    fake.calls = 0
    df2 = data.get_ticks("TEST", start, end)
    assert fake.calls == 0              # second request served from disk cache
    assert len(df2) == 3000


def test_get_ticks_caches_empty_result(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    fake = FakeIB([])
    data.ib = fake
    data._contracts["TEST"] = object()
    data._mark_data_success()  # a HEALTHY farm's empty answer is real evidence

    et = pytz.timezone("America/New_York")
    start = et.localize(datetime(2026, 4, 10, 7, 16, 50))
    end = start + timedelta(seconds=10)

    df = data.get_ticks("TEST", start, end)
    assert df.empty
    assert fake.calls == 1

    df2 = data.get_ticks("TEST", start, end)
    assert df2.empty
    assert fake.calls == 1              # empty result cached; IBKR not re-asked


class FlakyIB(FakeIB):
    """Returns empty batches for the first `fail_times` tick requests, then serves
    the feed -- models a transient HMDS/pacing failure that clears on retry."""

    def __init__(self, ticks, fail_times=1):
        super().__init__(ticks)
        self.fail_times = fail_times

    def reqHistoricalTicks(self, *args, **kwargs):
        if self.calls < self.fail_times:
            self.calls += 1
            return []
        return super().reqHistoricalTicks(*args, **kwargs)


def _tick_window():
    et = pytz.timezone("America/New_York")
    base = datetime(2026, 4, 10, 11, 16, 50, tzinfo=timezone.utc)
    return base, base.astimezone(et), (base + timedelta(seconds=10)).astimezone(et)


def test_get_ticks_expect_trades_retries_once_on_empty_then_caches(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    base, start, end = _tick_window()
    feed = [SimpleNamespace(time=base + timedelta(seconds=s), price=4.0 + s * 0.01, size=1, exchange="X") for s in range(5)]
    fake = FlakyIB(feed, fail_times=1)
    data.ib = fake
    data._contracts["TEST"] = object()

    df = data.get_ticks("TEST", start, end, expect_trades=True)
    assert len(df) == 5            # first request failed empty; the paced retry succeeded
    assert fake.calls == 2

    df2 = data.get_ticks("TEST", start, end, expect_trades=True)
    assert len(df2) == 5
    assert fake.calls == 2         # successful result was cached normally


def test_get_ticks_expect_trades_gap_memo_caps_retries_across_runs(tmp_path):
    """The bar traded but IBKR keeps answering empty: never cached as ticks, but a
    sidecar gap memo counts attempts across runs -- retries continue (transient
    failures heal) until max_empty_tick_attempts (default 4), after which the
    window is a confirmed IBKR gap and costs zero requests forever."""
    data = IBKRHistoricalData(_config(tmp_path))
    _, start, end = _tick_window()
    fake = FakeIB([])
    data.ib = fake
    data._contracts["TEST"] = object()
    data._mark_data_success()  # gap attempts only count against a proven-healthy farm
    cache = data._tick_cache_path("TEST", start, end)

    df = data.get_ticks("TEST", start, end, expect_trades=True)
    assert df.empty
    assert fake.calls == 2                     # first-ever attempt + immediate retry
    assert not cache.exists()                  # never cached as an (empty) tick result
    assert cache.with_suffix(".gap.json").exists()

    assert data.get_ticks("TEST", start, end, expect_trades=True).empty
    assert fake.calls == 3                     # one more attempt (3 of 4)
    assert data.get_ticks("TEST", start, end, expect_trades=True).empty
    assert fake.calls == 4                     # budget exhausted (4 of 4)

    assert data.get_ticks("TEST", start, end, expect_trades=True).empty
    assert fake.calls == 4                     # confirmed gap: no live request at all


def test_gap_memo_cleared_when_ticks_finally_arrive(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    base, start, end = _tick_window()
    data.ib = FakeIB([])
    data._contracts["TEST"] = object()
    assert data.get_ticks("TEST", start, end, expect_trades=True).empty  # gap memo written

    feed = [SimpleNamespace(time=base + timedelta(seconds=s), price=4.0 + s * 0.01, size=1, exchange="X") for s in range(5)]
    fake = FakeIB(feed)
    data.ib = fake
    df = data.get_ticks("TEST", start, end, expect_trades=True)
    assert len(df) == 5
    assert fake.calls == 1
    assert not data._tick_cache_path("TEST", start, end).with_suffix(".gap.json").exists()

    assert len(data.get_ticks("TEST", start, end, expect_trades=True)) == 5
    assert fake.calls == 1                     # healed: served from the tick cache


def test_get_ticks_expect_trades_distrusts_poisoned_empty_cache(tmp_path):
    """Empty tick caches written before the expect_trades guard existed (e.g. the
    HCAI 2026-04-07 windows) are contradictory for bars that traded: ignore them,
    re-ask IBKR, and overwrite the cache once real ticks come back."""
    data = IBKRHistoricalData(_config(tmp_path))
    base, start, end = _tick_window()

    # Legacy behavior writes the poisoned empty cache (no expect_trades hint).
    data.ib = FakeIB([])
    data._contracts["TEST"] = object()
    data._mark_data_success()  # empty caches only ever write against a healthy farm
    assert data.get_ticks("TEST", start, end).empty
    assert data._tick_cache_path("TEST", start, end).exists()

    # IBKR has the ticks after all (transient failure) -- expect_trades heals the cache.
    feed = [SimpleNamespace(time=base + timedelta(seconds=s), price=4.0 + s * 0.01, size=1, exchange="X") for s in range(5)]
    fake = FakeIB(feed)
    data.ib = fake
    df = data.get_ticks("TEST", start, end, expect_trades=True)
    assert len(df) == 5
    assert fake.calls == 1

    df2 = data.get_ticks("TEST", start, end, expect_trades=True)
    assert len(df2) == 5
    assert fake.calls == 1         # healed cache now serves the window


def test_connect_sets_request_timeout_so_requests_cannot_hang_forever(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))

    class FakeConn:
        def connect(self, host, port, clientId, readonly):
            pass

    data.ib = FakeConn()
    data.connect()
    assert data.ib.RequestTimeout == 60.0  # ib_insync default is 0 = wait forever


def test_connect_passes_configured_readonly_flag_not_a_hardcoded_true(tmp_path):
    """connect() must honor config.ibkr.readonly (it used to hardcode readonly=True,
    silently ignoring the field). The backtester keeps it True as a hard interlock;
    a live bot would set it False -- so the flag has to actually reach ib.connect()."""
    import dataclasses

    captured = {}

    class FakeConn:
        def connect(self, host, port, clientId, readonly):
            captured["readonly"] = readonly

    base = _config(tmp_path)
    data = IBKRHistoricalData(base)
    data.ib = FakeConn()
    data.connect()
    assert captured["readonly"] is True  # default

    cfg = dataclasses.replace(base, ibkr=dataclasses.replace(base.ibkr, readonly=False))
    data2 = IBKRHistoricalData(cfg)
    data2.ib = FakeConn()
    data2.connect()
    assert captured["readonly"] is False  # the configured value actually reaches ib.connect()


def test_status_callback_reports_live_tick_requests(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    base, start, end = _tick_window()
    feed = [SimpleNamespace(time=base + timedelta(seconds=s), price=4.0, size=1, exchange="X") for s in range(3)]
    data.ib = FakeIB(feed)
    data._contracts["TEST"] = object()
    seen = []
    data.on_status = seen.append

    data.get_ticks("TEST", start, end, expect_trades=True)

    assert any(m.startswith("ticks TEST") for m in seen)


def test_get_bid_ask_spread_computes_and_caches(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    at = datetime(2026, 4, 10, 11, 16, 50, tzinfo=timezone.utc)
    fake = FakeIB([SimpleNamespace(time=at, priceBid=4.00, priceAsk=4.03, sizeBid=1, sizeAsk=1)])
    data.ib = fake
    data._contracts["TEST"] = object()

    et = pytz.timezone("America/New_York")
    spread = data.get_bid_ask_spread("TEST", at.astimezone(et))
    assert spread == pytest.approx(0.03)
    assert fake.calls == 1

    fake.calls = 0
    spread2 = data.get_bid_ask_spread("TEST", at.astimezone(et))
    assert spread2 == pytest.approx(0.03)
    assert fake.calls == 0  # served from disk cache


def test_get_bid_ask_spread_returns_none_and_caches_when_no_quote(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    fake = FakeIB([])
    data.ib = fake
    data._contracts["TEST"] = object()

    et = pytz.timezone("America/New_York")
    at = et.localize(datetime(2026, 4, 10, 7, 16, 50))

    assert data.get_bid_ask_spread("TEST", at) is None
    assert fake.calls == 1

    fake.calls = 0
    assert data.get_bid_ask_spread("TEST", at) is None
    assert fake.calls == 0  # empty result cached, IBKR not re-asked


def test_get_bid_ask_spread_returns_none_for_invalid_quote(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    at = datetime(2026, 4, 10, 11, 16, 50, tzinfo=timezone.utc)
    # Crossed/invalid quote (ask below bid) must not produce a negative "spread".
    fake = FakeIB([SimpleNamespace(time=at, priceBid=4.03, priceAsk=4.00, sizeBid=1, sizeAsk=1)])
    data.ib = fake
    data._contracts["TEST"] = object()

    et = pytz.timezone("America/New_York")
    assert data.get_bid_ask_spread("TEST", at.astimezone(et)) is None


def _daily_bar(day_str: str, volume: float) -> BarData:
    return BarData(date=day_str, open=1.0, high=1.0, low=1.0, close=1.0, volume=volume)


def test_get_daily_volume_baseline_averages_trailing_days_and_caches(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar(f"202604{d:02d}", 1000.0 * d) for d in range(1, 10)]  # 04-01..04-09
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    # Day is 04-10; lookback_days=3 -> trailing 3 trading days strictly before it: 04-07,08,09.
    avg = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=3, min_history_days=1)
    assert avg == pytest.approx((7000.0 + 8000.0 + 9000.0) / 3)
    assert fake.daily_calls == 1

    fake.daily_calls = 0
    avg2 = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=3, min_history_days=1)
    assert avg2 == pytest.approx(avg)
    assert fake.daily_calls == 0  # served from disk cache


def test_get_daily_volume_baseline_ignores_bars_on_or_after_day(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar("20260408", 1000.0), _daily_bar("20260409", 2000.0), _daily_bar("20260410", 999_999.0)]
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    avg = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=5, min_history_days=1)
    assert avg == pytest.approx((1000.0 + 2000.0) / 2)  # the 04-10 bar itself is excluded


def test_get_daily_volume_baseline_returns_none_when_insufficient_history(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar("20260408", 1000.0), _daily_bar("20260409", 2000.0)]
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    avg = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=5)
    assert avg is None
    assert fake.daily_calls == 1

    # None result is cached too -- no re-request.
    fake.daily_calls = 0
    assert data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=5) is None
    assert fake.daily_calls == 0


def test_get_daily_volume_baseline_returns_none_when_no_bars(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    fake = FakeIB(daily_bars=[])
    data.ib = fake
    data._contracts["TEST"] = object()

    assert data.get_daily_volume_baseline("TEST", date(2026, 4, 10)) is None


def test_daily_volume_min_history_applied_at_read_time_not_baked_into_cache(tmp_path):
    """Changing rvol_min_history_days in config must take effect for already-cached
    tickers: the cache stores the raw (avg, n_days) facts and the threshold is
    applied on every read -- not decided once at write time."""
    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar("20260408", 1000.0), _daily_bar("20260409", 2000.0)]  # 2 days of history
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    # First read under a strict threshold: insufficient -> None (and cached).
    assert data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=5) is None
    assert fake.daily_calls == 1

    # User loosens the config: the SAME cache entry must now qualify, with no new
    # IBKR request -- this is exactly the "my config change did nothing" bug.
    fake.daily_calls = 0
    avg = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=1)
    assert avg == pytest.approx(1500.0)
    assert fake.daily_calls == 0

    # And tightening it again flips the same entry back to None.
    assert data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=5) is None
    assert fake.daily_calls == 0


def test_daily_volume_median_baseline_is_robust_to_one_outlier_day(tmp_path):
    """One prior halt/news day (33M shares) must not inflate a ~1K-share/day
    ticker's baseline: the default 'median' stat ignores it, while switching the
    config to 'mean' reads the OTHER stat from the SAME cache entry -- no new
    IBKR request (the stat is applied at read time, like min_history_days)."""
    import dataclasses

    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar("20260407", 1000.0), _daily_bar("20260408", 1200.0), _daily_bar("20260409", 33_000_000.0)]
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    med = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=3, min_history_days=1)
    assert med == pytest.approx(1200.0)  # not ~11M
    assert fake.daily_calls == 1

    fake.daily_calls = 0
    mean_cfg = dataclasses.replace(data.config, strategy=dataclasses.replace(data.config.strategy, rvol_baseline_stat="mean"))
    data.config = mean_cfg
    mean = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=3, min_history_days=1)
    assert mean == pytest.approx((1000.0 + 1200.0 + 33_000_000.0) / 3)
    assert fake.daily_calls == 0  # same cache entry, different column


def test_daily_volume_cache_without_median_column_is_refetched_once(tmp_path):
    """Cache files written before the median_volume column only hold the mean;
    they can't serve the median stat, so they're a miss and get rewritten in the
    current two-stat format."""
    import pandas as pd

    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar("20260408", 1000.0), _daily_bar("20260409", 2000.0)]
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    stale = data._daily_volume_cache_path("TEST", date(2026, 4, 10), 20)
    stale.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"avg_volume": 999_999.0, "n_days": 2}]).to_csv(stale, index=False)  # mean-only era

    value = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=1)
    assert value == pytest.approx(1500.0)
    assert fake.daily_calls == 1  # stale file ignored, refetched once

    fake.daily_calls = 0
    assert data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=1) == pytest.approx(1500.0)
    assert fake.daily_calls == 0  # rewritten with both stats; cached thereafter


def test_daily_volume_legacy_cache_without_n_days_is_refetched_once(tmp_path):
    """Cache files written before the n_days column had min_history_days baked in;
    they can't answer a different threshold, so they're treated as a miss and
    rewritten in the new format."""
    import pandas as pd

    data = IBKRHistoricalData(_config(tmp_path))
    bars = [_daily_bar("20260408", 1000.0), _daily_bar("20260409", 2000.0)]
    fake = FakeIB(daily_bars=bars)
    data.ib = fake
    data._contracts["TEST"] = object()

    legacy = data._daily_volume_cache_path("TEST", date(2026, 4, 10), 20)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"avg_volume": None}]).to_csv(legacy, index=False)  # old format: threshold-decided None

    avg = data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=1)
    assert avg == pytest.approx(1500.0)
    assert fake.daily_calls == 1  # legacy file ignored, refetched once

    fake.daily_calls = 0
    assert data.get_daily_volume_baseline("TEST", date(2026, 4, 10), lookback_days=20, min_history_days=1) == pytest.approx(1500.0)
    assert fake.daily_calls == 0  # rewritten in new format; cached thereafter


# --- share float from IBKR's Reuters company snapshot ---

_SNAPSHOT_XML = """<?xml version="1.0"?>
<ReportSnapshot>
  <CoGeneralInfo>
    <SharesOut Date="2026-06-30" TotalFloat="15000000.0">50000000.0</SharesOut>
  </CoGeneralInfo>
</ReportSnapshot>"""


def test_parse_total_float_static_helper():
    assert IBKRHistoricalData._parse_total_float(_SNAPSHOT_XML) == pytest.approx(15_000_000.0)
    assert IBKRHistoricalData._parse_total_float(None) is None
    assert IBKRHistoricalData._parse_total_float("not xml <<<") is None
    assert IBKRHistoricalData._parse_total_float('<R><SharesOut TotalFloat="0">1</SharesOut></R>') is None  # 0 -> None
