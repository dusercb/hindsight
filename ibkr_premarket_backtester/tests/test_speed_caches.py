"""Mass-run speed caches, all following the same self-healing budget pattern the
tick gap memos established (transient failures heal within N total live attempts
across runs; a later success always clears the memo):

1. get_float_YF disk+memory cache -- an uncached run made ~786 live yfinance
   calls (536 distinct tickers, 1-3s each) re-paid EVERY run, and that volume
   trips Yahoo's rate limiting, turning transient Nones into false
   INSUFFICIENT_DATA rejections.
2. get_day_bars empty-response memo -- empty bar fetches were never cached, so
   every run re-paid a live IBKR request per dead candidate (253 in one run).
3. reporting.trade_folders_for -- folder artifacts (json/bars/chart) only for
   result classes worth eyeballing; one run rendered 1,613 charts for 24 trades.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from ib_insync.objects import BarData

import premarket_backtester.ibkr_data as ibkr_data
from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig, load_config,
)
from premarket_backtester.engine import _await_farm_recovery
from premarket_backtester.ibkr_data import IBKRHistoricalData, MAX_FLOAT_ATTEMPTS
from premarket_backtester.models import TradeResult
from premarket_backtester.reporting import wants_trade_folder


def _config(tmp_path: Path, **data_overrides) -> AppConfig:
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(
            input_csv=Path("unused.csv"), reference_csv=None, cache_dir=tmp_path, output_dir=tmp_path,
            tick_request_pause_s=0.0, **data_overrides,
        ),
        session=SessionConfig(), strategy=StrategyConfig(), risk=RiskConfig(), reporting=ReportingConfig(),
    )


# --------------------------------------------------------------------------- #
# get_float_YF caching                                                        #
# --------------------------------------------------------------------------- #

class _FakeYF:
    """Stands in for the yfinance module: Ticker(t).info -> {"floatShares": ...}.
    `value` may be a number, None, or an Exception instance (raised on access)."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def Ticker(self, t):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return SimpleNamespace(info={"floatShares": self.value})


def test_float_found_is_cached_to_disk_and_memory(tmp_path, monkeypatch):
    fake = _FakeYF(12_500_000)
    monkeypatch.setattr(ibkr_data, "yf", fake)
    data = IBKRHistoricalData(_config(tmp_path))
    assert data.get_float_YF("TEST") == 12_500_000.0
    assert data.get_float_YF("test") == 12_500_000.0  # same run: memory, case-folded
    assert fake.calls == 1
    # A fresh instance (next run) reads the disk cache -- zero live calls.
    data2 = IBKRHistoricalData(_config(tmp_path))
    assert data2.get_float_YF("TEST") == 12_500_000.0
    assert fake.calls == 1


def test_float_unavailable_retries_within_budget_then_stops(tmp_path, monkeypatch):
    fake = _FakeYF(None)
    monkeypatch.setattr(ibkr_data, "yf", fake)
    # Each fresh instance simulates a new run; Nones retry live until the budget.
    for expected_calls in range(1, MAX_FLOAT_ATTEMPTS + 1):
        data = IBKRHistoricalData(_config(tmp_path))
        assert data.get_float_YF("GONE") is None
        assert fake.calls == expected_calls
    data = IBKRHistoricalData(_config(tmp_path))
    assert data.get_float_YF("GONE") is None
    assert fake.calls == MAX_FLOAT_ATTEMPTS  # budget exhausted: no more live calls


def test_float_success_after_transient_failures_heals_the_memo(tmp_path, monkeypatch):
    fake = _FakeYF(None)
    monkeypatch.setattr(ibkr_data, "yf", fake)
    data = IBKRHistoricalData(_config(tmp_path))
    assert data.get_float_YF("HEAL") is None  # one failed attempt recorded
    fake.value = 9_000_000  # Yahoo recovers (rate limit cleared)
    data2 = IBKRHistoricalData(_config(tmp_path))
    assert data2.get_float_YF("HEAL") == 9_000_000.0
    # And the success is now permanent -- next run is free.
    data3 = IBKRHistoricalData(_config(tmp_path))
    assert data3.get_float_YF("HEAL") == 9_000_000.0
    assert fake.calls == 2


def test_float_network_exception_is_an_unavailable_attempt_not_a_crash(tmp_path, monkeypatch):
    fake = _FakeYF(RuntimeError("connection reset"))
    monkeypatch.setattr(ibkr_data, "yf", fake)
    data = IBKRHistoricalData(_config(tmp_path))
    assert data.get_float_YF("ERR") is None
    assert fake.calls == 1


def test_float_legacy_cache_with_value_is_used_without_live_call(tmp_path, monkeypatch):
    fake = _FakeYF(999)
    monkeypatch.setattr(ibkr_data, "yf", fake)
    legacy = tmp_path / "float" / "OLD.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("float_shares\n45000000.0\n", encoding="utf-8")  # old IBKR-era format
    data = IBKRHistoricalData(_config(tmp_path))
    assert data.get_float_YF("OLD") == 45_000_000.0
    assert fake.calls == 0


def test_float_legacy_empty_cache_gets_yahoos_full_budget(tmp_path, monkeypatch):
    """The old IBKR fundamental fetcher cached 'unavailable' verdicts -- but ITS
    unavailability says nothing about Yahoo, so a legacy empty file must not lock
    the ticker out of a Yahoo lookup."""
    fake = _FakeYF(7_777_777)
    monkeypatch.setattr(ibkr_data, "yf", fake)
    legacy = tmp_path / "float" / "RETRY.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('float_shares\n""\n', encoding="utf-8")
    data = IBKRHistoricalData(_config(tmp_path))
    assert data.get_float_YF("RETRY") == 7_777_777.0
    assert fake.calls == 1


# --------------------------------------------------------------------------- #
# get_day_bars empty-response memo                                            #
# --------------------------------------------------------------------------- #

class _BarsIB:
    """reqHistoricalData returns the configured bars; counts live calls.
    `delay` (seconds of wall-clock stall before answering) simulates a
    throttled/timed-out farm -- ib_insync answers a timeout with an EMPTY list,
    so a slow empty is exactly what a timeout looks like to the caller."""

    def __init__(self, bars=(), delay: float = 0.0):
        self.bars = list(bars)
        self.delay = delay
        self.calls = 0

    def reqHistoricalData(self, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate=1, keepUpToDate=False):
        self.calls += 1
        if self.delay:
            import time as _t
            _t.sleep(self.delay)
        return list(self.bars)

    def sleep(self, seconds):
        # Real sleep: the bar-request pacing relies on ib.sleep actually waiting.
        import time as _t
        _t.sleep(seconds)


def _day_bars_data(tmp_path, ib, **data_overrides):
    data = IBKRHistoricalData(_config(tmp_path, **data_overrides))
    data.ib = ib
    data._contracts["DEAD"] = object()  # skip live contract qualification
    data._contracts["ALIVE"] = object()
    return data


def test_empty_bars_stop_being_refetched_after_the_attempt_budget(tmp_path):
    ib = _BarsIB(bars=())
    budget = 3
    for expected in range(1, budget + 1):
        data = _day_bars_data(tmp_path, ib, max_empty_bar_attempts=budget)
        df = data.get_day_bars("DEAD", datetime(2026, 4, 10).date())
        assert df.empty
        assert ib.calls == expected
    data = _day_bars_data(tmp_path, ib, max_empty_bar_attempts=budget)
    assert data.get_day_bars("DEAD", datetime(2026, 4, 10).date()).empty
    assert ib.calls == budget  # budget exhausted: served as known-empty, no live call


def test_slow_empty_bar_response_raises_instead_of_writing_a_memo(tmp_path):
    """ib_insync converts a request timeout into an EMPTY response. Recording
    that as 'no data' poisoned 776 empty-memos in one throttled-farm run
    (including COIN) -- an empty that burned ~the whole request timeout must
    raise TimeoutError (feeding the engine's circuit breaker) and write NOTHING."""
    ib = _BarsIB(bars=(), delay=0.1)  # empty AND slow == a timeout in disguise
    data = _day_bars_data(tmp_path, ib, request_timeout_s=0.05)
    with pytest.raises(TimeoutError):
        data.get_day_bars("DEAD", datetime(2026, 4, 10).date())
    assert not list(tmp_path.glob("*.empty.json"))  # no memo: not evidence of no-data


def test_fast_empty_bar_response_still_memos_normally(tmp_path):
    # A genuine no-data answer arrives quickly -- that IS evidence, and memos.
    ib = _BarsIB(bars=(), delay=0.0)
    data = _day_bars_data(tmp_path, ib, request_timeout_s=0.05)
    assert data.get_day_bars("DEAD", datetime(2026, 4, 10).date()).empty
    assert len(list(tmp_path.glob("DEAD_*.empty.json"))) == 1


def test_slow_empty_daily_volume_raises_instead_of_caching_no_history(tmp_path):
    ib = _BarsIB(bars=(), delay=0.1)
    data = _day_bars_data(tmp_path, ib, request_timeout_s=0.05)
    with pytest.raises(TimeoutError):
        data.get_daily_volume_baseline("DEAD", datetime(2026, 4, 10).date(), lookback_days=5, min_history_days=1)
    assert not list((tmp_path / "daily_volume").glob("*.csv"))  # nothing poisoned


def test_bar_request_pacing_spaces_live_fetches(tmp_path):
    """bar_request_pause_s throttles LIVE bar requests (cache hits skip it) --
    deliberate pacing keeps mass ingests out of IBKR's penalty box."""
    import time as _t
    ib = _BarsIB(bars=[_bar()])
    data = _day_bars_data(tmp_path, ib, bar_request_pause_s=0.2)
    data._contracts["ALIVE2"] = object()
    t0 = _t.monotonic()
    data.get_day_bars("ALIVE", datetime(2026, 4, 10).date())
    data.get_day_bars("ALIVE2", datetime(2026, 4, 10).date())
    assert _t.monotonic() - t0 >= 0.2  # second live fetch waited for the pace clock
    assert ib.calls == 2
    # A cache hit pays nothing.
    t0 = _t.monotonic()
    data.get_day_bars("ALIVE", datetime(2026, 4, 10).date())
    assert _t.monotonic() - t0 < 0.15
    assert ib.calls == 2


def test_fast_empty_for_qualified_ticker_needs_a_proven_farm(tmp_path):
    """A throttled farm also rejects INSTANTLY -- indistinguishable in shape from
    a genuine 'no data'. One such reject permanently condemned CRCA 2026-02-25
    (all memo attempts burned) though the ticker traded that month. Rule: a fast
    empty for a QUALIFIED contract is only evidence when the farm recently proved
    it can return real data; otherwise it raises like a timeout (ERROR row,
    breaker, retry later) and writes nothing."""
    ib = _BarsIB(bars=())  # instant empties
    data = _day_bars_data(tmp_path, ib)
    data._contracts["QUAL"] = SimpleNamespace(conId=12345)  # a real, listed instrument
    with pytest.raises(TimeoutError):
        data.get_day_bars("QUAL", datetime(2026, 4, 10).date())
    assert not list(tmp_path.glob("QUAL_*.empty.json"))

    # After the farm proves itself (a real fetch succeeds), the same fast empty
    # becomes trustworthy evidence and memos normally.
    ib.bars = [_bar()]
    data.get_day_bars("ALIVE", datetime(2026, 4, 10).date())   # marks farm healthy
    ib.bars = []
    assert data.get_day_bars("QUAL", datetime(2026, 4, 10).date()).empty
    assert len(list(tmp_path.glob("QUAL_*.empty.json"))) == 1


def test_unqualified_contract_empties_are_always_trustworthy(tmp_path):
    # conId 0 = delisted/unknown symbol: emptiness is EXPECTED, no health needed.
    # (The object() sentinel has no conId attribute -> treated as unqualified.)
    ib = _BarsIB(bars=())
    data = _day_bars_data(tmp_path, ib)
    assert data.get_day_bars("DEAD", datetime(2026, 4, 10).date()).empty
    assert len(list(tmp_path.glob("DEAD_*.empty.json"))) == 1


def test_successful_bar_fetch_clears_the_empty_memo_and_caches(tmp_path):
    day = datetime(2026, 4, 10).date()
    ib = _BarsIB(bars=())
    data = _day_bars_data(tmp_path, ib, max_empty_bar_attempts=3)
    assert data.get_day_bars("ALIVE", day).empty  # attempt 1 recorded
    memo = next(tmp_path.glob("ALIVE_*.empty.json"))
    assert memo.exists()

    ib.bars = [BarData(date=datetime(2026, 4, 10, 8, 0), open=4.0, high=4.1, low=3.9, close=4.05, volume=100, average=4.0, barCount=1)]
    data2 = _day_bars_data(tmp_path, ib, max_empty_bar_attempts=3)
    df = data2.get_day_bars("ALIVE", day)
    assert len(df) == 1
    assert not memo.exists()  # healed
    calls_before = ib.calls
    data3 = _day_bars_data(tmp_path, ib, max_empty_bar_attempts=3)
    assert len(data3.get_day_bars("ALIVE", day)) == 1  # now a plain cache hit
    assert ib.calls == calls_before


# --------------------------------------------------------------------------- #
# trade_folders_for gating                                                    #
# --------------------------------------------------------------------------- #

def _result(result: str, entry: float | None = None) -> TradeResult:
    tr = TradeResult(ticker="T", date="2026-04-10", result=result)
    tr.entry_price = entry
    return tr


@pytest.mark.parametrize("result,entry,trades,setups", [
    ("WIN", 4.0, True, True),
    ("LOSS", 4.0, True, True),
    ("EXIT_AT_OPEN", 4.0, True, True),
    ("STALE_HOD", None, False, True),
    ("INVALID_RETRACE", None, False, True),
    ("NO_TRIGGER", None, False, True),
    ("SKIPPED_PRICE_FILTER", None, False, False),
    ("SKIPPED_FLOAT_FILTER", None, False, False),
    ("DID_NOT_QUALIFY", None, False, False),
    ("INSUFFICIENT_DATA", None, False, False),
    ("ERROR", None, False, False),
])
def test_wants_trade_folder_matrix(result, entry, trades, setups):
    tr = _result(result, entry)
    assert wants_trade_folder(tr, "trades") is trades
    assert wants_trade_folder(tr, "setups") is setups
    assert wants_trade_folder(tr, "all") is True


def test_wants_trade_folder_always_keeps_a_filled_entry():
    # Whatever the label says, a candidate whose entry actually filled is a trade.
    tr = _result("ERROR", entry=4.0)
    assert wants_trade_folder(tr, "trades") is True


def test_new_keys_flow_from_yaml_and_validate(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "data:\n  max_empty_bar_attempts: 5\nreporting:\n  trade_folders_for: trades\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.data.max_empty_bar_attempts == 5
    assert cfg.reporting.trade_folders_for == "trades"
    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    cfg2 = load_config(p2)
    assert cfg2.data.max_empty_bar_attempts == 3   # default
    assert cfg2.reporting.trade_folders_for == "setups"  # default
    bad = tmp_path / "bad.yaml"
    bad.write_text("reporting:\n  trade_folders_for: everything\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trade_folders_for"):
        load_config(bad)


# --------------------------------------------------------------------------- #
# async bar pre-warmer                                                        #
# --------------------------------------------------------------------------- #


def _bar(day_hour=8):
    return BarData(date=datetime(2026, 4, 10, day_hour, 0), open=4.0, high=4.1, low=3.9, close=4.05, volume=100, average=4.0, barCount=1)


# --------------------------------------------------------------------------- #
# autonomous throttle/disconnect recovery                                     #
# --------------------------------------------------------------------------- #

class _RecoveryIB:
    """Fake ib for the recovery loop: connected, and sleep() is a NO-OP so the
    escalating backoff advances instantly in tests (the helper tracks nominal
    waited-time, not wall-clock)."""

    def __init__(self, connected=True):
        self._connected = connected
        self.sleeps = []

    def isConnected(self):
        return self._connected

    def sleep(self, s):
        self.sleeps.append(s)  # never actually sleeps


def test_probe_farm_health_true_on_data_false_on_empty(tmp_path):
    data = IBKRHistoricalData(_config(tmp_path))
    data._contracts["AAPL"] = object()  # skip live qualification
    data.ib = _BarsIB(bars=[_bar()])
    assert data.probe_farm_health("AAPL") is True
    assert data._last_data_success_at is not None       # health clock marked
    data.ib = _BarsIB(bars=[])
    data._last_data_success_at = None
    assert data.probe_farm_health("AAPL") is False       # empty = still down


def test_await_farm_recovery_returns_true_once_the_farm_answers(tmp_path, monkeypatch):
    data = IBKRHistoricalData(_config(tmp_path))
    data.ib = _RecoveryIB(connected=True)
    n = {"probes": 0}

    def fake_probe(symbol="AAPL"):
        n["probes"] += 1
        return n["probes"] >= 3          # down twice, healthy on the third probe

    monkeypatch.setattr(data, "probe_farm_health", fake_probe)
    assert _await_farm_recovery(data, _config(tmp_path), lambda m: None) is True
    assert n["probes"] == 3
    assert len(data.ib.sleeps) == 2                      # cooled down between the failed probes


def test_await_farm_recovery_gives_up_after_the_cap(tmp_path, monkeypatch):
    data = IBKRHistoricalData(_config(tmp_path, recovery_max_total_wait_s=500.0))
    data.ib = _RecoveryIB(connected=True)
    monkeypatch.setattr(data, "probe_farm_health", lambda symbol="AAPL": False)  # never recovers
    assert _await_farm_recovery(data, _config(tmp_path, recovery_max_total_wait_s=500.0), lambda m: None) is False


def test_await_farm_recovery_reconnects_when_disconnected(tmp_path, monkeypatch):
    data = IBKRHistoricalData(_config(tmp_path))
    ib = _RecoveryIB(connected=False)
    data.ib = ib
    reconnected = {"n": 0}

    def fake_connect():
        reconnected["n"] += 1
        ib._connected = True            # TWS came back

    monkeypatch.setattr(data, "connect", fake_connect)
    monkeypatch.setattr(data, "disconnect", lambda: None)
    monkeypatch.setattr(data, "probe_farm_health", lambda symbol="AAPL": ib._connected)
    assert _await_farm_recovery(data, _config(tmp_path), lambda m: None) is True
    assert reconnected["n"] == 1


def test_recovery_disabled_when_cap_is_zero(tmp_path, monkeypatch):
    data = IBKRHistoricalData(_config(tmp_path, recovery_max_total_wait_s=0.0))
    data.ib = _RecoveryIB(connected=True)
    monkeypatch.setattr(data, "probe_farm_health", lambda symbol="AAPL": True)
    # cap 0 = auto-recovery off -> returns False immediately, never probes/pauses.
    assert _await_farm_recovery(data, _config(tmp_path, recovery_max_total_wait_s=0.0), lambda m: None) is False


