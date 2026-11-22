from __future__ import annotations

import json
import time as time_mod
import xml.etree.ElementTree as ET
from datetime import date, datetime, time
from pathlib import Path
import pandas as pd
import pytz
from ib_insync import IB, Stock, util
import yfinance as yf #for (temp) func

from .config import AppConfig

# IBKR returns at most 1000 historical ticks per request; a hot 10-second bar can
# exceed that, so get_ticks paginates. The page cap only bounds a runaway loop.
MAX_TICKS_PER_REQUEST = 1000
MAX_TICK_PAGES = 20


def _parse_bar_dates(series: "pd.Series") -> "pd.Series":
    """Parse a bar/daily 'date' column to datetime, tolerating a fetch window that
    spans a DST change. A 2-day (or wider) window straddling the Nov/Mar DST
    boundary carries BOTH UTC offsets (e.g. -04:00 for the prior session and -05:00
    for the day), and pandas 3.0 refuses to parse a mixed-offset column without
    utc=True. Uniform-offset and tz-naive columns keep their original behavior
    (tz-aware stays convertible, naive stays localizable); only the mixed-offset
    case that used to raise "Mixed timezones detected" falls back to UTC
    normalization (which callers then tz_convert to the session timezone)."""
    try:
        return pd.to_datetime(series)
    except ValueError:
        return pd.to_datetime(series, utc=True)

# Total live attempts (across runs) for a ticker whose float lookup keeps coming
# back empty -- same self-healing budget pattern as the tick gap memos: transient
# failures (Yahoo rate limiting, network blips) heal within the budget, after
# which "no float available" is accepted and never re-queried. Without this, a
# mass run re-paid ~1-3s of yfinance per float-less candidate on every run.
MAX_FLOAT_ATTEMPTS = 3

# How recently a live request must have returned REAL data for this session to be
# allowed to record persistent negative evidence (empty-bar memos, tick gap
# attempts, "no daily history" caches). A throttled IBKR history farm doesn't
# only answer slowly -- it also rejects instantly, and an instant reject is
# indistinguishable in shape from a genuine "no data" answer. One sick run
# permanently marked CRCA 2026-02-25 "known empty" (all attempts burned) for a
# ticker with real prints that month. Rule: negative evidence only counts when
# the farm has recently proven it can say yes; the sole exception is an
# UNQUALIFIED contract (conId 0 -- delisted/unknown symbol), whose emptiness is
# genuine regardless of farm health.
FARM_HEALTH_WINDOW_S = 600.0


class IBKRHistoricalData:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ib = IB()
        self._contracts: dict[str, Stock] = {}
        # Per-run float memo: one ticker can appear on many candidate dates, and
        # a yfinance lookup costs 1-3s of live HTTP -- never pay it twice in a run.
        self._float_mem: dict[str, float | None] = {}
        self._last_tick_request_at: float | None = None
        # One shared clock for BAR request pacing (bar_request_pause_s): the sync
        # fetch and the async pre-warmer both stamp it, so total issuance stays
        # paced no matter which path fires.
        self._last_bar_request_at: float | None = None
        # When any live historical request last returned REAL data -- the farm
        # health signal gating all persistent negative evidence (see
        # FARM_HEALTH_WINDOW_S).
        self._last_data_success_at: float | None = None
        # Optional live-status hook: called with a short human-readable message
        # right before every live IBKR request, so the engine's progress display
        # can show exactly what the run is waiting on instead of appearing hung.
        self.on_status = None

    def _status(self, msg: str) -> None:
        if self.on_status is not None:
            self.on_status(msg)

    def connect(self) -> None:
        c = self.config.ibkr
        # Honor the configured readonly flag instead of hardcoding it. A readonly
        # IBKR session rejects all order submission at the API boundary, so this is
        # the backtester's hard safety interlock -- it defaults to True and
        # config.yaml keeps it True, so this backtester can never place an order.
        # (A live bot would set it False; that flag was previously ignored here,
        # which would have silently blocked order flow.)
        self.ib.connect(c.host, c.port, clientId=c.client_id, readonly=c.readonly)
        # ib_insync waits FOREVER for a response by default; one silently dropped
        # request used to hang the whole run. With a timeout, the stuck request
        # raises instead, callers treat it as "data unavailable", and the engine
        # survives it per candidate.
        self.ib.RequestTimeout = self.config.data.request_timeout_s

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def _cache_path(self, ticker: str, day: date, duration: str) -> Path:
        d = self.config.data.cache_dir
        suffix = "" if duration == "2 D" else f"_dur{duration.replace(' ', '')}"
        return d / f"{ticker.upper()}_{day.isoformat()}_{self.config.data.bar_size.replace(' ', '')}{suffix}.csv"

    def _contract(self, ticker: str) -> Stock:
        key = ticker.upper()
        if key not in self._contracts:
            self._status(f"qualifying contract {key}")
            contract = Stock(key, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            self._contracts[key] = contract
        return self._contracts[key]

    def get_day_bars(self, ticker: str, day: date, duration: str = "2 D") -> pd.DataFrame:
        """Bars covering `day` (and, via `duration`, prior sessions for previous-close context).

        `duration` normally stays at the "2 D" default, which is enough to reach the prior
        trading session across an ordinary weekend or single midweek holiday. Callers can
        widen it (e.g. get_wide_day_bars) for the rarer case of a holiday sitting right next
        to a weekend, where the prior session is 3-4 calendar days back.
        """
        cache = self._cache_path(ticker, day, duration)
        if self.config.data.use_cache and cache.exists():
            return pd.read_csv(cache, parse_dates=["date"])

        # Empty responses were never cached, so every run re-paid a live request
        # for each dead candidate (delisted/renamed ticker, no data that day) --
        # 253 of them in one measured run. Same self-healing budget pattern as the
        # tick gap memos: transient failures (drop, timeout) heal within
        # max_empty_bar_attempts total live attempts across runs, then the
        # candidate is accepted as no-data for free. A successful fetch clears it.
        empty_memo = cache.with_suffix(".empty.json")
        attempts = 0
        if self.config.data.use_cache:
            attempts = self._read_empty_attempts(empty_memo)
            if attempts >= self.config.data.max_empty_bar_attempts:
                self._status(f"bars {ticker.upper()} {day} known empty -- skipping")
                return self._empty_bars()

        self._status(f"bars {ticker.upper()} {day} ({duration}, {self.config.data.bar_size})")
        contract = self._contract(ticker)
        self._pace_bar_request()
        t0 = time_mod.monotonic()
        bars = self.ib.reqHistoricalData(contract, **self._bar_request_kwargs(day, duration))
        df = util.df(bars)
        if self._looks_like_timeout(df, t0):
            # ib_insync converts a request TIMEOUT into an EMPTY response (it
            # logs 'reqHistoricalData: Timeout', cancels the query, and returns
            # []). A genuine no-data answer arrives in seconds; an empty that
            # burned ~the whole RequestTimeout is a NON-answer. Recording those
            # as "no data" poisoned 776 empty-memos in one throttled-farm run
            # (including COIN). Raise instead: the engine records an ERROR row,
            # the wedged-farm circuit breaker counts it, and NO memo is written.
            raise TimeoutError(f"historical bars request timed out ({ticker} {day}, {duration})")
        if (df is None or df.empty) and not self._empty_is_trustworthy(contract):
            # FAST empty for a QUALIFIED contract with no recent proof the farm
            # can answer at all: a throttled farm's instant reject, not evidence
            # of no data. Same treatment as a timeout -- no memo, ERROR row,
            # breaker/cooldown -- so it retries later instead of being condemned.
            raise TimeoutError(f"empty response for qualified {ticker} {day} while farm unproven -- not trusted")
        if df is not None and not df.empty:
            self._mark_data_success()
        return self._store_bars_response(df, cache, empty_memo, attempts)

    def _looks_like_timeout(self, df: pd.DataFrame | None, t0: float) -> bool:
        """True when an EMPTY response took ~the whole request timeout -- a
        timeout masquerading as 'no data', not evidence about the ticker."""
        return (df is None or df.empty) and time_mod.monotonic() - t0 >= 0.9 * self.config.data.request_timeout_s

    def _mark_data_success(self) -> None:
        self._last_data_success_at = time_mod.monotonic()

    def _farm_recently_healthy(self) -> bool:
        return (
            self._last_data_success_at is not None
            and time_mod.monotonic() - self._last_data_success_at <= FARM_HEALTH_WINDOW_S
        )

    def probe_farm_health(self, symbol: str = "AAPL") -> bool:
        """Liveness check for IBKR's historical-data farm: one 1-day daily bar for
        a mega-liquid symbol. Returns True -- and marks the health clock -- only on
        real data, which is the signal the farm is serving again after a throttle
        or outage. Never raises; a throttled farm makes this time out or return
        empty, which reads as still-down. Used by the engine's recovery wait."""
        try:
            contract = self._contract(symbol)
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="1 D", barSizeSetting="1 day",
                whatToShow="TRADES", useRTH=True, formatDate=1, keepUpToDate=False,
            )
            df = util.df(bars)
            if df is not None and not df.empty:
                self._mark_data_success()
                return True
        except Exception:
            pass
        return False

    def _empty_is_trustworthy(self, contract) -> bool:
        """May an EMPTY (but fast, non-timeout) response be recorded as genuine
        'no data'? Yes for an unqualified contract (conId 0: delisted/unknown --
        emptiness is expected), otherwise only while the farm has recently proven
        it can return real data: a throttled farm also rejects INSTANTLY, and an
        instant reject looks exactly like a genuine empty."""
        if not getattr(contract, "conId", 0):
            return True
        return self._farm_recently_healthy()

    def _bar_pause_remaining(self) -> float:
        pause = self.config.data.bar_request_pause_s
        if not pause or self._last_bar_request_at is None:
            return 0.0
        return max(0.0, pause - (time_mod.monotonic() - self._last_bar_request_at))

    def _pace_bar_request(self) -> None:
        wait = self._bar_pause_remaining()
        if wait > 0:
            self.ib.sleep(wait)  # keeps the ib_insync event loop serviced
        self._last_bar_request_at = time_mod.monotonic()

    @staticmethod
    def _empty_bars() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    @staticmethod
    def _read_empty_attempts(memo: Path) -> int:
        if not memo.exists():
            return 0
        try:
            return int(json.loads(memo.read_text(encoding="utf-8")).get("attempts", 0))
        except (ValueError, OSError):
            return 0

    def _bar_request_kwargs(self, day: date, duration: str) -> dict:
        """The exact reqHistoricalData arguments for one candidate day -- shared
        by the sync fetch and the async pre-warmer so both issue byte-identical
        requests (fetch up to the regular open end plus a bit; duration reaches
        back for prior-close context)."""
        tz = pytz.timezone(self.config.session.timezone)
        return dict(
            endDateTime=tz.localize(datetime.combine(day, time(10, 0))),
            durationStr=duration,
            barSizeSetting=self.config.data.bar_size,
            whatToShow=self.config.data.what_to_show,
            useRTH=self.config.data.use_rth,
            formatDate=1,
            keepUpToDate=False,
        )

    def _store_bars_response(self, df: pd.DataFrame | None, cache: Path, empty_memo: Path, prior_attempts: int) -> pd.DataFrame:
        """Persist one bar response -- the single source of truth for cache/memo
        semantics, shared by get_day_bars and the async pre-warmer: a non-empty
        response is cached and clears any empty-memo; an empty one burns an
        attempt toward max_empty_bar_attempts."""
        if df is None or df.empty:
            empty_memo.parent.mkdir(parents=True, exist_ok=True)
            empty_memo.write_text(json.dumps({"attempts": prior_attempts + 1}), encoding="utf-8")
            return self._empty_bars()
        df["date"] = _parse_bar_dates(df["date"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        empty_memo.unlink(missing_ok=True)
        return df

    def _daily_volume_cache_path(self, ticker: str, day: date, lookback_days: int) -> Path:
        d = self.config.data.cache_dir / "daily_volume"
        return d / f"{ticker.upper()}_{day.isoformat()}_lb{lookback_days}.csv"

    def get_daily_volume_baseline(self, ticker: str, day: date, lookback_days: int = 20, min_history_days: int = 5) -> float | None:
        """This ticker's typical REGULAR-SESSION daily volume over the `lookback_days`
        trading days strictly before `day` -- the RVOL baseline traders mean by the term
        (vs. the ticker's own historical norm, not just that morning's own bars).
        strategy.rvol_baseline_stat picks the statistic: "median" is robust to a
        single prior halt/news day (one 33M-share day must not make a 66K-share/day
        ticker look like a 2.3M-share/day one, stalling qualification until the
        move is over), "mean" is the conservative alternative.

        Uses a single daily-bar request (barSizeSetting="1 day" has generous IBKR
        duration limits, unlike the 10-second bars used elsewhere), cached per
        (ticker, day, lookback_days). Returns None if fewer than `min_history_days`
        trading days of history are available (e.g. a recently listed ticker) --
        callers should treat that as "RVOL cannot be computed", not "RVOL is zero".

        The cache stores the raw facts (mean + median + how many prior days they
        were computed from); min_history_days and the stat choice are applied at
        READ time, so changing either in config takes effect immediately, without
        clearing the cache. Older cache files predate the median_volume/n_days
        columns -- those are treated as a miss and refetched once (a single cheap
        daily-bar request), then rewritten in the current format.
        """
        stat = getattr(self.config.strategy, "rvol_baseline_stat", "median")
        col = "median_volume" if stat == "median" else "avg_volume"
        cache = self._daily_volume_cache_path(ticker, day, lookback_days)
        if self.config.data.use_cache and cache.exists():
            df = pd.read_csv(cache)
            if not df.empty and {"n_days", "median_volume"}.issubset(df.columns):
                n_days = int(df.iloc[0]["n_days"])
                value = df.iloc[0][col]
                if n_days < min_history_days or pd.isna(value):
                    return None
                return float(value)

        tz = pytz.timezone(self.config.session.timezone)
        end_dt = tz.localize(datetime.combine(day, time(0, 0)))
        self._status(f"daily volume {ticker.upper()} (RVOL baseline, {lookback_days}d)")
        contract = self._contract(ticker)
        t0 = time_mod.monotonic()
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=f"{lookback_days * 3} D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
            keepUpToDate=False,
        )
        df = util.df(bars)
        if self._looks_like_timeout(df, t0):
            # Same timeout-masquerading-as-empty trap as get_day_bars: caching a
            # non-answer as {n_days: 0} would permanently mislabel the ticker as
            # having no volume history (INSUFFICIENT_DATA on every future run).
            raise TimeoutError(f"daily-volume baseline request timed out ({ticker})")
        if (df is None or df.empty) and not self._empty_is_trustworthy(contract):
            # Instant reject from an unproven farm: caching {n_days: 0} now would
            # permanently mislabel a real ticker as history-less.
            raise TimeoutError(f"empty daily-volume response for qualified {ticker} while farm unproven -- not trusted")
        if df is not None and not df.empty:
            self._mark_data_success()
        avg_volume = None
        median_volume = None
        n_days = 0
        if df is not None and not df.empty:
            df["date"] = _parse_bar_dates(df["date"])
            prior = df[df["date"].dt.date < day].tail(lookback_days)
            n_days = len(prior)
            if n_days > 0:
                avg_volume = float(prior["volume"].mean())
                median_volume = float(prior["volume"].median())
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"avg_volume": avg_volume, "median_volume": median_volume, "n_days": n_days}]).to_csv(cache, index=False)
        value = median_volume if stat == "median" else avg_volume
        if n_days < min_history_days or value is None:
            return None
        return value

    def _float_cache_path(self, ticker: str) -> Path:
        d = self.config.data.cache_dir / "float"
        return d / f"{ticker.upper()}.csv"

    @staticmethod
    def _parse_total_float(xml_text: str | None) -> float | None:
        """Pull TotalFloat out of a Reuters 'ReportSnapshot' fundamental report.
        The snapshot XML carries `<SharesOut ... TotalFloat="45000000.0">` (the
        element text is total shares outstanding; the attribute is the float).
        Returns None on any missing/unparseable value."""
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None
        el = root.find(".//SharesOut")
        if el is None:
            return None
        raw = el.get("TotalFloat")
        try:
            val = float(raw) if raw not in (None, "") else None
        except ValueError:
            return None
        return val if (val is not None and val > 0) else None

    def get_float_YF(self, ticker: str) -> float | None:
        """Ticker float from Yahoo Finance, cached in memory (per run) and on disk
        (per ticker, data/cache/float/) -- an uncached run made ~786 live yfinance
        calls (~1-3s each of HTTP) for 536 distinct tickers, re-paid EVERY run, and
        that volume also trips Yahoo's rate limiting, turning rate-limited Nones
        into false INSUFFICIENT_DATA rejections.

        Disk format: float_shares (empty when unavailable) + attempts. A found
        float is cached permanently (floats drift slowly -- same point-in-time
        caveat as the config documents for the IBKR snapshot; supply
        reference_floats.csv when precision matters). An UNAVAILABLE float is
        retried up to MAX_FLOAT_ATTEMPTS total live attempts across runs (healing
        transient rate limits), then accepted as having no float data. yfinance
        network errors are caught and treated as an unavailable attempt rather
        than crashing the candidate into an ERROR row."""
        key = ticker.upper()
        if key in self._float_mem:
            return self._float_mem[key]

        cache = self._float_cache_path(key)
        attempts = 0
        if self.config.data.use_cache and cache.exists():
            try:
                df = pd.read_csv(cache)
            except Exception:
                df = pd.DataFrame()
            if not df.empty and "float_shares" in df.columns:
                raw = df.iloc[0]["float_shares"]
                if pd.notna(raw) and float(raw) > 0:
                    self._float_mem[key] = float(raw)
                    return self._float_mem[key]
                # Unavailable so far: honor the attempt budget. Cache files that
                # predate the attempts column are relics of the old IBKR
                # fundamental fetcher -- ITS "unavailable" says nothing about
                # Yahoo, so those get Yahoo's full budget (attempts 0) and the
                # file is rewritten in the current format after the live call.
                attempts = int(df.iloc[0]["attempts"]) if "attempts" in df.columns and pd.notna(df.iloc[0].get("attempts")) else 0
                if attempts >= MAX_FLOAT_ATTEMPTS:
                    self._float_mem[key] = None
                    return None

        self._status(f"float {key} (yfinance)")
        try:
            total_float = yf.Ticker(key).info.get("floatShares")
        except Exception:
            total_float = None
        value = float(total_float) if total_float else None

        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"float_shares": value, "attempts": 0 if value is not None else attempts + 1}]).to_csv(cache, index=False)
        self._float_mem[key] = value
        return value

    def _tick_cache_path(self, ticker: str, start: datetime, end: datetime) -> Path:
        d = self.config.data.cache_dir / "ticks"
        return d / f"{ticker.upper()}_{start:%Y-%m-%d_%H%M%S}_{end:%H%M%S}_ticks.csv"

    def _empty_ticks(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["time", "price", "size", "exchange"])

    def _pace_tick_request(self) -> None:
        pause = self.config.data.tick_request_pause_s
        now = time_mod.monotonic()
        if self._last_tick_request_at is not None:
            wait = pause - (now - self._last_tick_request_at)
            if wait > 0:
                # ib.sleep keeps the ib_insync event loop serviced while waiting.
                self.ib.sleep(wait)
        self._last_tick_request_at = time_mod.monotonic()

    def get_ticks(self, ticker: str, start: datetime, end: datetime, expect_trades: bool = False) -> pd.DataFrame:
        """Historical trade ticks in [start, end), sorted chronologically.

        IBKR rate-limits historical tick requests, so this is meant to be called
        sparingly, and results — including empty ones — are cached to disk.

        `expect_trades=True` means the caller KNOWS trades printed in this window
        (it holds a bar with volume covering it), so an empty tick response is
        contradictory — almost always a failed/denied/paced-out request, not a
        quiet market. In that mode: a cached EMPTY result is distrusted and
        re-fetched (self-healing windows poisoned before this guard existed), and
        a still-empty window is never cached as ticks. Instead, a small sidecar
        "gap memo" counts live attempts ACROSS runs; transient failures heal
        within `max_empty_tick_attempts` total attempts, after which the window
        is accepted as a permanent gap in IBKR's tick store and skipped for free
        — otherwise every run would re-pay paced live requests for it forever.
        """
        tz = pytz.timezone(self.config.session.timezone)
        cache = self._tick_cache_path(ticker, start, end)
        gap = cache.with_suffix(".gap.json")
        if self.config.data.use_cache and cache.exists():
            df = pd.read_csv(cache)
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(tz)
                return df
            if not expect_trades:
                return self._empty_ticks()
            # expect_trades with a cached empty: contradictory -- fall through to a
            # live re-fetch; a successful one overwrites the poisoned cache below.

        attempts = 0
        if expect_trades and gap.exists():
            try:
                attempts = int(json.loads(gap.read_text(encoding="utf-8")).get("attempts", 0))
            except (ValueError, OSError):
                attempts = 0
            if attempts >= self.config.data.max_empty_tick_attempts:
                # IBKR has answered "no ticks" enough separate times for a window
                # whose bar shows real volume: accept it as a permanent gap in
                # IBKR's tick store and stop spending paced requests on it.
                self._status(f"ticks {ticker.upper()} {start:%H:%M:%S} known IBKR gap -- skipping")
                return self._empty_ticks()

        t0 = time_mod.monotonic()
        df = self._fetch_ticks_paginated(ticker, start, end, tz)
        if df.empty and self._looks_like_timeout(df, t0):
            # A timeout masquerading as an empty tick window must not burn a gap
            # attempt -- it says the farm was slow, not that the window has no
            # trades. Fall back to bar-based fills for this window, memo untouched.
            return df
        if df.empty and expect_trades and attempts == 0:
            # One paced retry on the first-ever attempt: transient HMDS/pacing
            # failures surface as empty batches and often clear immediately.
            self._status(f"ticks {ticker.upper()} {start:%H:%M:%S} came back empty -- retrying")
            t0 = time_mod.monotonic()
            df = self._fetch_ticks_paginated(ticker, start, end, tz)
            if df.empty and self._looks_like_timeout(df, t0):
                return df

        if not df.empty:
            self._mark_data_success()
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
            gap.unlink(missing_ok=True)
        elif not self._farm_recently_healthy():
            # Empty from an unproven farm (instant throttle reject): not evidence
            # about this window -- no cache, no gap attempt burned. The caller
            # falls back to bar-based fills for now and the window retries later.
            pass
        elif not expect_trades:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
        else:
            attempts += 2 if attempts == 0 else 1  # first call burns attempt + retry
            gap.parent.mkdir(parents=True, exist_ok=True)
            gap.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
        return df

    def _fetch_ticks_paginated(self, ticker: str, start: datetime, end: datetime, tz) -> pd.DataFrame:
        """One live paginated tick fetch for [start, end) — no caching, no retries."""
        contract = self._contract(ticker)
        rows: list[tuple] = []
        req_start: datetime = start
        last_time = None
        n_at_last_time = 0
        for page in range(MAX_TICK_PAGES):
            self._status(f"ticks {ticker.upper()} {start:%H:%M:%S}" + (f" (page {page + 1})" if page else ""))
            self._pace_tick_request()
            batch = self.ib.reqHistoricalTicks(
                contract,
                startDateTime=req_start,
                endDateTime="",
                numberOfTicks=MAX_TICKS_PER_REQUEST,
                whatToShow="TRADES",
                useRth=self.config.data.use_rth,
                ignoreSize=False,
            )
            if not batch:
                break
            # startDateTime is inclusive and tick times have 1-second resolution, so
            # the next page re-sends ticks from the boundary second; skip the ones
            # already collected.
            skip = n_at_last_time
            new = []
            for t in batch:
                if last_time is not None and t.time == last_time and skip > 0:
                    skip -= 1
                    continue
                new.append(t)
            if not new:
                break
            rows.extend((t.time, t.price, t.size, t.exchange) for t in new)
            last_time = new[-1].time
            n_at_last_time = sum(1 for r in rows if r[0] == last_time)
            if last_time >= end or len(batch) < MAX_TICKS_PER_REQUEST:
                break
            req_start = last_time

        df = pd.DataFrame(rows, columns=["time", "price", "size", "exchange"])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(tz)
            df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)
        return df

    def _bid_ask_cache_path(self, ticker: str, at: datetime) -> Path:
        d = self.config.data.cache_dir / "bid_ask"
        return d / f"{ticker.upper()}_{at:%Y-%m-%d_%H%M%S}_spread.csv"

    def get_bid_ask_spread(self, ticker: str, at: datetime) -> float | None:
        """Best-effort bid/ask spread (dollars) closest to `at`.

        Uses a single-tick historical BID_ASK request (reqHistoricalTicks with
        whatToShow="BID_ASK"), which reports unambiguous priceBid/priceAsk fields --
        unlike whatToShow="BID_ASK" bar aggregates, whose OHLC semantics aren't
        clearly documented. Shares the same tick request pacing as get_ticks, since
        both draw from IBKR's historical-tick pacing budget. Returns None (never
        raises) if no quote is available; callers should treat that as "unknown"
        rather than "spread is zero".
        """
        cache = self._bid_ask_cache_path(ticker, at)
        if self.config.data.use_cache and cache.exists():
            df = pd.read_csv(cache)
            if df.empty or pd.isna(df.iloc[0]["spread"]):
                return None
            return float(df.iloc[0]["spread"])

        self._status(f"bid/ask spread {ticker.upper()} {at:%H:%M:%S}")
        contract = self._contract(ticker)
        self._pace_tick_request()
        ticks = self.ib.reqHistoricalTicks(
            contract,
            startDateTime=at,
            endDateTime="",
            numberOfTicks=1,
            whatToShow="BID_ASK",
            useRth=self.config.data.use_rth,
            ignoreSize=True,
        )
        spread = None
        if ticks:
            t = ticks[0]
            if t.priceBid > 0 and t.priceAsk > 0 and t.priceAsk >= t.priceBid:
                spread = round(t.priceAsk - t.priceBid, 4)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"spread": spread}]).to_csv(cache, index=False)
        return spread


def split_sessions(df: pd.DataFrame, day: date, tz_name: str, premarket_start: str, regular_open: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    tz = pytz.timezone(tz_name)
    d = _parse_bar_dates(df["date"])
    if d.dt.tz is None:
        d = d.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        d = d.dt.tz_convert(tz)
    work = df.copy()
    work["dt_et"] = d
    start_h, start_m = [int(x) for x in premarket_start.split(":")]
    open_h, open_m = [int(x) for x in regular_open.split(":")]
    pm_start = tz.localize(datetime.combine(day, time(start_h, start_m)))
    rth_open = tz.localize(datetime.combine(day, time(open_h, open_m)))
    premarket = work[(work["dt_et"] >= pm_start) & (work["dt_et"] < rth_open)].copy()

    # The previous *trading* session is not simply "day - 1 calendar day": that
    # breaks on Mondays (previous day is Sunday) and on any day after a market
    # holiday. Instead, use whichever session date in the fetched bars is the
    # most recent one strictly before `day` -- IBKR's returned bars already
    # skip weekends/holidays, so this reflects the real last trading day.
    session_dates = sorted({dt.date() for dt in work["dt_et"] if dt.date() < day})
    if not session_dates:
        return premarket, work.iloc[0:0].copy()
    prev_day = session_dates[-1]
    prev_start = tz.localize(datetime.combine(prev_day, time(9, 30)))
    prev_end = tz.localize(datetime.combine(prev_day, time(16, 1)))
    prev_rth = work[(work["dt_et"] >= prev_start) & (work["dt_et"] < prev_end)].copy()
    return premarket, prev_rth
