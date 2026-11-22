"""Every 1-minute indicator -- EMA9/EMA20/VWAP/MACD -- is computed once
(add_indicators) and shared by both the decision gates and the trade chart's
plotted lines, so the chart never shows something different from what the
strategy actually saw. The retrace-support gates (block_retrace_below_ema9/
ema20/vwap) block a trade if the retrace (HOD through the current touch) traded
below that indicator's own causal (prior-completed-minute) value at ANY point
along the way -- not just at the touch itself -- each independently switchable,
off by default. MACD periods (12/26/9 default) are configurable."""
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
    load_config,
)
from premarket_backtester.strategy import _retrace_breached_indicator, add_indicators, backtest_one

TZ = "America/New_York"
DAY = date(2026, 4, 10)
_DAILY_VOLUME_FETCHER = lambda ticker, day: 100.0  # noqa: E731


def _bar(t, o, h, low, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": low, "close": c, "volume": v}


# --- add_indicators: hand-computable EMA9/EMA20/VWAP/MACD ---

def _sample_one_min():
    return pd.DataFrame({
        "dt_et": pd.date_range("2026-04-10 07:00", periods=3, freq="1min", tz=TZ),
        "open": [10.0, 12.0, 11.0], "high": [11.0, 13.0, 12.0], "low": [9.0, 11.0, 10.0],
        "close": [10.0, 12.0, 11.0], "volume": [100.0, 200.0, 100.0],
    })


def test_add_indicators_ema9_ema20_vwap():
    result = add_indicators(_sample_one_min())
    # ewm(span=9, adjust=False): alpha=0.2 -> 10.0, 10.4, 10.52
    assert result["ema9"].tolist() == pytest.approx([10.0, 10.4, 10.52])
    # ewm(span=20, adjust=False): alpha=2/21 -> 10.0, 10.190476, 10.267574
    assert result["ema20"].tolist() == pytest.approx([10.0, 10.190476, 10.267574], abs=1e-5)
    # typical_price == close here (high/low are symmetric around it); vwap =
    # cumulative(typical*volume)/cumulative(volume) -> 10.0, 11.3333, 11.25
    assert result["vwap"].tolist() == pytest.approx([10.0, 11.333333, 11.25], abs=1e-5)


def test_add_indicators_default_macd_matches_the_old_12_26_9():
    """The consolidated function must reproduce the grandfathered add_macd's
    output byte-for-byte at the default 12/26/9 periods -- macd = EMA12 - EMA26
    of the close, signal = EMA9 of macd, hist = macd - signal."""
    df = _sample_one_min()
    result = add_indicators(df)
    close = df["close"]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    assert result["macd"].tolist() == pytest.approx(macd.tolist())
    assert result["signal"].tolist() == pytest.approx(signal.tolist())
    assert result["hist"].tolist() == pytest.approx((macd - signal).tolist())


def test_add_indicators_respects_configured_macd_periods():
    df = _sample_one_min()
    result = add_indicators(df, macd_fast=3, macd_slow=6, macd_signal=2)
    close = df["close"]
    macd = close.ewm(span=3, adjust=False).mean() - close.ewm(span=6, adjust=False).mean()
    signal = macd.ewm(span=2, adjust=False).mean()
    assert result["macd"].tolist() == pytest.approx(macd.tolist())
    assert result["signal"].tolist() == pytest.approx(signal.tolist())
    # And these differ from the default-period MACD, proving the params flow through.
    default_macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    assert result["macd"].tolist() != pytest.approx(default_macd.tolist())


# --- _retrace_breached_indicator: causal lookup, any-bar breach, NaN-safe ---

def test_retrace_breached_indicator_true_when_any_bar_dips_below_its_own_prior_minute_value():
    one_min = pd.DataFrame({
        "dt_et": [pd.Timestamp("2026-04-10 07:00", tz=TZ), pd.Timestamp("2026-04-10 07:01", tz=TZ)],
        "level": [5.0, 6.0],
    })
    move_bars = pd.DataFrame({
        "dt_et": [pd.Timestamp("2026-04-10 07:01:30", tz=TZ), pd.Timestamp("2026-04-10 07:02:15", tz=TZ)],
        "low": [4.9, 6.5],  # first bar's prior-minute (07:00) level is 5.0 -> 4.9 breaches it
    })
    assert _retrace_breached_indicator(move_bars, one_min, "level") is True


def test_retrace_breached_indicator_false_when_every_bar_holds_above_its_level():
    one_min = pd.DataFrame({
        "dt_et": [pd.Timestamp("2026-04-10 07:00", tz=TZ), pd.Timestamp("2026-04-10 07:01", tz=TZ)],
        "level": [5.0, 6.0],
    })
    move_bars = pd.DataFrame({
        "dt_et": [pd.Timestamp("2026-04-10 07:01:30", tz=TZ), pd.Timestamp("2026-04-10 07:02:15", tz=TZ)],
        "low": [5.1, 6.5],
    })
    assert _retrace_breached_indicator(move_bars, one_min, "level") is False


def test_retrace_breached_indicator_missing_prior_minute_is_not_a_breach():
    """A bar whose prior minute has no row (e.g. the session's first minute)
    can't be judged against a value that doesn't exist -- NaN comparisons are
    always False, so it fails open rather than blocking on missing data."""
    one_min = pd.DataFrame({"dt_et": [pd.Timestamp("2026-04-10 07:05", tz=TZ)], "level": [5.0]})
    move_bars = pd.DataFrame({"dt_et": [pd.Timestamp("2026-04-10 07:00:10", tz=TZ)], "low": [0.01]})
    assert _retrace_breached_indicator(move_bars, one_min, "level") is False


# --- strategy-level: the gates actually block a breaching retrace's entry ---

def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0, min_premarket_gain_pct=10.0, target_offset=0.20, stop_offset=0.10,
        intrabar_policy="conservative", scale_out=False,
        max_pullback_retrace_pct=70.0,  # loosened so this fixture's deep dip stays a VALID retrace
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


def _premarket_with_deep_pullback():
    """Climb 3.00 -> 4.00 (100 bars), then a pullback to 3.40 -- deep enough to
    dip below the 1-minute EMA9 (~3.73), EMA20 (~3.52), AND VWAP (~3.49) as of
    the prior completed minute, but still only a 59% retrace of the ~1.01 rip
    (valid under this fixture's loosened 70% max), then a clean single-bar
    trigger/exit via the bar fallback."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.01, o - 0.01, o + 0.01, 5000))
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 3.99, 3.40, 3.60, 1000))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 4.45, 3.98, 4.30, 6000))
    return pd.DataFrame(rows)


def _prev_rth():
    return pd.DataFrame([_bar(pd.Timestamp("2026-04-09 15:59:00", tz=TZ), 2.0, 2.0, 2.0, 2.0, 1000)])


def test_gates_off_by_default_let_a_breaching_retrace_trade():
    premarket = _premarket_with_deep_pullback()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.result == "WIN"
    assert tr.entry_price is not None
    assert tr.skipped_retrace_below_ema9 == 0
    assert tr.skipped_retrace_below_ema20 == 0
    assert tr.skipped_retrace_below_vwap == 0


def test_block_retrace_below_ema9_invalidates_the_breaching_touch():
    # The breaching touch IS a HOD break; failing this gate means the HOD broke
    # without a valid entry, so under first_setup_only the setup goes STALE_HOD.
    premarket = _premarket_with_deep_pullback()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(block_retrace_below_ema9=True),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"
    assert tr.skipped_retrace_below_ema9 == 1
    assert tr.stale_hod_breaks == 1


def test_block_retrace_below_ema20_invalidates_the_breaching_touch():
    premarket = _premarket_with_deep_pullback()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(block_retrace_below_ema20=True),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"
    assert tr.skipped_retrace_below_ema20 == 1
    assert tr.stale_hod_breaks == 1


def test_block_retrace_below_vwap_invalidates_the_breaching_touch():
    premarket = _premarket_with_deep_pullback()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(block_retrace_below_vwap=True),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    assert tr.entry_price is None
    assert tr.result == "STALE_HOD"
    assert tr.skipped_retrace_below_vwap == 1
    assert tr.stale_hod_breaks == 1


# --- config ---

def test_retrace_indicator_gates_flow_from_yaml_and_default_false(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "strategy:\n  block_retrace_below_ema9: true\n  block_retrace_below_ema20: true\n"
        "  block_retrace_below_vwap: true\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.strategy.block_retrace_below_ema9 is True
    assert cfg.strategy.block_retrace_below_ema20 is True
    assert cfg.strategy.block_retrace_below_vwap is True

    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    cfg2 = load_config(p2)
    assert cfg2.strategy.block_retrace_below_ema9 is False
    assert cfg2.strategy.block_retrace_below_ema20 is False
    assert cfg2.strategy.block_retrace_below_vwap is False


def test_macd_periods_flow_from_yaml_default_12_26_9_and_validate(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  macd_fast: 8\n  macd_slow: 21\n  macd_signal: 5\n", encoding="utf-8")
    cfg = load_config(p)
    assert (cfg.strategy.macd_fast, cfg.strategy.macd_slow, cfg.strategy.macd_signal) == (8, 21, 5)

    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    cfg2 = load_config(p2)
    assert (cfg2.strategy.macd_fast, cfg2.strategy.macd_slow, cfg2.strategy.macd_signal) == (12, 26, 9)

    for bad in ("macd_fast: 0", "macd_slow: -1", "macd_signal: 2.5"):
        pb = tmp_path / "bad.yaml"
        pb.write_text(f"strategy:\n  {bad}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="macd_"):
            load_config(pb)


# --- chart: the same indicators get plotted, not just decided on ---

def test_trade_chart_plots_ema9_ema20_vwap_without_error(tmp_path):
    from premarket_backtester.reporting import write_trade_folder

    premarket = _premarket_with_deep_pullback()
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(), daily_volume_fetcher=_DAILY_VOLUME_FETCHER)
    write_trade_folder(tr, premarket, tmp_path, write_chart=True)

    safe = f"{tr.ticker}_{tr.date}"
    assert (tmp_path / "trade_folders" / safe / "chart.png").exists()
    assert not (tmp_path / "trade_folders" / safe / "chart_error.txt").exists()
