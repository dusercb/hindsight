"""Share-float band (min_float / max_float): a candidate whose float is outside
the band is skipped (SKIPPED_FLOAT_FILTER). Float comes from the input/reference
CSV when supplied, otherwise from IBKR's Reuters snapshot via a float_fetcher; if
a float gate is set but float can't be determined, the candidate is
INSUFFICIENT_DATA rather than silently traded. The gate is off (no fetch) unless
min_float or max_float is configured."""
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
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 4, 10)
_DAILY_VOLUME_FETCHER = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=5.0, min_premarket_gain_pct=10.0, target_offset=0.20, stop_offset=0.10,
        intrabar_policy="conservative", scale_out=False, min_price=1.0, max_price=100.0,
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


def _bar(t, o, h, low, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": low, "close": c, "volume": v}


def _winning_premarket():
    """Climb 3.00 -> 4.00, pullback to 3.55, then a single-bar trigger+exit WIN."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.01, o - 0.01, o + 0.01, 5000))
    pullback_at = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(pullback_at, 3.98, 3.99, 3.55, 3.60, 1000))
    trigger_at = pullback_at + pd.Timedelta(seconds=10)
    rows.append(_bar(trigger_at, 4.02, 4.45, 3.98, 4.30, 6000))
    return pd.DataFrame(rows)


def _prev_rth():
    return pd.DataFrame([_bar(pd.Timestamp("2026-04-09 15:59:00", tz=TZ), 2.0, 2.0, 2.0, 2.0, 1000)])


class _Fetcher:
    """float_fetcher that records how many times it was called."""
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, ticker):
        self.calls += 1
        return self.value


def test_no_float_filter_configured_never_fetches_float():
    premarket = _winning_premarket()
    fetcher = _Fetcher(999_000_000.0)  # huge; would be blocked IF the gate ran
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER, float_fetcher=fetcher)
    assert tr.result == "WIN"
    assert fetcher.calls == 0  # gate off by default -> no fundamental request


def test_float_above_max_is_skipped():
    premarket = _winning_premarket()
    fetcher = _Fetcher(45_000_000.0)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_float=30_000_000),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER, float_fetcher=fetcher)
    assert tr.result == "SKIPPED_FLOAT_FILTER"
    assert tr.float_shares == pytest.approx(45_000_000.0)
    assert tr.float_source == "fetched"
    assert "above configured max_float" in tr.reason
    assert fetcher.calls == 1


def test_float_within_band_trades():
    premarket = _winning_premarket()
    fetcher = _Fetcher(15_000_000.0)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_float=30_000_000),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER, float_fetcher=fetcher)
    assert tr.result == "WIN"
    assert tr.float_shares == pytest.approx(15_000_000.0)
    assert tr.float_source == "fetched"


def test_float_below_min_is_skipped():
    premarket = _winning_premarket()
    fetcher = _Fetcher(500_000.0)
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(min_float=1_000_000),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER, float_fetcher=fetcher)
    assert tr.result == "SKIPPED_FLOAT_FILTER"
    assert "below configured min_float" in tr.reason


def test_csv_float_takes_precedence_and_skips_the_ibkr_fetch():
    """A float supplied via the input/reference CSV (float_shares) is used as-is;
    IBKR is not queried, and its historically-accurate value is what gates."""
    premarket = _winning_premarket()
    fetcher = _Fetcher(45_000_000.0)  # would be blocked; but CSV float wins
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_float=30_000_000),
                       float_shares=8_000_000.0, daily_volume_fetcher=_DAILY_VOLUME_FETCHER, float_fetcher=fetcher)
    assert tr.result == "WIN"
    assert tr.float_shares == pytest.approx(8_000_000.0)
    assert tr.float_source == "input/reference_csv"
    assert fetcher.calls == 0  # CSV value present -> no IBKR request


def test_unavailable_float_with_gate_set_is_insufficient_data():
    premarket = _winning_premarket()
    fetcher = _Fetcher(None)  # no subscription / Reuters has no float
    tr = backtest_one("TEST", DAY, premarket, premarket, _prev_rth(), _config(max_float=30_000_000),
                       daily_volume_fetcher=_DAILY_VOLUME_FETCHER, float_fetcher=fetcher)
    assert tr.result == "INSUFFICIENT_DATA"
    assert "Float unavailable" in tr.reason


def _write_yaml(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_float_band_flows_from_yaml_and_defaults_null(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, "strategy:\n  min_float: 500000\n  max_float: 30000000\n"))
    assert cfg.strategy.min_float == 500000
    assert cfg.strategy.max_float == 30000000
    cfg2 = load_config(_write_yaml(tmp_path, "strategy:\n  min_rvol: 4.0\n"))
    assert cfg2.strategy.min_float is None and cfg2.strategy.max_float is None


def test_invalid_float_config_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="max_float"):
        load_config(_write_yaml(tmp_path, "strategy:\n  max_float: 0\n"))
    with pytest.raises(ValueError, match="min_float"):
        load_config(_write_yaml(tmp_path, "strategy:\n  min_float: -5\n"))
    with pytest.raises(ValueError, match="min_float"):  # min > max
        load_config(_write_yaml(tmp_path, "strategy:\n  min_float: 40000000\n  max_float: 30000000\n"))
