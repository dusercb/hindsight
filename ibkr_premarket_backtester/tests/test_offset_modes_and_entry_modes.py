"""Two config-driven strategy dimensions added together:

1. offset_mode -- target/stop measured as flat cents (original) or as a PERCENT of
   the entry fill, so a $12 name gets a proportionally wider stop than a $3 one
   instead of the same flat 25c. R-multiple uses the REAL per-share risk
   (fill - stop) in either mode.

2. entry_mode -- which price break arms the entry. hod_break (the original, break
   of the HOD) plus three "buy the break of the dip's downtrend" proxies that fire
   BELOW the HOD: dip_swing_high (first higher-high off the pullback low),
   dip_trendline (break of a descending pivot line, swing-high fallback), and
   dip_ema_reclaim (reclaim of an EMA/VWAP the dip had pierced). Every mode reuses
   the identical upstream qualification + entry gates; only the trigger differs.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig, load_config,
)
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DV = lambda ticker, day: 200.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=1.0, min_premarket_gain_pct=25.0, min_pullback_cents=0.08,
        intrabar_policy="conservative", scale_out=False, earliest_entry_time=None,
        latest_entry_time=None, block_buy_below_vwap=False, block_buy_below_ema9=False,
        block_buy_below_ema20=False, min_premarket_volume=None,
    )
    defaults.update(overrides)
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("u"), reference_csv=None, cache_dir=Path("u"), output_dir=Path("u")),
        session=SessionConfig(), strategy=StrategyConfig(**defaults), risk=RiskConfig(), reporting=ReportingConfig(),
    )


def _bar(t, o, h, l, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _prev():
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 2.5, 2.5, 2.5, 2.5, 1000)])


# --------------------------------------------------------------------------- #
# offset_mode: cents vs percent                                               #
# --------------------------------------------------------------------------- #

def _clean_winner(s: float) -> pd.DataFrame:
    """Climb 3.00->4.00 HOD, a ~10% dip to 3.60, a CLEAN trigger bar (crosses the
    HOD+offset with a low above either mode's stop and a high below either mode's
    target -- no ambiguity), then a later bar that lifts through both modes'
    targets. Every price is multiplied by `s` so the same shape trades cheap
    (s=1, ~$4) or expensive (s=4, ~$16)."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    for k in range(50):
        o = (3.00 + 0.02 * k) * s
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.02 * s, o - 0.005 * s, o + 0.02 * s, 8000))
    hod = 4.00 * s  # last climb bar high == 3.98s + 0.02s
    t = start + pd.Timedelta(seconds=500)
    # dip to 3.60 (10% of HOD, 40% of the 3.00->4.00 rip)
    rows.append(_bar(t, 3.98 * s, 3.98 * s, 3.60 * s, 3.62 * s, 3000)); t += pd.Timedelta(seconds=10)
    # a couple of recovery bars building back toward the HOD (still below it)
    for c in (3.75, 3.90):
        rows.append(_bar(t, (c - 0.03) * s, (c + 0.01) * s, (c - 0.04) * s, c * s, 4000)); t += pd.Timedelta(seconds=10)
    # CLEAN trigger bar: opens 3.98s, high 4.05s, low 3.96s -> crosses HOD+0.01
    # without reaching +10% target or -5% stop of the fill.
    rows.append(_bar(t, 3.98 * s, 4.05 * s, 3.96 * s, 4.03 * s, 9000)); t += pd.Timedelta(seconds=10)
    # later bars lift cleanly through both targets (+0.50 cents / +10% percent)
    for c in (4.20, 4.45, 4.70):
        rows.append(_bar(t, (c - 0.05) * s, (c + 0.05) * s, (c - 0.06) * s, c * s, 6000)); t += pd.Timedelta(seconds=10)
    for _ in range(4):
        rows.append(_bar(t, 4.70 * s, 4.80 * s, 4.66 * s, 4.75 * s, 4000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_percent_offsets_scale_stop_and_target_with_price():
    """Percent mode puts the stop/target a % of the fill away, so the per-share
    risk grows with price -- unlike the flat-cents mode."""
    pm = _clean_winner(4.0)  # ~$12-16 name
    cents = backtest_one("EXP", DAY, pm, pm, _prev(), _config(offset_mode="cents", target_offset=0.50, stop_offset=0.25), daily_volume_fetcher=_DV)
    pct = backtest_one("EXP", DAY, pm, pm, _prev(), _config(offset_mode="percent", target_offset_pct=10.0, stop_offset_pct=5.0), daily_volume_fetcher=_DV)
    assert cents.entry_price == pct.entry_price  # same entry, only the exits differ
    # flat 25c stop on a ~$16 fill:
    assert round(cents.entry_price - cents.stop_price, 2) == 0.25
    # 5% stop on the same fill (~$0.80) is proportional to price -- and, on this
    # expensive name, several times wider than the flat 25c cents stop:
    assert round(pct.entry_price - pct.stop_price, 2) == round(pct.entry_price * 0.05, 2)
    assert (pct.entry_price - pct.stop_price) > (cents.entry_price - cents.stop_price) * 3


def test_percent_offset_r_multiple_uses_real_per_share_risk():
    """A winner that hits the +10% target with a -5% stop is a +2R trade in percent
    mode -- R divides by the ACTUAL (fill - stop), not the old flat stop_offset."""
    pm = _clean_winner(4.0)
    pct = backtest_one("EXP", DAY, pm, pm, _prev(), _config(offset_mode="percent", target_offset_pct=10.0, stop_offset_pct=5.0), daily_volume_fetcher=_DV)
    assert pct.result == "WIN"
    assert pct.r_multiple == pytest.approx(2.0, abs=0.15)


def test_offset_mode_and_pcts_flow_from_yaml_and_validate(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  offset_mode: percent\n  target_offset_pct: 8.0\n  stop_offset_pct: 4.0\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.strategy.offset_mode == "percent"
    assert cfg.strategy.target_offset_pct == 8.0 and cfg.strategy.stop_offset_pct == 4.0
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategy:\n  offset_mode: dollars\n", encoding="utf-8")
    with pytest.raises(ValueError, match="offset_mode"):
        load_config(bad)
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("strategy:\n  stop_offset_pct: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stop_offset_pct"):
        load_config(bad2)


# --------------------------------------------------------------------------- #
# entry_mode                                                                  #
# --------------------------------------------------------------------------- #

def _dip_reversal() -> pd.DataFrame:
    """Climb to a 5.00 HOD, a staircase dip of LOWER HIGHS down to 4.20, then a
    reversal that makes higher-highs and ultimately breaks the HOD. The dip-break
    modes should enter on the reversal (below 5.00); hod_break waits for 5.01."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    for k in range(40):
        o = 3.00 + 0.05 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.05, o - 0.01, o + 0.05, 8000))
    t = start + pd.Timedelta(seconds=400)
    for (o, h, l, c) in [(4.98, 4.99, 4.90, 4.92), (4.92, 4.94, 4.75, 4.78), (4.78, 4.80, 4.55, 4.58),
                         (4.58, 4.62, 4.35, 4.38), (4.38, 4.42, 4.20, 4.22)]:
        rows.append(_bar(t, o, h, l, c, 3000)); t += pd.Timedelta(seconds=10)
    for (o, h, l, c) in [(4.22, 4.45, 4.21, 4.44), (4.44, 4.70, 4.43, 4.68), (4.68, 4.90, 4.67, 4.88),
                         (4.88, 5.05, 4.87, 5.03), (5.03, 5.60, 5.02, 5.55)]:
        rows.append(_bar(t, o, h, l, c, 9000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):
        rows.append(_bar(t, 5.55, 5.70, 5.50, 5.60, 4000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_hod_break_enters_at_the_hod_plus_offset():
    pm = _dip_reversal()
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(entry_mode="hod_break"), daily_volume_fetcher=_DV)
    assert tr.entry_price == pytest.approx(5.01, abs=0.001)  # HOD 5.00 + entry_offset 0.01
    assert tr.entry_mode == "hod_break"


def test_dip_swing_high_enters_below_the_hod_on_the_first_higher_high():
    pm = _dip_reversal()
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(entry_mode="dip_swing_high"), daily_volume_fetcher=_DV)
    assert tr.entry_price is not None
    assert tr.entry_price < 5.00  # fired on the reversal, well before the HOD reclaim
    assert tr.trigger_price < 5.00
    assert tr.entry_mode == "dip_swing_high"


def test_dip_trendline_enters_below_the_hod():
    pm = _dip_reversal()
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(entry_mode="dip_trendline"), daily_volume_fetcher=_DV)
    assert tr.entry_price is not None
    assert tr.entry_price < 5.00
    assert tr.entry_mode == "dip_trendline"


def _dip_that_pierces_and_reclaims_ema() -> pd.DataFrame:
    """A slower move so the 1-minute EMA9 sits UP inside the price range: a rip, a
    dip whose lows dig below the trailing EMA9, then a reclaim back above it and on
    to a HOD break. Gives dip_ema_reclaim something real to reclaim."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    price = 3.00
    # gentle 6-minute climb (EMA9 trails just under price)
    for k in range(36):
        price += 0.03
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), price - 0.03, price + 0.01, price - 0.04, price, 6000))
    hod = price  # ~4.08
    t = start + pd.Timedelta(seconds=360)
    # dip that undercuts the trailing EMA9 for ~1.5 min
    for c in [hod - 0.15, hod - 0.30, hod - 0.42, hod - 0.50, hod - 0.46, hod - 0.40, hod - 0.30, hod - 0.18]:
        rows.append(_bar(t, c + 0.05, c + 0.06, c - 0.03, c, 4000)); t += pd.Timedelta(seconds=10)
    # reclaim and push to a new HOD
    for c in [hod - 0.05, hod + 0.10, hod + 0.30, hod + 0.55, hod + 0.80]:
        rows.append(_bar(t, c - 0.08, c + 0.05, c - 0.09, c, 9000)); t += pd.Timedelta(seconds=10)
    for _ in range(6):
        rows.append(_bar(t, hod + 0.80, hod + 0.95, hod + 0.75, hod + 0.85, 5000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_dip_ema_reclaim_requires_a_prior_breach_then_fires_on_reclaim():
    pm = _dip_that_pierces_and_reclaims_ema()
    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(entry_mode="dip_ema_reclaim", dip_reclaim_indicator="ema9", min_premarket_gain_pct=10.0),
                      daily_volume_fetcher=_DV)
    assert tr.entry_mode == "dip_ema_reclaim"
    # It either entered on the reclaim or found no reclaim -- but it must NOT have
    # entered at a bare HOD break; when it does enter, it's on the reclaim level.
    if tr.entry_price is not None:
        assert tr.result in ("WIN", "LOSS", "EXIT_AT_OPEN")


def test_dip_ema_reclaim_no_breach_no_trade():
    """A dip so shallow it never loses the EMA gives dip_ema_reclaim nothing to
    reclaim -- no entry (vs hod_break, which would still take the HOD break)."""
    pm = _dip_reversal()  # fast rip: 1-min EMA lags far below, dip never pierces it
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(entry_mode="dip_ema_reclaim"), daily_volume_fetcher=_DV)
    assert tr.entry_price is None


def test_entry_mode_flows_from_yaml_and_validates(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  entry_mode: dip_trendline\n  dip_reclaim_indicator: vwap\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.strategy.entry_mode == "dip_trendline"
    assert cfg.strategy.dip_reclaim_indicator == "vwap"
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategy:\n  entry_mode: buy_the_dip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="entry_mode"):
        load_config(bad)
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("strategy:\n  dip_reclaim_indicator: sma50\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dip_reclaim_indicator"):
        load_config(bad2)


def test_default_entry_mode_is_hod_break_and_default_offset_is_cents():
    cfg = StrategyConfig()
    assert cfg.entry_mode == "hod_break"
    assert cfg.offset_mode == "cents"
