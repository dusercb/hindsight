"""trade_winners_qty ("press winners") + continuation_vwap_above_entry_pct:

A winning trade may unlock the hunt for a SECOND, genuinely fresh HOD/pullback
(the new HOD must exceed the true session high), up to trade_winners_qty total
trades -- any LOSS ends the day immediately. Unlike max_trades_per_day (which
re-hunts after ANY completed trade), this only ever adds trades on strength.

The continuation break may additionally require the session VWAP to have lifted
continuation_vwap_above_entry_pct above the prior winning entry -- VWAP is
volume-weighted, so it only climbs that far this fast when heavy volume keeps
printing at higher prices (a volatile, heavily traded long), which is exactly
the kind of run worth pressing.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig, load_config,
)
from premarket_backtester.strategy import backtest_all_trades

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DV = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=2.5, min_premarket_gain_pct=25.0, target_offset=0.20, stop_offset=0.10,
        min_pullback_cents=0.08, intrabar_policy="conservative", scale_out=False,
        earliest_entry_time=None, latest_entry_time=None, min_premarket_volume=None,
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


def _two_setup_session(first_wins: bool = True, continuation_volume: int = 500_000) -> pd.DataFrame:
    """Move 1: climb 3.00 -> 4.00 HOD, pullback to 3.55, break fills 4.01, then a
    WIN (target-only bar, exit 4.21) or a LOSS (stop-only bar). Move 2: a fresh
    climb to a 5.00 HOD (above the 4.25 session high), pullback to 4.65, break
    fills 5.01, target-only exit 5.21 -- a clean second winner IF it is hunted.
    continuation_volume controls the move-2 climb's volume: 500k/bar drags the
    session VWAP above move 1's 4.01 entry (heavy accumulation); 5,000/bar leaves
    VWAP down near the 3.00-4.00 base (thin drift)."""
    start = pd.Timestamp(f"{DAY} 07:00:00", tz=TZ)
    rows = []
    for k in range(100):
        o = 3.00 + 0.01 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.01, o - 0.01, o + 0.01, 5000))
    t = start + pd.Timedelta(seconds=1000)
    rows.append(_bar(t, 3.98, 3.99, 3.55, 3.60, 1000)); t += pd.Timedelta(seconds=10)          # pullback
    rows.append(_bar(t, 3.92, 4.05, 3.92, 4.02, 6000)); t += pd.Timedelta(seconds=10)          # break: fill 4.01
    if first_wins:
        rows.append(_bar(t, 4.05, 4.25, 4.02, 4.20, 5000)); t += pd.Timedelta(seconds=10)      # target-only: WIN 4.21
    else:
        rows.append(_bar(t, 4.00, 4.02, 3.80, 3.85, 5000)); t += pd.Timedelta(seconds=10)      # stop-only: LOSS 3.91
    # Move 2: fresh climb through the old session high to a 5.00 HOD.
    for k in range(16):
        o = 4.20 + 0.05 * k
        rows.append(_bar(t, o, o + 0.05, o - 0.01, o + 0.05, continuation_volume)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.98, 4.99, 4.65, 4.70, 4000)); t += pd.Timedelta(seconds=10)          # pullback
    rows.append(_bar(t, 4.95, 5.05, 4.95, 5.02, 8000)); t += pd.Timedelta(seconds=10)          # break: fill 5.01
    rows.append(_bar(t, 5.05, 5.25, 5.00, 5.20, 5000)); t += pd.Timedelta(seconds=10)          # target-only: WIN 5.21
    for _ in range(3):
        rows.append(_bar(t, 5.18, 5.20, 5.15, 5.18, 2000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def _run(pm, **cfg):
    return backtest_all_trades("TEST", DAY, pm, pm, _prev(), _config(**cfg), daily_volume_fetcher=_DV)


def test_default_qty_one_stops_after_the_first_win():
    trs = _run(_two_setup_session(first_wins=True))
    assert len(trs) == 1
    assert trs[0].result == "WIN"


def test_a_win_unlocks_a_second_fresh_setup():
    trs = _run(_two_setup_session(first_wins=True), trade_winners_qty=2)
    assert [tr.result for tr in trs] == ["WIN", "WIN"]
    assert [tr.setup_index for tr in trs] == [1, 2]
    assert trs[1].entry_price == pytest.approx(5.01)
    assert trs[1].premarket_high == pytest.approx(5.00)  # a genuine NEW high of day
    assert trs[1].entry_price > trs[0].exit_price        # pressed strength, not a re-test


def test_a_loss_ends_the_day_even_with_qty_remaining():
    trs = _run(_two_setup_session(first_wins=False), trade_winners_qty=3)
    assert len(trs) == 1
    assert trs[0].result == "LOSS"


def test_max_trades_per_day_still_rehunts_after_a_loss():
    # The unconditional knob keeps its existing semantics, independent of the
    # win-gated one: a loss does NOT end the day under max_trades_per_day.
    trs = _run(_two_setup_session(first_wins=False), max_trades_per_day=2, trade_winners_qty=1)
    assert [tr.result for tr in trs] == ["LOSS", "WIN"]


def test_continuation_vwap_gate_blocks_when_vwap_has_not_extended():
    # Heavy continuation volume puts session VWAP around ~4.5 at the second
    # break; requiring it 20% above the 4.01 entry (>= 4.81) blocks the
    # continuation, ends the day with just the first win, and the diagnostic
    # rides the day's last trade row instead of vanishing.
    trs = _run(_two_setup_session(first_wins=True), trade_winners_qty=2,
               continuation_vwap_above_entry_pct=20.0)
    assert [tr.result for tr in trs] == ["WIN"]
    assert trs[0].skipped_continuation_vwap >= 1


def test_continuation_vwap_gate_passes_on_heavy_accumulation():
    trs = _run(_two_setup_session(first_wins=True), trade_winners_qty=2,
               continuation_vwap_above_entry_pct=0.0)
    assert [tr.result for tr in trs] == ["WIN", "WIN"]
    assert trs[0].skipped_continuation_vwap == 0
    assert trs[1].skipped_continuation_vwap == 0


def test_continuation_vwap_gate_blocks_a_thin_drift():
    # Same second setup shape, but the continuation climb prints only 5,000
    # shares/bar -- VWAP stays anchored near the 3.00-4.00 base, below the 4.01
    # prior entry, so even the 0.0 ("merely above entry") requirement fails.
    trs = _run(_two_setup_session(first_wins=True, continuation_volume=5_000),
               trade_winners_qty=2, continuation_vwap_above_entry_pct=0.0)
    assert [tr.result for tr in trs] == ["WIN"]
    assert trs[0].skipped_continuation_vwap >= 1


def test_yaml_flow_and_validation(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("strategy:\n  trade_winners_qty: 3\n  continuation_vwap_above_entry_pct: 2.5\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.strategy.trade_winners_qty == 3
    assert cfg.strategy.continuation_vwap_above_entry_pct == 2.5
    p2 = tmp_path / "config2.yaml"
    p2.write_text("strategy:\n  min_rvol: 4.0\n", encoding="utf-8")
    cfg2 = load_config(p2)
    assert cfg2.strategy.trade_winners_qty == 1                       # default: off
    assert cfg2.strategy.continuation_vwap_above_entry_pct is None    # default: no VWAP condition
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategy:\n  trade_winners_qty: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trade_winners_qty"):
        load_config(bad)
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("strategy:\n  continuation_vwap_above_entry_pct: -150\n", encoding="utf-8")
    with pytest.raises(ValueError, match="continuation_vwap_above_entry_pct"):
        load_config(bad2)
