"""The on_armed hook (Phase B): backtest_all_trades reports each armed dip setup
-- the trigger + frozen structure the live path watches tick-by-tick. The hook
is decision-free (byte-identical backtest), so these tests only assert it reports
the SAME trigger the trade actually fired on, and never fires spuriously.
"""
import pandas as pd

from premarket_backtester.strategy import ArmedSetup, backtest_all_trades

# reuse the dip fixture + config from the structure-stop suite
from test_dip_red_break_structure_stop import _config, _dip_session, _prev, _DV, DAY


def _run_with_armed(session, cfg):
    armed: list[ArmedSetup] = []
    trades = backtest_all_trades("TST", DAY, session, session, _prev(), cfg,
                                 daily_volume_fetcher=_DV, on_armed=armed.append)
    return trades, armed


def test_on_armed_reports_the_trigger_the_trade_fires_on():
    cfg = _config(entry_mode="dip_red_break")
    trades, armed = _run_with_armed(_dip_session(red_dip=True, wins=True), cfg)
    entered = [t for t in trades if t.entry_price is not None]
    assert entered, "fixture should enter a dip trade"
    tr = entered[0]
    assert armed, "a setup should have been reported as armed before the cross"
    # the armed trigger the live path would watch matches the trade's trigger
    triggers = {round(a.trigger, 2) for a in armed}
    assert round(tr.trigger_price, 2) in triggers
    # the frozen structure is coherent: stop basis (pullback low) below the trigger
    last = armed[-1]
    assert last.pullback_low < last.trigger <= last.armed_hod


def test_on_armed_is_decision_free_same_trade_with_and_without_it():
    cfg = _config(entry_mode="dip_red_break")
    session = _dip_session(red_dip=True, wins=True)
    with_hook, _ = _run_with_armed(session, cfg)
    without = backtest_all_trades("TST", DAY, session, session, _prev(), cfg, daily_volume_fetcher=_DV)
    # identical entries whether or not the observer hook is attached
    a = [(t.entry_price, t.stop_price, t.target_price, t.result) for t in with_hook if t.entry_price]
    b = [(t.entry_price, t.stop_price, t.target_price, t.result) for t in without if t.entry_price]
    assert a == b and a


def test_no_armed_report_when_no_setup_forms():
    """A session that never forms a qualifying dip reports no armed setups."""
    cfg = _config(entry_mode="dip_red_break")
    # a green-only 'dip' never produces a red candle to break -> no dip_red trigger
    trades, armed = _run_with_armed(_dip_session(red_dip=False, wins=True), cfg)
    assert not any(t.entry_price for t in trades)
    # dip_red_break needs a red high; with none, no dip_red trigger is ever armed
    assert armed == []
