"""Strategy routing -- the shared truth table (premarket_backtester/routing.py).

Which engines run is a two-flag interaction, and the backtester, the live
observer, and the nightly parity replay must all resolve it identically: a
disagreement means the bot trades a strategy the backtest never ran (or misses
one it did), which is a Class-0 parity divergence on its own. These tests pin
the table and that signal_core re-exports the SAME function object the engine
uses.
"""
from dataclasses import replace

import pytest

from premarket_backtester.config import WedgeConfig
from premarket_backtester.routing import active_strategies


@pytest.mark.parametrize("enabled,concurrent,expected", [
    (False, False, ("dip",)),
    (False, True,  ("dip",)),      # concurrent is IGNORED while the wedge is off
    (True,  False, ("wedge",)),    # wedge REPLACES dip
    (True,  True,  ("dip", "wedge")),
])
def test_truth_table(enabled, concurrent, expected):
    w = replace(WedgeConfig(), enabled=enabled, concurrent_with_dip=concurrent)
    assert active_strategies(w) == expected


def test_dip_is_first_when_both_run():
    """Row order matters: the engine appends dip results before the wedge's, and
    the recorded/parity ordering follows it."""
    w = replace(WedgeConfig(), enabled=True, concurrent_with_dip=True)
    assert active_strategies(w)[0] == "dip"


def test_engine_and_signal_core_share_one_routing_implementation():
    import signal_core
    from premarket_backtester import engine, routing
    assert engine.active_strategies is routing.active_strategies
    assert signal_core.active_strategies is routing.active_strategies


def test_shipped_config_runs_the_dip_only():
    """The shipped config is the live intent: the dip alone. The wedge was switched
    off live on 2026-08-15, and the live bot refuses any other strategy set."""
    from premarket_backtester.config import load_config
    from pathlib import Path
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    assert active_strategies(cfg.wedge) == ("dip",)
