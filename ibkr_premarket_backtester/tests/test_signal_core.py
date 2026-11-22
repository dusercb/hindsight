"""signal_core -- the Stage-1 parity gates (PORTING_TO_LIVE.md).

Two properties define the package; both are enforced here, not documented-only:

  1. IMPORT PURITY: `import signal_core` must not load the IBKR data layer
     (ibkr_data -> ib_insync). The live bot runs ib_async; ib_insync leaking
     into its process via the signal import chain is exactly the dependency
     collision the extraction exists to prevent. Checked in a SUBPROCESS so
     this test is immune to whatever other tests already imported.

  2. OBJECT IDENTITY: every exported symbol IS the backtester's object (not a
     copy). Parity is then true by construction -- there is no second
     implementation to drift.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_import_purity_no_ibkr_layer_in_subprocess():
    code = (
        "import sys, signal_core;"
        "leaked = [m for m in ('ib_insync', 'ib_async', 'premarket_backtester.ibkr_data') if m in sys.modules];"
        "print('LEAKED:' + ','.join(leaked) if leaked else 'PURE')"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "PURE", out.stdout + out.stderr


def test_every_export_is_the_backtesters_object():
    import signal_core
    from premarket_backtester import calendar_blackout, config, models, sizing, strategy, wedge

    same = {
        "load_config": config.load_config, "AppConfig": config.AppConfig,
        "StrategyConfig": config.StrategyConfig, "WedgeConfig": config.WedgeConfig,
        "RiskConfig": config.RiskConfig, "SessionConfig": config.SessionConfig,
        "BlackoutConfig": config.BlackoutConfig, "parse_hhmm": config.parse_hhmm,
        "Result": models.Result, "TradeResult": models.TradeResult,
        "blackout_reason": calendar_blackout.blackout_reason,
        "add_indicators": strategy.add_indicators, "add_rvol": strategy.add_rvol,
        "build_1m_bars": strategy.build_1m_bars,
        "backtest_all_trades": strategy.backtest_all_trades,
        "first_price_at_or_above": strategy.first_price_at_or_above,
        "previous_close": strategy.previous_close,
        "resolve_intrabar": strategy.resolve_intrabar,
        "Pennant": wedge.Pennant, "backtest_wedge": wedge.backtest_wedge,
        "detect_pennant": wedge.detect_pennant,
        "pivot_positions": wedge.pivot_positions, "theil_sen": wedge.theil_sen,
        "capped_shares": sizing._capped_shares,
        "apply_position_sizing": sizing.apply_position_sizing,
    }
    for name, src in same.items():
        assert getattr(signal_core, name) is src, f"signal_core.{name} is not the backtester's object"


def test_all_is_complete_and_resolvable():
    import signal_core
    for name in signal_core.__all__:
        assert getattr(signal_core, name, None) is not None, f"__all__ lists unresolvable {name!r}"


def test_engine_and_signal_core_share_one_sizing_implementation():
    """The regression that would silently break parity: engine growing its own
    sizing copy again. Both must be the sizing module's functions."""
    import signal_core
    from premarket_backtester import engine, sizing
    assert engine.apply_position_sizing is sizing.apply_position_sizing
    assert engine._capped_shares is sizing._capped_shares
    assert signal_core.apply_position_sizing is sizing.apply_position_sizing


def test_wedge_engine_is_exported_for_the_observer():
    """The bot trades BOTH strategies concurrently (bot README): the wedge
    engine must be part of the facade, not an afterthought."""
    import signal_core
    assert callable(signal_core.backtest_wedge) and callable(signal_core.backtest_all_trades)
    assert "backtest_wedge" in signal_core.__all__
