"""signal_core -- the shared signal package (PORTING_TO_LIVE.md Stage 1).

THE PARITY CONTRACT: this package is the live bot's ONLY import surface into
the backtester repo. Every trading decision the bot makes -- qualification,
pennant/dip detection, entry triggers, stops, targets, exit management,
position sizing, calendar blackout -- comes from the exact code objects the
backtester validates, at one pinned commit SHA. The bot repo pins this repo at
a SHA, imports ONLY `signal_core`, and refuses to start on a pin mismatch
(PORTING_TO_LIVE.md D1/S5). Nothing here may import the IBKR data layer:
`premarket_backtester.ibkr_data` pulls ib_insync, the bot runs ib_async, and
the two must never meet in one process. tests/test_signal_core.py enforces
both properties (import purity + facade/backtester object identity).

Shape: a curated FACADE over the pure `premarket_backtester` modules rather
than a physical code move. The plan sketched physical extraction; the facade
achieves the same guarantees -- one SHA pins both programs' logic, and parity
is by construction because the exported names ARE the backtester's objects
(not copies that could drift). Physical extraction remains open as a later,
behavior-free refactor; this interface would not change.

The two batch entry points (`backtest_all_trades`, `backtest_wedge`) are
exported deliberately: the Stage-3 observer replays live-accumulated session
frames through them to produce intended-trade decisions, and the nightly
parity replay does the same over recorded feeds. Live event-driven wrappers
(signal_runtime) come later and must delegate every decision here.
"""
from __future__ import annotations

# --- configuration (the strategy IS the config: current config.yaml values go live)
from premarket_backtester.config import (
    AppConfig,
    BlackoutConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
    WedgeConfig,
    load_config,
    parse_hhmm,
)

# --- decision/result vocabulary
from premarket_backtester.models import Result, TradeResult

# --- calendar blackout gate (session_controller's first check of the day)
from premarket_backtester.calendar_blackout import blackout_reason

# --- WHICH strategies run (the one truth table both programs share; a
#     disagreement here would be a Class-0 divergence by itself)
from premarket_backtester.routing import active_strategies

# --- indicators + frame builders (1-minute EMA9/EMA20/VWAP/MACD, RVOL, resampling)
from premarket_backtester.strategy import (
    add_indicators,
    add_rvol,
    build_1m_bars,
)

# --- dip_red_break strategy: the full validated decision engine + pure helpers
from premarket_backtester.strategy import (
    backtest_all_trades,
    first_price_at_or_above,
    previous_close,
    resolve_intrabar,
)

# --- Phase B: minimal-latency live entry. evaluate_touch is the pure per-touch
#     fire-check the backtest and live path share; ArmedSetup is what the
#     on_armed hook reports (the trigger + frozen structure to watch tick-by-
#     tick); TouchDecision is evaluate_touch's return. The live trigger-watch
#     arms from bars, watches the tick stream, and calls evaluate_touch on the
#     crossing print -- same gate stack as the backtest, sub-second.
from premarket_backtester.strategy import ArmedSetup, TouchDecision, evaluate_touch

# --- wedge (symmetrical pennant) strategy: detector geometry + decision engine.
#     evaluate_wedge_touch is the wedge's counterpart to evaluate_touch: the pure
#     per-break gate stack both the backtest loop and the live trigger-watch call,
#     with WedgeArmedSetup carrying the FROZEN pennant geometry (both triggers and
#     the lines the stops sit on), reported via backtest_wedge(on_armed=...).
#     Freezing is a CORRECTNESS requirement, not just a latency one: a pennant
#     re-fitted on a part-formed candle can pick the opposite break side.
from premarket_backtester.wedge import (
    Pennant,
    WedgeArmedSetup,
    WedgeTouchDecision,
    arm_wedge,
    backtest_wedge,
    detect_pennant,
    evaluate_wedge_touch,
    pivot_positions,
    theil_sen,
)

# --- position sizing (notional/risk/volume caps + concurrent funding walk)
from premarket_backtester.sizing import (
    _capped_shares as capped_shares,
    apply_position_sizing,
)

__all__ = [
    # config
    "AppConfig", "BlackoutConfig", "RiskConfig", "SessionConfig",
    "StrategyConfig", "WedgeConfig", "load_config", "parse_hhmm",
    # vocabulary
    "Result", "TradeResult",
    # blackout + strategy routing
    "blackout_reason", "active_strategies",
    # indicators
    "add_indicators", "add_rvol", "build_1m_bars",
    # dip engine
    "backtest_all_trades", "first_price_at_or_above", "previous_close",
    "resolve_intrabar",
    # Phase B live-entry building blocks
    "evaluate_touch", "ArmedSetup", "TouchDecision",
    # wedge engine (+ its live-entry building blocks)
    "Pennant", "backtest_wedge", "detect_pennant", "pivot_positions",
    "theil_sen", "arm_wedge", "evaluate_wedge_touch", "WedgeArmedSetup", "WedgeTouchDecision",
    # sizing
    "capped_shares", "apply_position_sizing",
]
