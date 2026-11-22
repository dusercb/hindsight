"""Which strategy engine(s) run for a session -- the SINGLE source of truth,
shared by the backtester and the live bot through signal_core.

This exists because the routing is a two-flag interaction that is easy to
re-derive slightly differently in two places, and any such difference is a
Class-0 parity divergence (the bot would trade a strategy the backtest never
ran, or miss one it did). Both programs call active_strategies(); neither
re-implements the truth table:

    wedge.enabled  concurrent_with_dip   engines
    -------------  -------------------   ------------------------
    false          (ignored)             ("dip",)
    true           false                 ("wedge",)          <- wedge REPLACES dip
    true           true                  ("dip", "wedge")    <- both, concurrently

"dip" is emitted first when both run, matching the backtester's row order
(dip results are prepended to the wedge's in engine.py).
"""
from __future__ import annotations


def active_strategies(wedge_cfg) -> tuple[str, ...]:
    """The engine names that run for this config: "dip" and/or "wedge"."""
    if not wedge_cfg.enabled:
        return ("dip",)
    if wedge_cfg.concurrent_with_dip:
        return ("dip", "wedge")
    return ("wedge",)
