"""Failed-retest early exit state machine (WedgeConfig.exit_retest_failure).

A swing extreme in the trade's favour is retested and HOLDS; the moment a later
candle breaks back through the attempt candle, exit. Pure/causal -- each candle is
complete when stepped, and the exit only fires on a candle AFTER the attempt.
"""
from premarket_backtester.wedge import _RetestExit


def _run(direction, entry, candles, bounce=0.5, tol=1.0):
    r = _RetestExit(direction, entry, bounce, tol)
    return [r.step(*c) for c in candles]


def test_short_failed_retest_covers_at_the_attempt_high():
    # swing low 9.50, bounce to ~9.92, RED candle retests 9.55 and holds -> arm at
    # its high 9.92; the next candle trades above 9.92 -> cover there.
    out = _run("short", 10.00, [
        (9.90, 10.00, 9.80, 9.85),
        (9.85, 9.90, 9.50, 9.55),      # drive to the swing low
        (9.55, 9.92, 9.55, 9.90),      # bounce up
        (9.90, 9.92, 9.55, 9.58),      # red retest, low holds -> arm @ 9.92
        (9.60, 9.95, 9.58, 9.93),      # breaks the attempt high -> exit
    ])
    assert out[-1] == 9.92 and all(x is None for x in out[:-1])


def test_long_failed_retest_sells_at_the_attempt_low():
    out = _run("long", 10.00, [
        (10.10, 10.50, 10.05, 10.45),  # swing high 10.50
        (10.45, 10.48, 10.10, 10.12),  # pull back
        (10.12, 10.49, 10.12, 10.46),  # green retest, high holds -> arm @ low 10.12
        (10.40, 10.42, 10.05, 10.08),  # breaks the attempt low -> exit
    ])
    assert out[-1] == 10.12 and all(x is None for x in out[:-1])


def test_no_exit_while_the_trade_keeps_extending():
    # continuous new lows: the short is working, never a failed retest -> never exits
    out = _run("short", 10.00, [
        (9.9, 9.95, 9.7, 9.72),
        (9.72, 9.75, 9.5, 9.52),
        (9.52, 9.55, 9.3, 9.32),
        (9.32, 9.35, 9.1, 9.12),
    ])
    assert all(x is None for x in out)


def test_no_exit_without_a_bounce_off_the_low():
    # price pins near the low (never bounces >= bounce_min) -> a retest never arms
    out = _run("short", 10.00, [
        (9.90, 9.95, 9.50, 9.52),      # low 9.50
        (9.52, 9.54, 9.51, 9.53),      # high 9.54 < 9.5*(1+0.5%) -> no bounce
        (9.53, 9.54, 9.51, 9.515),     # red, near low, but not armed (no bounce)
        (9.60, 9.80, 9.55, 9.78),      # rises, but nothing armed -> no exit
    ])
    assert all(x is None for x in out)


def test_a_new_low_after_arming_disarms_the_retest():
    # arm a retest, then price makes a NEW low (the short is still working): the
    # armed level is discarded, so a later up-candle must NOT trigger an exit.
    out = _run("short", 10.00, [
        (9.85, 9.90, 9.50, 9.55),      # swing low 9.50
        (9.55, 9.92, 9.55, 9.90),      # bounce
        (9.90, 9.92, 9.55, 9.58),      # red retest -> arm @ 9.92
        (9.58, 9.60, 9.40, 9.42),      # NEW low 9.40 -> disarm, re-base
        (9.45, 9.95, 9.44, 9.93),      # would have broken 9.92, but disarmed -> no exit
    ])
    assert all(x is None for x in out)
