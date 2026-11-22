"""Close-confirmed breakout entry (strategy.require_breakout_close_confirm).

OFF (default): a bar whose HIGH crosses the trigger fills intrabar at the touch --
the original behavior. ON: the crossing bar must also CLOSE at/above the trigger
AND in the top (1 - breakout_close_min_frac) of its range; only then does the
entry land on the NEXT bar's open. A weak/wicky break is deferred (the level stays
armed), never condemned. Same downstream stop/target/exit path either way.
"""
from datetime import date

import pandas as pd
import pytest

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig,
)
from premarket_backtester.strategy import backtest_one

TZ = "America/New_York"
DAY = date(2026, 5, 5)
_DV = lambda ticker, day: 100.0  # noqa: E731


def _config(**overrides) -> AppConfig:
    defaults = dict(
        min_rvol=2.5, min_premarket_gain_pct=25.0, offset_mode="cents",
        entry_mode="dip_red_break", stop_mode="pullback_low", target_mode="r_multiple",
        target_r_multiple=1.0, min_pullback_cents=0.08, intrabar_policy="conservative",
        scale_out=False, earliest_entry_time=None, latest_entry_time=None,
        min_premarket_volume=None,
    )
    defaults.update(overrides)
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv="u", reference_csv=None, cache_dir="u", output_dir="u"),
        session=SessionConfig(), strategy=StrategyConfig(**defaults), risk=RiskConfig(), reporting=ReportingConfig(),
    )


def _bar(t, o, h, l, c, v):
    return {"dt_et": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _prev():
    return pd.DataFrame([_bar(pd.Timestamp("2026-05-04 15:59:00", tz=TZ), 2.5, 2.5, 2.5, 2.5, 1000)])


def _sess(breaker: str) -> pd.DataFrame:
    """Rip 3.00 -> 5.00 HOD, a red dip to 4.20 (last red high 4.60 -> trigger 4.61),
    a green pause below the trigger, then a breakout bar that crosses 4.61 with a
    close that is 'strong' (top of range), 'weak' (bottom of range, below trigger),
    or 'churn' (a run of weak-closing pokes that never confirm), and a run to 5.60."""
    start = pd.Timestamp(f"{DAY} 08:00:00", tz=TZ)
    rows = []
    for k in range(40):
        o = 3.00 + 0.05 * k
        rows.append(_bar(start + pd.Timedelta(seconds=10 * k), o, o + 0.05, o - 0.01, o + 0.05, 8000))
    t = start + pd.Timedelta(seconds=400)
    for (o, h, l, c) in [(4.95, 4.96, 4.75, 4.78), (4.78, 4.80, 4.55, 4.58), (4.58, 4.60, 4.20, 4.25)]:
        rows.append(_bar(t, o, h, l, c, 3000)); t += pd.Timedelta(seconds=10)
    rows.append(_bar(t, 4.25, 4.35, 4.24, 4.33, 2000)); t += pd.Timedelta(seconds=10)  # green pause, below trigger
    if breaker == "churn":
        # Every bar pokes above 4.61 but closes in the bottom of its range -> never confirms.
        for _ in range(6):
            rows.append(_bar(t, 4.55, 4.70, 4.54, 4.56, 9000)); t += pd.Timedelta(seconds=10)
        return pd.DataFrame(rows)
    if breaker == "weak":
        # First crossing bar closes below the trigger (deferred), the next closes
        # strong and confirms; both the confirm and its next-bar open stay within
        # the 5% slippage cap of the 4.61 trigger (entry 4.80 <= limit 4.84).
        rows.append(_bar(t, 4.55, 4.70, 4.54, 4.56, 9000)); t += pd.Timedelta(seconds=10)  # deferred
        rows.append(_bar(t, 4.60, 4.78, 4.58, 4.76, 9000)); t += pd.Timedelta(seconds=10)  # confirms (close_pos 0.90)
        rows.append(_bar(t, 4.80, 5.10, 4.75, 5.05, 6000)); t += pd.Timedelta(seconds=10)  # entry bar, open 4.80
        rows.append(_bar(t, 5.05, 5.60, 5.00, 5.55, 7000)); t += pd.Timedelta(seconds=10)
        for _ in range(3):
            rows.append(_bar(t, 5.50, 5.55, 5.45, 5.50, 2000)); t += pd.Timedelta(seconds=10)
        return pd.DataFrame(rows)
    # "strong": the crossing bar itself closes in the top of its range and confirms.
    rows.append(_bar(t, 4.55, 4.70, 4.54, 4.66, 9000)); t += pd.Timedelta(seconds=10)  # close_pos 0.75 -> confirms
    rows.append(_bar(t, 4.72, 5.00, 4.66, 4.95, 6000)); t += pd.Timedelta(seconds=10)  # next open 4.72
    rows.append(_bar(t, 4.95, 5.60, 4.90, 5.55, 7000)); t += pd.Timedelta(seconds=10)
    for _ in range(3):
        rows.append(_bar(t, 5.50, 5.55, 5.45, 5.50, 2000)); t += pd.Timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_off_fills_intrabar_on_the_touch():
    pm = _sess("strong")
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(require_breakout_close_confirm=False), daily_volume_fetcher=_DV)
    assert tr.entry_price == pytest.approx(4.61)          # the trigger, on the breakout bar itself
    assert tr.entry_fill_source == "bar"
    assert tr.result == "WIN"


def test_on_enters_next_bar_open_after_a_strong_close():
    pm = _sess("strong")
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(require_breakout_close_confirm=True), daily_volume_fetcher=_DV)
    # The breakout bar (close 4.66, top of its range) confirms; entry is the NEXT
    # bar's open, 4.72 -- a worse price than the 4.61 touch, in exchange for proof.
    assert tr.entry_price == pytest.approx(4.72)
    assert tr.entry_fill_source == "bar_confirm"
    assert tr.trigger_price == pytest.approx(4.61)
    assert tr.trigger_time != tr.entry_time              # signal bar precedes the fill bar
    assert tr.stop_price == pytest.approx(4.20)          # structure stop unchanged by the entry timing
    assert tr.result == "WIN"


def test_on_defers_a_weak_close_then_confirms_later():
    pm = _sess("weak")
    # The confirmed fill (4.80) is $0.19 above the 4.61 trigger; widen the fill cap
    # here so this test isolates close-confirm DEFERRAL, not the slippage cap (which
    # would otherwise correctly reject a 19c-slipped fill under the default 15c).
    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(require_breakout_close_confirm=True, entry_limit_slippage_cents=0.50),
                      daily_volume_fetcher=_DV)
    # The first crossing bar closes 4.56 (below the 4.61 trigger) -> deferred, NOT
    # a trade and NOT stale. The following bar closes strong (4.95, top third) and
    # confirms, so entry lands on the bar after that at its 4.95 open.
    assert tr.trigger_touches >= 2
    assert tr.entry_price == pytest.approx(4.80)
    assert tr.entry_fill_source == "bar_confirm"


def test_on_never_confirms_takes_no_trade():
    pm = _sess("churn")
    tr = backtest_one("TST", DAY, pm, pm, _prev(), _config(require_breakout_close_confirm=True), daily_volume_fetcher=_DV)
    assert tr.trigger_touches >= 1                        # the level was crossed repeatedly
    assert tr.entry_price is None                         # but no bar ever closed strong -> no fill

    # The identical tape DOES trade with confirmation off (fills the first touch).
    off = backtest_one("TST", DAY, pm, pm, _prev(), _config(require_breakout_close_confirm=False), daily_volume_fetcher=_DV)
    assert off.entry_price == pytest.approx(4.61)


def test_max_risk_pct_rejects_a_too_wide_structural_stop():
    # touch entry 4.61 with a pullback-low stop at 4.20 -> risk = 0.41 = 8.9% of entry.
    pm = _sess("strong")
    ok = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(require_breakout_close_confirm=False, first_setup_only=True, max_risk_pct=10.0),
                      daily_volume_fetcher=_DV)
    assert ok.entry_price == pytest.approx(4.61)          # 8.9% < 10% cap -> trades
    assert ok.skipped_risk_too_wide == 0

    tight = backtest_one("TST", DAY, pm, pm, _prev(),
                         _config(require_breakout_close_confirm=False, first_setup_only=True, max_risk_pct=5.0),
                         daily_volume_fetcher=_DV)
    assert tight.entry_price is None                       # 8.9% > 5% cap -> rejected
    assert tight.skipped_risk_too_wide >= 1
    assert tight.result == "STALE_HOD"


def test_min_frac_threshold_gates_the_confirm():
    pm = _sess("strong")
    # Demand a close in the top 1% of range: the 0.75-close breaker (and every
    # later bar on this tape) fails, so nothing confirms and no trade is taken.
    tr = backtest_one("TST", DAY, pm, pm, _prev(),
                      _config(require_breakout_close_confirm=True, breakout_close_min_frac=0.99),
                      daily_volume_fetcher=_DV)
    assert tr.entry_price is None
