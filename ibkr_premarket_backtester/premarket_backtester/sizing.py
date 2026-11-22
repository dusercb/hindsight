"""Position sizing -- the pure decision core, physically separate from engine.py
so the live bot can import it (via signal_core) without dragging in the IBKR
historical-data layer: engine.py imports ibkr_data -> ib_insync, and the bot
runs ib_async instead, so the two must never meet in one import chain.

Moved verbatim from engine.py (Stage 1 of PORTING_TO_LIVE.md); engine re-imports
both names, so every existing caller (engine's progress readout, tests,
montecarlo's vectorized mirror) is unchanged. Behavior is byte-identical.
"""
from __future__ import annotations

from .config import RiskConfig
from .models import TradeResult


def _capped_shares(equity: float, entry_price: float, stop_price: float | None, volume_at_entry: float | None, risk: RiskConfig, vol_pct: float | None = None, lookback_volume: float | None = None, lookback_pct: float | None = None) -> tuple[str, int]:
    """Share count for ONE trade at a given equity, as the MINIMUM of every
    applicable cap -- the notional % (position_pct_of_equity) or fixed shares, the
    fixed-fractional risk cap (risk_pct_of_equity, entry-to-stop), the
    session-volume cap, and (wedge only) the lookback-window volume cap -- plus
    which cap bound it ("notional"/"shares"/"risk"/"volume"/"lookback_vol").
    `vol_pct` overrides risk.max_pct_of_volume_at_entry for this trade (the wedge
    strategy's independent participation cap); None = use the shared risk value.
    `lookback_pct`+`lookback_volume` add the wedge's lookback-window cap (at most
    that % of the volume traded across the pivot-lookback window) -- independent
    of, and stacking with, the session-volume cap; both must be present to apply.
    The single source of truth shared by both apply_position_sizing (the
    chronological sizing pass) and the live progress readout in engine.py, so a
    trade sizes identically in both. Ties resolve to the earlier-appended cap
    (notional/shares -> risk -> volume -> lookback_vol)."""
    caps: list[tuple[str, int]] = []
    if getattr(risk, "size_to_risk", False) and risk.max_leverage is not None:
        # SIZE-TO-RISK: the per-trade cash cap is lifted to the leverage ceiling
        # (max_leverage x equity), so the risk cap below governs -- a tight stop is
        # funded on margin to reach the full risk target instead of being clipped
        # by cash. Total exposure is still bounded by max_leverage in the funding
        # walk (single-trade and concurrent).
        caps.append(("notional", max(int((equity * risk.max_leverage) // entry_price), 0)))
    elif risk.position_pct_of_equity is not None:
        caps.append(("notional", max(int((equity * risk.position_pct_of_equity / 100.0) // entry_price), 0)))
    else:
        caps.append(("shares", risk.shares))
    if risk.risk_pct_of_equity is not None and stop_price is not None:
        # abs() so it works both ways: long stops sit BELOW the fill, short stops
        # ABOVE it -- the per-share risk is the distance either way.
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share > 0:
            caps.append(("risk", max(int((equity * risk.risk_pct_of_equity / 100.0) // risk_per_share), 0)))
    effective_vol_pct = vol_pct if vol_pct is not None else risk.max_pct_of_volume_at_entry
    if effective_vol_pct is not None and volume_at_entry:
        caps.append(("volume", int(volume_at_entry * effective_vol_pct / 100.0)))
    if lookback_pct is not None and lookback_volume:
        caps.append(("lookback_vol", int(lookback_volume * lookback_pct / 100.0)))
    return min(caps, key=lambda c: c[1])


def apply_position_sizing(results: list[TradeResult], risk: RiskConfig, wedge_vol_pct: float | None = None, wedge_lookback_pct: float | None = None) -> float:
    """Size every executed trade from the account's compounding equity, walking the
    trades in the CHRONOLOGICAL order they happened (candidates are processed in
    input-file order, which is arbitrary -- compounding must follow time).

    With position_pct_of_equity set: each trade goes all-in with that % of the
    current account value, buying as many whole shares as that affords at the
    actual entry fill price (95% rather than 100% leaves slippage headroom). With
    it null, the fixed `shares` count applies. Either way, risk.max_pct_of_volume_
    at_entry then caps that count against the ticker's own volume_at_entry (the
    fewer of the two wins) -- an equity-driven size can otherwise demand more
    shares than the ticker's actual session volume could plausibly fill without
    the order itself moving price, which only gets worse as equity/position size
    scale up. Updates shares/pnl_total/equity_after in place; pnl_per_share is
    untouched -- per-share economics don't depend on position size.

    The walk is EVENT-driven over each trade's [entry_time, exit_time] interval,
    because trades overlap -- two tickers' entries can be open at once, and under
    wedge.concurrent_with_dip so can two strategies'. Two consequences:
      * P&L is realized (equity updated) at each trade's EXIT, never at its entry
        row -- an overlapping entry can't be sized against profit that hadn't
        happened yet when it filled (sizing it off entry-ordered compounding was
        itself a small lookahead);
      * risk.max_leverage, when set, caps the COMMITTED NOTIONAL across all open
        positions at that multiple of realized equity (1.0 = cash-only, 2.0 = a
        Reg-T-style margin account). The per-trade caps are still computed
        against full equity -- the risk per trade is unchanged -- the funding cap
        only decides whether that size can be BOUGHT while other positions hold
        capital; a trade it clips gets size_basis "funding". null = unlimited
        parallel funding, the original behavior. Either way each entry's
        margin_used / leverage_at_entry / open_positions_at_entry are stamped so
        a run shows the leverage it actually demanded.

    wedge_vol_pct: the wedge strategy's INDEPENDENT volume-participation cap
    (wedge.max_pct_of_volume_at_entry). When set, trades with entry_mode
    "wedge_break" use it instead of risk.max_pct_of_volume_at_entry; None (or a
    non-wedge trade) keeps the shared cap.
    """
    trades = [r for r in results if r.entry_price is not None]
    equity = risk.starting_equity
    committed = 0.0          # notional currently tied up in open positions
    open_count = 0
    total_monthly_fees = 0.0
    prev_month: str | None = None  # "YYYY-MM" of the last month already charged
    entered: set[int] = set()

    def _enter(r: TradeResult) -> None:
        nonlocal equity, committed, open_count, total_monthly_fees, prev_month
        # Flat monthly platform fee: charge once per calendar month over the
        # trading span (first trade's month through the last, gap months batched)
        # BEFORE sizing that month's first trade, so the drag compounds into every
        # later position size. equity_after therefore already nets these out.
        if risk.monthly_fees:
            ym = r.date[:7]
            months = 1 if prev_month is None else (int(ym[:4]) - int(prev_month[:4])) * 12 + (int(ym[5:7]) - int(prev_month[5:7]))
            if months > 0:
                fee = round(risk.monthly_fees * months, 2)
                equity = round(equity - fee, 2)
                total_monthly_fees = round(total_monthly_fees + fee, 2)
                prev_month = ym
        # The position is the MINIMUM of every applicable cap (notional/shares, the
        # fixed-fractional risk cap, the volume cap); whichever binds is recorded as
        # size_basis. _capped_shares is the shared sizing core -- the same one the
        # live progress readout uses, so both agree. Wedge trades may carry their
        # own participation cap (wedge_vol_pct), independent of the shared one.
        is_wedge = r.entry_mode == "wedge_break"
        vol_pct = wedge_vol_pct if (wedge_vol_pct is not None and is_wedge) else None
        lb_pct = wedge_lookback_pct if (wedge_lookback_pct is not None and is_wedge) else None
        r.size_basis, shares = _capped_shares(equity, r.entry_price, r.stop_price, r.volume_at_entry, risk,
                                              vol_pct=vol_pct, lookback_volume=r.lookback_volume, lookback_pct=lb_pct)
        # Funding: what the account can actually BUY right now, with `committed`
        # dollars already holding other open positions.
        if risk.max_leverage is not None:
            headroom = max(equity * risk.max_leverage - committed, 0.0)
            fundable = int(headroom // r.entry_price)
            if fundable < shares:
                shares = fundable
                r.size_basis = "funding"
        notional = shares * r.entry_price
        cash_free = max(equity - committed, 0.0)
        r.open_positions_at_entry = open_count
        r.margin_used = round(max(notional - cash_free, 0.0), 2)
        r.leverage_at_entry = round((committed + notional) / equity, 3) if equity > 0 else None
        r.shares = shares
        r.pnl_total = round(r.pnl_per_share * shares, 2)
        committed += notional
        open_count += 1
        entered.add(id(r))

    def _exit(r: TradeResult) -> None:
        nonlocal equity, committed, open_count
        committed -= (r.shares or 0) * r.entry_price
        open_count -= 1
        equity = round(equity + r.pnl_total, 2)
        r.equity_after = equity

    # ISO time strings within one day share a UTC offset, so lexicographic order
    # IS time order; pairing with the date avoids parsing/timezone headaches. A
    # missing exit_time (engine rows always have one; hand-built test rows may
    # not) falls back to the entry instant -- the trade is treated as releasing
    # its capital immediately rather than inventing an overlap. Exits sort BEFORE
    # entries at the same instant (capital freed by a closing trade is available
    # to one entering on the same print); a trade whose exit event arrives before
    # its own entry has been processed (that same-instant case) is entered inline
    # first, so its lifecycle order is always entry -> exit.
    events: list[tuple[str, str, int, TradeResult]] = []
    for r in trades:
        entry_key = r.entry_time or ""
        exit_key = r.exit_time or entry_key
        if (r.date, exit_key) < (r.date, entry_key):
            exit_key = entry_key
        events.append((r.date, entry_key, 1, r))
        events.append((r.date, exit_key, 0, r))
    events.sort(key=lambda e: e[:3])
    for _date, _t, kind, r in events:
        if kind == 1:
            if id(r) not in entered:
                _enter(r)
        else:
            if id(r) not in entered:
                _enter(r)
            _exit(r)
    return total_monthly_fees
