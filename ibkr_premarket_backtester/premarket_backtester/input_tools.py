from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
import pandas as pd

from .models import Candidate

# Prefer US-style MM/DD/YYYY because the input file uses DATES like 03/27/2026.
DATE_PATTERNS = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]


def parse_date(value: str):
    value = value.strip()
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {value!r}. Expected MM/DD/YYYY or YYYY-MM-DD.")


def _col(df: pd.DataFrame, *names: str) -> str | None:
    lookup = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _parse_float_shares(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().replace(",", "")
    if not s or s.upper() in {"NAN", "NONE", "NULL", "UNKNOWN"}:
        return None
    mult = 1.0
    if s[-1:].upper() == "K":
        mult, s = 1_000.0, s[:-1]
    elif s[-1:].upper() == "M":
        mult, s = 1_000_000.0, s[:-1]
    elif s[-1:].upper() == "B":
        mult, s = 1_000_000_000.0, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def load_reference_floats(reference_csv: str | Path | None) -> dict[str, float]:
    """Load optional ticker -> float_shares lookup.

    CSV may use columns: ticker/stock/symbol and float/float_shares/shares_float.
    Values may be raw numbers or shorthand such as 7.5M.
    """
    if not reference_csv:
        return {}
    path = Path(reference_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    tcol = _col(df, "ticker", "tickers", "symbol", "stock")
    fcol = _col(df, "float", "float_shares", "shares_float", "stock_float")
    if not tcol or not fcol:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = str(row[tcol]).strip().upper()
        val = _parse_float_shares(row[fcol])
        if ticker and ticker != "NAN" and val is not None:
            out[ticker] = val
    return out


def load_borrow_fees(borrow_csv: str | Path | None) -> dict[str, float]:
    """Load an optional ticker -> annualized borrow-fee %% lookup for the wedge
    strategy's short-side `max_borrow_fee_pct` gate. CSV columns: ticker/symbol/stock
    and fee/borrow_fee/fee_pct/rate. A missing file (the common case -- historical
    borrow rates aren't in IBKR/Yahoo) returns {}, and shorts are then treated as
    borrowable-but-assumed (flagged) so research isn't blocked; live enforces the
    real rate. Values are plain percents (e.g. 12.5 for 12.5%)."""
    if not borrow_csv:
        return {}
    path = Path(borrow_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    tcol = _col(df, "ticker", "tickers", "symbol", "stock")
    fcol = _col(df, "fee", "borrow_fee", "fee_pct", "borrow_fee_pct", "rate")
    if not tcol or not fcol:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = str(row[tcol]).strip().upper()
        if not ticker or ticker == "NAN" or pd.isna(row[fcol]):
            continue
        try:
            out[ticker] = float(str(row[fcol]).strip().replace("%", ""))
        except ValueError:
            continue
    return out


def load_candidates(csv_path: str | Path, reference_csv: str | Path | None = None) -> list[Candidate]:
    df = pd.read_csv(csv_path)
    tcol = _col(df, "ticker", "tickers", "symbol", "stock")
    dcol = _col(df, "date", "dates", "day")
    fcol = _col(df, "float", "float_shares", "shares_float", "stock_float")
    ncol = _col(df, "notes", "note")
    if not tcol or not dcol:
        raise ValueError("Input CSV must contain columns such as TICKERS,DATES or ticker,date")
    reference = load_reference_floats(reference_csv)
    out: list[Candidate] = []
    for _, row in df.iterrows():
        ticker = str(row[tcol]).strip().upper()
        if not ticker or ticker == "NAN":
            continue
        row_float = _parse_float_shares(row[fcol]) if fcol else None
        float_shares = row_float if row_float is not None else reference.get(ticker)
        notes = str(row[ncol]).strip() if ncol and not pd.isna(row[ncol]) else None
        out.append(Candidate(ticker=ticker, day=parse_date(str(row[dcol])), float_shares=float_shares, notes=notes))
    return out


def format_raw_ticker_dates(raw_path: str | Path, out_path: str | Path) -> Path:
    """Turn loose user text into TICKERS,DATES CSV using MM/DD/YYYY dates.

    Accepts lines like:
      KITT 05/14/2025
      2025-05-15 ICCT
      KITT, ICCT - 05/14/2025   (uses the first parseable date on the line)
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    text = raw_path.read_text(encoding="utf-8")
    rows = []
    date_re = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    ticker_re = r"\b[A-Z]{1,5}(?:\.[A-Z])?\b"
    for line in text.splitlines():
        clean = re.sub(r"[^A-Z0-9/,\-\s]", "", line.upper()).strip()
        clean = re.sub(r",+", ",", clean)
        if not clean or clean.startswith("#"):
            continue
        date_match = re.search(date_re, clean)
        if not date_match:
            continue
        day = parse_date(date_match.group(1))
        without_date = clean.replace(date_match.group(1), " ")
        for ticker in re.findall(ticker_re, without_date):
            if ticker in {"DATE", "DATES", "TICKER", "TICKERS", "NASDAQ", "NYSE", "AMEX"}:
                continue
            rows.append({"TICKERS": ticker, "DATES": day.strftime("%m/%d/%Y")})
    if not rows:
        raise ValueError("No ticker/date pairs found.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(out_path, index=False)
    return out_path
