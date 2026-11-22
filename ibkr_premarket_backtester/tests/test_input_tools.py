from pathlib import Path

from premarket_backtester.input_tools import format_raw_ticker_dates, load_candidates


def test_format_raw_ticker_dates(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("ARTL, 03/27/2026\nONCO 03/27/2026\n")
    out = tmp_path / "ticker_dates.csv"
    format_raw_ticker_dates(raw, out)
    text = out.read_text()
    assert "TICKERS,DATES" in text
    assert "ARTL,03/27/2026" in text


def test_load_candidates_with_tickers_dates_and_float(tmp_path: Path):
    csv = tmp_path / "ticker_dates.csv"
    csv.write_text("TICKERS,DATES,FLOAT\nARTL,03/27/2026,7.5M\n")
    items = load_candidates(csv)
    assert items[0].ticker == "ARTL"
    assert items[0].day.isoformat() == "2026-03-27"
    assert items[0].float_shares == 7_500_000


def test_load_candidates_uses_reference_float(tmp_path: Path):
    csv = tmp_path / "ticker_dates.csv"
    ref = tmp_path / "reference_floats.csv"
    csv.write_text("TICKERS,DATES\nARTL,03/27/2026\n")
    ref.write_text("TICKERS,FLOAT\nARTL,8M\n")
    items = load_candidates(csv, ref)
    assert items[0].float_shares == 8_000_000
