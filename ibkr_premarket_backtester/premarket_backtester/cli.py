from __future__ import annotations

import argparse
from pathlib import Path
from rich.console import Console

from .config import load_config
from .engine import run_backtest
from .input_tools import format_raw_ticker_dates

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR premarket HOD first-pullback backtester. Backtest/paper research only; does not place orders.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="Run the ticker/date backtest using IBKR historical bars.")
    b.add_argument("--config", default="config.yaml")
    b.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="RUN_DIR",
        help="Carry over a previous run's completed results and only process the candidates it didn't "
             "finish (ERROR rows retry; the merged report covers the whole dataset). With no value, "
             "resumes the newest run folder that wrote a trades CSV. Requires the same result-affecting "
             "config (strategy/session/bar settings) -- refuses loudly otherwise.",
    )

    f = sub.add_parser("format-input", help="Convert loose ticker/date text into data/ticker_dates.csv.")
    f.add_argument("--raw", required=True, help="Text file containing tickers and dates.")
    f.add_argument("--out", default="data/ticker_dates.csv")

    args = parser.parse_args()
    if args.cmd == "backtest":
        cfg = load_config(args.config)
        resume = None
        if args.resume == "latest":
            runs = sorted(p for p in cfg.data.output_dir.glob("run_*") if p.is_dir() and any(p.glob("backtest_trades_*.csv")))
            if not runs:
                console.print("[red]--resume: no previous run with a trades CSV found under "
                              f"{cfg.data.output_dir} -- nothing to resume.[/red]")
                return
            resume = runs[-1]
        elif args.resume:
            resume = Path(args.resume)
        try:
            run_backtest(cfg, resume_from=resume)
        except (ValueError, FileNotFoundError) as e:
            if resume is None:
                raise
            # A resume refusal (config mismatch / nothing to carry) is an
            # expected answer, not a crash -- print it readably, no traceback.
            console.print(f"[bold red]Resume refused:[/bold red] {e}")
    elif args.cmd == "format-input":
        out = format_raw_ticker_dates(Path(args.raw), Path(args.out))
        console.print(f"Wrote formatted CSV to [green]{out}[/green]")


if __name__ == "__main__":
    main()
