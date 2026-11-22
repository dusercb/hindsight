"""--resume: continue an interrupted run. A previous run's non-ERROR result rows
are carried over (ERROR rows retry, unreached candidates run fresh), the merged
report covers the whole dataset, and resuming refuses loudly when any
result-affecting config differs (strategy/session/bar settings). risk is
deliberately NOT compared: position sizing recomputes over the merged results
every run, so it can change freely between the original run and the resume."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from premarket_backtester.config import (
    AppConfig, DataConfig, IBKRConfig, ReportingConfig, RiskConfig, SessionConfig, StrategyConfig, config_to_dict,
)
from premarket_backtester.engine import (
    _append_partial_rows, _trade_result_from_row, load_resume_results, resume_config_mismatches,
)
from premarket_backtester.models import TradeResult
from premarket_backtester.reporting import jsonify_cells


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ibkr=IBKRConfig(),
        data=DataConfig(input_csv=Path("unused.csv"), reference_csv=None, cache_dir=tmp_path, output_dir=tmp_path),
        session=SessionConfig(), strategy=StrategyConfig(), risk=RiskConfig(), reporting=ReportingConfig(),
    )


def _trade(ticker, day, result, entry=None, **kw):
    tr = TradeResult(ticker=ticker, date=day, result=result)
    tr.entry_price = entry
    for k, v in kw.items():
        setattr(tr, k, v)
    return tr


def _fake_run_dir(tmp_path: Path, cfg: AppConfig, rows: list[TradeResult]) -> Path:
    run = tmp_path / "run_20260101_000000"
    run.mkdir()
    (run / "config_used.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    pd.DataFrame(jsonify_cells([r.to_dict() for r in rows])).to_csv(run / "backtest_trades_20260101_000000.csv", index=False)
    return run


def test_row_roundtrip_preserves_types_through_the_csv(tmp_path):
    src = _trade("AAA", "2026-03-09", "WIN", entry=4.61,
                 exit_price=5.43, pnl_per_share=0.82, r_multiple=2.0, setup_index=2,
                 trigger_touches=3, path_ambiguous=True, stale_hod_breaks=1,
                 exit_legs=[{"kind": "target", "frac": 1.0, "price": 5.43, "time": "t"}])
    csv = tmp_path / "t.csv"
    pd.DataFrame(jsonify_cells([src.to_dict()])).to_csv(csv, index=False)
    back = _trade_result_from_row(pd.read_csv(csv).iloc[0].to_dict())
    assert back.ticker == "AAA" and back.result == "WIN"
    assert back.entry_price == pytest.approx(4.61)
    assert back.setup_index == 2 and isinstance(back.setup_index, int)
    assert back.trigger_touches == 3 and back.stale_hod_breaks == 1
    assert back.path_ambiguous is True
    assert back.exit_legs == [{"kind": "target", "frac": 1.0, "price": 5.43, "time": "t"}]
    assert back.stop_price is None  # never set -> NaN in CSV -> back to None


def test_resume_carries_done_rows_retries_errors_and_drops_strangers(tmp_path):
    cfg = _config(tmp_path)
    run = _fake_run_dir(tmp_path, cfg, [
        _trade("AAA", "2026-03-09", "WIN", entry=4.61),
        _trade("AAA", "2026-03-09", "LOSS", entry=5.10, setup_index=2),  # multi-trade day: both carried
        _trade("BBB", "2026-03-09", "ERROR"),                            # timeout casualty: retries
        _trade("CCC", "2026-03-09", "NO_TRIGGER"),                       # informational: carried
        _trade("ZZZ", "2026-03-09", "WIN", entry=2.0),                   # not in current input: dropped
    ])
    keys = {("AAA", "2026-03-09"), ("BBB", "2026-03-09"), ("CCC", "2026-03-09"), ("DDD", "2026-03-09")}
    carried, mismatches = load_resume_results(run, config_to_dict(cfg), keys)
    assert mismatches == []
    assert set(carried) == {("AAA", "2026-03-09"), ("CCC", "2026-03-09")}
    assert [r.result for r in carried[("AAA", "2026-03-09")]] == ["WIN", "LOSS"]
    # BBB (ERROR) and DDD (never reached) are left for the caller to process.


def test_resume_without_a_trades_csv_refuses(tmp_path):
    cfg = _config(tmp_path)
    run = tmp_path / "run_dead"
    run.mkdir()
    (run / "config_used.yaml").write_text(yaml.safe_dump(config_to_dict(cfg)), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="backtest_trades"):
        load_resume_results(run, config_to_dict(cfg), set())


def test_partial_checkpoint_appends_incrementally_and_roundtrips(tmp_path):
    path = tmp_path / "backtest_trades_partial_x.csv"
    _append_partial_rows(path, [_trade("AAA", "2026-03-09", "WIN", entry=4.61, setup_index=1)])
    _append_partial_rows(path, [_trade("BBB", "2026-03-09", "NO_TRIGGER")])
    df = pd.read_csv(path)
    assert list(df["ticker"]) == ["AAA", "BBB"]  # one header, two appends
    back = _trade_result_from_row(df.iloc[0].to_dict())
    assert back.result == "WIN" and back.entry_price == pytest.approx(4.61)


def test_resume_works_from_a_partial_checkpoint_alone(tmp_path):
    """A killed/crashed run writes no final CSV -- its incremental checkpoint
    must be enough to resume from (this was the original gap: four interrupted
    runs in a row left nothing to continue)."""
    cfg = _config(tmp_path)
    run = tmp_path / "run_killed"
    run.mkdir()
    (run / "config_used.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    _append_partial_rows(run / "backtest_trades_partial_x.csv", [
        _trade("AAA", "2026-03-09", "WIN", entry=4.61),
        _trade("BBB", "2026-03-09", "ERROR"),
    ])
    carried, mismatches = load_resume_results(run, config_to_dict(cfg), {("AAA", "2026-03-09"), ("BBB", "2026-03-09")})
    assert mismatches == []
    assert set(carried) == {("AAA", "2026-03-09")}  # ERROR still retries


def test_final_csv_is_preferred_over_a_leftover_partial(tmp_path):
    cfg = _config(tmp_path)
    run = _fake_run_dir(tmp_path, cfg, [_trade("AAA", "2026-03-09", "WIN", entry=4.61)])
    # A stale partial disagreeing with the final must be ignored.
    _append_partial_rows(run / "backtest_trades_partial_x.csv", [_trade("AAA", "2026-03-09", "LOSS", entry=9.99)])
    carried, _ = load_resume_results(run, config_to_dict(cfg), {("AAA", "2026-03-09")})
    assert [r.result for r in carried[("AAA", "2026-03-09")]] == ["WIN"]


def test_config_mismatch_detection_is_scoped_to_result_affecting_keys(tmp_path):
    cfg = _config(tmp_path)
    snap = config_to_dict(cfg)

    changed = config_to_dict(cfg)
    changed["strategy"]["min_rvol"] = 99.0
    assert any("strategy.min_rvol" in d for d in resume_config_mismatches(snap, changed))

    bar_changed = config_to_dict(cfg)
    bar_changed["data"]["bar_size"] = "30 secs"
    assert any("data.bar_size" in d for d in resume_config_mismatches(snap, bar_changed))

    # risk / reporting / plumbing changes do NOT block a resume.
    benign = config_to_dict(cfg)
    benign["risk"]["starting_equity"] = 999_999.0
    benign["reporting"]["trade_folders_for"] = "all"
    benign["data"]["request_timeout_s"] = 300.0
    assert resume_config_mismatches(snap, benign) == []


def test_knob_added_since_the_old_run_matches_only_at_its_default(tmp_path):
    cfg = _config(tmp_path)
    old = config_to_dict(cfg)
    del old["strategy"]["trade_winners_qty"]  # the old run predates this knob

    cur_default = config_to_dict(cfg)  # trade_winners_qty == schema default (1)
    assert resume_config_mismatches(old, cur_default) == []

    cur_changed = config_to_dict(cfg)
    cur_changed["strategy"]["trade_winners_qty"] = 3  # non-default: NOT comparable
    assert any("trade_winners_qty" in d for d in resume_config_mismatches(old, cur_changed))
