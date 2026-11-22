"""config.yaml is the single source of truth for settings; these tests pin down
that (a) yaml values really do override the dataclass fallbacks, and (b) a typo'd
key or section fails loudly instead of silently running with defaults -- which
from the outside looks exactly like "my config change had no effect"."""

from pathlib import Path

import pytest

from premarket_backtester.config import config_to_dict, load_config


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_yaml_values_override_dataclass_defaults(tmp_path):
    p = _write_yaml(
        tmp_path,
        """
strategy:
  min_rvol: 7.5
  min_premarket_gain_pct: 42.0
  first_setup_only: false
risk:
  shares: 250
""",
    )
    cfg = load_config(p)
    assert cfg.strategy.min_rvol == 7.5
    assert cfg.strategy.min_premarket_gain_pct == 42.0
    assert cfg.strategy.first_setup_only is False
    assert cfg.risk.shares == 250
    # Keys omitted from the yaml fall back to the dataclass defaults.
    assert cfg.strategy.stop_offset == 0.10


def test_unknown_strategy_key_fails_loudly(tmp_path):
    p = _write_yaml(tmp_path, "strategy:\n  min_rvoll: 5.0\n")
    with pytest.raises(ValueError, match="min_rvoll"):
        load_config(p)


def test_unknown_top_level_section_fails_loudly(tmp_path):
    p = _write_yaml(tmp_path, "stratagy:\n  min_rvol: 5.0\n")
    with pytest.raises(ValueError, match="stratagy"):
        load_config(p)


def test_unknown_data_key_fails_loudly(tmp_path):
    # The data section is built field-by-field (for path resolution), which used
    # to swallow unknown keys silently instead of raising like the ** sections.
    p = _write_yaml(tmp_path, "data:\n  tick_request_pauses: 5.0\n")
    with pytest.raises(ValueError, match="tick_request_pauses"):
        load_config(p)


def test_scale_out_flows_from_yaml(tmp_path):
    p = _write_yaml(tmp_path, "strategy:\n  scale_out: false\n  scale_out_step: 0.05\n  scale_out_max_sells: 4\n  scale_out_stop_offset: 0.02\n")
    cfg = load_config(p)
    assert cfg.strategy.scale_out is False
    assert cfg.strategy.scale_out_step == 0.05
    assert cfg.strategy.scale_out_max_sells == 4
    assert cfg.strategy.scale_out_stop_offset == 0.02
    p2 = _write_yaml(tmp_path, "strategy:\n  min_rvol: 4.0\n")
    assert load_config(p2).strategy.scale_out is True  # default: ladder on
    assert load_config(p2).strategy.scale_out_sell_pct == 50.0
    assert load_config(p2).strategy.scale_out_max_sells == 3   # 50% -> 25% -> final 25%
    assert load_config(p2).strategy.scale_out_stop_offset == 0.01


def test_position_sizing_flows_from_yaml(tmp_path):
    p = _write_yaml(tmp_path, "risk:\n  position_pct_of_equity: null\n  shares: 250\n")
    cfg = load_config(p)
    assert cfg.risk.position_pct_of_equity is None
    assert cfg.risk.shares == 250
    p2 = _write_yaml(tmp_path, "risk:\n  starting_equity: 15000\n")
    cfg2 = load_config(p2)
    assert cfg2.risk.position_pct_of_equity == 50.0  # default: percent-of-equity sizing on
    assert cfg2.risk.starting_equity == 15000


def test_red_volume_check_flows_and_validates(tmp_path):
    p = _write_yaml(tmp_path, "strategy:\n  red_volume_check: \"largest_bar\"\n")
    assert load_config(p).strategy.red_volume_check == "largest_bar"
    p2 = _write_yaml(tmp_path, "strategy:\n  min_rvol: 4.0\n")
    assert load_config(p2).strategy.red_volume_check == "total"  # flow-based default
    p3 = _write_yaml(tmp_path, "strategy:\n  red_volume_check: \"sum\"\n")
    with pytest.raises(ValueError, match="red_volume_check"):
        load_config(p3)


def test_rvol_baseline_stat_flows_and_validates(tmp_path):
    p = _write_yaml(tmp_path, "strategy:\n  rvol_baseline_stat: \"mean\"\n")
    assert load_config(p).strategy.rvol_baseline_stat == "mean"
    p2 = _write_yaml(tmp_path, "strategy:\n  min_rvol: 4.0\n")
    assert load_config(p2).strategy.rvol_baseline_stat == "median"  # robust default
    p3 = _write_yaml(tmp_path, "strategy:\n  rvol_baseline_stat: \"average\"\n")
    with pytest.raises(ValueError, match="rvol_baseline_stat"):
        load_config(p3)


def test_request_timeout_flows_from_yaml(tmp_path):
    p = _write_yaml(tmp_path, "data:\n  request_timeout_s: 25\n")
    assert load_config(p).data.request_timeout_s == 25.0
    p2 = _write_yaml(tmp_path, "strategy:\n  min_rvol: 4.0\n")
    assert load_config(p2).data.request_timeout_s == 60.0  # default


def test_config_to_dict_snapshot_is_plain_and_complete(tmp_path):
    p = _write_yaml(tmp_path, "strategy:\n  min_rvol: 9.0\n")
    snapshot = config_to_dict(load_config(p))
    assert snapshot["strategy"]["min_rvol"] == 9.0
    assert set(snapshot) == {"ibkr", "data", "session", "strategy", "risk", "reporting", "monte_carlo", "blackout", "wedge"}
    # Paths are serialized as plain strings so the snapshot yaml-dumps cleanly.
    assert isinstance(snapshot["data"]["input_csv"], str)
    assert snapshot["data"]["reference_csv"] is None or isinstance(snapshot["data"]["reference_csv"], str)
