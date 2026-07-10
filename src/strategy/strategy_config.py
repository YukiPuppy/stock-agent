from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY_CONFIG: dict[str, dict[str, Any]] = {
    "trend_pullback": {
        "enabled": True,
        "version": "v1",
        "min_pct_chg_5d": 0.03,
        "max_pct_chg_1d": 0.06,
        "min_close_position_20": 0.55,
        "min_volume_ratio_5": 1.0,
        "require_above_ma5": True,
        "require_above_ma10": True,
    },
    "breakout_volume": {
        "enabled": True,
        "version": "v1",
        "min_pct_chg_5d": 0.05,
        "min_volume_ratio_5": 1.3,
        "min_close_position_20": 0.70,
        "max_pct_chg_1d": 0.095,
        "require_above_ma5": True,
    },
    "support_rebound": {
        "enabled": True,
        "version": "v1",
        "min_pct_chg_1d": -0.095,
        "max_pct_chg_1d": -0.02,
        "min_close_position_20": 0.35,
        "max_close_position_20": 0.75,
        "require_above_ma20": True,
        # min_amount_ma5 uses amount_ma5 in thousand yuan.
        "min_amount_ma5": 0,
    },
    "industry_rotation": {
        "enabled": False,
        "version": "v1_strength_follow",
        "min_amount_ma5": 10000.0,
        "min_turnover_rate": 0.3,
        "min_industry_strength_score": 60.0,
        "min_industry_return_3d": 0.0,
        "min_industry_return_5d": 0.015,
        "min_industry_amount_ratio_5": 0.9,
        "min_pct_chg_5d": 0.015,
        "min_close_position_20": 0.55,
        "min_moneyflow_score": 0.0,
        "max_pct_chg_3d": 0.10,
        "max_pct_chg_5d": 0.14,
        "max_close_position_20": 0.90,
        "min_relative_strength_5d": 0.015,
    },
    "moneyflow_accumulation": {
        "enabled": False,
        "version": "v1_main_inflow",
        "min_amount_ma5": 10000.0,
        "min_turnover_rate": 0.3,
        "min_moneyflow_score": 12.0,
        "min_main_net_amount_ratio": 2.0,
        "min_pct_chg_1d": -0.03,
        "max_pct_chg_1d": 0.05,
        "max_pct_chg_3d": 0.08,
        "min_pct_chg_5d": -0.04,
        "max_pct_chg_5d": 0.10,
        "max_turnover_rate": 8.0,
    },
    "low_vol_trend": {
        "enabled": False,
        "version": "v1_ma_alignment",
        "min_amount_ma5": 10000.0,
        "min_turnover_rate": 0.3,
        "require_above_ma5": True,
        "require_above_ma10": True,
        "require_above_ma20": True,
        "min_pct_chg_5d": 0.01,
        "max_pct_chg_5d": 0.10,
        "max_abs_pct_chg_3d": 0.06,
        "max_abs_pct_chg_1d": 0.035,
        "max_pct_chg_1d": 0.03,
        "min_close_position_20": 0.45,
        "max_close_position_20": 0.88,
        "min_volume_ratio_5": 0.7,
        "max_volume_ratio_5": 1.8,
        "max_turnover_rate": 6.0,
    },
    "oversold_rebound": {
        "enabled": False,
        "version": "v1_mild_oversold",
        "min_amount_ma5": 10000.0,
        "min_turnover_rate": 0.3,
        "min_pct_chg_3d": -0.09,
        "max_pct_chg_3d": -0.015,
        "min_pct_chg_5d": -0.14,
        "min_pct_chg_10d": -0.22,
        "min_close_position_20": 0.12,
        "max_close_position_20": 0.60,
        "min_moneyflow_score": 3.0,
        "min_industry_strength_score": 45.0,
        "max_down_pct_chg_3d": -0.10,
        "max_down_pct_chg_5d": -0.16,
        "max_down_pct_chg_10d": -0.25,
    },
    "volume_dryup_breakout": {
        "enabled": False,
        "version": "v1_dryup_recover",
        "min_amount_ma5": 10000.0,
        "min_turnover_rate": 0.3,
        "require_above_ma5": True,
        "require_above_ma10": True,
        "min_volume_ratio_5": 0.85,
        "max_volume_ratio_5": 1.8,
        "min_volume_ratio_daily_basic": 1.0,
        "max_volume_ratio_daily_basic": 2.2,
        "min_pct_chg_1d": 0.01,
        "max_pct_chg_1d": 0.065,
        "max_pct_chg_3d": 0.10,
        "min_close_position_20": 0.60,
        "confirm_min_close_position_20": 0.70,
    },
    "relative_strength_pullback": {
        "enabled": False,
        "version": "v1_rs_pullback",
        "min_amount_ma5": 10000.0,
        "min_turnover_rate": 0.3,
        "require_above_ma10": True,
        "require_above_ma20": True,
        "min_relative_strength_5d": 0.015,
        "min_pct_chg_10d": 0.02,
        "min_pct_chg_5d": -0.02,
        "max_pct_chg_5d": 0.07,
        "min_close_position_20": 0.35,
        "max_close_position_20": 0.85,
        "min_industry_strength_score": 50.0,
        "min_moneyflow_score": 5.0,
    },
}


def load_strategy_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path is not None else _default_config_path()
    defaults = deepcopy(DEFAULT_STRATEGY_CONFIG)
    if not path.exists():
        return defaults

    try:
        parsed = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    except OSError:
        return defaults

    if not isinstance(parsed, dict):
        return defaults

    for strategy_name, values in parsed.items():
        if not isinstance(values, dict):
            continue
        merged = defaults.get(strategy_name, {}).copy()
        merged.update(values)
        defaults[strategy_name] = merged
    return defaults


def get_strategy_config(strategy_name: str, config: dict | None = None) -> dict:
    source = load_strategy_config() if config is None else config
    defaults = DEFAULT_STRATEGY_CONFIG.get(strategy_name, {}).copy()
    values = source.get(strategy_name, {}) if isinstance(source, dict) else {}
    if isinstance(values, dict):
        defaults.update(values)
    return defaults


def is_strategy_enabled(strategy_name: str, config: dict | None = None) -> bool:
    strategy_config = get_strategy_config(strategy_name, config)
    return bool(strategy_config.get("enabled", True))


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "strategies.yaml"


def _parse_simple_yaml(content: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    current_section: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            key = line.strip()
            if not key.endswith(":"):
                continue
            current_section = key[:-1].strip()
            parsed[current_section] = {}
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        parsed[current_section][key.strip()] = _parse_scalar(value.strip())
    return parsed


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(marker in value for marker in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value
