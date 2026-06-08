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
