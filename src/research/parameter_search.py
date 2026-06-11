from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path


DEFAULT_PARAMETER_SEARCH_SPACE: dict = {
    "trend_pullback": {
        "enabled": True,
        "max_combinations": 50,
        "base_params": {
            "require_above_ma5": True,
            "require_above_ma10": True,
        },
        "search_space": {
            "min_pct_chg_5d": [0.02, 0.03, 0.05],
            "max_pct_chg_1d": [0.04, 0.06, 0.08],
            "min_close_position_20": [0.45, 0.55, 0.65],
            "min_volume_ratio_5": [0.8, 1.0, 1.2],
        },
    },
    "breakout_volume": {
        "enabled": True,
        "max_combinations": 40,
        "base_params": {
            "require_above_ma5": True,
        },
        "search_space": {
            "min_pct_chg_5d": [0.04, 0.05, 0.08],
            "min_volume_ratio_5": [1.2, 1.5, 2.0],
            "min_close_position_20": [0.65, 0.75, 0.85],
            "max_pct_chg_1d": [0.06, 0.08, 0.095],
        },
    },
    "support_rebound": {
        "enabled": True,
        "max_combinations": 40,
        "base_params": {
            "require_above_ma20": True,
            "min_amount_ma5": 0,
        },
        "search_space": {
            "min_pct_chg_1d": [-0.095, -0.07],
            "max_pct_chg_1d": [-0.02, -0.025, -0.03],
            "min_close_position_20": [0.30, 0.35, 0.40],
            "max_close_position_20": [0.70, 0.75, 0.80],
        },
    },
}


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "parameter_search_space.json"


def load_parameter_search_space(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        return deepcopy(DEFAULT_PARAMETER_SEARCH_SPACE)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_param_combinations(strategy_name: str, strategy_config: dict) -> list[dict]:
    base_params = dict(strategy_config.get("base_params", {}))
    search_space = strategy_config.get("search_space", {}) or {}
    max_combinations = int(strategy_config.get("max_combinations", 0) or 0)

    keys = list(search_space.keys())
    value_lists = [list(search_space[key]) for key in keys]
    raw_combinations = product(*value_lists) if keys else [()]

    versions: list[dict] = []
    for index, values in enumerate(raw_combinations, start=1):
        params = dict(base_params)
        params.update(dict(zip(keys, values, strict=True)))
        versions.append(
            {
                "strategy_name": strategy_name,
                "strategy_version": f"search_{index:03d}",
                "enabled": True,
                "params": params,
            }
        )
        if max_combinations > 0 and len(versions) >= max_combinations:
            break

    return versions


def generate_search_versions(
    search_space_config: dict,
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
) -> list[dict]:
    versions: list[dict] = []
    strategy_count = 0
    for strategy_name, strategy_config in search_space_config.items():
        if not strategy_config.get("enabled", False):
            continue
        strategy_count += 1
        limited_config = deepcopy(strategy_config)
        if limit_param_combinations is not None:
            current_max = int(limited_config.get("max_combinations", 0) or 0)
            limited_max = max(0, int(limit_param_combinations))
            limited_config["max_combinations"] = limited_max if current_max <= 0 else min(current_max, limited_max)
        versions.extend(generate_param_combinations(strategy_name, limited_config))
        if limit_strategies is not None and strategy_count >= int(limit_strategies):
            break
    return versions
