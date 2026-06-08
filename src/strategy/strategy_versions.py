from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY_VERSIONS: dict[str, list[dict[str, Any]]] = {
    "trend_pullback": [
        {
            "version": "v1",
            "enabled": True,
            "params": {
                "min_pct_chg_5d": 0.03,
                "max_pct_chg_1d": 0.06,
                "min_close_position_20": 0.55,
                "min_volume_ratio_5": 1.0,
                "require_above_ma5": True,
                "require_above_ma10": True,
            },
        },
        {
            "version": "v2_conservative",
            "enabled": True,
            "params": {
                "min_pct_chg_5d": 0.05,
                "max_pct_chg_1d": 0.04,
                "min_close_position_20": 0.50,
                "min_volume_ratio_5": 1.0,
                "require_above_ma5": True,
                "require_above_ma10": True,
            },
        },
        {
            "version": "v3_loose",
            "enabled": True,
            "params": {
                "min_pct_chg_5d": 0.02,
                "max_pct_chg_1d": 0.08,
                "min_close_position_20": 0.45,
                "min_volume_ratio_5": 0.8,
                "require_above_ma5": True,
                "require_above_ma10": False,
            },
        },
    ],
    "breakout_volume": [
        {
            "version": "v1",
            "enabled": True,
            "params": {
                "min_pct_chg_5d": 0.05,
                "min_volume_ratio_5": 1.3,
                "min_close_position_20": 0.70,
                "max_pct_chg_1d": 0.095,
                "require_above_ma5": True,
            },
        },
        {
            "version": "v2_no_chase",
            "enabled": True,
            "params": {
                "min_pct_chg_5d": 0.04,
                "min_volume_ratio_5": 1.5,
                "min_close_position_20": 0.65,
                "max_pct_chg_1d": 0.06,
                "require_above_ma5": True,
            },
        },
    ],
    "support_rebound": [
        {
            "version": "v1",
            "enabled": True,
            "params": {
                "min_pct_chg_1d": -0.095,
                "max_pct_chg_1d": -0.02,
                "min_close_position_20": 0.35,
                "max_close_position_20": 0.75,
                "require_above_ma20": True,
                "min_amount_ma5": 0,
            },
        },
        {
            "version": "v2_strict_support",
            "enabled": True,
            "params": {
                "min_pct_chg_1d": -0.07,
                "max_pct_chg_1d": -0.025,
                "min_close_position_20": 0.40,
                "max_close_position_20": 0.70,
                "require_above_ma20": True,
                "min_amount_ma5": 0,
            },
        },
    ],
}


def load_strategy_versions(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        return deepcopy(DEFAULT_STRATEGY_VERSIONS)

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(DEFAULT_STRATEGY_VERSIONS)

    return parsed if isinstance(parsed, dict) else deepcopy(DEFAULT_STRATEGY_VERSIONS)


def iter_strategy_versions(config: dict) -> list[dict]:
    versions: list[dict] = []
    for strategy_name, strategy_versions in config.items():
        if not isinstance(strategy_versions, list):
            continue
        for item in strategy_versions:
            if not isinstance(item, dict):
                continue
            enabled = bool(item.get("enabled", True))
            if not enabled:
                continue
            params = item.get("params", {})
            versions.append(
                {
                    "strategy_name": str(strategy_name),
                    "strategy_version": str(item.get("version", "v1")),
                    "enabled": enabled,
                    "params": params.copy() if isinstance(params, dict) else {},
                }
            )
    return versions


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "strategy_versions.json"
