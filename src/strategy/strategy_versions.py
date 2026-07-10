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
    "industry_rotation": [
        {
            "version": "v1_strength_follow",
            "enabled": True,
            "status": "research",
            "params": {
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
        },
        {
            "version": "v2_no_overheat",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "min_industry_strength_score": 62.0,
                "min_industry_return_3d": 0.0,
                "min_industry_return_5d": 0.012,
                "min_industry_amount_ratio_5": 0.9,
                "min_pct_chg_5d": 0.01,
                "min_close_position_20": 0.50,
                "min_moneyflow_score": 0.0,
                "max_pct_chg_3d": 0.07,
                "max_pct_chg_5d": 0.10,
                "max_close_position_20": 0.85,
                "min_relative_strength_5d": 0.015,
            },
        },
        {
            "version": "v3_industry_leader",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "min_industry_strength_score": 58.0,
                "min_industry_return_3d": 0.0,
                "min_industry_return_5d": 0.01,
                "min_industry_amount_ratio_5": 0.9,
                "min_pct_chg_5d": 0.02,
                "min_close_position_20": 0.55,
                "min_moneyflow_score": 3.0,
                "max_pct_chg_3d": 0.10,
                "max_pct_chg_5d": 0.14,
                "max_close_position_20": 0.90,
                "min_relative_strength_5d": 0.02,
            },
        },
    ],
    "moneyflow_accumulation": [
        {
            "version": "v1_main_inflow",
            "enabled": True,
            "status": "research",
            "params": {
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
        },
        {
            "version": "v2_big_order_confirm",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "min_moneyflow_score": 10.0,
                "min_main_net_amount_ratio": 1.0,
                "min_pct_chg_1d": -0.03,
                "max_pct_chg_1d": 0.055,
                "max_pct_chg_3d": 0.08,
                "min_pct_chg_5d": -0.04,
                "max_pct_chg_5d": 0.11,
                "max_turnover_rate": 8.0,
            },
        },
        {
            "version": "v3_price_not_chased",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "min_moneyflow_score": 12.0,
                "min_main_net_amount_ratio": 1.5,
                "min_pct_chg_1d": -0.025,
                "max_pct_chg_1d": 0.035,
                "max_pct_chg_3d": 0.055,
                "min_pct_chg_5d": -0.04,
                "max_pct_chg_5d": 0.07,
                "max_turnover_rate": 6.0,
            },
        },
    ],
    "low_vol_trend": [
        {
            "version": "v1_ma_alignment",
            "enabled": True,
            "status": "research",
            "params": {
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
        },
        {
            "version": "v2_low_chase",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "require_above_ma5": True,
                "require_above_ma10": True,
                "require_above_ma20": True,
                "min_pct_chg_5d": 0.005,
                "max_pct_chg_5d": 0.08,
                "max_abs_pct_chg_3d": 0.05,
                "max_abs_pct_chg_1d": 0.03,
                "max_pct_chg_1d": 0.025,
                "min_close_position_20": 0.42,
                "max_close_position_20": 0.82,
                "min_volume_ratio_5": 0.7,
                "max_volume_ratio_5": 1.6,
                "max_turnover_rate": 6.0,
            },
        },
        {
            "version": "v3_steady_volume",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.5,
                "require_above_ma5": True,
                "require_above_ma10": True,
                "require_above_ma20": True,
                "min_pct_chg_5d": 0.01,
                "max_pct_chg_5d": 0.09,
                "max_abs_pct_chg_3d": 0.05,
                "max_abs_pct_chg_1d": 0.03,
                "max_pct_chg_1d": 0.03,
                "min_close_position_20": 0.45,
                "max_close_position_20": 0.85,
                "min_volume_ratio_5": 0.8,
                "max_volume_ratio_5": 1.5,
                "max_turnover_rate": 5.0,
            },
        },
    ],
    "oversold_rebound": [
        {
            "version": "v1_mild_oversold",
            "enabled": True,
            "status": "research",
            "params": {
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
        },
        {
            "version": "v2_moneyflow_repair",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "min_pct_chg_3d": -0.10,
                "max_pct_chg_3d": -0.01,
                "min_pct_chg_5d": -0.15,
                "min_pct_chg_10d": -0.22,
                "min_close_position_20": 0.10,
                "max_close_position_20": 0.58,
                "min_moneyflow_score": 5.0,
                "min_industry_strength_score": 45.0,
                "max_down_pct_chg_3d": -0.10,
                "max_down_pct_chg_5d": -0.16,
                "max_down_pct_chg_10d": -0.25,
            },
        },
        {
            "version": "v3_industry_repair",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "min_pct_chg_3d": -0.10,
                "max_pct_chg_3d": -0.01,
                "min_pct_chg_5d": -0.15,
                "min_pct_chg_10d": -0.22,
                "min_close_position_20": 0.10,
                "max_close_position_20": 0.60,
                "min_moneyflow_score": 3.0,
                "min_industry_strength_score": 50.0,
                "max_down_pct_chg_3d": -0.10,
                "max_down_pct_chg_5d": -0.16,
                "max_down_pct_chg_10d": -0.25,
            },
        },
    ],
    "volume_dryup_breakout": [
        {
            "version": "v1_dryup_recover",
            "enabled": True,
            "status": "research",
            "params": {
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
        },
        {
            "version": "v2_breakout_confirm",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "require_above_ma5": True,
                "require_above_ma10": True,
                "min_volume_ratio_5": 1.0,
                "max_volume_ratio_5": 2.0,
                "min_volume_ratio_daily_basic": 1.1,
                "max_volume_ratio_daily_basic": 2.4,
                "min_pct_chg_1d": 0.015,
                "max_pct_chg_1d": 0.07,
                "max_pct_chg_3d": 0.11,
                "min_close_position_20": 0.65,
                "confirm_min_close_position_20": 0.72,
            },
        },
    ],
    "relative_strength_pullback": [
        {
            "version": "v1_rs_pullback",
            "enabled": True,
            "status": "research",
            "params": {
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
        },
        {
            "version": "v2_industry_strong_pullback",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "require_above_ma10": True,
                "require_above_ma20": True,
                "min_relative_strength_5d": 0.01,
                "min_pct_chg_10d": 0.02,
                "min_pct_chg_5d": -0.025,
                "max_pct_chg_5d": 0.065,
                "min_close_position_20": 0.32,
                "max_close_position_20": 0.82,
                "min_industry_strength_score": 55.0,
                "min_moneyflow_score": 5.0,
            },
        },
        {
            "version": "v3_moneyflow_confirm",
            "enabled": True,
            "status": "research",
            "params": {
                "min_amount_ma5": 10000.0,
                "min_turnover_rate": 0.3,
                "require_above_ma10": True,
                "require_above_ma20": True,
                "min_relative_strength_5d": 0.01,
                "min_pct_chg_10d": 0.02,
                "min_pct_chg_5d": -0.025,
                "max_pct_chg_5d": 0.065,
                "min_close_position_20": 0.32,
                "max_close_position_20": 0.82,
                "min_industry_strength_score": 50.0,
                "min_moneyflow_score": 8.0,
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
