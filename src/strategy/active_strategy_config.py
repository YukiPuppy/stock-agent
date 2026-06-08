from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


EMPTY_ACTIVE_STRATEGY_CANDIDATE_CONFIG = {
    "note": "No active strategy candidate config found.",
    "active_strategy_candidates": [],
}


def load_active_strategy_candidates(config_path: str = "configs/active_strategies_candidate.json") -> dict:
    try:
        path = Path(config_path)
        if not path.exists():
            return deepcopy(EMPTY_ACTIVE_STRATEGY_CANDIDATE_CONFIG)
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(EMPTY_ACTIVE_STRATEGY_CANDIDATE_CONFIG)

    if not isinstance(parsed, dict):
        return deepcopy(EMPTY_ACTIVE_STRATEGY_CANDIDATE_CONFIG)
    candidates = parsed.get("active_strategy_candidates", [])
    if not isinstance(candidates, list):
        parsed = parsed.copy()
        parsed["active_strategy_candidates"] = []
    return parsed


def get_active_strategy_version_set(config: dict) -> set[tuple[str, str]]:
    candidates = config.get("active_strategy_candidates", []) if isinstance(config, dict) else []
    if not isinstance(candidates, list):
        return set()

    result: set[tuple[str, str]] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        strategy_name = item.get("strategy_name")
        strategy_version = item.get("strategy_version")
        if strategy_name is None or strategy_version is None:
            continue
        result.add((str(strategy_name), str(strategy_version)))
    return result


def filter_versions_by_active_candidates(
    versions: list[dict],
    active_config: dict,
) -> list[dict]:
    active_versions = get_active_strategy_version_set(active_config)
    if not active_versions:
        return []

    filtered: list[dict[str, Any]] = []
    for version in versions:
        key = (str(version.get("strategy_name")), str(version.get("strategy_version")))
        if key in active_versions:
            item = version.copy()
            params = item.get("params", {})
            item["params"] = params.copy() if isinstance(params, dict) else {}
            filtered.append(item)
    return filtered
