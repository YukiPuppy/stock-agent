import json

from src.strategy.active_strategy_config import (
    filter_versions_by_active_candidates,
    get_active_strategy_version_set,
    load_active_strategy_candidates,
)


def test_load_active_strategy_candidates_missing_file_returns_empty_config(tmp_path):
    config = load_active_strategy_candidates(str(tmp_path / "missing.json"))

    assert config == {
        "note": "No active strategy candidate config found.",
        "active_strategy_candidates": [],
    }


def test_get_active_strategy_version_set_extracts_versions():
    config = {
        "active_strategy_candidates": [
            {"strategy_name": "trend_pullback", "strategy_version": "search_001"},
            {"strategy_name": "support_rebound", "strategy_version": "v1"},
            {"strategy_name": "missing_version"},
        ]
    }

    assert get_active_strategy_version_set(config) == {
        ("trend_pullback", "search_001"),
        ("support_rebound", "v1"),
    }


def test_filter_versions_by_active_candidates_filters_without_mutating_source(tmp_path):
    active_path = tmp_path / "active.json"
    active_path.write_text(
        json.dumps(
            {
                "active_strategy_candidates": [
                    {"strategy_name": "trend_pullback", "strategy_version": "v1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    versions = [
        {"strategy_name": "trend_pullback", "strategy_version": "v1", "enabled": True, "params": {"x": 1}},
        {"strategy_name": "support_rebound", "strategy_version": "v1", "enabled": True, "params": {}},
    ]

    result = filter_versions_by_active_candidates(versions, load_active_strategy_candidates(str(active_path)))
    result[0]["params"]["x"] = 2

    assert len(result) == 1
    assert result[0]["strategy_name"] == "trend_pullback"
    assert versions[0]["params"]["x"] == 1


def test_filter_versions_by_active_candidates_empty_candidates_returns_empty():
    versions = [{"strategy_name": "trend_pullback", "strategy_version": "v1", "enabled": True, "params": {}}]

    assert filter_versions_by_active_candidates(versions, {"active_strategy_candidates": []}) == []
