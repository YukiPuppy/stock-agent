import json

from src.research.parameter_search import (
    generate_param_combinations,
    generate_search_versions,
    load_parameter_search_space,
)


def test_load_parameter_search_space_uses_default_when_file_missing(tmp_path):
    config = load_parameter_search_space(str(tmp_path / "missing.json"))

    assert "trend_pullback" in config
    assert config["trend_pullback"]["enabled"] is True


def test_generate_param_combinations_is_stable_and_merges_base_params():
    config = {
        "base_params": {"require_above_ma5": True},
        "search_space": {
            "min_pct_chg_5d": [0.02, 0.03],
            "max_pct_chg_1d": [0.04, 0.06],
        },
    }

    result = generate_param_combinations("trend_pullback", config)

    assert [item["strategy_version"] for item in result] == [
        "search_001",
        "search_002",
        "search_003",
        "search_004",
    ]
    assert result[0]["params"] == {
        "require_above_ma5": True,
        "min_pct_chg_5d": 0.02,
        "max_pct_chg_1d": 0.04,
    }
    assert result[1]["params"]["max_pct_chg_1d"] == 0.06
    assert result[2]["params"]["min_pct_chg_5d"] == 0.03


def test_generate_param_combinations_respects_max_combinations():
    config = {
        "max_combinations": 2,
        "search_space": {
            "a": [1, 2],
            "b": [3, 4],
        },
    }

    result = generate_param_combinations("trend_pullback", config)

    assert len(result) == 2
    assert result[-1]["strategy_version"] == "search_002"


def test_generate_search_versions_skips_disabled_and_combines_strategies():
    config = {
        "trend_pullback": {
            "enabled": True,
            "search_space": {"a": [1, 2]},
        },
        "breakout_volume": {
            "enabled": True,
            "search_space": {"b": [3]},
        },
        "support_rebound": {
            "enabled": False,
            "search_space": {"c": [4]},
        },
    }

    result = generate_search_versions(config)

    assert [item["strategy_name"] for item in result] == [
        "trend_pullback",
        "trend_pullback",
        "breakout_volume",
    ]


def test_load_parameter_search_space_reads_json(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(json.dumps({"trend_pullback": {"enabled": False}}), encoding="utf-8")

    assert load_parameter_search_space(str(path)) == {"trend_pullback": {"enabled": False}}
