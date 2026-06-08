import json

from src.strategy.strategy_versions import iter_strategy_versions, load_strategy_versions


def test_load_strategy_versions_reads_default_config():
    config = load_strategy_versions()

    assert "trend_pullback" in config
    assert config["trend_pullback"][0]["version"] == "v1"
    assert config["trend_pullback"][0]["params"]["min_pct_chg_5d"] == 0.03


def test_iter_strategy_versions_expands_enabled_versions():
    config = {
        "trend_pullback": [
            {"version": "v1", "enabled": True, "params": {"min_pct_chg_5d": 0.03}},
            {"version": "off", "enabled": False, "params": {"min_pct_chg_5d": 0.10}},
        ]
    }

    result = iter_strategy_versions(config)

    assert result == [
        {
            "strategy_name": "trend_pullback",
            "strategy_version": "v1",
            "enabled": True,
            "params": {"min_pct_chg_5d": 0.03},
        }
    ]


def test_load_strategy_versions_reads_custom_json(tmp_path):
    config_path = tmp_path / "strategy_versions.json"
    config_path.write_text(
        json.dumps(
            {
                "breakout_volume": [
                    {"version": "custom", "enabled": True, "params": {"min_volume_ratio_5": 2.0}}
                ]
            }
        ),
        encoding="utf-8",
    )

    config = load_strategy_versions(str(config_path))

    assert config["breakout_volume"][0]["version"] == "custom"
