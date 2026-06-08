from src.strategy.strategy_config import (
    get_strategy_config,
    is_strategy_enabled,
    load_strategy_config,
)


def test_default_strategy_config_can_be_loaded():
    config = load_strategy_config()

    assert config["trend_pullback"]["enabled"] is True
    assert config["trend_pullback"]["version"] == "v1"
    assert config["breakout_volume"]["min_volume_ratio_5"] == 1.3
    assert config["support_rebound"]["min_amount_ma5"] == 0


def test_missing_strategy_config_file_uses_builtin_defaults(tmp_path):
    config = load_strategy_config(str(tmp_path / "missing.yaml"))

    assert config["trend_pullback"]["min_pct_chg_5d"] == 0.03
    assert get_strategy_config("support_rebound", config)["version"] == "v1"
    assert is_strategy_enabled("breakout_volume", config) is True


def test_partial_strategy_config_merges_with_defaults(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
trend_pullback:
  enabled: false
  version: test
  min_pct_chg_5d: 0.10
""",
        encoding="utf-8",
    )

    config = load_strategy_config(str(config_path))

    assert config["trend_pullback"]["enabled"] is False
    assert config["trend_pullback"]["version"] == "test"
    assert config["trend_pullback"]["max_pct_chg_1d"] == 0.06
    assert config["breakout_volume"]["version"] == "v1"
