import json

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_strategy_signals import build_strategy_signals
from src.strategy.base_strategy import SIGNAL_COLUMNS


def test_build_strategy_signals_reads_daily_factors_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "close": [11.0],
                "pct_chg_1d": [0.02],
                "pct_chg_3d": [0.03],
                "pct_chg_5d": [0.06],
                "pct_chg_10d": [0.08],
                "ma5": [10.0],
                "ma10": [10.0],
                "ma20": [10.0],
                "volume_ma5": [1000.0],
                "amount_ma5": [200000000.0],
                "volume_ratio_5": [1.5],
                "high_20": [12.0],
                "low_20": [8.0],
                "close_position_20": [0.75],
                "above_ma5": [True],
                "above_ma10": [True],
                "above_ma20": [True],
            }
        )
    )

    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
trend_pullback:
  enabled: true
  version: custom
breakout_volume:
  enabled: false
support_rebound:
  enabled: false
""",
        encoding="utf-8",
    )

    result = build_strategy_signals(
        trade_date="2026-01-02",
        db_path=str(db_path),
        config_path=str(config_path),
    )
    saved = store.load_strategy_signals(trade_date="2026-01-02")

    assert not result.empty
    assert "trend_pullback" in result["strategy_name"].tolist()
    assert result["strategy_version"].tolist() == ["custom"]
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), result.reset_index(drop=True))


def test_build_strategy_signals_active_candidates_empty_returns_standard_empty_dataframe(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "close": [11.0],
            }
        )
    )
    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps({"active_strategy_candidates": []}), encoding="utf-8")

    result = build_strategy_signals(
        trade_date="2026-01-02",
        db_path=str(db_path),
        use_active_candidates=True,
        active_config_path=str(active_path),
    )

    assert result.empty
    assert result.columns.tolist() == SIGNAL_COLUMNS


def test_build_strategy_signals_use_active_candidates_only_generates_matching_versions(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-02"],
                "code": ["600000", "000001"],
                "close": [11.0, 19.5],
                "pct_chg_1d": [0.02, -0.04],
                "pct_chg_3d": [0.03, -0.05],
                "pct_chg_5d": [0.06, 0.01],
                "pct_chg_10d": [0.08, 0.01],
                "ma5": [10.0, 19.8],
                "ma10": [10.0, 19.9],
                "ma20": [10.0, 19.0],
                "volume_ma5": [1000.0, 1000.0],
                "amount_ma5": [200000000.0, 200000000.0],
                "volume_ratio_5": [1.5, 1.2],
                "high_20": [12.0, 22.0],
                "low_20": [8.0, 18.0],
                "close_position_20": [0.75, 0.38],
                "above_ma5": [True, False],
                "above_ma10": [True, False],
                "above_ma20": [True, True],
            }
        )
    )
    versions_path = tmp_path / "versions.json"
    versions_path.write_text(
        json.dumps(
            {
                "trend_pullback": [
                    {
                        "version": "candidate",
                        "enabled": True,
                        "params": {
                            "min_pct_chg_5d": 0.03,
                            "max_pct_chg_1d": 0.06,
                            "min_close_position_20": 0.55,
                            "min_volume_ratio_5": 1.0,
                            "require_above_ma5": True,
                            "require_above_ma10": True,
                        },
                    }
                ],
                "support_rebound": [
                    {
                        "version": "not_candidate",
                        "enabled": True,
                        "params": {
                            "min_pct_chg_1d": -0.095,
                            "max_pct_chg_1d": -0.02,
                            "min_close_position_20": 0.35,
                            "max_close_position_20": 0.75,
                            "require_above_ma20": True,
                            "min_amount_ma5": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    active_path = tmp_path / "active.json"
    active_path.write_text(
        json.dumps(
            {
                "active_strategy_candidates": [
                    {"strategy_name": "trend_pullback", "strategy_version": "candidate"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_strategy_signals(
        trade_date="2026-01-02",
        db_path=str(db_path),
        use_active_candidates=True,
        active_config_path=str(active_path),
        versions_config_path=str(versions_path),
    )

    assert not result.empty
    assert set(result["strategy_name"]) == {"trend_pullback"}
    assert set(result["strategy_version"]) == {"candidate"}


def test_build_strategy_signals_default_ignores_active_candidates(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "close": [11.0],
                "pct_chg_1d": [0.02],
                "pct_chg_3d": [0.03],
                "pct_chg_5d": [0.06],
                "pct_chg_10d": [0.08],
                "ma5": [10.0],
                "ma10": [10.0],
                "ma20": [10.0],
                "volume_ma5": [1000.0],
                "amount_ma5": [200000000.0],
                "volume_ratio_5": [1.5],
                "high_20": [12.0],
                "low_20": [8.0],
                "close_position_20": [0.75],
                "above_ma5": [True],
                "above_ma10": [True],
                "above_ma20": [True],
            }
        )
    )
    active_path = tmp_path / "active.json"
    active_path.write_text(json.dumps({"active_strategy_candidates": []}), encoding="utf-8")

    result = build_strategy_signals(
        trade_date="2026-01-02",
        db_path=str(db_path),
        active_config_path=str(active_path),
    )

    assert not result.empty
