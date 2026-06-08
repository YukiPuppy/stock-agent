import json

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.evaluate_strategy_versions import (
    export_active_strategy_config,
    run_strategy_version_evaluation,
)


def _performance_df():
    return pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v2"],
            "sample_count": [40, 40],
            "valid_count": [40, 40],
            "win_rate_1d": [0.5, 0.4],
            "win_rate_3d": [0.6, 0.4],
            "win_rate_5d": [0.5, 0.4],
            "avg_return_1d": [0.01, -0.01],
            "avg_return_3d": [0.02, -0.01],
            "avg_return_5d": [0.01, -0.01],
            "median_return_1d": [0.01, -0.01],
            "median_return_3d": [0.01, -0.01],
            "median_return_5d": [0.01, -0.01],
            "avg_max_drawdown_1d": [-0.02, -0.02],
            "avg_max_drawdown_3d": [-0.03, -0.03],
            "avg_max_drawdown_5d": [-0.03, -0.03],
        }
    )


def test_run_strategy_version_evaluation_reads_performance_and_saves_evaluation(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_strategy_version_performance(_performance_df())

    evaluation = run_strategy_version_evaluation(db_path=str(db_path))
    saved = store.load_strategy_version_evaluation()

    assert len(evaluation) == 2
    assert len(saved) == 2
    assert saved.loc[0, "recommendation"] == "enable_observation"
    assert set(saved["recommendation"]) == {"enable_observation", "pause"}


def test_export_active_strategy_config_writes_json(tmp_path):
    output_path = tmp_path / "active_strategies.json"
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v2"],
            "recommendation": ["enable_observation", "pause"],
            "evaluation_score": [25.0, 10.0],
        }
    )

    config = export_active_strategy_config(evaluation, str(output_path))

    assert config["active_strategies"] == [
        {
            "strategy_name": "trend_pullback",
            "strategy_version": "v1",
            "recommendation": "enable_observation",
            "evaluation_score": 25.0,
        }
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == config
