import json

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_strategy_admission import run_strategy_admission


def test_run_strategy_admission_reads_duckdb_saves_and_exports_candidate_config(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    config_path = tmp_path / "configs" / "active_strategies_candidate.json"
    store = StockAgentStore(str(db_path))
    store.save_strategy_version_evaluation(
        pd.DataFrame(
            {
                "strategy_name": ["trend"],
                "strategy_version": ["v1"],
                "valid_count": [40],
                "evaluation_score": [45.0],
                "recommendation": ["enable_observation"],
            }
        )
    )
    store.save_parameter_search_results(pd.DataFrame())
    store.save_walk_forward_validation(
        pd.DataFrame(
            {
                "strategy_name": ["trend"],
                "strategy_version": ["v1"],
                "validation_status": ["passed_oos"],
                "overfit_risk": ["low"],
                "stability_score": [25.0],
            }
        )
    )
    store.save_trade_plan_backtest_performance(
        pd.DataFrame(
            {
                "strategy_names": ["trend"],
                "strategy_versions": ["v1"],
                "action": ["低吸"],
                "plan_count": [20],
                "triggered_count": [8],
                "valid_count": [12],
                "trigger_rate": [0.4],
                "win_rate": [0.55],
                "avg_return": [0.02],
                "avg_max_drawdown": [-0.03],
            }
        )
    )

    admission = run_strategy_admission(
        db_path=str(db_path),
        export_candidate_config=True,
        candidate_config_path=str(config_path),
    )
    saved = store.load_strategy_admission()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert len(admission) == 1
    assert len(saved) == 1
    assert saved.loc[0, "admission_recommendation"] == "enable_observation_candidate"
    assert config["active_strategy_candidates"][0]["strategy_name"] == "trend"
