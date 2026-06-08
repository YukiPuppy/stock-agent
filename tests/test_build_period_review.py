import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_period_review import build_period_review


def test_build_period_review_reads_inputs_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10", "2025-01-13"],
                "trade_time": ["10:00:00", "10:30:00"],
                "code": ["600000", "000001"],
                "side": ["buy", "buy"],
                "price": [10.0, 20.0],
                "volume": [100, 100],
            }
        )
    )
    store.save_execution_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10", "2025-01-13"],
                "trade_time": ["10:00:00", "10:30:00"],
                "code": ["600000", "000001"],
                "side": ["buy", "buy"],
                "execution_status": ["follow_plan", "off_plan"],
                "execution_flags": ["price_in_range", "plan_not_found"],
            }
        )
    )
    store.save_actual_trade_performance(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10", "2025-01-13"],
                "trade_time": ["10:00:00", "10:30:00"],
                "code": ["600000", "000001"],
                "side": ["buy", "buy"],
                "entry_price": [10.0, 20.0],
                "entry_volume": [100, 100],
                "execution_status": ["follow_plan", "off_plan"],
                "execution_flags": ["price_in_range", "plan_not_found"],
                "return_1d": [0.01, -0.01],
                "return_3d": [0.03, -0.02],
                "return_5d": [0.05, -0.03],
                "is_valid": [True, True],
            }
        )
    )
    store.save_daily_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10", "2025-01-13"],
                "actual_trade_count": [1, 1],
                "buy_count": [1, 1],
                "sell_count": [0, 0],
                "planned_trade_count": [1, 1],
                "matched_plan_count": [1, 0],
                "off_plan_count": [0, 1],
                "follow_plan_count": [1, 0],
                "deviation_count": [0, 0],
                "chase_count": [0, 0],
                "over_position_count": [0, 0],
                "bought_watch_only_count": [0, 0],
                "execution_score": [100, 80],
                "main_issues": ["未发现明显执行偏差", "存在计划外交易"],
                "review_summary": ["执行良好", "存在偏差"],
                "next_action_suggestion": ["继续保持", "减少计划外交易"],
            }
        )
    )

    result = build_period_review(db_path=str(db_path))
    saved = store.load_period_review("2025-01-10", "2025-01-13")

    assert len(result) == 1
    assert result.loc[0, "start_date"] == "2025-01-10"
    assert result.loc[0, "end_date"] == "2025-01-13"
    assert result.loc[0, "actual_trade_count"] == 2
    assert len(saved) == 1
    assert saved.loc[0, "off_plan_count"] == 1
