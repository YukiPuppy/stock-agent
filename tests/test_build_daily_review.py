import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_daily_review import build_daily_review


def test_build_daily_review_reads_inputs_and_saves_result(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["09:45:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "price": [10.5],
                "volume": [100],
                "position_ratio": [0.1],
            }
        )
    )
    store.save_trade_plan(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "rank": [1],
                "action": ["回踩低吸"],
                "entry_low": [10.0],
                "entry_high": [11.0],
                "position_low": [0.05],
                "position_high": [0.2],
            }
        )
    )
    store.save_execution_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["09:45:00"],
                "code": ["600000"],
                "side": ["buy"],
                "plan_match_status": ["matched"],
                "execution_status": ["follow_plan"],
                "execution_flags": ["price_in_range"],
            }
        )
    )

    result = build_daily_review(trade_date="2025-01-10", db_path=str(db_path))
    saved = store.load_daily_review("2025-01-10")

    assert result.loc[0, "trade_date"] == "2025-01-10"
    assert result.loc[0, "follow_plan_count"] == 1
    assert saved.loc[0, "execution_score"] == 100
