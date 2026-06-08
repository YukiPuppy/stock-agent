import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.review_execution import run_execution_review


def test_run_execution_review_reads_local_tables_generates_and_saves_review(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["09:45:00"],
                "code": ["600000"],
                "side": ["buy"],
                "price": [10.5],
                "volume": [100],
                "position_ratio": [0.15],
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
                "position_low": [0.1],
                "position_high": [0.2],
            }
        )
    )

    result = run_execution_review(trade_date="2025-01-10", db_path=str(db_path))
    saved = store.load_execution_review(trade_date="2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "execution_status"] == "follow_plan"
    assert saved.loc[0, "execution_status"] == "follow_plan"
