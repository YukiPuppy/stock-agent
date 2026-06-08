import pandas as pd
import pytest

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_trade_performance import build_trade_performance


def test_build_trade_performance_reads_inputs_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "price": [10.0],
                "volume": [100],
                "amount": [1000.0],
                "position_ratio": [0.1],
                "strategy_name": ["trend_pullback"],
                "plan_rank": [1],
            }
        )
    )
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16"],
                "code": ["600000"] * 5,
                "open": [10.0] * 5,
                "high": [10.5, 10.8, 11.3, 11.5, 11.8],
                "low": [9.8, 9.9, 10.0, 10.5, 10.8],
                "close": [10.2, 10.4, 10.6, 10.8, 11.0],
                "volume": [1000] * 5,
                "amount": [10000] * 5,
            }
        )
    )
    store.save_execution_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "side": ["buy"],
                "plan_match_status": ["matched"],
                "execution_status": ["follow_plan"],
                "execution_flags": ["price_in_range"],
            }
        )
    )

    result = build_trade_performance(trade_date="2025-01-10", db_path=str(db_path))
    saved = store.load_actual_trade_performance("2025-01-10")

    assert len(result) == 1
    assert len(saved) == 1
    assert saved.loc[0, "return_5d"] == pytest.approx(0.1)
    assert saved.loc[0, "execution_status"] == "follow_plan"
