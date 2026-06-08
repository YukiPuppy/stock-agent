import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_positions import build_positions


def test_build_positions_reads_duckdb_and_saves_results(tmp_path):
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
                "position_ratio": [0.3],
                "strategy_name": ["trend"],
                "plan_rank": [1],
            }
        )
    )
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "code": ["600000"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [9.0],
                "volume": [1000],
                "amount": [9000],
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
                "stop_loss": [9.5],
                "take_profit_1": [11.0],
                "take_profit_2": [12.0],
            }
        )
    )

    positions, review = build_positions(as_of_date="2025-01-10", db_path=str(db_path))

    assert len(positions) == 1
    assert len(review) == 1
    assert len(store.load_positions(as_of_date="2025-01-10")) == 1
    assert "below_stop_loss" in store.load_position_review(as_of_date="2025-01-10").loc[0, "position_flags"]
