import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.import_actual_trades import import_actual_trades


def test_import_actual_trades_reads_csv_normalizes_and_saves(tmp_path):
    csv_path = tmp_path / "actual_trades.csv"
    db_path = tmp_path / "stock_agent.duckdb"
    pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "code": ["1"],
            "side": ["BUY"],
            "price": [10.0],
            "volume": [100],
        }
    ).to_csv(csv_path, index=False)

    result = import_actual_trades(str(csv_path), db_path=str(db_path))
    saved = StockAgentStore(str(db_path)).load_actual_trades()

    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "amount"] == 1000.0
    assert saved.loc[0, "code"] == "000001"
    assert saved.loc[0, "amount"] == 1000.0
