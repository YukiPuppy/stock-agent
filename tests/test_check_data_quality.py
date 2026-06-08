import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.check_data_quality import run_data_quality_check


def test_run_data_quality_check_reads_and_saves_report(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000],
                "amount": [10200],
            }
        )
    )

    result = run_data_quality_check(db_path=store.db_path)
    saved = store.load_data_quality_report()

    assert not result.empty
    assert len(saved) == len(result)
    assert "enriched_daily_factors_missing_rate" in result["check_name"].tolist()
