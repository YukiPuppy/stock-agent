import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_industry_strength import build_and_save_industry_strength


def test_build_industry_strength_pipeline_saves_result(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    rows = [
        {
            "trade_date": f"2025-01-{index + 1:02d}",
            "industry_code": "801780.SI",
            "industry_name": "银行",
            "close": 100 + index,
            "pct_change": 1.0,
            "amount": 1000 + index,
        }
        for index in range(6)
    ]
    store.save_sw_daily(pd.DataFrame(rows))

    result, sw_daily_count, _ = build_and_save_industry_strength(str(db_path))

    assert sw_daily_count == 6
    assert len(result) == 6
    assert store.load_industry_strength().loc[0, "industry_code"] == "801780.SI"
