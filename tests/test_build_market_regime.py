import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_market_regime import build_market_regime


def test_build_market_regime_pipeline_reads_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_index_daily(
        pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=5).strftime("%Y-%m-%d"),
                "index_code": ["000001.SH"] * 5,
                "close": [3000, 3010, 3020, 3030, 3040],
                "pct_chg": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
    )
    store.save_limit_list_daily(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-05"],
                "code": ["000001"],
                "limit_type": ["U"],
                "open_times": [0],
            }
        )
    )

    result = build_market_regime(db_path=str(db_path))
    saved = store.load_market_regime()

    assert len(result) == 5
    assert len(saved) == 5
    assert saved.iloc[-1]["trade_date"] == "2025-01-05"
