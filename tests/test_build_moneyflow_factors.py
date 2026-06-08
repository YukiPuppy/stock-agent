import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_moneyflow_factors import build_and_save_moneyflow_factors


def test_build_moneyflow_factors_pipeline_reads_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_moneyflow(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "buy_sm_amount": [10.0],
                "sell_sm_amount": [5.0],
                "buy_md_amount": [20.0],
                "buy_lg_amount": [100.0],
                "sell_lg_amount": [20.0],
                "buy_elg_amount": [50.0],
                "sell_elg_amount": [10.0],
                "net_mf_amount": [125.0],
            }
        )
    )

    result = build_and_save_moneyflow_factors(db_path=str(db_path))
    saved = store.load_moneyflow_factors()

    assert result.loc[0, "main_net_amount"] == 120.0
    assert saved.loc[0, "moneyflow_score"] > 0
