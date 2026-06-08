import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_factor_diagnostics import run_build_factor_diagnostics


def test_run_build_factor_diagnostics_saves_result(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "close": [10.0],
                "turnover_rate": [1.2],
            }
        )
    )

    result = run_build_factor_diagnostics(db_path=store.db_path)
    loaded = store.load_factor_diagnostics()

    assert "turnover_rate" in set(result["factor_name"])
    assert len(loaded) == len(result)
