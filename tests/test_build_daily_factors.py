import pandas as pd
import pytest

from src.database.duckdb_store import StockAgentStore
from src.factors.technical_factors import DAILY_FACTOR_COLUMNS
from src.pipeline.build_daily_factors import build_daily_factors


def test_build_daily_factors_reads_bars_computes_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260101", "20260102"],
                "code": ["600000", "600000"],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 11.0],
                "volume": [1000.0, 2000.0],
                "amount": [10000.0, 22000.0],
            }
        )
    )

    result = build_daily_factors(db_path=str(db_path))
    saved = store.load_daily_factors()

    assert len(result) == 2
    assert result.loc[1, "pct_chg_1d"] == pytest.approx(0.1)
    pd.testing.assert_frame_equal(saved, result)


def test_build_daily_factors_handles_empty_daily_bars(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.init_tables()

    result = build_daily_factors(db_path=str(db_path))
    saved = store.load_daily_factors()

    assert result.empty
    assert result.columns.tolist() == DAILY_FACTOR_COLUMNS
    assert saved.empty
