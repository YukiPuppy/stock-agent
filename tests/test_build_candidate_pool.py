import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_candidate_pool import build_candidate_pool
from src.strategy.candidate_selector import CANDIDATE_COLUMNS


def _factor_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "000001"],
            "close": [11.0, 20.0],
            "pct_chg_1d": [0.02, 0.03],
            "pct_chg_3d": [0.03, 0.01],
            "pct_chg_5d": [0.05, 0.20],
            "pct_chg_10d": [0.06, 0.02],
            "ma5": [10.0, 19.0],
            "ma10": [10.0, 21.0],
            "ma20": [10.0, 21.0],
            "volume_ma5": [1000.0, 2000.0],
            "amount_ma5": [200000000.0, 200000000.0],
            "volume_ratio_5": [2.0, 4.0],
            "high_20": [12.0, 21.0],
            "low_20": [8.0, 16.0],
            "close_position_20": [0.8, 0.6],
            "above_ma5": [True, True],
            "above_ma10": [True, False],
            "above_ma20": [True, False],
        }
    )


def test_build_candidate_pool_reads_generates_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(_factor_rows())
    store.save_stock_basic(
        pd.DataFrame(
            {
                "code": ["600000", "000001"],
                "name": ["浦发银行", "平安银行"],
                "market": ["SH", "SZ"],
                "board": ["main", "main"],
                "list_status": ["L", "L"],
            }
        )
    )

    result = build_candidate_pool(
        trade_date="2026-01-02",
        top_n=1,
        min_amount_ma5=100000000.0,
        db_path=str(db_path),
    )
    saved = store.load_candidate_pool(trade_date="2026-01-02")

    assert len(result) == 1
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "name"] == "平安银行"
    pd.testing.assert_frame_equal(saved, result)


def test_build_candidate_pool_handles_empty_daily_factors(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.init_tables()

    result = build_candidate_pool(db_path=str(db_path))
    saved = store.load_candidate_pool()

    assert result.empty
    assert result.columns.tolist() == CANDIDATE_COLUMNS
    assert saved.empty
