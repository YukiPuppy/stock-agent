import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_trade_plan import build_trade_plan
from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02", "2026-01-02"],
            "code": ["600000", "600000", "000001"],
            "name": ["旧日期", "趋势股", "观察股"],
            "close": [9.0, 10.0, 20.0],
            "pct_chg_1d": [0.01, 0.01, 0.01],
            "pct_chg_3d": [0.02, 0.02, 0.02],
            "pct_chg_5d": [0.03, 0.03, 0.03],
            "pct_chg_10d": [0.04, 0.04, 0.04],
            "volume_ratio_5": [1.0, 1.0, 1.0],
            "close_position_20": [0.6, 0.6, 0.4],
            "above_ma5": [True, True, False],
            "above_ma10": [True, True, False],
            "above_ma20": [True, True, False],
            "amount_ma5": [200000000.0, 200000000.0, 200000000.0],
            "score": [80.0, 90.0, 70.0],
            "rank": [1, 1, 2],
            "reason": ["旧日期", "趋势较强", "条件不足"],
        }
    )


def test_build_trade_plan_reads_candidate_pool_generates_and_saves_latest_date(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_candidate_pool(_candidate_rows())

    result = build_trade_plan(max_items=1, db_path=str(db_path))
    saved = store.load_trade_plan(trade_date="2026-01-02")

    assert len(result) == 1
    assert result.loc[0, "trade_date"] == "2026-01-02"
    assert result.loc[0, "code"] == "600000"
    assert result.loc[0, "strategy_type"] == "trend_pullback"
    pd.testing.assert_frame_equal(saved, result)


def test_build_trade_plan_handles_empty_candidate_pool(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.init_tables()

    result = build_trade_plan(db_path=str(db_path))
    saved = store.load_trade_plan()

    assert result.empty
    assert result.columns.tolist() == TRADE_PLAN_COLUMNS
    assert saved.empty
