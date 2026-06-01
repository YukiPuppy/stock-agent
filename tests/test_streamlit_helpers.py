import duckdb
import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.ui.streamlit_app import (
    get_latest_trade_date,
    list_report_files,
    safe_load_table,
)


def test_get_latest_trade_date_returns_none_for_empty_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()

    assert get_latest_trade_date(store) is None


def test_get_latest_trade_date_uses_candidate_pool_and_trade_plan(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_candidate_pool(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "close": [10.5],
                "pct_chg_5d": [0.07],
                "volume_ratio_5": [1.5],
                "close_position_20": [0.75],
                "score": [44.5],
                "rank": [1],
                "reason": ["趋势较强"],
            }
        )
    )
    store.save_trade_plan(
        pd.DataFrame(
            {
                "trade_date": ["20260103"],
                "code": ["000001"],
                "name": ["平安银行"],
                "rank": [1],
                "close": [20.5],
                "strategy_type": ["watch_only"],
                "action": ["仅观察"],
                "position_low": [0.0],
                "position_high": [0.0],
                "invalid_condition": ["等待新的量价确认。"],
                "t_plus_1_risk": ["T+1 风险"],
                "plan_reason": ["条件不足"],
            }
        )
    )

    assert get_latest_trade_date(store) == "20260103"


def test_list_report_files_finds_daily_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "daily_report_2026-01-02.md"
    report_2 = reports_dir / "daily_report_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "other.md").write_text("# other", encoding="utf-8")

    assert list_report_files(str(reports_dir)) == [str(report_1), str(report_2)]


def test_safe_load_table_returns_empty_dataframe_for_missing_table(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    with duckdb.connect(str(db_path)):
        pass

    store = StockAgentStore(str(db_path))
    result = safe_load_table(store, "candidate_pool")

    assert result.empty
