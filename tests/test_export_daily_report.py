from pathlib import Path

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_daily_report import export_daily_report


def _trade_plan(trade_date: str, code: str = "600000") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "code": [code],
            "name": ["浦发银行" if code == "600000" else "平安银行"],
            "rank": [1],
            "close": [10.5],
            "strategy_type": ["trend_pullback"],
            "action": ["回踩低吸"],
            "entry_low": [10.24],
            "entry_high": [10.45],
            "position_low": [0.1],
            "position_high": [0.2],
            "stop_loss": [9.97],
            "take_profit_1": [10.92],
            "take_profit_2": [11.34],
            "invalid_condition": ["计划失效条件"],
            "t_plus_1_risk": ["T+1 风险"],
            "plan_reason": ["趋势较强"],
        }
    )


def _candidate_pool(trade_date: str, code: str = "600000") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "code": [code],
            "name": ["浦发银行" if code == "600000" else "平安银行"],
            "close": [10.5],
            "pct_chg_1d": [0.01],
            "pct_chg_3d": [0.03],
            "pct_chg_5d": [0.05],
            "pct_chg_10d": [0.08],
            "volume_ratio_5": [1.3],
            "close_position_20": [0.8],
            "above_ma5": [True],
            "above_ma10": [True],
            "above_ma20": [True],
            "amount_ma5": [1000.0],
            "score": [88.0],
            "rank": [1],
            "reason": ["趋势评分靠前"],
        }
    )


def test_export_daily_report_writes_markdown_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_trade_plan(_trade_plan("2026-01-02"))
    store.save_candidate_pool(_candidate_pool("2026-01-02"))

    output_path = export_daily_report(db_path=str(db_path), output_dir=str(output_dir))

    path = Path(output_path)
    assert path.name == "daily_report_2026-01-02.md"
    assert path.exists()
    assert "A股日度交易计划报告" in path.read_text(encoding="utf-8")


def test_export_daily_report_filters_by_trade_date(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_trade_plan(pd.concat([_trade_plan("2026-01-02"), _trade_plan("2026-01-03", "000001")]))
    store.save_candidate_pool(pd.concat([_candidate_pool("2026-01-02"), _candidate_pool("2026-01-03", "000001")]))

    output_path = export_daily_report(
        trade_date="2026-01-03",
        db_path=str(db_path),
        output_dir=str(output_dir),
    )

    content = Path(output_path).read_text(encoding="utf-8")
    assert "000001" in content
    assert "平安银行" in content
    assert "600000" not in content
