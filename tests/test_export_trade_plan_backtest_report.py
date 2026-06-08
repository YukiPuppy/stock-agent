import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_trade_plan_backtest_report import export_trade_plan_backtest_report


def test_export_trade_plan_backtest_report_writes_file(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_trade_plan_backtest_results(
        pd.DataFrame(
            {
                "plan_date": ["2026-01-01"],
                "code": ["600000"],
                "name": ["测试股"],
                "action": ["回踩低吸"],
                "strategy_names": ["trend"],
                "strategy_versions": ["v1"],
                "is_triggered": [True],
                "is_valid": [True],
                "return_pct": [0.1],
                "exit_reason": ["take_profit_1"],
            }
        )
    )
    store.save_trade_plan_backtest_performance(
        pd.DataFrame(
            {
                "strategy_names": ["trend"],
                "strategy_versions": ["v1"],
                "action": ["回踩低吸"],
                "plan_count": [1],
            }
        )
    )

    output_path = export_trade_plan_backtest_report(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    content = (tmp_path / "reports" / "trade_plan_backtest_2026-01-02.md").read_text(encoding="utf-8")
    assert output_path.endswith("trade_plan_backtest_2026-01-02.md")
    assert "# 交易计划规则回测报告" in content
