import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_trade_performance_report import export_trade_performance_report


def test_export_trade_performance_report_writes_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_actual_trade_performance(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "entry_price": [10.0],
                "entry_volume": [100],
                "entry_amount": [1000.0],
                "position_ratio": [0.1],
                "strategy_name": ["trend_pullback"],
                "plan_rank": [1],
                "plan_match_status": ["matched"],
                "execution_status": ["follow_plan"],
                "execution_flags": ["price_in_range"],
                "return_1d": [0.02],
                "return_3d": [0.04],
                "return_5d": [0.06],
                "max_drawdown_1d": [-0.01],
                "max_drawdown_3d": [-0.02],
                "max_drawdown_5d": [-0.03],
                "max_favorable_1d": [0.03],
                "max_favorable_3d": [0.05],
                "max_favorable_5d": [0.08],
                "is_valid": [True],
                "invalid_reason": [""],
                "performance_comment": ["短期表现较稳"],
            }
        )
    )

    output_path = export_trade_performance_report(
        trade_date="2025-01-10",
        db_path=str(db_path),
        output_dir=str(output_dir),
    )

    assert output_path.endswith("trade_performance_2025-01-10.md")
    content = (output_dir / "trade_performance_2025-01-10.md").read_text(encoding="utf-8")
    assert "# A股实盘交易表现复盘报告" in content
    assert "600000" in content
