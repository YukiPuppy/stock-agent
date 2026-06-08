import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_position_review_report import export_position_review_report


def test_export_position_review_report_writes_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_positions(
        pd.DataFrame(
            {
                "as_of_date": ["2025-01-10"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "holding_volume": [100],
                "available_volume": [100],
                "frozen_volume": [0],
                "cost_amount": [1000.0],
                "cost_price": [10.0],
                "latest_price": [10.5],
                "market_value": [1050.0],
                "floating_pnl": [50.0],
                "floating_pnl_pct": [0.05],
                "position_ratio": [0.1],
                "first_buy_date": ["2025-01-09"],
                "latest_trade_date": ["2025-01-09"],
                "strategy_name": ["trend"],
                "plan_rank": [1],
                "t_plus_1_status": ["sellable"],
                "position_status": ["profit_watch"],
            }
        )
    )
    store.save_position_review(
        pd.DataFrame(
            {
                "as_of_date": ["2025-01-10"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "holding_volume": [100],
                "available_volume": [100],
                "frozen_volume": [0],
                "cost_amount": [1000.0],
                "cost_price": [10.0],
                "latest_price": [10.5],
                "market_value": [1050.0],
                "floating_pnl": [50.0],
                "floating_pnl_pct": [0.05],
                "position_ratio": [0.1],
                "first_buy_date": ["2025-01-09"],
                "latest_trade_date": ["2025-01-09"],
                "strategy_name": ["trend"],
                "plan_rank": [1],
                "t_plus_1_status": ["sellable"],
                "position_status": ["profit_watch"],
                "planned_stop_loss": [9.0],
                "planned_take_profit_1": [10.5],
                "planned_take_profit_2": [12.0],
                "position_risk_level": ["low"],
                "position_flags": ["take_profit_zone"],
                "position_comment": ["当前价格进入第一止盈观察区间"],
                "next_action_hint": ["可关注止盈或移动止盈条件"],
            }
        )
    )

    path = export_position_review_report(
        as_of_date="2025-01-10",
        db_path=str(db_path),
        output_dir=str(output_dir),
    )

    assert path.endswith("position_review_2025-01-10.md")
    content = (output_dir / "position_review_2025-01-10.md").read_text(encoding="utf-8")
    assert "# A股持仓与T+1风险检查报告" in content
