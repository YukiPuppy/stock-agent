import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_period_review_report import export_period_review_report


def test_export_period_review_report_writes_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_period_review(
        pd.DataFrame(
            {
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "actual_trade_count": [1],
                "buy_count": [1],
                "sell_count": [0],
                "follow_plan_count": [1],
                "off_plan_count": [0],
                "deviation_count": [0],
                "chase_count": [0],
                "over_position_count": [0],
                "bought_watch_only_count": [0],
                "valid_performance_count": [1],
                "avg_return_3d": [0.02],
                "main_issues": ["未发现明显执行偏差。"],
                "period_summary": ["本周期执行较好。"],
                "next_period_suggestion": ["执行良好，建议继续保持，并等待更多样本评估策略有效性。"],
            }
        )
    )

    output_path = export_period_review_report(
        start_date="2025-01-01",
        end_date="2025-01-31",
        db_path=str(db_path),
        output_dir=str(output_dir),
    )

    assert output_path.endswith("period_review_2025-01-01_to_2025-01-31.md")
    content = (output_dir / "period_review_2025-01-01_to_2025-01-31.md").read_text(encoding="utf-8")
    assert "# A股周期执行复盘报告" in content
    for phrase in ["保证" + "盈利", "稳" + "赚", "满" + "仓"]:
        assert phrase not in content
