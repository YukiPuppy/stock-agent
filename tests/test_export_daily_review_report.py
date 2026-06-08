from pathlib import Path

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_daily_review_report import export_daily_review_report


def test_export_daily_review_report_writes_markdown_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_daily_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "actual_trade_count": [1],
                "buy_count": [1],
                "sell_count": [0],
                "planned_trade_count": [1],
                "matched_plan_count": [1],
                "off_plan_count": [0],
                "follow_plan_count": [1],
                "deviation_count": [0],
                "chase_count": [0],
                "over_position_count": [0],
                "bought_watch_only_count": [0],
                "execution_score": [100],
                "main_issues": ["未发现明显执行偏差"],
                "review_summary": ["当日共记录 1 笔实际交易，整体执行与计划匹配度较好。"],
                "next_action_suggestion": ["执行良好，建议继续保持，后续结合收益结果评估策略有效性。"],
            }
        )
    )

    output_path = export_daily_review_report(
        trade_date="2025-01-10",
        db_path=str(db_path),
        output_dir=str(output_dir),
    )

    path = Path(output_path)
    content = path.read_text(encoding="utf-8")
    assert path.name == "daily_review_2025-01-10.md"
    assert "# A股盘后执行复盘报告" in content
    for phrase in ("保证" + "盈利", "稳" + "赚", "满" + "仓"):
        assert phrase not in content
