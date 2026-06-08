import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_parameter_search_report import export_parameter_search_report


def test_export_parameter_search_report_writes_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    reports_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_parameter_search_results(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "valid_count": [30],
                "win_rate_3d": [0.6],
                "avg_return_3d": [0.02],
                "median_return_3d": [0.01],
                "avg_max_drawdown_3d": [-0.03],
                "evaluation_score": [24.5],
                "evaluation_status": ["qualified"],
                "recommendation": ["enable_observation"],
            }
        )
    )
    store.save_parameter_search_performance(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "sample_count": [30],
                "valid_count": [30],
            }
        )
    )

    path = export_parameter_search_report(
        db_path=str(db_path),
        output_dir=str(reports_dir),
        report_date="2026-06-01",
    )

    content = (reports_dir / "parameter_search_2026-06-01.md").read_text(encoding="utf-8")
    assert path.endswith("parameter_search_2026-06-01.md")
    assert "# 策略参数搜索报告" in content
    assert "过拟合风险" in content
