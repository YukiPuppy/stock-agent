from pathlib import Path

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_strategy_evaluation_report import export_strategy_evaluation_report


def test_export_strategy_evaluation_report_writes_markdown_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_strategy_version_performance(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "sample_count": [50],
                "valid_count": [40],
                "win_rate_1d": [0.55],
                "win_rate_3d": [0.6],
                "win_rate_5d": [0.65],
                "avg_return_1d": [0.01],
                "avg_return_3d": [0.03],
                "avg_return_5d": [0.05],
                "median_return_1d": [0.01],
                "median_return_3d": [0.02],
                "median_return_5d": [0.04],
                "avg_max_drawdown_1d": [-0.01],
                "avg_max_drawdown_3d": [-0.02],
                "avg_max_drawdown_5d": [-0.03],
            }
        )
    )
    store.save_strategy_version_evaluation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "sample_count": [50],
                "valid_count": [40],
                "win_rate_3d": [0.6],
                "avg_return_3d": [0.03],
                "median_return_3d": [0.02],
                "avg_max_drawdown_3d": [-0.02],
                "evaluation_score": [0.8],
                "evaluation_status": ["qualified"],
                "risk_level": ["low"],
                "recommendation": ["enable_observation"],
                "evaluation_reason": ["满足观察条件。"],
            }
        )
    )

    output_path = export_strategy_evaluation_report(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    path = Path(output_path)
    assert path.name == "strategy_evaluation_2026-01-02.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# 策略版本评价报告" in content
    assert "trend_pullback" in content
