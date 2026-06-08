import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_walk_forward_validation_report import export_walk_forward_validation_report


def test_export_walk_forward_validation_report_writes_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_walk_forward_validation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "train_valid_count": [30],
                "train_win_rate_3d": [0.60],
                "train_avg_return_3d": [0.02],
                "train_avg_drawdown_3d": [-0.01],
                "validation_valid_count": [12],
                "validation_win_rate_3d": [0.58],
                "validation_avg_return_3d": [0.01],
                "validation_avg_drawdown_3d": [-0.02],
                "return_decay": [-0.01],
                "win_rate_decay": [-0.02],
                "drawdown_worsening": [-0.01],
                "stability_score": [20.0],
                "overfit_risk": ["low"],
                "validation_status": ["passed_oos"],
                "validation_reason": ["样本外表现基本稳定。"],
            }
        )
    )

    path = export_walk_forward_validation_report(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-06-01",
    )

    content = (output_dir / "walk_forward_validation_2026-06-01.md").read_text(encoding="utf-8")
    assert path == str(output_dir / "walk_forward_validation_2026-06-01.md")
    assert "# 策略样本外验证报告" in content
    for phrase in ["保证盈利", "稳赚", "满仓"]:
        assert phrase not in content
