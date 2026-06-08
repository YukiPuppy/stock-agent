import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.export_strategy_admission_report import export_strategy_admission_report


def test_export_strategy_admission_report_writes_file(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_strategy_admission(
        pd.DataFrame(
            {
                "strategy_name": ["trend"],
                "strategy_version": ["v1"],
                "source": ["manual_version"],
                "valid_count": [40],
                "evaluation_recommendation": ["enable_observation"],
                "evaluation_score": [45.0],
                "oos_status": ["passed_oos"],
                "oos_risk": ["low"],
                "oos_stability_score": [25.0],
                "admission_score": [100.0],
                "admission_status": ["qualified_for_observation"],
                "admission_recommendation": ["enable_observation_candidate"],
                "admission_reason": ["满足观察候选条件。"],
            }
        )
    )

    path = export_strategy_admission_report(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    assert path == str(output_dir / "strategy_admission_2026-01-02.md")
    assert "# 策略准入与观察候选报告" in (output_dir / "strategy_admission_2026-01-02.md").read_text(
        encoding="utf-8"
    )
