from pathlib import Path

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline.run_report_agent import run_report_agent_pipeline


def test_run_report_agent_pipeline_writes_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_data_quality_report(
        pd.DataFrame([{"check_name": "daily_bars", "status": "ok", "issue_count": 0, "message": "ok"}])
    )
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    output_path = run_report_agent_pipeline(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    assert output_path == str(output_dir / "llm_report_summary_2026-01-02.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert "LLM is disabled" in content
