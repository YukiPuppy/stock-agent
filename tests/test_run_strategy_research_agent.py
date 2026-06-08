import json
from pathlib import Path

from src.config import settings
from src.pipeline.run_strategy_research_agent import run_strategy_research_agent_pipeline


def test_pipeline_generates_placeholder_report_when_llm_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)

    summary = run_strategy_research_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    report_path = Path(summary["strategy_research_report_path"])
    assert report_path.exists()
    assert report_path.name == "llm_strategy_research_2026-01-02.md"
    assert "LLM 当前未启用或未配置" in report_path.read_text(encoding="utf-8")
    assert summary["requires_human_review"] is True


def test_pipeline_writes_strategy_research_suggestions_json(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)

    summary = run_strategy_research_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    suggestions_path = Path(summary["strategy_research_suggestions_path"])
    assert suggestions_path.exists()
    assert suggestions_path.name == "strategy_research_suggestions_2026-01-02.json"
    payload = json.loads(suggestions_path.read_text(encoding="utf-8"))
    assert payload["requires_human_review"] is True
    assert payload["strategy_hypotheses"] == []
