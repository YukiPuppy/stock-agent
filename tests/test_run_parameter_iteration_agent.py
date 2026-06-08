import json
from pathlib import Path

from src.agents.llm_client import DisabledLLMClient
from src.config import settings
from src.pipeline import run_parameter_iteration_agent as pipeline


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return "# 参数迭代报告\n\n## 七、建议下一轮参数搜索空间\n候选。"


def test_pipeline_generates_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: DisabledLLMClient())

    summary = pipeline.run_parameter_iteration_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    report_path = Path(summary["parameter_iteration_report_path"])
    assert report_path.exists()
    assert "LLM 当前未启用或未配置" in report_path.read_text(encoding="utf-8")
    assert summary["requires_human_review"] is True


def test_pipeline_writes_parameter_search_space_candidate_json(tmp_path, monkeypatch):
    suggestions = tmp_path / "reports" / "strategy_research_suggestions_2026-01-01.json"
    suggestions.parent.mkdir()
    suggestions.write_text('{"parameter_search_suggestions": ["候选"]}', encoding="utf-8")
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: FakeLLMClient())

    summary = pipeline.run_parameter_iteration_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    candidate_path = Path(summary["parameter_search_space_candidate_path"])
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate_path.name == "parameter_search_space_candidate_2026-01-02.json"
    assert payload["requires_human_review"] is True
    assert payload["do_not_auto_apply"] is True
    assert summary["exported_candidate_json"] is True
