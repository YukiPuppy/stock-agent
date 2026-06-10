from pathlib import Path

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_risk_review_agent as pipeline


class FakeLLMClient:
    def __init__(self):
        self.prompts = []
        self.generate_kwargs = []

    def generate(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        self.generate_kwargs.append(dict(kwargs))
        return "## 一、总体风险结论\n仅供人工复核参考。"


def test_run_risk_review_agent_pipeline_writes_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_data_quality_report(
        pd.DataFrame([{"check_name": "daily_bars", "status": "ok", "issue_count": 0, "message": "ok"}])
    )
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    output_path = pipeline.run_risk_review_agent_pipeline(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    assert output_path == str(output_dir / "llm_risk_review_2026-01-02.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert "LLM 当前未启用或未配置" in content


def test_run_risk_review_agent_pipeline_uses_fake_llm_client(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_data_quality_report(
        pd.DataFrame([{"check_name": "daily_bars", "status": "warning", "issue_count": 1, "message": "missing"}])
    )
    fake_client = FakeLLMClient()
    client_calls = []
    monkeypatch.setattr(
        pipeline,
        "get_llm_client",
        lambda *args, **kwargs: client_calls.append((args, kwargs)) or fake_client,
    )

    output_path = pipeline.run_risk_review_agent_pipeline(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    content = Path(output_path).read_text(encoding="utf-8")
    assert content.startswith("## 一、总体风险结论")
    assert "仅供人工复核参考" in content
    assert client_calls == [(("RiskReviewAgent",), {})]
    assert fake_client.prompts
    assert fake_client.generate_kwargs == [{}]
    assert "warning" in fake_client.prompts[0]
