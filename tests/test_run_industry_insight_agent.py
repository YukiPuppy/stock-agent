from pathlib import Path

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_industry_insight_agent as pipeline


class FakeLLMClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "## 一、行业强弱总体判断\nFake Industry Insight"


def test_run_industry_insight_agent_pipeline_writes_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    output_path = pipeline.run_industry_insight_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    assert output_path == str(tmp_path / "reports" / "llm_industry_insight_2026-01-02.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert "LLM 当前未启用或未配置" in content
    assert "只做行业强弱解释，不做交易决策" in content


def test_run_industry_insight_agent_pipeline_uses_fake_llm_client(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_industry_strength(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-01-02",
                    "industry_code": "801780.SI",
                    "industry_name": "银行",
                    "industry_strength_level": "strong",
                    "industry_strength_score": 88.0,
                }
            ]
        )
    )
    store.save_candidate_pool(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-01-02",
                    "code": "000001",
                    "name": "平安银行",
                    "risk_flags": "strong_industry",
                }
            ]
        )
    )
    fake_client = FakeLLMClient()
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: fake_client)

    output_path = pipeline.run_industry_insight_agent_pipeline(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    content = Path(output_path).read_text(encoding="utf-8")
    assert content.startswith("## 一、行业强弱总体判断")
    assert fake_client.prompts
    assert "801780.SI" in fake_client.prompts[0]
    assert "strong_industry" in fake_client.prompts[0]


def test_safe_llm_client_falls_back_to_disabled(monkeypatch):
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: (_ for _ in ()).throw(ValueError("bad config")))

    assert pipeline._safe_llm_client().__class__.__name__ == "DisabledLLMClient"
