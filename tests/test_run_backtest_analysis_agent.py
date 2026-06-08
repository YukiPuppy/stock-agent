from pathlib import Path

import pandas as pd

from src.agents.llm_client import DisabledLLMClient
from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_backtest_analysis_agent as pipeline


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        assert "BacktestAnalysisAgent" in prompt
        return "## 一、总体结论\nFake Markdown"


def test_run_backtest_analysis_agent_pipeline_writes_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    output_path = pipeline.run_backtest_analysis_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    assert output_path == str(tmp_path / "reports" / "llm_backtest_analysis_2026-01-02.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert "LLM 当前未启用" in content
    assert "未调用远程模型" in content


def test_run_backtest_analysis_agent_pipeline_uses_fake_llm_client(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_strategy_version_evaluation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "evaluation_score": [80.0],
                "evaluation_status": ["ready"],
                "risk_level": ["low"],
                "recommendation": ["continue_research"],
            }
        )
    )
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: FakeLLMClient())

    output_path = pipeline.run_backtest_analysis_agent_pipeline(
        db_path=str(db_path),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-03",
    )

    content = Path(output_path).read_text(encoding="utf-8")
    assert content == "## 一、总体结论\nFake Markdown"


def test_safe_llm_client_falls_back_to_disabled(monkeypatch):
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: (_ for _ in ()).throw(ValueError("bad config")))

    assert isinstance(pipeline._safe_llm_client(), DisabledLLMClient)
