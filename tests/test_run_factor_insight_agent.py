import pandas as pd

from src.agents.llm_client import DisabledLLMClient
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_factor_insight_agent as pipeline


def test_pipeline_generates_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "code": ["600000"],
                "close": [10.0],
                "turnover_rate": [1.0],
            }
        )
    )
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: DisabledLLMClient())

    path = pipeline.run_factor_insight_agent_pipeline(
        db_path=store.db_path,
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    content = (tmp_path / "reports" / "llm_factor_insight_2026-01-02.md").read_text(encoding="utf-8")
    assert path.endswith("llm_factor_insight_2026-01-02.md")
    assert "LLM 当前未启用或未配置" in content
    assert "FactorInsightAgent" in content
    assert not store.load_factor_diagnostics().empty
