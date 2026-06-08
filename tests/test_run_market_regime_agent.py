from pathlib import Path

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_market_regime_agent as pipeline


class FakeLLMClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "## 一、市场环境总体判断\nFake Market Regime"


def test_run_market_regime_agent_pipeline_writes_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    output_path = pipeline.run_market_regime_agent_pipeline(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    assert output_path == str(tmp_path / "reports" / "llm_market_regime_2026-01-02.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert "LLM 当前未启用或未配置" in content
    assert "只做市场环境解释，不做交易决策" in content


def test_run_market_regime_agent_pipeline_uses_fake_llm_client(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_market_regime(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-01-02",
                    "market_regime": "weak",
                    "risk_level": "high",
                    "regime_reason": "市场风险升高",
                }
            ]
        )
    )
    store.save_index_daily(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-01-02",
                    "index_code": "000001.SH",
                    "close": 3000.0,
                    "pct_chg": -1.2,
                }
            ]
        )
    )
    fake_client = FakeLLMClient()
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: fake_client)

    output_path = pipeline.run_market_regime_agent_pipeline(
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-02",
    )

    content = Path(output_path).read_text(encoding="utf-8")
    assert content.startswith("## 一、市场环境总体判断")
    assert fake_client.prompts
    assert "weak" in fake_client.prompts[0]
    assert "000001.SH" in fake_client.prompts[0]


def test_safe_llm_client_falls_back_to_disabled(monkeypatch):
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: (_ for _ in ()).throw(ValueError("bad config")))

    assert pipeline._safe_llm_client().__class__.__name__ == "DisabledLLMClient"
