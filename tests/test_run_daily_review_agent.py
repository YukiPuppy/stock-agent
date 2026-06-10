from pathlib import Path

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_daily_review_agent as pipeline


class FakeLLMClient:
    def __init__(self):
        self.prompts: list[str] = []
        self.generate_kwargs: list[dict] = []

    def generate(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        self.generate_kwargs.append(dict(kwargs))
        return "## 一、当日执行总体评价\nFake Daily Review"


def test_run_daily_review_agent_pipeline_writes_placeholder_when_llm_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    output_path = pipeline.run_daily_review_agent_pipeline(
        trade_date="2026-01-02",
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )

    assert output_path == str(tmp_path / "reports" / "llm_daily_review_2026-01-02.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert "LLM 当前未启用或未配置" in content
    assert "只做交易复盘和执行纪律分析" in content


def test_run_daily_review_agent_pipeline_uses_fake_llm_client(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    output_dir = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-03"],
                "trade_time": ["10:00:00", "10:00:00"],
                "code": ["600000", "000001"],
                "side": ["buy", "buy"],
                "price": [10.0, 20.0],
                "volume": [100, 100],
            }
        )
    )
    store.save_execution_review(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "side": ["buy"],
                "execution_status": ["off_plan"],
            }
        )
    )
    fake_client = FakeLLMClient()
    client_calls = []
    monkeypatch.setattr(
        pipeline,
        "get_llm_client",
        lambda *args, **kwargs: client_calls.append((args, kwargs)) or fake_client,
    )

    output_path = pipeline.run_daily_review_agent_pipeline(
        trade_date="2026-01-02",
        db_path=str(db_path),
        output_dir=str(output_dir),
        report_date="2026-01-04",
    )

    assert output_path == str(output_dir / "llm_daily_review_2026-01-04.md")
    content = Path(output_path).read_text(encoding="utf-8")
    assert content.startswith("## 一、当日执行总体评价")
    assert "Fake Daily Review" in content
    assert client_calls == [(("DailyReviewAgent",), {})]
    assert fake_client.prompts
    assert fake_client.generate_kwargs == [{}]
    assert "600000" in fake_client.prompts[0]
    assert "000001" not in fake_client.prompts[0]
    assert "off_plan" in fake_client.prompts[0]


def test_safe_llm_client_falls_back_to_disabled(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "get_llm_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    )

    assert pipeline._safe_llm_client().__class__.__name__ == "DisabledLLMClient"
