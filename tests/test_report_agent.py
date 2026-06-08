import pandas as pd
import pytest

from src.agents.report_agent import (
    build_report_agent_context,
    build_report_agent_prompt,
    run_report_agent,
)


class FakeLLMClient:
    def __init__(self, response: str = "# LLM 总结\n\n## 系统状态概览\n正常。"):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_build_report_agent_context_handles_empty_dataframe():
    context = build_report_agent_context(data_quality=pd.DataFrame(columns=["status"]))

    assert context["data_quality"]["is_empty"] is True
    assert context["data_quality"]["columns"] == ["status"]
    assert context["data_quality"]["rows"] == []


def test_build_report_agent_context_limits_dataframe_rows_and_removes_sensitive_columns():
    df = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(25)],
            "api_key": ["secret"] * 25,
        }
    )

    context = build_report_agent_context(candidate_pool=df)

    assert context["candidate_pool"]["row_count"] == 25
    assert context["candidate_pool"]["included_rows"] == 20
    assert "api_key" not in context["candidate_pool"]["columns"]


def test_build_report_agent_prompt_contains_safety_constraints():
    prompt = build_report_agent_prompt({"trade_plan": {"rows": []}})

    assert "不得直接给出自动交易指令" in prompt
    assert "不得承诺收益" in prompt
    assert "不得建议绕过风控" in prompt
    assert "不得修改 active_strategies.json" in prompt
    assert "不得输出 API key、token" in prompt
    assert "系统状态、数据质量、策略研究、交易计划风险" in prompt
    assert "中文 Markdown" in prompt


def test_run_report_agent_uses_fake_client_and_returns_markdown():
    client = FakeLLMClient("# 报告\n\n## 主要风险\n需要人工复核。")

    result = run_report_agent(client, trade_plan=pd.DataFrame([{"code": "600000"}]))

    assert result.startswith("# 报告")
    assert client.prompts
    assert "600000" in client.prompts[0]


@pytest.mark.parametrize("term", ["保证盈利", "稳赚", "满仓", "自动下单"])
def test_run_report_agent_blocks_forbidden_terms(term):
    client = FakeLLMClient(f"# 报告\n\n{term}")

    with pytest.raises(ValueError, match=term):
        run_report_agent(client)
