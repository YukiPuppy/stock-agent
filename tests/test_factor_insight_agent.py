import pandas as pd

from src.agents.factor_insight_agent import (
    build_factor_insight_context,
    build_factor_insight_prompt,
    run_factor_insight_agent,
)


class FakeLLMClient:
    def __init__(self, text="# 因子诊断\n\n正常输出"):
        self.text = text
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.text


def test_build_factor_insight_context_handles_empty_dataframes():
    context = build_factor_insight_context(
        factor_diagnostics=pd.DataFrame(),
        daily_factors=pd.DataFrame(),
        candidate_pool=pd.DataFrame(),
        trade_plan=pd.DataFrame(),
    )

    assert context["factor_diagnostics"]["is_empty"] is True
    assert context["daily_factors"]["rows"] == []


def test_prompt_contains_required_constraints():
    prompt = build_factor_insight_prompt({})

    assert "当前诊断不等于因子有效性证明" in prompt
    assert "不得承诺收益" in prompt
    assert "自动下单" in prompt
    assert "不得直接修改策略参数" in prompt


def test_run_factor_insight_agent_uses_fake_llm_client():
    client = FakeLLMClient("# Markdown\n\n因子诊断")

    result = run_factor_insight_agent(client, factor_diagnostics=pd.DataFrame({"factor_name": ["turnover_rate"]}))

    assert result.startswith("# Markdown")
    assert "FactorInsightAgent" in client.prompt


def test_run_factor_insight_agent_neutralizes_forbidden_terms():
    client = FakeLLMClient("保证盈利，稳赚，满仓，自动下单")

    result = run_factor_insight_agent(client)

    assert "保证盈利" not in result
    assert "稳赚" not in result
    assert "满仓" not in result
    assert "自动下单" not in result
