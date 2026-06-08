import pandas as pd

from src.agents.strategy_research_agent import (
    build_strategy_research_context,
    build_strategy_research_prompt,
    run_strategy_research_agent,
)


class FakeLLMClient:
    def __init__(self, text="# 策略研究\n\n普通建议"):
        self.text = text
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.text


def test_build_strategy_research_context_handles_empty_dataframe():
    context = build_strategy_research_context(strategy_evaluation=pd.DataFrame())

    assert context["strategy_evaluation"]["is_empty"] is True
    assert context["strategy_evaluation"]["rows"] == []


def test_context_prioritizes_strategy_admission_statuses():
    admission = pd.DataFrame(
        [
            {"strategy_version": "v_other", "admission_status": "enabled"},
            {"strategy_version": "v_continue", "admission_status": "continue_research"},
            {"strategy_version": "v_observe", "admission_status": "observation_candidate"},
            {"strategy_version": "v_reject", "admission_status": "do_not_enable"},
        ]
    )

    context = build_strategy_research_context(strategy_admission=admission)
    versions = [row["strategy_version"] for row in context["strategy_admission"]["rows"]]

    assert versions[:3] == ["v_reject", "v_continue", "v_observe"]


def test_strategy_research_prompt_contains_scope_and_validation_constraints():
    prompt = build_strategy_research_prompt(build_strategy_research_context())

    assert "只做策略研究建议，不做交易决策" in prompt
    assert "所有建议必须经过回测、样本外验证、交易计划级回测" in prompt
    assert "不得承诺收益" in prompt
    assert "自动下单" in prompt
    assert "不得直接修改 active_strategies.json" in prompt
    assert "不得直接修改 parameter_search_space.json" in prompt


def test_run_strategy_research_agent_uses_fake_llm_client():
    client = FakeLLMClient("# 一、当前策略研究状态判断\n\n候选建议")

    markdown = run_strategy_research_agent(client)

    assert "候选建议" in markdown
    assert "StrategyResearchAgent" in client.prompt


def test_run_strategy_research_agent_neutralizes_forbidden_terms():
    client = FakeLLMClient("保证盈利，稳赚，满仓，自动下单")

    markdown = run_strategy_research_agent(client)

    assert "保证盈利" not in markdown
    assert "稳赚" not in markdown
    assert "满仓" not in markdown
    assert "自动下单" not in markdown
    assert "收益存在不确定性" in markdown
