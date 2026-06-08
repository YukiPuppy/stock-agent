import pandas as pd

from src.agents.parameter_iteration_agent import (
    build_parameter_iteration_context,
    build_parameter_iteration_prompt,
    run_parameter_iteration_agent,
)


class FakeLLMClient:
    def __init__(self, response: str = "# 参数迭代\n\n仅输出候选建议。"):
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_build_parameter_iteration_context_handles_empty_dataframe():
    context = build_parameter_iteration_context(
        strategy_evaluation=pd.DataFrame(),
        parameter_search_results=pd.DataFrame(),
        walk_forward_validation=pd.DataFrame(),
    )

    assert context["strategy_evaluation"]["is_empty"] is True
    assert context["parameter_search_results"]["rows"] == []
    assert context["walk_forward_validation"]["row_count"] == 0


def test_build_parameter_iteration_context_includes_strategy_research_suggestions():
    context = build_parameter_iteration_context(
        strategy_research_suggestions={
            "parameter_search_suggestions": [{"name": "lookback_window"}],
            "LLM_API_KEY": "secret",
        }
    )

    assert context["strategy_research_suggestions"]["parameter_search_suggestions"] == [{"name": "lookback_window"}]
    assert "LLM_API_KEY" not in context["strategy_research_suggestions"]


def test_build_parameter_iteration_context_prioritizes_key_rows_and_columns():
    parameter_results = pd.DataFrame({"name": ["a", "b"], "evaluation_score": [0.1, 0.9]})
    walk_forward = pd.DataFrame({"name": ["low", "high", "mid"], "stability_score": [0.1, 0.9, 0.5]})
    trade_plan = pd.DataFrame(
        {
            "extra": [1],
            "trigger_rate": [0.2],
            "win_rate": [0.6],
            "avg_return": [0.03],
            "max_drawdown": [-0.08],
        }
    )

    context = build_parameter_iteration_context(
        parameter_search_results=parameter_results,
        walk_forward_validation=walk_forward,
        trade_plan_backtest_performance=trade_plan,
    )

    assert context["parameter_search_results"]["rows"][0]["name"] == "b"
    assert context["walk_forward_validation"]["rows"][0]["name"] == "low"
    assert context["trade_plan_backtest_performance"]["columns"][:4] == [
        "trigger_rate",
        "win_rate",
        "avg_return",
        "max_drawdown",
    ]


def test_build_parameter_iteration_prompt_contains_required_guardrails():
    prompt = build_parameter_iteration_prompt(build_parameter_iteration_context())

    assert "只做参数搜索空间候选建议，不做交易决策" in prompt
    assert "不能直接修改 parameter_search_space.json" in prompt
    assert "所有参数建议必须重新回测" in prompt


def test_run_parameter_iteration_agent_uses_fake_llm_client():
    client = FakeLLMClient("# 一、当前参数研究状态\n\n候选建议")

    result = run_parameter_iteration_agent(client)

    assert "候选建议" in result
    assert "ParameterIterationAgent" in client.prompt


def test_run_parameter_iteration_agent_neutralizes_forbidden_terms():
    client = FakeLLMClient("保证盈利，稳赚，满仓，自动下单")

    result = run_parameter_iteration_agent(client)

    assert "保证盈利" not in result
    assert "稳赚" not in result
    assert "满仓" not in result
    assert "自动下单" not in result
