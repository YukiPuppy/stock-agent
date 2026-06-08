import pandas as pd

from src.agents.backtest_analysis_agent import (
    build_backtest_analysis_context,
    build_backtest_analysis_prompt,
    run_backtest_analysis_agent,
)


class FakeLLMClient:
    def __init__(self, response: str = "## 一、总体结论\n仅做回测分析。"):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_build_backtest_analysis_context_handles_empty_dataframe():
    context = build_backtest_analysis_context(
        strategy_evaluation=pd.DataFrame(),
        parameter_search_results=None,
    )

    assert context["strategy_evaluation"]["is_empty"] is True
    assert context["strategy_evaluation"]["row_count"] == 0
    assert context["parameter_search_results"]["is_empty"] is True


def test_build_backtest_analysis_context_sorts_by_score_and_limits_30_rows():
    df = pd.DataFrame(
        {
            "strategy_name": [f"s{i}" for i in range(35)],
            "evaluation_score": list(range(35)),
            "api_key": ["secret"] * 35,
        }
    )

    context = build_backtest_analysis_context(strategy_evaluation=df)
    rows = context["strategy_evaluation"]["rows"]

    assert len(rows) == 30
    assert rows[0]["evaluation_score"] == 34
    assert rows[-1]["evaluation_score"] == 5
    assert "api_key" not in context["strategy_evaluation"]["columns"]
    assert "secret" not in str(context)


def test_build_backtest_analysis_context_sorts_all_configured_sources():
    context = build_backtest_analysis_context(
        parameter_search_results=pd.DataFrame({"evaluation_score": [1, 3, 2]}),
        walk_forward_validation=pd.DataFrame({"stability_score": [0.1, 0.9, 0.2]}),
        trade_plan_backtest_performance=pd.DataFrame({"avg_return": [-0.1, 0.2, 0.0]}),
        strategy_admission=pd.DataFrame({"admission_score": [60, 90, 70]}),
    )

    assert context["parameter_search_results"]["rows"][0]["evaluation_score"] == 3
    assert context["walk_forward_validation"]["rows"][0]["stability_score"] == 0.9
    assert context["trade_plan_backtest_performance"]["rows"][0]["avg_return"] == 0.2
    assert context["strategy_admission"]["rows"][0]["admission_score"] == 90


def test_build_backtest_analysis_prompt_contains_scope_and_forbidden_constraints():
    prompt = build_backtest_analysis_prompt(build_backtest_analysis_context())

    assert "只做分析，不做交易决策" in prompt
    assert "不得承诺收益" in prompt
    assert "不得使用“保证盈利”“稳赚”“满仓”“自动下单”" in prompt
    assert "不得直接启用策略" in prompt


def test_run_backtest_analysis_agent_uses_fake_llm_client():
    client = FakeLLMClient("## 一、总体结论\n当前仅供人工复核参考。")

    result = run_backtest_analysis_agent(client, strategy_evaluation=pd.DataFrame({"evaluation_score": [1]}))

    assert result.startswith("## 一、总体结论")
    assert client.prompts
    assert "evaluation_score" in client.prompts[0]


def test_run_backtest_analysis_agent_filters_forbidden_terms():
    client = FakeLLMClient("保证盈利，稳赚，满仓，自动下单。")

    result = run_backtest_analysis_agent(client)

    assert "保证盈利" not in result
    assert "稳赚" not in result
    assert "满仓" not in result
    assert "自动下单" not in result
    assert "收益存在不确定性" in result
