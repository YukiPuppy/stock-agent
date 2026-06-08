import pandas as pd

from src.agents.risk_review_agent import (
    build_risk_review_context,
    build_risk_review_prompt,
    run_risk_review_agent,
)


class FakeLLMClient:
    def __init__(self, response: str = "## 一、总体风险结论\n仅做风险审查。"):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_build_risk_review_context_handles_empty_dataframe():
    context = build_risk_review_context(data_quality_report=pd.DataFrame(columns=["status"]))

    assert context["data_quality_report"]["is_empty"] is True
    assert context["data_quality_report"]["columns"] == ["status"]
    assert context["data_quality_report"]["rows"] == []


def test_build_risk_review_context_prioritizes_risk_records_and_limits_30_rows():
    data_quality = pd.DataFrame(
        {
            "check_name": [f"check_{index}" for index in range(35)] + ["bad_check"],
            "status": ["ok"] * 35 + ["error"],
            "api_key": ["secret"] * 36,
        }
    )
    trade_plan = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(35)] + ["600000"],
            "risk_flags": [""] * 35 + ["risk"],
        }
    )

    context = build_risk_review_context(data_quality_report=data_quality, trade_plan=trade_plan)

    quality_rows = context["data_quality_report"]["rows"]
    plan_rows = context["trade_plan"]["rows"]
    assert len(quality_rows) == 30
    assert quality_rows[0]["status"] == "error"
    assert plan_rows[0]["risk_flags"] == "risk"
    assert "api_key" not in context["data_quality_report"]["columns"]
    assert "secret" not in str(context)


def test_build_risk_review_context_sorts_configured_risk_sources():
    context = build_risk_review_context(
        strategy_admission=pd.DataFrame({"admission_recommendation": ["enable", "continue_research"]}),
        position_review=pd.DataFrame({"position_risk_level": ["low", "high", "medium"]}),
        execution_review=pd.DataFrame({"execution_status": ["matched", "off_plan", "deviation"]}),
        daily_review=pd.DataFrame({"execution_score": [80, 30, 60]}),
        period_review=pd.DataFrame({"off_plan_count": [1, 5, 5], "deviation_count": [9, 1, 3]}),
    )

    assert context["strategy_admission"]["rows"][0]["admission_recommendation"] == "continue_research"
    assert context["position_review"]["rows"][0]["position_risk_level"] == "high"
    assert context["execution_review"]["rows"][0]["execution_status"] == "deviation"
    assert context["daily_review"]["rows"][0]["execution_score"] == 30
    assert context["period_review"]["rows"][0]["off_plan_count"] == 5
    assert context["period_review"]["rows"][0]["deviation_count"] == 3


def test_build_risk_review_prompt_contains_scope_and_constraints():
    prompt = build_risk_review_prompt(build_risk_review_context())

    assert "只做风险审查，不做交易决策" in prompt
    assert "不得承诺收益" in prompt
    assert "不得使用“保证盈利”“稳赚”“满仓”“自动下单”" in prompt
    assert "不得建议绕过止损、仓位限制或风控" in prompt
    assert "不得直接启用策略" in prompt
    assert "中文 Markdown" in prompt


def test_run_risk_review_agent_uses_fake_client_and_returns_markdown():
    client = FakeLLMClient("## 一、总体风险结论\n当前仅供人工复核参考。")

    result = run_risk_review_agent(client, data_quality_report=pd.DataFrame({"status": ["warning"]}))

    assert result.startswith("## 一、总体风险结论")
    assert client.prompts
    assert "warning" in client.prompts[0]


def test_run_risk_review_agent_filters_forbidden_terms():
    client = FakeLLMClient("保证盈利，稳赚，满仓，自动下单。")

    result = run_risk_review_agent(client)

    assert "保证盈利" not in result
    assert "稳赚" not in result
    assert "满仓" not in result
    assert "自动下单" not in result
    assert "收益存在不确定性" in result
