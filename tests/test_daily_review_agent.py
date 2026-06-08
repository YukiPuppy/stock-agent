import pandas as pd

from src.agents.daily_review_agent import (
    build_daily_review_agent_context,
    build_daily_review_agent_prompt,
    run_daily_review_agent,
)


class FakeLLMClient:
    def __init__(self, response: str = "## 一、当日执行总体评价\n仅供人工复核参考。"):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_build_daily_review_agent_context_handles_empty_dataframes():
    context = build_daily_review_agent_context(
        actual_trades=pd.DataFrame(),
        execution_review=pd.DataFrame(),
        daily_review=pd.DataFrame(),
        period_review=pd.DataFrame(),
        actual_trade_performance=pd.DataFrame(),
        positions=pd.DataFrame(),
        position_review=pd.DataFrame(),
    )

    assert context["actual_trades"]["is_empty"] is True
    assert context["execution_review"]["rows"] == []
    assert context["position_review"]["row_count"] == 0


def test_build_daily_review_agent_context_prioritizes_risk_records_and_limits_rows():
    execution_review = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(35)],
            "execution_status": ["matched"] * 33 + ["off_plan", "deviation"],
            "api_key": ["secret"] * 35,
        }
    )
    daily_review = pd.DataFrame({"trade_date": [f"2026-01-{index:02d}" for index in range(1, 36)], "execution_score": list(range(35, 0, -1))})
    period_review = pd.DataFrame(
        {
            "start_date": ["a", "b", "c"],
            "off_plan_count": [1, 9, 2],
            "deviation_count": [1, 1, 7],
            "chase_count": [0, 0, 5],
            "over_position_count": [0, 0, 4],
        }
    )
    performance = pd.DataFrame(
        {
            "code": ["weak", "strong", "invalid"],
            "is_valid": [True, True, False],
            "return_3d": [-0.08, 0.12, -0.2],
        }
    )
    position_review = pd.DataFrame(
        {"code": ["low", "high", "medium"], "position_risk_level": ["low", "high", "medium"]}
    )

    context = build_daily_review_agent_context(
        execution_review=execution_review,
        daily_review=daily_review,
        period_review=period_review,
        actual_trade_performance=performance,
        position_review=position_review,
    )

    execution_rows = context["execution_review"]["rows"]
    assert len(execution_rows) == 30
    assert [row["execution_status"] for row in execution_rows[:2]] == ["deviation", "off_plan"]
    assert "api_key" not in context["execution_review"]["columns"]
    assert context["daily_review"]["rows"][0]["execution_score"] == 1
    assert context["period_review"]["rows"][0]["off_plan_count"] == 9
    assert context["actual_trade_performance"]["rows"][0]["code"] == "weak"
    assert any(row["code"] == "strong" for row in context["actual_trade_performance"]["rows"])
    assert [row["position_risk_level"] for row in context["position_review"]["rows"][:2]] == ["high", "medium"]


def test_build_daily_review_agent_prompt_contains_scope_and_constraints():
    prompt = build_daily_review_agent_prompt(build_daily_review_agent_context())

    assert "只做交易复盘和执行纪律分析，不做交易决策" in prompt
    assert "不得承诺收益" in prompt
    assert "自动下单" in prompt
    assert "不得建议绕过止损、仓位限制或风控" in prompt
    assert "策略问题、执行问题、数据样本不足问题" in prompt


def test_run_daily_review_agent_uses_fake_llm_client():
    fake = FakeLLMClient()

    markdown = run_daily_review_agent(fake, actual_trades=pd.DataFrame({"code": ["600000"]}))

    assert markdown.startswith("## 一、当日执行总体评价")
    assert fake.prompts
    assert "600000" in fake.prompts[0]


def test_run_daily_review_agent_neutralizes_forbidden_terms():
    fake = FakeLLMClient("保证盈利，稳赚，满仓，自动下单")

    markdown = run_daily_review_agent(fake)

    assert "保证盈利" not in markdown
    assert "稳赚" not in markdown
    assert "满仓" not in markdown
    assert "自动下单" not in markdown
