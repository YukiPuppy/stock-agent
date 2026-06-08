import pandas as pd

from src.agents.industry_insight_agent import (
    build_industry_insight_context,
    build_industry_insight_prompt,
    run_industry_insight_agent,
)


class FakeLLMClient:
    def __init__(self, response: str = "## 一、行业强弱总体判断\nFake Industry Insight"):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_build_industry_insight_context_handles_empty_dataframe():
    context = build_industry_insight_context(
        industry_strength=pd.DataFrame(),
        sw_daily=pd.DataFrame(),
        stock_industry_map=pd.DataFrame(),
        candidate_pool=pd.DataFrame(),
        trade_plan=pd.DataFrame(),
    )

    assert context["industry_strength"]["is_empty"] is True
    assert context["candidate_pool"]["rows"] == []


def test_context_prefers_latest_industry_strength_and_strong_weak():
    df = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "industry_code": "old", "industry_strength_level": "strong", "industry_strength_score": 99},
            {"trade_date": "2026-01-02", "industry_code": "strong", "industry_strength_level": "strong", "industry_strength_score": 60},
            {"trade_date": "2026-01-02", "industry_code": "neutral", "industry_strength_level": "neutral", "industry_strength_score": 90},
            {"trade_date": "2026-01-02", "industry_code": "weak", "industry_strength_level": "weak", "industry_strength_score": 10},
        ]
    )

    rows = build_industry_insight_context(industry_strength=df)["industry_strength"]["rows"]

    assert {row["trade_date"] for row in rows} == {"2026-01-02"}
    assert [row["industry_code"] for row in rows[:2]] == ["strong", "weak"]


def test_context_prefers_candidate_pool_industry_risk_flags():
    df = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "code": "000001", "risk_flags": "normal"},
            {"trade_date": "2026-01-02", "code": "000002", "risk_flags": "weak_industry"},
            {"trade_date": "2026-01-02", "code": "000003", "risk_flags": "strong_industry"},
        ]
    )

    rows = build_industry_insight_context(candidate_pool=df)["candidate_pool"]["rows"]

    assert [row["code"] for row in rows[:2]] == ["000002", "000003"]


def test_prompt_contains_scope_and_safety_constraints():
    prompt = build_industry_insight_prompt(build_industry_insight_context())

    assert "只做行业强弱解释，不做交易决策" in prompt
    assert "不得承诺收益" in prompt
    assert "自动下单" in prompt
    assert "不得建议绕过止损、仓位限制或风控" in prompt
    assert "行业强度只用于调整策略置信度和风险提示" in prompt


def test_run_industry_insight_agent_uses_fake_llm_client():
    fake = FakeLLMClient()

    markdown = run_industry_insight_agent(fake, industry_strength=pd.DataFrame([{"industry_strength_level": "strong"}]))

    assert markdown.startswith("## 一、行业强弱总体判断")
    assert fake.prompts
    assert "strong" in fake.prompts[0]


def test_run_industry_insight_agent_neutralizes_forbidden_terms():
    fake = FakeLLMClient("保证盈利，稳赚，满仓，自动下单")

    markdown = run_industry_insight_agent(fake)

    assert "保证盈利" not in markdown
    assert "稳赚" not in markdown
    assert "满仓" not in markdown
    assert "自动下单" not in markdown
    assert "收益存在不确定性" in markdown
