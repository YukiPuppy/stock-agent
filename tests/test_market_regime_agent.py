import pandas as pd

from src.agents.market_regime_agent import (
    build_market_regime_agent_context,
    build_market_regime_agent_prompt,
    run_market_regime_agent,
)


class FakeLLMClient:
    def __init__(self, output: str = "## 一、市场环境总体判断\n仅供人工复核参考。"):
        self.output = output
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.output


def test_build_market_regime_agent_context_handles_empty_dataframes():
    context = build_market_regime_agent_context(
        market_regime=pd.DataFrame(),
        index_daily=pd.DataFrame(),
        limit_list_daily=pd.DataFrame(),
        candidate_pool=pd.DataFrame(),
        trade_plan=pd.DataFrame(),
    )

    assert context["market_regime"]["is_empty"] is True
    assert context["index_daily"]["rows"] == []
    assert context["limit_list_daily"]["rows"] == []


def test_context_prioritizes_latest_market_regime():
    context = build_market_regime_agent_context(
        market_regime=pd.DataFrame(
            [
                {"trade_date": "2026-01-01", "market_regime": "weak"},
                {"trade_date": "2026-01-03", "market_regime": "strong"},
                {"trade_date": "2026-01-02", "market_regime": "neutral"},
            ]
        )
    )

    rows = context["market_regime"]["rows"]
    assert rows[0]["trade_date"] == "2026-01-03"
    assert rows[0]["market_regime"] == "strong"


def test_context_prioritizes_market_high_risk_candidate_records():
    context = build_market_regime_agent_context(
        candidate_pool=pd.DataFrame(
            [
                {"code": "000001", "risk_flags": ""},
                {"code": "600000", "risk_flags": "market_high_risk,liquidity"},
            ]
        )
    )

    assert context["candidate_pool"]["rows"][0]["code"] == "600000"


def test_prompt_contains_scope_and_risk_constraints():
    prompt = build_market_regime_agent_prompt(build_market_regime_agent_context())

    assert "只做市场环境解释，不做交易决策" in prompt
    assert "不得承诺收益" in prompt
    assert "不得使用“保证盈利”“稳赚”“满仓”“自动下单”" in prompt
    assert "不得建议绕过止损、仓位限制或风控" in prompt
    assert "strong 不代表一定上涨" in prompt
    assert "weak 不代表一定下跌" in prompt


def test_run_market_regime_agent_uses_fake_llm_client():
    fake_client = FakeLLMClient()

    markdown = run_market_regime_agent(
        fake_client,
        market_regime=pd.DataFrame([{"trade_date": "2026-01-02", "market_regime": "neutral"}]),
    )

    assert markdown.startswith("## 一、市场环境总体判断")
    assert fake_client.prompts
    assert "neutral" in fake_client.prompts[0]


def test_run_market_regime_agent_neutralizes_forbidden_terms():
    fake_client = FakeLLMClient("保证盈利，稳赚，满仓，自动下单")

    markdown = run_market_regime_agent(fake_client)

    assert "保证盈利" not in markdown
    assert "稳赚" not in markdown
    assert "满仓" not in markdown
    assert "自动下单" not in markdown
    assert "收益存在不确定性" in markdown
