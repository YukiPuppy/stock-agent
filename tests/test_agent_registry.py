import pytest

from src.agents.agent_registry import (
    ALLOWED_PERMISSIONS,
    REQUIRED_FIELDS,
    get_agent,
    list_agents,
    list_agents_by_category,
    list_agents_by_workflow,
)


EXPECTED_AGENTS = {
    "ReportAgent",
    "RiskReviewAgent",
    "DailyReviewAgent",
    "MarketRegimeAgent",
    "IndustryInsightAgent",
    "FactorInsightAgent",
    "BacktestAnalysisAgent",
    "StrategyResearchAgent",
    "ParameterIterationAgent",
}
EXPECTED_MODEL_ENVS = {
    "ReportAgent": "REPORT_AGENT_MODEL",
    "RiskReviewAgent": "RISK_REVIEW_AGENT_MODEL",
    "DailyReviewAgent": "DAILY_REVIEW_AGENT_MODEL",
    "MarketRegimeAgent": "MARKET_REGIME_AGENT_MODEL",
    "IndustryInsightAgent": "INDUSTRY_INSIGHT_AGENT_MODEL",
    "FactorInsightAgent": "FACTOR_INSIGHT_AGENT_MODEL",
    "BacktestAnalysisAgent": "BACKTEST_ANALYSIS_AGENT_MODEL",
    "StrategyResearchAgent": "STRATEGY_RESEARCH_AGENT_MODEL",
    "ParameterIterationAgent": "PARAMETER_ITERATION_AGENT_MODEL",
}


def test_list_agents_contains_expected_agents_and_fields():
    agents = list_agents()

    assert {agent["name"] for agent in agents} == EXPECTED_AGENTS
    for agent in agents:
        assert set(agent) == REQUIRED_FIELDS
        assert all(str(value).strip() for value in agent.values())


def test_agent_registry_permissions_are_read_only_or_proposal_only():
    agents = list_agents()

    assert {agent["permission"] for agent in agents}.issubset(ALLOWED_PERMISSIONS)
    assert get_agent("StrategyResearchAgent")["permission"] == "proposal_only"
    assert get_agent("ParameterIterationAgent")["permission"] == "proposal_only"
    assert get_agent("ReportAgent")["permission"] == "read_only"


def test_agent_registry_uses_agent_specific_model_envs_with_default_fallback():
    for agent_name, model_env in EXPECTED_MODEL_ENVS.items():
        agent = get_agent(agent_name)
        assert agent["default_model_env"] == model_env
        assert agent["fallback_model_env"] == "DEFAULT_LLM_MODEL"


def test_agent_registry_does_not_allow_protected_config_or_trade_outputs():
    protected_outputs = (
        "configs/active_strategies.json",
        "configs/parameter_search_space.json",
        "trade",
        "order",
        "broker",
    )

    for agent in list_agents():
        output_pattern = agent["output_pattern"].lower()
        assert "reports/" in output_pattern
        for protected_output in protected_outputs:
            assert protected_output not in output_pattern


def test_get_agent_returns_copy_by_name():
    agent = get_agent("RiskReviewAgent")
    agent["permission"] = "modified"

    assert get_agent("RiskReviewAgent")["permission"] == "read_only"


def test_get_agent_raises_for_unknown_name():
    with pytest.raises(KeyError, match="Unknown agent"):
        get_agent("UnknownAgent")


def test_list_agents_by_category_filters_registry():
    agents = list_agents_by_category("research")

    assert {agent["name"] for agent in agents} == {
        "BacktestAnalysisAgent",
        "StrategyResearchAgent",
        "ParameterIterationAgent",
    }


def test_list_agents_by_workflow_filters_comma_separated_workflows():
    agents = list_agents_by_workflow("strategy_ops")

    assert {agent["name"] for agent in agents} == {
        "BacktestAnalysisAgent",
        "StrategyResearchAgent",
        "ParameterIterationAgent",
    }
    assert all("strategy_ops" in agent["workflow"] for agent in agents)
