from pathlib import Path

from src.config import settings
from src.pipeline import run_llm_agents_workflow as workflow


def test_default_runs_all_agent_pipelines(tmp_path, monkeypatch):
    calls = []

    def fake_runner(name):
        def _run(**kwargs):
            calls.append((name, kwargs))
            return str(tmp_path / "reports" / f"{name}.md")

        return _run

    monkeypatch.setattr(workflow, "run_report_agent_pipeline", fake_runner("report"))
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", fake_runner("backtest"))
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", fake_runner("market"))
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", fake_runner("industry"))
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", fake_runner("factor"))
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", fake_runner("strategy"))
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", fake_runner("parameter"))
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", fake_runner("risk"))
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", fake_runner("daily"))
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "stock_agent.duckdb"))
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(settings, "LLM_MODEL", "deepseek-v4-flash")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    assert [call[0] for call in calls] == ["report", "backtest", "market", "industry", "factor", "strategy", "parameter", "risk", "daily"]
    assert summary["db_path"] == str(tmp_path / "stock_agent.duckdb")
    assert summary["llm_provider"] == "deepseek"
    assert summary["llm_model"] == "deepseek-v4-flash"
    assert summary["generated_report_count"] == 9
    assert summary["errors"] == []


def test_skip_parameters_skip_selected_agents(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: calls.append("report") or "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        run_backtest_analysis_agent=False,
        run_market_regime_agent=False,
        run_industry_insight_agent=False,
        run_factor_insight_agent=False,
        run_strategy_research_agent=False,
        run_parameter_iteration_agent=False,
        run_daily_review_agent=False,
    )

    assert calls == ["report", "risk"]
    assert summary["skipped_agents"] == ["BacktestAnalysisAgent", "MarketRegimeAgent", "IndustryInsightAgent", "FactorInsightAgent", "StrategyResearchAgent", "ParameterIterationAgent", "DailyReviewAgent"]
    assert summary["generated_report_count"] == 2
    assert summary["backtest_analysis_agent_path"] is None
    assert summary["market_regime_agent_path"] is None
    assert summary["industry_insight_agent_path"] is None
    assert summary["factor_insight_agent_path"] is None
    assert summary["strategy_research_agent_path"] is None
    assert summary["parameter_iteration_agent_path"] is None
    assert summary["daily_review_agent_path"] is None


def test_trade_date_is_passed_to_daily_review_agent(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: "risk.md")

    def fake_daily(**kwargs):
        captured.update(kwargs)
        return "daily.md"

    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", fake_daily)

    workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        trade_date="2026-01-01",
    )

    assert captured["trade_date"] == "2026-01-01"


def test_agent_error_does_not_stop_other_agents(tmp_path, monkeypatch):
    calls = []

    def failing_report(**kwargs):
        calls.append("report")
        raise RuntimeError("report failed")

    monkeypatch.setattr(workflow, "run_report_agent_pipeline", failing_report)
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    assert calls == ["report", "backtest", "market", "industry", "factor", "strategy", "parameter", "risk", "daily"]
    assert summary["report_agent_path"] is None
    assert summary["backtest_analysis_agent_path"] == "backtest.md"
    assert summary["market_regime_agent_path"] == "market.md"
    assert summary["industry_insight_agent_path"] == "industry.md"
    assert summary["factor_insight_agent_path"] == "factor.md"
    assert summary["strategy_research_agent_path"] == "strategy.md"
    assert summary["strategy_research_suggestions_path"] == "suggestions.json"
    assert summary["parameter_iteration_agent_path"] == "parameter.md"
    assert summary["parameter_search_space_candidate_path"] == "candidate.json"
    assert summary["risk_review_agent_path"] == "risk.md"
    assert summary["daily_review_agent_path"] == "daily.md"
    assert summary["generated_report_count"] == 8
    assert summary["errors"] == [{"agent": "ReportAgent", "error": "report failed"}]


def test_workflow_generates_index_report(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: "reports/llm_report_summary_2026-01-02.md")
    monkeypatch.setattr(
        workflow,
        "run_backtest_analysis_agent_pipeline",
        lambda **kwargs: "reports/llm_backtest_analysis_2026-01-02.md",
    )
    monkeypatch.setattr(
        workflow,
        "run_market_regime_agent_pipeline",
        lambda **kwargs: "reports/llm_market_regime_2026-01-02.md",
    )
    monkeypatch.setattr(
        workflow,
        "run_industry_insight_agent_pipeline",
        lambda **kwargs: "reports/llm_industry_insight_2026-01-02.md",
    )
    monkeypatch.setattr(
        workflow,
        "run_factor_insight_agent_pipeline",
        lambda **kwargs: "reports/llm_factor_insight_2026-01-02.md",
    )
    monkeypatch.setattr(
        workflow,
        "run_strategy_research_agent_pipeline",
        lambda **kwargs: {
            "strategy_research_report_path": "reports/llm_strategy_research_2026-01-02.md",
            "strategy_research_suggestions_path": "reports/strategy_research_suggestions_2026-01-02.json",
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_parameter_iteration_agent_pipeline",
        lambda **kwargs: {
            "parameter_iteration_report_path": "reports/llm_parameter_iteration_2026-01-02.md",
            "parameter_search_space_candidate_path": "reports/parameter_search_space_candidate_2026-01-02.json",
        },
    )
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: "reports/llm_risk_review_2026-01-02.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: "reports/llm_daily_review_2026-01-02.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
    )

    index_path = Path(summary["llm_agents_index_path"])
    assert index_path == tmp_path / "reports" / "llm_agents_index_2026-01-02.md"
    content = index_path.read_text(encoding="utf-8")
    assert "# LLM Agent 报告索引" in content
    assert "MarketRegimeAgent 报告路径：reports/llm_market_regime_2026-01-02.md" in content
    assert "IndustryInsightAgent 报告路径：reports/llm_industry_insight_2026-01-02.md" in content
    assert "FactorInsightAgent 报告路径：reports/llm_factor_insight_2026-01-02.md" in content
    assert "StrategyResearchAgent 报告路径：reports/llm_strategy_research_2026-01-02.md" in content
    assert "StrategyResearchAgent 候选研究建议 JSON 路径：reports/strategy_research_suggestions_2026-01-02.json" in content
    assert "ParameterIterationAgent 报告路径：reports/llm_parameter_iteration_2026-01-02.md" in content
    assert "ParameterIterationAgent 候选参数搜索空间 JSON 路径：reports/parameter_search_space_candidate_2026-01-02.json" in content
    assert "只是候选参数研究建议，不能直接用于实盘" in content
    assert "只是候选研究建议，不能直接用于实盘" in content
    assert "不构成投资建议" in content
    assert "不代表自动交易" in content


def test_skip_market_regime_agent_skips_selected_agent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: calls.append("report") or "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        run_market_regime_agent=False,
    )

    assert calls == ["report", "backtest", "industry", "factor", "strategy", "parameter", "risk", "daily"]
    assert summary["market_regime_agent_path"] is None
    assert "MarketRegimeAgent" in summary["skipped_agents"]


def test_skip_industry_insight_agent_skips_selected_agent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: calls.append("report") or "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        run_industry_insight_agent=False,
    )

    assert calls == ["report", "backtest", "market", "factor", "strategy", "parameter", "risk", "daily"]
    assert summary["industry_insight_agent_path"] is None
    assert "IndustryInsightAgent" in summary["skipped_agents"]


def test_skip_factor_insight_agent_skips_selected_agent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: calls.append("report") or "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        run_factor_insight_agent=False,
    )

    assert calls == ["report", "backtest", "market", "industry", "strategy", "parameter", "risk", "daily"]
    assert summary["factor_insight_agent_path"] is None
    assert "FactorInsightAgent" in summary["skipped_agents"]


def test_skip_strategy_research_agent_skips_selected_agent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: calls.append("report") or "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        run_strategy_research_agent=False,
    )

    assert calls == ["report", "backtest", "market", "industry", "factor", "parameter", "risk", "daily"]
    assert summary["strategy_research_agent_path"] is None
    assert "StrategyResearchAgent" in summary["skipped_agents"]


def test_parse_args_supports_skip_strategy_research_agent():
    args = workflow._parse_args(["--skip-strategy-research-agent"])

    assert args.skip_strategy_research_agent is True


def test_skip_parameter_iteration_agent_skips_selected_agent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_report_agent_pipeline", lambda **kwargs: calls.append("report") or "report.md")
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", lambda **kwargs: calls.append("backtest") or "backtest.md")
    monkeypatch.setattr(workflow, "run_market_regime_agent_pipeline", lambda **kwargs: calls.append("market") or "market.md")
    monkeypatch.setattr(workflow, "run_industry_insight_agent_pipeline", lambda **kwargs: calls.append("industry") or "industry.md")
    monkeypatch.setattr(workflow, "run_factor_insight_agent_pipeline", lambda **kwargs: calls.append("factor") or "factor.md")
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", lambda **kwargs: calls.append("strategy") or {"strategy_research_report_path": "strategy.md", "strategy_research_suggestions_path": "suggestions.json"})
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", lambda **kwargs: calls.append("parameter") or {"parameter_iteration_report_path": "parameter.md", "parameter_search_space_candidate_path": "candidate.json"})
    monkeypatch.setattr(workflow, "run_risk_review_agent_pipeline", lambda **kwargs: calls.append("risk") or "risk.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: calls.append("daily") or "daily.md")

    summary = workflow.run_llm_agents_workflow(
        output_dir=str(tmp_path / "reports"),
        report_date="2026-01-02",
        run_parameter_iteration_agent=False,
    )

    assert calls == ["report", "backtest", "market", "industry", "factor", "strategy", "risk", "daily"]
    assert summary["parameter_iteration_agent_path"] is None
    assert "ParameterIterationAgent" in summary["skipped_agents"]


def test_parse_args_supports_skip_parameter_iteration_agent():
    args = workflow._parse_args(["--skip-parameter-iteration-agent"])

    assert args.skip_parameter_iteration_agent is True
