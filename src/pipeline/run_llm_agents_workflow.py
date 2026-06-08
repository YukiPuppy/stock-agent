"""Run all low-risk LLM agent report pipelines and build an index report."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from src.config import settings
from src.pipeline.run_backtest_analysis_agent import run_backtest_analysis_agent_pipeline
from src.pipeline.run_daily_review_agent import run_daily_review_agent_pipeline
from src.pipeline.run_factor_insight_agent import run_factor_insight_agent_pipeline
from src.pipeline.run_industry_insight_agent import run_industry_insight_agent_pipeline
from src.pipeline.run_market_regime_agent import run_market_regime_agent_pipeline
from src.pipeline.run_parameter_iteration_agent import run_parameter_iteration_agent_pipeline
from src.pipeline.run_report_agent import run_report_agent_pipeline
from src.pipeline.run_risk_review_agent import run_risk_review_agent_pipeline
from src.pipeline.run_strategy_research_agent import run_strategy_research_agent_pipeline


AGENT_PATH_KEYS = {
    "ReportAgent": "report_agent_path",
    "BacktestAnalysisAgent": "backtest_analysis_agent_path",
    "MarketRegimeAgent": "market_regime_agent_path",
    "IndustryInsightAgent": "industry_insight_agent_path",
    "FactorInsightAgent": "factor_insight_agent_path",
    "StrategyResearchAgent": "strategy_research_agent_path",
    "ParameterIterationAgent": "parameter_iteration_agent_path",
    "RiskReviewAgent": "risk_review_agent_path",
    "DailyReviewAgent": "daily_review_agent_path",
}


def run_llm_agents_workflow(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
    trade_date: str | None = None,
    run_report_agent: bool = True,
    run_backtest_analysis_agent: bool = True,
    run_market_regime_agent: bool = True,
    run_industry_insight_agent: bool = True,
    run_factor_insight_agent: bool = True,
    run_strategy_research_agent: bool = True,
    run_parameter_iteration_agent: bool = True,
    run_risk_review_agent: bool = True,
    run_daily_review_agent: bool = True,
) -> dict:
    """Run selected LLM report agents and return a structured workflow summary."""
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_report_date = report_date or date.today().isoformat()
    summary: dict[str, Any] = {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "report_date": resolved_report_date,
        "trade_date": trade_date,
        "llm_provider": str(getattr(settings, "LLM_PROVIDER", "none") or "none"),
        "llm_model": str(getattr(settings, "LLM_MODEL", "") or ""),
        "report_agent_path": None,
        "backtest_analysis_agent_path": None,
        "market_regime_agent_path": None,
        "industry_insight_agent_path": None,
        "factor_insight_agent_path": None,
        "strategy_research_agent_path": None,
        "strategy_research_suggestions_path": None,
        "parameter_iteration_agent_path": None,
        "parameter_search_space_candidate_path": None,
        "risk_review_agent_path": None,
        "daily_review_agent_path": None,
        "generated_report_count": 0,
        "skipped_agents": [],
        "errors": [],
        "llm_agents_index_path": None,
    }

    _run_or_skip(
        summary,
        enabled=run_report_agent,
        agent_name="ReportAgent",
        runner=run_report_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_backtest_analysis_agent,
        agent_name="BacktestAnalysisAgent",
        runner=run_backtest_analysis_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_market_regime_agent,
        agent_name="MarketRegimeAgent",
        runner=run_market_regime_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_industry_insight_agent,
        agent_name="IndustryInsightAgent",
        runner=run_industry_insight_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_factor_insight_agent,
        agent_name="FactorInsightAgent",
        runner=run_factor_insight_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_strategy_research_agent,
        agent_name="StrategyResearchAgent",
        runner=run_strategy_research_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_parameter_iteration_agent,
        agent_name="ParameterIterationAgent",
        runner=run_parameter_iteration_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    _run_or_skip(
        summary,
        enabled=run_risk_review_agent,
        agent_name="RiskReviewAgent",
        runner=run_risk_review_agent_pipeline,
        kwargs={"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date},
    )
    daily_kwargs = {"db_path": resolved_db_path, "output_dir": output_dir, "report_date": resolved_report_date}
    if trade_date is not None:
        daily_kwargs["trade_date"] = trade_date
    _run_or_skip(
        summary,
        enabled=run_daily_review_agent,
        agent_name="DailyReviewAgent",
        runner=run_daily_review_agent_pipeline,
        kwargs=daily_kwargs,
    )

    summary["generated_report_count"] = sum(1 for key in AGENT_PATH_KEYS.values() if summary.get(key))
    summary["llm_agents_index_path"] = _write_index_report(summary)
    return summary


def _run_or_skip(
    summary: dict[str, Any],
    *,
    enabled: bool,
    agent_name: str,
    runner: Callable[..., str],
    kwargs: dict[str, Any],
) -> None:
    path_key = AGENT_PATH_KEYS[agent_name]
    if not enabled:
        summary["skipped_agents"].append(agent_name)
        return
    try:
        result = runner(**kwargs)
        if isinstance(result, dict):
            summary[path_key] = (
                result.get("strategy_research_report_path")
                or result.get("parameter_iteration_report_path")
                or result.get(path_key)
            )
            if "strategy_research_suggestions_path" in result:
                summary["strategy_research_suggestions_path"] = result.get("strategy_research_suggestions_path")
            if "parameter_search_space_candidate_path" in result:
                summary["parameter_search_space_candidate_path"] = result.get("parameter_search_space_candidate_path")
        else:
            summary[path_key] = result
    except Exception as exc:
        summary["errors"].append({"agent": agent_name, "error": str(exc)})


def _write_index_report(summary: dict[str, Any]) -> str:
    output_dir = str(summary["output_dir"])
    report_date = str(summary["report_date"])
    output_path = Path(output_dir) / f"llm_agents_index_{report_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_index_markdown(summary), encoding="utf-8")
    return str(output_path)


def _build_index_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LLM Agent 报告索引",
        "",
        "## 一、运行说明",
        "本流程只做结构化结果的分析总结和报告索引，不做自动交易、策略启用、参数调整或交易执行。",
        "",
        "## 二、生成报告",
        f"- ReportAgent 报告路径：{summary.get('report_agent_path') or '未生成'}",
        f"- BacktestAnalysisAgent 报告路径：{summary.get('backtest_analysis_agent_path') or '未生成'}",
        f"- MarketRegimeAgent 报告路径：{summary.get('market_regime_agent_path') or '未生成'}",
        f"- IndustryInsightAgent 报告路径：{summary.get('industry_insight_agent_path') or '未生成'}",
        f"- FactorInsightAgent 报告路径：{summary.get('factor_insight_agent_path') or '未生成'}",
        f"- StrategyResearchAgent 报告路径：{summary.get('strategy_research_agent_path') or '未生成'}",
        f"- StrategyResearchAgent 候选研究建议 JSON 路径：{summary.get('strategy_research_suggestions_path') or '未生成'}",
        f"- ParameterIterationAgent 报告路径：{summary.get('parameter_iteration_agent_path') or '未生成'}",
        f"- ParameterIterationAgent 候选参数搜索空间 JSON 路径：{summary.get('parameter_search_space_candidate_path') or '未生成'}",
        f"- RiskReviewAgent 报告路径：{summary.get('risk_review_agent_path') or '未生成'}",
        f"- DailyReviewAgent 报告路径：{summary.get('daily_review_agent_path') or '未生成'}",
        "",
        "## 三、失败 Agent",
    ]
    errors = summary.get("errors") or []
    if errors:
        for item in errors:
            lines.append(f"- {item.get('agent', 'UnknownAgent')}：{item.get('error', '')}")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 四、风险提示",
            "- LLM 报告只用于辅助理解结构化结果；",
            "- 不构成投资建议；",
            "- 不代表自动交易；",
            "- StrategyResearchAgent JSON 只是候选研究建议，不能直接用于实盘；",
            "- ParameterIterationAgent JSON 只是候选参数研究建议，不能直接用于实盘，不能直接写入正式 parameter_search_space.json；",
            "- 不应绕过系统风控和人工确认。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all LLM agent report pipelines.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--skip-report-agent", action="store_true")
    parser.add_argument("--skip-backtest-analysis-agent", action="store_true")
    parser.add_argument("--skip-market-regime-agent", action="store_true")
    parser.add_argument("--skip-industry-insight-agent", action="store_true")
    parser.add_argument("--skip-factor-insight-agent", action="store_true")
    parser.add_argument("--skip-strategy-research-agent", action="store_true")
    parser.add_argument("--skip-parameter-iteration-agent", action="store_true")
    parser.add_argument("--skip-risk-review-agent", action="store_true")
    parser.add_argument("--skip-daily-review-agent", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_llm_agents_workflow(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
        trade_date=args.trade_date,
        run_report_agent=not args.skip_report_agent,
        run_backtest_analysis_agent=not args.skip_backtest_analysis_agent,
        run_market_regime_agent=not args.skip_market_regime_agent,
        run_industry_insight_agent=not args.skip_industry_insight_agent,
        run_factor_insight_agent=not args.skip_factor_insight_agent,
        run_strategy_research_agent=not args.skip_strategy_research_agent,
        run_parameter_iteration_agent=not args.skip_parameter_iteration_agent,
        run_risk_review_agent=not args.skip_risk_review_agent,
        run_daily_review_agent=not args.skip_daily_review_agent,
    )
    print("LLM agents workflow finished.")
    for key in [
        "llm_provider",
        "llm_model",
        "report_agent_path",
        "backtest_analysis_agent_path",
        "market_regime_agent_path",
        "industry_insight_agent_path",
        "factor_insight_agent_path",
        "strategy_research_agent_path",
        "strategy_research_suggestions_path",
        "parameter_iteration_agent_path",
        "parameter_search_space_candidate_path",
        "risk_review_agent_path",
        "daily_review_agent_path",
        "llm_agents_index_path",
        "generated_report_count",
        "errors",
    ]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
