"""Run research-advice agents against one completed run without mutating research state."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.backtest_analysis_agent import run_backtest_analysis_agent
from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.agents.parameter_iteration_agent import run_parameter_iteration_agent
from src.agents.strategy_research_agent import run_strategy_research_agent
from src.agents.strategy_research_outputs import extract_strategy_research_suggestions
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.pipeline.memory import log_memory


INCOMPLETE_WARNING = "准入结论不完整，不建议实盘"


def run_strategy_research_agents_readonly(
    *,
    run_id: str,
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> dict:
    resolved_db_path = db_path or DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)
    log_memory("strategy_research_agents_readonly", "before_load")
    evaluation = store.load_strategy_version_evaluation(run_id=run_id)
    parameter_results = store.load_parameter_search_results(run_id=run_id)
    walk_forward = store.load_walk_forward_validation(run_id=run_id)
    trade_plan_performance = store.load_trade_plan_backtest_performance(run_id=run_id)
    admission = store.load_strategy_admission(run_id=run_id)
    incomplete = admission.empty or int(
        admission.get("trade_plan_win_rate", pd.Series(dtype=float)).notna().sum()
    ) == 0
    warning = f"> **{INCOMPLETE_WARNING}。**\n\n" if incomplete else ""

    backtest_client = _safe_client("BacktestAnalysisAgent")
    strategy_client = _safe_client("StrategyResearchAgent")
    parameter_client = _safe_client("ParameterIterationAgent")
    if isinstance(backtest_client, DisabledLLMClient):
        backtest_markdown = _placeholder("backtest_analysis", resolved_date)
    else:
        backtest_markdown = run_backtest_analysis_agent(
            backtest_client,
            strategy_evaluation=evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward,
            trade_plan_backtest_performance=trade_plan_performance,
            strategy_admission=admission,
        )
    if isinstance(strategy_client, DisabledLLMClient):
        strategy_markdown = _placeholder("strategy_research", resolved_date)
    else:
        strategy_markdown = run_strategy_research_agent(
            strategy_client,
            strategy_evaluation=evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward,
            trade_plan_backtest_performance=trade_plan_performance,
            strategy_admission=admission,
        )
    suggestions = extract_strategy_research_suggestions(strategy_markdown)
    if isinstance(parameter_client, DisabledLLMClient):
        parameter_markdown = _placeholder("parameter_iteration", resolved_date)
    else:
        parameter_markdown = run_parameter_iteration_agent(
            parameter_client,
            strategy_research_suggestions=suggestions,
            strategy_evaluation=evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward,
            trade_plan_backtest_performance=trade_plan_performance,
            strategy_admission=admission,
        )

    output_paths = {
        "backtest_analysis_report_path": _write_report(
            output_dir, f"backtest_analysis_{run_id}_{resolved_date}.md", warning + backtest_markdown
        ),
        "strategy_research_report_path": _write_report(
            output_dir, f"strategy_research_{run_id}_{resolved_date}.md", warning + strategy_markdown
        ),
        "parameter_iteration_proposal_path": _write_report(
            output_dir,
            f"parameter_iteration_proposal_{run_id}_{resolved_date}.md",
            warning
            + parameter_markdown
            + "\n\n> 本文件仅为 proposal，未写入正式参数、策略配置或候选配置。\n",
        ),
    }
    log_memory("strategy_research_agents_readonly", "finished")
    return {
        "run_id": run_id,
        "read_only": True,
        "admission_incomplete": incomplete,
        "trade_plan_win_rate_nonnull_rows": int(
            admission.get("trade_plan_win_rate", pd.Series(dtype=float)).notna().sum()
        ),
        **output_paths,
    }


def _safe_client(agent_name: str):
    try:
        return get_llm_client(agent_name)
    except Exception:
        return DisabledLLMClient()


def _placeholder(report_type: str, report_date: str) -> str:
    labels = {
        "backtest_analysis": "回测分析建议",
        "strategy_research": "策略研究建议",
        "parameter_iteration": "参数迭代 proposal",
    }
    return (
        f"# {labels[report_type]}\n\n报告日期：{report_date}\n\n"
        "LLM 当前未启用；本只读入口未重跑研究步骤，也未修改任何配置。\n\n"
        "本报告不启用策略、不生成交易指令、不执行交易。\n"
    )


def _write_report(output_dir: str, filename: str, markdown: str) -> str:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown.strip() + "\n", encoding="utf-8")
    return str(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate read-only research-agent advice for an existing run; never rerun backtests or modify configs."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_strategy_research_agents_readonly(
        run_id=args.run_id,
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
