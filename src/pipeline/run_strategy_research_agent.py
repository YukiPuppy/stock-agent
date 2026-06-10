"""Run StrategyResearchAgent and export research candidate reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.agents.strategy_research_agent import run_strategy_research_agent
from src.agents.strategy_research_outputs import extract_strategy_research_suggestions
from src.config import settings
from src.database.duckdb_store import StockAgentStore


def run_strategy_research_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
    export_candidate_json: bool = True,
) -> dict:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    strategy_evaluation = _safe_load(store.load_strategy_version_evaluation)
    parameter_results = _safe_load(store.load_parameter_search_results)
    walk_forward = _safe_load(store.load_walk_forward_validation)
    trade_plan_performance = _safe_load(store.load_trade_plan_backtest_performance)
    admission = _safe_load(store.load_strategy_admission)
    factor_diagnostics = _safe_load(store.load_factor_diagnostics)
    market_regime = _safe_load(store.load_market_regime)
    industry_strength = _safe_load(store.load_industry_strength)
    moneyflow_factors = _safe_load(store.load_moneyflow_factors)

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_placeholder(resolved_date)
    else:
        markdown = run_strategy_research_agent(
            llm_client,
            strategy_evaluation=strategy_evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward,
            trade_plan_backtest_performance=trade_plan_performance,
            strategy_admission=admission,
            factor_diagnostics=factor_diagnostics,
            market_regime=market_regime,
            industry_strength=industry_strength,
            moneyflow_factors=moneyflow_factors,
        )

    output_path = Path(output_dir) / f"llm_strategy_research_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    suggestions_path: str | None = None
    requires_human_review = True
    if export_candidate_json:
        suggestions = extract_strategy_research_suggestions(markdown)
        requires_human_review = bool(suggestions.get("requires_human_review", True))
        suggestions_output = Path(output_dir) / f"strategy_research_suggestions_{resolved_date}.json"
        suggestions_output.write_text(json.dumps(suggestions, ensure_ascii=False, indent=2), encoding="utf-8")
        suggestions_path = str(suggestions_output)

    return {
        "strategy_research_report_path": str(output_path),
        "strategy_research_suggestions_path": suggestions_path,
        "exported_candidate_json": bool(export_candidate_json),
        "requires_human_review": requires_human_review,
    }


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _safe_llm_client():
    try:
        return get_llm_client("StrategyResearchAgent")
    except Exception:
        return DisabledLLMClient()


def _disabled_placeholder(report_date: str) -> str:
    return (
        "# LLM 策略研究建议报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "StrategyResearchAgent 只做策略研究建议，不做交易决策；"
        "不直接修改策略配置，不启用策略，不推荐买入股票，不执行交易。\n\n"
        "## 一、当前策略研究状态判断\n"
        "当前未调用 LLM，仅确认报告链路可用。\n\n"
        "## 八、下一轮研究计划\n"
        "所有建议必须经过回测、样本外验证、交易计划级回测；"
        "小样本结果不能用于实盘判断，Agent 建议不能直接进入正式策略配置。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StrategyResearchAgent and export research reports.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    parser.add_argument("--no-candidate-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_strategy_research_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
        export_candidate_json=not args.no_candidate_json,
    )
    for key in [
        "strategy_research_report_path",
        "strategy_research_suggestions_path",
        "exported_candidate_json",
        "requires_human_review",
    ]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
