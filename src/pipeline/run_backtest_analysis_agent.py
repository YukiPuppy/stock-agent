"""Run BacktestAnalysisAgent and export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.backtest_analysis_agent import run_backtest_analysis_agent
from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.config import settings
from src.database.duckdb_store import StockAgentStore


def run_backtest_analysis_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    strategy_evaluation = _safe_load(store.load_strategy_version_evaluation)
    parameter_results = _safe_load(store.load_parameter_search_results)
    walk_forward = _safe_load(store.load_walk_forward_validation)
    trade_plan_performance = _safe_load(store.load_trade_plan_backtest_performance)
    admission = _safe_load(store.load_strategy_admission)

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_placeholder(resolved_date)
    else:
        markdown = run_backtest_analysis_agent(
            llm_client,
            strategy_evaluation=strategy_evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward,
            trade_plan_backtest_performance=trade_plan_performance,
            strategy_admission=admission,
        )

    output_path = Path(output_dir) / f"llm_backtest_analysis_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return str(output_path)


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _safe_llm_client():
    try:
        return get_llm_client()
    except Exception:
        return DisabledLLMClient()


def _disabled_placeholder(report_date: str) -> str:
    return (
        f"# LLM 回测分析报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用，未调用远程模型。本占位报告用于看板和文件链路测试。\n\n"
        "BacktestAnalysisAgent 只分析程序计算出的回测结果，不做交易决策，"
        "不选股，不自动调参，不启用策略，不执行交易。\n"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BacktestAnalysisAgent and export a Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_backtest_analysis_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
