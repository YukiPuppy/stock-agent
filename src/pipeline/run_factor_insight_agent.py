"""Run FactorInsightAgent and export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.factor_insight_agent import run_factor_insight_agent
from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.research.factor_diagnostics import build_factor_diagnostics


def run_factor_insight_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    factor_diagnostics = _safe_load(store.load_factor_diagnostics)
    daily_factors = _safe_load(store.load_daily_factors)
    candidate_pool = _safe_load(lambda: store.load_candidate_pool(trade_date=report_date))
    trade_plan = _safe_load(lambda: store.load_trade_plan(trade_date=report_date))
    if factor_diagnostics.empty:
        factor_diagnostics = build_factor_diagnostics(
            daily_factors=daily_factors,
            candidate_pool=candidate_pool,
            trade_plan=trade_plan,
        )
        _safe_save_factor_diagnostics(store, factor_diagnostics)

    strategy_admission = _safe_load(store.load_strategy_admission)
    trade_plan_backtest_performance = _safe_load(store.load_trade_plan_backtest_performance)

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_llm_placeholder(resolved_date)
    else:
        markdown = run_factor_insight_agent(
            llm_client,
            factor_diagnostics=factor_diagnostics,
            daily_factors=daily_factors,
            candidate_pool=candidate_pool,
            trade_plan=trade_plan,
            strategy_admission=strategy_admission,
            trade_plan_backtest_performance=trade_plan_backtest_performance,
        )

    output_path = Path(output_dir) / f"llm_factor_insight_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return str(output_path)


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _safe_save_factor_diagnostics(store: StockAgentStore, df: pd.DataFrame) -> None:
    try:
        store.save_factor_diagnostics(df)
    except Exception:
        return


def _safe_llm_client():
    try:
        return get_llm_client("FactorInsightAgent")
    except Exception:
        return DisabledLLMClient()


def _disabled_llm_placeholder(report_date: str) -> str:
    return (
        "# LLM 因子诊断报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "FactorInsightAgent 只做因子诊断和研究建议，不做交易决策；"
        "不选股、不调参、不启用策略、不执行交易。\n\n"
        "## 一、因子覆盖情况概览\n"
        "当前未调用 LLM，仅确认报告链路可用。\n\n"
        "## 七、下一轮因子研究建议\n"
        "当前诊断不等于因子有效性证明；因子是否有效需要通过回测、样本外验证和交易计划级回测确认。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FactorInsightAgent and export a Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_factor_insight_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
