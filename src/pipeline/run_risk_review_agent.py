"""Run RiskReviewAgent and export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.agents.risk_review_agent import run_risk_review_agent
from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.diagnostics.system_health import run_system_health_check


def run_risk_review_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    system_health = _safe_system_health(resolved_db_path, output_dir)
    data_quality_report = _safe_load(store.load_data_quality_report)
    strategy_admission = _safe_load(store.load_strategy_admission)
    trade_plan = _safe_load(lambda: store.load_trade_plan(trade_date=report_date))
    position_review = _safe_load(lambda: store.load_position_review(as_of_date=report_date))
    execution_review = _safe_load(lambda: store.load_execution_review(trade_date=report_date))
    daily_review = _safe_load(lambda: store.load_daily_review(trade_date=report_date))
    period_review = _safe_load(store.load_period_review)

    llm_client = get_llm_client("RiskReviewAgent")
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_llm_placeholder(resolved_date)
    else:
        markdown = run_risk_review_agent(
            llm_client,
            system_health=system_health,
            data_quality_report=data_quality_report,
            strategy_admission=strategy_admission,
            trade_plan=trade_plan,
            position_review=position_review,
            execution_review=execution_review,
            daily_review=daily_review,
            period_review=period_review,
        )

    output_path = Path(output_dir) / f"llm_risk_review_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return str(output_path)


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _safe_system_health(db_path: str, output_dir: str) -> dict:
    try:
        return run_system_health_check(db_path=db_path, reports_dir=output_dir)
    except Exception as exc:
        return {"status": "unavailable", "message": str(exc)}


def _disabled_llm_placeholder(report_date: str) -> str:
    return (
        "# LLM 风险审查报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "## 一、总体风险结论\n"
        "暂无 LLM 风险审查结论，请启用并配置 LLM 后重新生成。\n\n"
        "## 八、下一步风险控制建议\n"
        "仅供人工复核参考，不构成交易指令。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RiskReviewAgent and export a Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_risk_review_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
