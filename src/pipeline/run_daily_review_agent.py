"""Run DailyReviewAgent and export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.daily_review_agent import run_daily_review_agent
from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.config import settings
from src.database.duckdb_store import StockAgentStore


def run_daily_review_agent_pipeline(
    trade_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or trade_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    actual_trades = _safe_load(lambda: store.load_actual_trades(trade_date=trade_date))
    execution_review = _safe_load(lambda: store.load_execution_review(trade_date=trade_date))
    daily_review = _safe_load(lambda: store.load_daily_review(trade_date=trade_date))
    period_review = _safe_load(store.load_period_review)
    actual_trade_performance = _safe_load(lambda: store.load_actual_trade_performance(trade_date=trade_date))
    positions = _safe_load(lambda: store.load_positions(as_of_date=trade_date))
    position_review = _safe_load(lambda: store.load_position_review(as_of_date=trade_date))

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_llm_placeholder(resolved_date, trade_date)
    else:
        markdown = run_daily_review_agent(
            llm_client,
            actual_trades=actual_trades,
            execution_review=execution_review,
            daily_review=daily_review,
            period_review=period_review,
            actual_trade_performance=actual_trade_performance,
            positions=positions,
            position_review=position_review,
        )

    output_path = Path(output_dir) / f"llm_daily_review_{resolved_date}.md"
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
        return get_llm_client("DailyReviewAgent")
    except Exception:
        return DisabledLLMClient()


def _disabled_llm_placeholder(report_date: str, trade_date: str | None = None) -> str:
    target_date = trade_date or report_date
    return (
        "# LLM 每日执行复盘报告\n\n"
        f"报告日期：{report_date}\n\n"
        f"复盘交易日：{target_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "DailyReviewAgent 只做交易复盘和执行纪律分析，不做交易决策；"
        "不选股、不调参、不启用策略、不执行交易。\n\n"
        "## 八、下一交易日执行纪律建议\n"
        "仅供人工复核参考，不构成交易指令。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DailyReviewAgent and export a Markdown report.")
    parser.add_argument("--trade-date", default=None, help="Trade date in YYYY-MM-DD format.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_daily_review_agent_pipeline(
        trade_date=args.trade_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
