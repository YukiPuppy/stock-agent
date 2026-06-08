"""Run IndustryInsightAgent and export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.industry_insight_agent import run_industry_insight_agent
from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.config import settings
from src.database.duckdb_store import StockAgentStore


def run_industry_insight_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    industry_strength = _safe_load(store.load_industry_strength)
    sw_daily = _safe_load(store.load_sw_daily)
    stock_industry_map = _safe_load(store.load_stock_industry_map)
    candidate_pool = _safe_load(lambda: store.load_candidate_pool(trade_date=report_date))
    trade_plan = _safe_load(lambda: store.load_trade_plan(trade_date=report_date))

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_llm_placeholder(resolved_date)
    else:
        markdown = run_industry_insight_agent(
            llm_client,
            industry_strength=industry_strength,
            sw_daily=sw_daily,
            stock_industry_map=stock_industry_map,
            candidate_pool=candidate_pool,
            trade_plan=trade_plan,
        )

    output_path = Path(output_dir) / f"llm_industry_insight_{resolved_date}.md"
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


def _disabled_llm_placeholder(report_date: str) -> str:
    return (
        "# LLM 行业洞察报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "IndustryInsightAgent 只做行业强弱解释，不做交易决策；"
        "不选股、不调参、不启用策略、不执行交易。\n\n"
        "## 七、下一步研究建议\n"
        "仅供人工复核参考，不构成交易指令。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IndustryInsightAgent and export a Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_industry_insight_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
