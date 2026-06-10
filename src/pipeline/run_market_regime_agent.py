"""Run MarketRegimeAgent and export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.agents.market_regime_agent import run_market_regime_agent
from src.config import settings
from src.database.duckdb_store import StockAgentStore


def run_market_regime_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    market_regime = _safe_load(store.load_market_regime)
    index_daily = _safe_load(store.load_index_daily)
    limit_list_daily = _safe_load(store.load_limit_list_daily)
    candidate_pool = _safe_load(lambda: store.load_candidate_pool(trade_date=report_date))
    trade_plan = _safe_load(lambda: store.load_trade_plan(trade_date=report_date))

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_llm_placeholder(resolved_date)
    else:
        markdown = run_market_regime_agent(
            llm_client,
            market_regime=market_regime,
            index_daily=index_daily,
            limit_list_daily=limit_list_daily,
            candidate_pool=candidate_pool,
            trade_plan=trade_plan,
        )

    output_path = Path(output_dir) / f"llm_market_regime_{resolved_date}.md"
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
        return get_llm_client("MarketRegimeAgent")
    except Exception:
        return DisabledLLMClient()


def _disabled_llm_placeholder(report_date: str) -> str:
    return (
        "# LLM 市场环境解释报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "MarketRegimeAgent 只做市场环境解释，不做交易决策；"
        "不选股、不调参、不启用策略、不执行交易。\n\n"
        "## 七、下一步观察建议\n"
        "仅供人工复核参考，不构成交易指令。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MarketRegimeAgent and export a Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_market_regime_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
