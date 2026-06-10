"""Run the low-risk LLM ReportAgent and export a Markdown summary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.llm_client import get_llm_client
from src.agents.report_agent import run_report_agent
from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.diagnostics.system_health import run_system_health_check


def run_report_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    system_health = run_system_health_check(db_path=resolved_db_path, reports_dir=output_dir)
    data_quality = _safe_load(store.load_data_quality_report)
    strategy_admission = _safe_load(store.load_strategy_admission)
    trade_plan = _safe_load(lambda: store.load_trade_plan(trade_date=report_date))
    candidate_pool = _safe_load(lambda: store.load_candidate_pool(trade_date=report_date))

    llm_client = get_llm_client("ReportAgent")
    markdown = run_report_agent(
        llm_client,
        system_health=system_health,
        data_quality=data_quality,
        strategy_admission=strategy_admission,
        trade_plan=trade_plan,
        candidate_pool=candidate_pool,
    )

    output_path = Path(output_dir) / f"llm_report_summary_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return str(output_path)


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReportAgent and export a Markdown summary.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = run_report_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"report_path: {output_path}")


if __name__ == "__main__":
    main()
