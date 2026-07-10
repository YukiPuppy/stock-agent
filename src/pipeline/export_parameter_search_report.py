"""Export parameter-search reports to Markdown files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.reports.parameter_search_report import generate_parameter_search_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_parameter_search_report(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
    evaluation: pd.DataFrame | None = None,
    performance: pd.DataFrame | None = None,
    run_id: str | None = None,
) -> str:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    if evaluation is None:
        evaluation = store.load_parameter_search_results(run_id=run_id)
    if performance is None:
        performance = store.load_parameter_search_performance(run_id=run_id)
    resolved_report_date = report_date or date.today().isoformat()
    report = generate_parameter_search_report(
        evaluation=evaluation,
        performance=performance,
        report_date=resolved_report_date,
    )

    output_path = Path(output_dir) / f"parameter_search_{resolved_report_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export parameter-search Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None, help="Optional report date, format YYYY-MM-DD.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolved_db_path = _resolve_db_path(args.db_path)
    store = StockAgentStore(resolved_db_path)
    evaluation = store.load_parameter_search_results()
    performance = store.load_parameter_search_performance()
    output_path = export_parameter_search_report(
        db_path=resolved_db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    content = Path(output_path).read_text(encoding="utf-8")
    print(f"parameter_search_results 行数: {len(evaluation)}")
    print(f"parameter_search_performance 行数: {len(performance)}")
    print(f"输出文件路径: {output_path}")
    print(f"报告字符数: {len(content)}")


if __name__ == "__main__":
    main()
