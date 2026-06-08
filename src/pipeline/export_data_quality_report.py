"""Export data quality checks as a Markdown report."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.config import settings
from src.data_quality.provider_compare import summarize_provider_compare
from src.database.duckdb_store import StockAgentStore
from src.reports.data_quality_report import generate_data_quality_report


def export_data_quality_report(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
) -> str:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)
    quality_report = store.load_data_quality_report()
    compare_result = store.load_provider_compare_result()
    compare_summary = summarize_provider_compare(compare_result)
    report = generate_data_quality_report(
        quality_report=quality_report,
        compare_result=compare_result,
        compare_summary=compare_summary,
        report_date=resolved_date,
    )
    output_path = Path(output_dir) / f"data_quality_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export data quality report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = export_data_quality_report(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"Data quality report exported: {output_path}")


if __name__ == "__main__":
    main()

