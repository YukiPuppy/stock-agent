"""Export strategy admission reports to Markdown files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.reports.strategy_admission_report import generate_strategy_admission_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_strategy_admission_report(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
    admission: pd.DataFrame | None = None,
    run_id: str | None = None,
) -> str:
    resolved_report_date = report_date or date.today().isoformat()
    store = StockAgentStore(_resolve_db_path(db_path))
    if admission is None:
        admission = store.load_strategy_admission(run_id=run_id)
    report = generate_strategy_admission_report(admission, report_date=resolved_report_date)

    output_path = Path(output_dir) / f"strategy_admission_{resolved_report_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export strategy admission Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None, help="Optional report date, format YYYY-MM-DD.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = export_strategy_admission_report(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"报告已写入: {output_path}")


if __name__ == "__main__":
    main()
