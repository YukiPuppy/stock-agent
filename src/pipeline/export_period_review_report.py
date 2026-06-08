from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.reports.period_review_report import generate_period_review_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_period_review_report(
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
) -> str:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    period_review = store.load_period_review(start_date=start_date, end_date=end_date)
    if not period_review.empty and (start_date is not None or end_date is not None):
        period_review = _select_matching_period(period_review, start_date, end_date)
    execution_review = _filter_by_date(store.load_execution_review(), start_date, end_date)
    trade_performance = _filter_by_date(store.load_actual_trade_performance(), start_date, end_date)
    resolved_start, resolved_end = _resolve_report_period(period_review, start_date, end_date)

    report = generate_period_review_report(
        period_review=period_review,
        execution_review=execution_review,
        trade_performance=trade_performance,
        start_date=resolved_start,
        end_date=resolved_end,
    )
    output_path = Path(output_dir) / f"period_review_{_format_date_for_filename(resolved_start)}_to_{_format_date_for_filename(resolved_end)}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _select_matching_period(
    period_review: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    result = period_review.copy()
    if start_date is not None and "start_date" in result.columns:
        result = result[result["start_date"].astype(str) == str(start_date)]
    if end_date is not None and "end_date" in result.columns:
        result = result[result["end_date"].astype(str) == str(end_date)]
    if result.empty:
        return period_review.head(1).copy()
    return result.head(1).copy()


def _filter_by_date(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if df.empty or "trade_date" not in df.columns:
        return df
    result = df.copy()
    dates = result["trade_date"].astype(str)
    if start_date is not None:
        result = result[dates >= str(start_date)]
        dates = result["trade_date"].astype(str)
    if end_date is not None:
        result = result[dates <= str(end_date)]
    return result.copy()


def _resolve_report_period(
    period_review: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    if not period_review.empty:
        row = period_review.iloc[0]
        return str(start_date or row.get("start_date") or "unknown"), str(end_date or row.get("end_date") or "unknown")
    return str(start_date or "unknown"), str(end_date or "unknown")


def _format_date_for_filename(value: str | None) -> str:
    if not value:
        return "unknown"
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export period execution review Markdown report.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = export_period_review_report(
        start_date=args.start_date,
        end_date=args.end_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
    )
    content = Path(output_path).read_text(encoding="utf-8")
    print(f"输出文件路径: {output_path}")
    print(f"报告字符数: {len(content)}")


if __name__ == "__main__":
    main()
