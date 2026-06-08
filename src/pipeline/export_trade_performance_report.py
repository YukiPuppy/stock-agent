from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.reports.trade_performance_report import generate_trade_performance_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_trade_performance_report(
    trade_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
) -> str:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    trade_performance = store.load_actual_trade_performance(trade_date=trade_date)
    daily_review = store.load_daily_review(trade_date=trade_date)

    report = generate_trade_performance_report(
        trade_performance=trade_performance,
        daily_review=daily_review,
        trade_date=trade_date,
    )
    report_date = _resolve_report_date(trade_performance, daily_review, trade_date)
    output_path = Path(output_dir) / f"trade_performance_{_format_date_for_filename(report_date)}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _resolve_report_date(*items) -> str | None:
    trade_date = items[-1]
    if trade_date:
        return str(trade_date)
    for df in items[:-1]:
        if df is not None and not df.empty and "trade_date" in df.columns:
            values = df["trade_date"].dropna()
            if not values.empty:
                return str(values.max())
    return None


def _format_date_for_filename(trade_date: str | None) -> str:
    if not trade_date:
        return "unknown"
    text = str(trade_date)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export actual trade performance Markdown report.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = export_trade_performance_report(
        trade_date=args.trade_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
    )
    content = Path(output_path).read_text(encoding="utf-8")
    print(f"输出文件路径: {output_path}")
    print(f"报告字符数: {len(content)}")


if __name__ == "__main__":
    main()
