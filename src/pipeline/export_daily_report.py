"""Export deterministic daily trade-plan reports to Markdown files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.reports.daily_report import generate_daily_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_daily_report(
    trade_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
) -> str:
    """Read persisted plan data, render Markdown, write it, and return the file path."""
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    trade_plan = store.load_trade_plan(trade_date=trade_date)
    candidate_pool = store.load_candidate_pool(trade_date=trade_date)
    report = generate_daily_report(
        trade_plan=trade_plan,
        candidate_pool=candidate_pool,
        trade_date=trade_date,
    )

    report_date = _resolve_report_date(trade_plan, candidate_pool, trade_date)
    file_date = _format_date_for_filename(report_date)
    output_path = Path(output_dir) / f"daily_report_{file_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _resolve_report_date(trade_plan, candidate_pool, trade_date: str | None) -> str | None:
    if trade_date:
        return str(trade_date)
    for df in (trade_plan, candidate_pool):
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
    parser = argparse.ArgumentParser(description="Export daily A-share trade-plan Markdown report.")
    parser.add_argument("--trade-date", default=None, help="Optional trade date, format YYYY-MM-DD.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = export_daily_report(
        trade_date=args.trade_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
    )
    content = Path(output_path).read_text(encoding="utf-8")
    report_date = _extract_report_date(content)
    print(f"报告日期: {report_date}")
    print(f"输出文件路径: {output_path}")
    print(f"报告字符数: {len(content)}")


def _extract_report_date(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("报告日期："):
            return line.removeprefix("报告日期：").strip()
    return "暂无可用交易日期"


if __name__ == "__main__":
    main()
