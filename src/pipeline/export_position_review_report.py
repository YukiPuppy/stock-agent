from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.reports.position_review_report import generate_position_review_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_position_review_report(
    as_of_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
) -> str:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    positions = store.load_positions(as_of_date=as_of_date)
    position_review = store.load_position_review(as_of_date=as_of_date)

    report_date = _resolve_report_date(positions, position_review, as_of_date)
    report = generate_position_review_report(
        positions=positions,
        position_review=position_review,
        as_of_date=report_date,
    )

    output_path = Path(output_dir) / f"position_review_{_format_date_for_filename(report_date)}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _resolve_report_date(*items) -> str | None:
    as_of_date = items[-1]
    if as_of_date:
        return str(as_of_date)
    for df in items[:-1]:
        if df is not None and not df.empty and "as_of_date" in df.columns:
            values = df["as_of_date"].dropna()
            if not values.empty:
                return str(values.max())
    return None


def _format_date_for_filename(as_of_date: str | None) -> str:
    if not as_of_date:
        return "unknown"
    text = str(as_of_date)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export position review Markdown report.")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = export_position_review_report(
        as_of_date=args.as_of_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
    )
    content = Path(output_path).read_text(encoding="utf-8")
    print(f"输出文件路径: {output_path}")
    print(f"报告字符数: {len(content)}")


if __name__ == "__main__":
    main()
