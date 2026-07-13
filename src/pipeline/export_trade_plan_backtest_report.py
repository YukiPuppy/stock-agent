from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.pipeline.rerun_trade_plan_and_admission import export_trade_plan_backtest_report_low_memory
from src.reports.trade_plan_backtest_report import generate_trade_plan_backtest_report


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def export_trade_plan_backtest_report(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
    backtest_results: pd.DataFrame | None = None,
    performance: pd.DataFrame | None = None,
    run_id: str | None = None,
) -> str:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    if backtest_results is None:
        backtest_results = store.load_trade_plan_backtest_results(run_id=run_id)
    if performance is None:
        performance = store.load_trade_plan_backtest_performance(run_id=run_id)
    if run_id is not None and int(backtest_results.attrs.get("row_count", 0)) > 0 and backtest_results.empty:
        return export_trade_plan_backtest_report_low_memory(store, run_id, performance, output_dir)
    resolved_report_date = report_date or _resolve_report_date(backtest_results)
    report = generate_trade_plan_backtest_report(backtest_results, performance, report_date=resolved_report_date)
    output_path = Path(output_dir) / f"trade_plan_backtest_{_format_date_for_filename(resolved_report_date)}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _resolve_report_date(backtest_results) -> str | None:
    if backtest_results is not None and not backtest_results.empty and "plan_date" in backtest_results.columns:
        values = backtest_results["plan_date"].dropna()
        if not values.empty:
            return str(values.max())
    return None


def _format_date_for_filename(report_date: str | None) -> str:
    if not report_date:
        return "unknown"
    text = str(report_date)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trade-plan backtest Markdown report.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = export_trade_plan_backtest_report(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
    )
    print(f"输出文件路径: {output_path}")
    print(f"报告字符数: {len(Path(output_path).read_text(encoding='utf-8'))}")


if __name__ == "__main__":
    main()
