from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline.build_daily_review import build_daily_review
from src.pipeline.build_positions import build_positions
from src.pipeline.build_trade_performance import build_trade_performance
from src.pipeline.export_daily_review_report import export_daily_review_report
from src.pipeline.export_position_review_report import export_position_review_report
from src.pipeline.export_trade_performance_report import export_trade_performance_report
from src.pipeline.review_execution import run_execution_review
from src.pipeline.run_daily_review_agent import run_daily_review_agent_pipeline


def run_after_market_review(
    trade_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
    export_reports: bool = True,
    run_llm_daily_review: bool = False,
) -> dict:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    actual_trades_all = store.load_actual_trades()
    resolved_trade_date = trade_date or _latest_trade_date(actual_trades_all)
    actual_trades = _filter_by_trade_date(actual_trades_all, resolved_trade_date)

    summary = _empty_summary(
        trade_date=resolved_trade_date,
        db_path=resolved_db_path,
        actual_trades_rows=len(actual_trades),
    )

    if resolved_trade_date is None:
        return summary

    execution_review = run_execution_review(trade_date=resolved_trade_date, db_path=resolved_db_path)
    trade_performance = build_trade_performance(trade_date=resolved_trade_date, db_path=resolved_db_path)
    daily_review = build_daily_review(trade_date=resolved_trade_date, db_path=resolved_db_path)
    positions, position_review = build_positions(as_of_date=resolved_trade_date, db_path=resolved_db_path)

    summary.update(
        {
            "execution_review_rows": _row_count(execution_review),
            "actual_trade_performance_rows": _row_count(trade_performance),
            "daily_review_rows": _row_count(daily_review),
            "positions_rows": _row_count(positions),
            "position_review_rows": _row_count(position_review),
        }
    )

    if export_reports:
        try:
            summary["daily_review_report_path"] = export_daily_review_report(
                trade_date=resolved_trade_date,
                db_path=resolved_db_path,
                output_dir=output_dir,
            )
            summary["trade_performance_report_path"] = export_trade_performance_report(
                trade_date=resolved_trade_date,
                db_path=resolved_db_path,
                output_dir=output_dir,
            )
            summary["position_review_report_path"] = export_position_review_report(
                as_of_date=resolved_trade_date,
                db_path=resolved_db_path,
                output_dir=output_dir,
            )
        except Exception as exc:
            raise RuntimeError(f"导出盘后复盘报告失败: {exc}") from exc

    if run_llm_daily_review:
        try:
            summary["llm_daily_review_report_path"] = run_daily_review_agent_pipeline(
                trade_date=resolved_trade_date,
                db_path=resolved_db_path,
                output_dir=output_dir,
                report_date=resolved_trade_date,
            )
        except Exception as exc:
            raise RuntimeError(f"生成 LLM 每日执行复盘报告失败: {exc}") from exc

    return summary


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _latest_trade_date(df: pd.DataFrame) -> str | None:
    if df.empty or "trade_date" not in df.columns:
        return None
    values = df["trade_date"].dropna()
    if values.empty:
        return None
    return str(values.max())


def _filter_by_trade_date(df: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    if df.empty or trade_date is None or "trade_date" not in df.columns:
        return df
    return df[df["trade_date"].astype(str) == str(trade_date)].copy()


def _row_count(value: object) -> int:
    if value is None:
        return 0
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _empty_summary(trade_date: str | None, db_path: str, actual_trades_rows: int = 0) -> dict:
    return {
        "trade_date": trade_date,
        "db_path": db_path,
        "actual_trades_rows": actual_trades_rows,
        "execution_review_rows": 0,
        "actual_trade_performance_rows": 0,
        "daily_review_rows": 0,
        "positions_rows": 0,
        "position_review_rows": 0,
        "daily_review_report_path": None,
        "trade_performance_report_path": None,
        "position_review_report_path": None,
        "llm_daily_review_report_path": None,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local after-market trade review workflow.")
    parser.add_argument("--trade-date", default=None, help="Trade date in YYYY-MM-DD format.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--no-report", action="store_true", help="Do not export Markdown reports.")
    parser.add_argument("--run-llm-daily-review", action="store_true", help="Run LLM DailyReviewAgent.")
    return parser.parse_args(argv)


def _print_summary(summary: dict) -> None:
    print("After-market review finished.")
    for key in (
        "trade_date",
        "db_path",
        "actual_trades_rows",
        "execution_review_rows",
        "actual_trade_performance_rows",
        "daily_review_rows",
        "positions_rows",
        "position_review_rows",
        "daily_review_report_path",
        "trade_performance_report_path",
        "position_review_report_path",
        "llm_daily_review_report_path",
    ):
        print(f"{key}: {summary.get(key)}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_after_market_review(
        trade_date=args.trade_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
        export_reports=not args.no_report,
        run_llm_daily_review=args.run_llm_daily_review,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
