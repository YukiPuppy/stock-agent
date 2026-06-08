"""Optional diagnostic comparison for a small daily_bars sample from two providers."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import settings
from src.data_providers.factory import get_data_provider
from src.data_quality.provider_compare import COMPARE_COLUMNS, compare_daily_bars, summarize_provider_compare
from src.database.duckdb_store import StockAgentStore


def run_provider_compare(
    code: str,
    start_date: str,
    end_date: str,
    left_provider: str = "akshare",
    right_provider: str = "tushare",
    db_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = get_data_provider(left_provider).get_daily_bars(code, start_date, end_date)
    right = get_data_provider(right_provider).get_daily_bars(code, start_date, end_date)
    compare_result = compare_daily_bars(left, right, left_name=left_provider, right_name=right_provider)
    summary = summarize_provider_compare(compare_result)

    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    result_to_save = compare_result if not compare_result.empty else pd.DataFrame(columns=COMPARE_COLUMNS)
    StockAgentStore(resolved_db_path).save_provider_compare_result(result_to_save)
    return compare_result, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally diagnose daily-bar differences between two providers.")
    parser.add_argument("--code", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--left-provider", default="akshare")
    parser.add_argument("--right-provider", default="tushare")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("当前流程会访问配置的数据源，仅作为可选诊断工具，不是主工作流必需步骤。")
    compare_result, summary = run_provider_compare(
        code=args.code,
        start_date=args.start_date,
        end_date=args.end_date,
        left_provider=args.left_provider,
        right_provider=args.right_provider,
        db_path=args.db_path,
    )
    print(f"异常明细行数: {len(compare_result)}")
    print("汇总:")
    print(summary.to_string(index=False) if not summary.empty else "暂无异常。")
    if not compare_result.empty:
        print("前 50 条异常明细:")
        print(compare_result.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
