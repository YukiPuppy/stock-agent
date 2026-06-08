"""Run daily_bars data quality checks."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import settings
from src.data_quality.daily_bar_quality import (
    check_daily_bars_quality,
    check_enriched_daily_factors_quality,
    check_factor_diagnostics_quality,
    check_industry_strength_quality,
    check_moneyflow_quality,
)
from src.database.duckdb_store import StockAgentStore


def run_data_quality_check(db_path: str | None = None) -> pd.DataFrame:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    store = StockAgentStore(resolved_db_path)
    daily_bars = store.load_daily_bars()
    daily_factors = store.load_daily_factors()
    moneyflow = store.load_moneyflow()
    stock_industry_map = store.load_stock_industry_map()
    factor_diagnostics = store.load_factor_diagnostics()
    report = pd.concat(
        [
            check_daily_bars_quality(daily_bars),
            check_enriched_daily_factors_quality(daily_factors),
            check_moneyflow_quality(moneyflow, daily_factors),
            check_industry_strength_quality(stock_industry_map, daily_factors),
            check_factor_diagnostics_quality(factor_diagnostics, daily_factors),
        ],
        ignore_index=True,
    )
    store.save_data_quality_report(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local daily_bars data quality.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolved_db_path = args.db_path if args.db_path is not None else settings.DB_PATH
    store = StockAgentStore(resolved_db_path)
    daily_bars = store.load_daily_bars()
    daily_factors = store.load_daily_factors()
    moneyflow = store.load_moneyflow()
    stock_industry_map = store.load_stock_industry_map()
    factor_diagnostics = store.load_factor_diagnostics()
    report = pd.concat(
        [
            check_daily_bars_quality(daily_bars),
            check_enriched_daily_factors_quality(daily_factors),
            check_moneyflow_quality(moneyflow, daily_factors),
            check_industry_strength_quality(stock_industry_map, daily_factors),
            check_factor_diagnostics_quality(factor_diagnostics, daily_factors),
        ],
        ignore_index=True,
    )
    store.save_data_quality_report(report)
    error_count = int((report["status"] == "error").sum()) if not report.empty else 0
    warning_count = int((report["status"] == "warning").sum()) if not report.empty else 0

    print(f"daily_bars 行数: {len(daily_bars)}")
    print("检查结果表:")
    print(report.to_string(index=False))
    print(f"error 数量: {error_count}")
    print(f"warning 数量: {warning_count}")


if __name__ == "__main__":
    main()
