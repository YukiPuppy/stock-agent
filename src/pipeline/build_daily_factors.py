"""Build local A-share daily technical factors."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.factors.technical_factors import compute_daily_factors


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def _build_and_save_daily_factors(db_path: str | None = None) -> tuple[pd.DataFrame, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_bars = store.load_daily_bars()
    daily_factors = compute_daily_factors(daily_bars)
    store.save_daily_factors(daily_factors)
    return daily_factors, len(daily_bars), resolved_db_path


def build_daily_factors(db_path: str | None = None) -> pd.DataFrame:
    """Read daily bars, compute technical factors, persist them, and return the result."""
    daily_factors, _, _ = _build_and_save_daily_factors(db_path)
    return daily_factors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local A-share daily technical factors.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    daily_factors, daily_bars_count, resolved_db_path = _build_and_save_daily_factors(args.db_path)

    print(f"daily_bars 行数: {daily_bars_count}")
    print(f"daily_factors 行数: {len(daily_factors)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条因子数据:")
    print(daily_factors.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
