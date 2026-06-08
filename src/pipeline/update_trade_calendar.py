"""Update local Tushare trade calendar."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import settings
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import StockAgentStore
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def update_trade_calendar(
    start_date: str,
    end_date: str,
    exchange: str = "SSE",
    db_path: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    provider = get_data_provider("tushare")
    result = provider.get_trade_calendar(start_date=start_date, end_date=end_date, exchange=exchange)
    StockAgentStore(resolved_db_path).save_trade_calendar(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update local Tushare trade calendar.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--exchange", default="SSE")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    clear_proxy_env_for_process()
    args = _parse_args()
    result = update_trade_calendar(args.start_date, args.end_date, args.exchange, args.db_path)
    print(f"trade_calendar 行数: {len(result)}")
    print(f"日期范围: {args.start_date} - {args.end_date}")
    print(f"交易所: {args.exchange}")
    print(f"保存数据库路径: {_resolve_db_path(args.db_path)}")


if __name__ == "__main__":
    main()
