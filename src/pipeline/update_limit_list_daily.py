"""Update local Tushare limit_list_d data."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import pandas as pd

from src.config import settings
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import LIMIT_LIST_DAILY_COLUMNS, StockAgentStore
from src.pipeline.update_daily_basic import _format_date, _trade_dates, _yyyymmdd
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def update_limit_list_daily(
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    sleep_seconds: float = 0.3,
    limit_days: int | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    provider = get_data_provider("tushare")
    frames = []
    dates = _trade_dates(store, start_date, end_date, limit_days)
    for index, trade_date in enumerate(dates, start=1):
        df = provider.get_limit_list_daily(trade_date=_yyyymmdd(trade_date))
        print(f"{trade_date} limit_list_daily 行数: {len(df)}")
        if not df.empty:
            frames.append(df)
        if sleep_seconds > 0 and index < len(dates):
            time.sleep(sleep_seconds)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LIMIT_LIST_DAILY_COLUMNS)
    store.save_limit_list_daily(result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update local Tushare limit_list_d data.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--limit-days", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    clear_proxy_env_for_process()
    args = _parse_args()
    result = update_limit_list_daily(args.start_date, args.end_date, args.db_path, args.sleep_seconds, args.limit_days)
    print(f"limit_list_daily 合计行数: {len(result)}")
    print(f"日期范围: {_format_date(args.start_date)} - {_format_date(args.end_date)}")
    print(f"保存数据库路径: {_resolve_db_path(args.db_path)}")


if __name__ == "__main__":
    main()
