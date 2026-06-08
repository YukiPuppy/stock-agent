"""Update local Tushare index daily bars."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import pandas as pd

from src.config import settings
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import INDEX_DAILY_COLUMNS, StockAgentStore
from src.pipeline.update_daily_basic import _format_date
from src.utils.network import clear_proxy_env_for_process


DEFAULT_INDEX_CODES = ["000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH"]


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def update_index_daily(
    start_date: str,
    end_date: str,
    index_codes: Sequence[str] | None = None,
    db_path: str | None = None,
    sleep_seconds: float = 0.3,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    provider = get_data_provider("tushare")
    frames = []
    codes = list(index_codes or DEFAULT_INDEX_CODES)
    for index, index_code in enumerate(codes, start=1):
        df = provider.get_index_daily(index_code=index_code, start_date=start_date, end_date=end_date)
        print(f"{index_code} index_daily 行数: {len(df)}")
        if not df.empty:
            frames.append(df)
        if sleep_seconds > 0 and index < len(codes):
            time.sleep(sleep_seconds)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_DAILY_COLUMNS)
    StockAgentStore(resolved_db_path).save_index_daily(result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update local Tushare index_daily data.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--index-codes", default=",".join(DEFAULT_INDEX_CODES))
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    return parser.parse_args(argv)


def main() -> None:
    clear_proxy_env_for_process()
    args = _parse_args()
    index_codes = [item.strip().upper() for item in str(args.index_codes).split(",") if item.strip()]
    result = update_index_daily(args.start_date, args.end_date, index_codes, args.db_path, args.sleep_seconds)
    print(f"index_daily 合计行数: {len(result)}")
    print(f"日期范围: {_format_date(args.start_date)} - {_format_date(args.end_date)}")
    print(f"保存数据库路径: {_resolve_db_path(args.db_path)}")


if __name__ == "__main__":
    main()
