"""Update local Tushare daily_basic data."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import settings
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import DAILY_BASIC_COLUMNS, StockAgentStore
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _yyyymmdd(value: str) -> str:
    return str(value).replace("-", "")


def _trade_dates(store: StockAgentStore, start_date: str, end_date: str, limit_days: int | None) -> list[str]:
    calendar = store.load_trade_calendar(start_date=_format_date(start_date), end_date=_format_date(end_date))
    if calendar.empty:
        print("trade_calendar 为空，按自然日循环；建议先更新 trade_calendar。")
        dates = pd.date_range(_format_date(start_date), _format_date(end_date)).strftime("%Y-%m-%d").tolist()
    else:
        dates = calendar[calendar["is_open"].astype(int) == 1]["trade_date"].dropna().astype(str).tolist()
    return dates[:limit_days] if limit_days is not None else dates


def _format_date(value: str) -> str:
    text = str(value).strip()
    return pd.to_datetime(text, format="%Y%m%d").strftime("%Y-%m-%d") if text.isdigit() and len(text) == 8 else str(pd.to_datetime(text).date())


def update_daily_basic(
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
        df = provider.get_daily_basic(trade_date=_yyyymmdd(trade_date))
        print(f"{trade_date} daily_basic 行数: {len(df)}")
        if not df.empty:
            frames.append(df)
        if sleep_seconds > 0 and index < len(dates):
            time.sleep(sleep_seconds)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_BASIC_COLUMNS)
    store.save_daily_basic(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update local Tushare daily_basic data.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--limit-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    clear_proxy_env_for_process()
    args = _parse_args()
    result = update_daily_basic(args.start_date, args.end_date, args.db_path, args.sleep_seconds, args.limit_days)
    print(f"daily_basic 合计行数: {len(result)}")
    print(f"保存数据库路径: {_resolve_db_path(args.db_path)}")


if __name__ == "__main__":
    main()
