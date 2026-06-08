"""Update local A-share daily bars."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import settings
from src.data_providers.akshare_provider import DAILY_BAR_COLUMNS
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import StockAgentStore
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _resolve_provider(provider: str | None) -> str:
    return str(provider if provider is not None else settings.DEFAULT_DATA_PROVIDER).strip().lower()


def _fetch_and_save_daily_bars(
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0,
    provider: str | None = None,
) -> tuple[pd.DataFrame, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    resolved_provider = _resolve_provider(provider)
    data_provider = get_data_provider(resolved_provider)
    store = StockAgentStore(resolved_db_path)
    stock_basic = store.load_stock_basic()

    codes = stock_basic["code"].dropna().astype(str).tolist()
    if limit is not None:
        codes = codes[:limit]

    daily_bars = []
    failures: list[tuple[str, str]] = []

    for index, code in enumerate(codes, start=1):
        print(f"[{index}/{len(codes)}] fetching {code}")
        try:
            bars = data_provider.get_daily_bars(code, start_date, end_date)
        except Exception as exc:
            failures.append((code, str(exc)))
            print(f"[{index}/{len(codes)}] failed code={code} error={exc}")
        else:
            print(f"[{index}/{len(codes)}] success rows={len(bars)}")
            if not bars.empty:
                daily_bars.append(bars)

        if sleep_seconds > 0 and index < len(codes):
            time.sleep(sleep_seconds)

    if daily_bars:
        result = pd.concat(daily_bars, ignore_index=True)
        store.save_daily_bars(result)
    else:
        result = pd.DataFrame(columns=DAILY_BAR_COLUMNS)
        print("no daily bars fetched, nothing saved")

    success_count = len(codes) - len(failures)
    failure_count = len(failures)

    return result, len(codes), success_count, failure_count, resolved_db_path


def update_daily_bars(
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0,
    provider: str | None = None,
) -> pd.DataFrame:
    """Fetch, persist, and return daily bars for stocks in stock_basic."""
    result, _, success_count, failure_count, _ = _fetch_and_save_daily_bars(
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        limit=limit,
        sleep_seconds=sleep_seconds,
        provider=provider,
    )

    print(f"成功数量: {success_count}")
    print(f"失败数量: {failure_count}")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update local A-share daily bars.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    parser.add_argument(
        "--provider",
        default=settings.DEFAULT_DATA_PROVIDER,
        help="默认数据源来自 DEFAULT_DATA_PROVIDER，当前推荐使用 tushare。",
    )
    return parser.parse_args()


def main() -> None:
    clear_proxy_env_for_process()
    args = _parse_args()
    result, stock_count, success_count, failure_count, resolved_db_path = _fetch_and_save_daily_bars(
        start_date=args.start_date,
        end_date=args.end_date,
        db_path=args.db_path,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        provider=args.provider,
    )

    print(f"开始日期: {args.start_date}")
    print(f"结束日期: {args.end_date}")
    print(f"股票数量: {stock_count}")
    print(f"成功数量: {success_count}")
    print(f"失败数量: {failure_count}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条行情数据:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
