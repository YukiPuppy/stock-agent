"""Update SW industry daily bars."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config.settings import DB_PATH
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import StockAgentStore


def update_sw_daily(
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    sleep_seconds: float = 0.3,
    limit_days: int | None = None,
) -> tuple[pd.DataFrame, str]:
    resolved_db_path = db_path if db_path is not None else DB_PATH
    store = StockAgentStore(resolved_db_path)
    provider = get_data_provider("tushare")
    trade_dates = _trade_dates(store, start_date, end_date)
    if limit_days is not None:
        trade_dates = trade_dates[: int(limit_days)]

    frames = []
    for trade_date in trade_dates:
        daily = provider.get_sw_daily(trade_date=trade_date.replace("-", ""))
        frames.append(daily)
        print(f"{trade_date} sw_daily 行数: {len(daily)}")
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    store.save_sw_daily(result)
    return result, resolved_db_path


def _trade_dates(store: StockAgentStore, start_date: str, end_date: str) -> list[str]:
    start = _format_date(start_date)
    end = _format_date(end_date)
    calendar = store.load_trade_calendar(start_date=start, end_date=end)
    if not calendar.empty and "is_open" in calendar.columns:
        dates = calendar[calendar["is_open"].astype(int) == 1]["trade_date"].dropna().astype(str).tolist()
        return sorted(dict.fromkeys(dates))
    print("trade_calendar 为空，按自然日循环；建议先运行 update_trade_calendar。")
    return pd.date_range(start=start, end=end, freq="D").strftime("%Y-%m-%d").tolist()


def _format_date(value: str) -> str:
    text = str(value).strip()
    return pd.to_datetime(text, format="%Y%m%d" if text.isdigit() and len(text) == 8 else None).strftime("%Y-%m-%d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update SW industry daily bars.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--limit-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result, resolved_db_path = update_sw_daily(
        start_date=args.start_date,
        end_date=args.end_date,
        db_path=args.db_path,
        sleep_seconds=args.sleep_seconds,
        limit_days=args.limit_days,
    )
    print(f"sw_daily 合计行数: {len(result)}")
    print(f"保存数据库路径: {resolved_db_path}")


if __name__ == "__main__":
    main()
