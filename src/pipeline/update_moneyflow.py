"""Update Tushare moneyflow data into local DuckDB."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.data_providers.tushare_provider import TushareProvider


def update_moneyflow(
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    sleep_seconds: float = 0.3,
    limit_days: int | None = None,
    provider: TushareProvider | None = None,
) -> tuple[pd.DataFrame, str]:
    resolved_db_path = db_path if db_path is not None else DB_PATH
    store = StockAgentStore(resolved_db_path)
    dates, used_calendar = _trade_dates(store, start_date, end_date)
    if limit_days is not None:
        dates = dates[:limit_days]
    if not used_calendar:
        print("trade_calendar 为空或无开市日期，按自然日循环；建议先运行 update_trade_calendar。")

    data_provider = provider if provider is not None else TushareProvider()
    frames = []
    total_rows = 0
    for index, trade_date in enumerate(dates):
        api_date = trade_date.replace("-", "")
        df = data_provider.get_moneyflow(trade_date=api_date)
        store.save_moneyflow(df)
        frames.append(df)
        total_rows += len(df)
        print(f"{trade_date} moneyflow 行数: {len(df)}")
        if sleep_seconds > 0 and index < len(dates) - 1:
            time.sleep(sleep_seconds)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"moneyflow 合计行数: {total_rows}")
    print(f"保存数据库路径: {resolved_db_path}")
    return result, resolved_db_path


def _trade_dates(store: StockAgentStore, start_date: str, end_date: str) -> tuple[list[str], bool]:
    start = _format_date(start_date)
    end = _format_date(end_date)
    try:
        calendar = store.load_trade_calendar(start_date=start, end_date=end)
    except Exception:
        calendar = pd.DataFrame()
    if not calendar.empty and {"trade_date", "is_open"} <= set(calendar.columns):
        open_dates = calendar[pd.to_numeric(calendar["is_open"], errors="coerce").fillna(0).astype(int) == 1]
        dates = sorted(open_dates["trade_date"].dropna().astype(str).unique().tolist())
        if dates:
            return dates, True
    return [date.strftime("%Y-%m-%d") for date in pd.date_range(start=start, end=end, freq="D")], False


def _format_date(value: str) -> str:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d").strftime("%Y-%m-%d")
    return pd.to_datetime(text).strftime("%Y-%m-%d")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Tushare moneyflow data.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--limit-days", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    update_moneyflow(
        start_date=args.start_date,
        end_date=args.end_date,
        db_path=args.db_path,
        sleep_seconds=args.sleep_seconds,
        limit_days=args.limit_days,
    )


if __name__ == "__main__":
    main()
