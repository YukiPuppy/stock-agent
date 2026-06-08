from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.trading.trade_performance import calculate_actual_trade_performance


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def build_trade_performance(
    trade_date: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)

    actual_trades = store.load_actual_trades()
    daily_bars = store.load_daily_bars()
    execution_review = store.load_execution_review()

    if trade_date is not None:
        actual_trades = _filter_by_trade_date(actual_trades, trade_date)
        execution_review = _filter_by_trade_date(execution_review, trade_date)

    result = calculate_actual_trade_performance(
        actual_trades=actual_trades,
        daily_bars=daily_bars,
        execution_review=execution_review,
    )
    store.save_actual_trade_performance(result)
    return result


def _filter_by_trade_date(df: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    if df.empty or trade_date is None or "trade_date" not in df.columns:
        return df
    return df[df["trade_date"].astype(str) == str(trade_date)].copy()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build actual trade performance review from local DuckDB data.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolved_db_path = _resolve_db_path(args.db_path)
    store = StockAgentStore(resolved_db_path)
    actual_trades = store.load_actual_trades()
    daily_bars = store.load_daily_bars()
    execution_review = store.load_execution_review()
    filtered_actual = _filter_by_trade_date(actual_trades, args.trade_date)
    filtered_execution = _filter_by_trade_date(execution_review, args.trade_date)

    result = build_trade_performance(trade_date=args.trade_date, db_path=resolved_db_path)

    print(f"使用交易日期: {args.trade_date or '全部'}")
    print(f"actual_trades 行数: {len(filtered_actual)}")
    print(f"daily_bars 行数: {len(daily_bars)}")
    print(f"execution_review 行数: {len(filtered_execution)}")
    print(f"actual_trade_performance 行数: {len(result)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条结果:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
