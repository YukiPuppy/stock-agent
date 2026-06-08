from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.trading.period_review import generate_period_review


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def build_period_review(
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)

    actual_trades = store.load_actual_trades()
    execution_review = store.load_execution_review()
    trade_performance = store.load_actual_trade_performance()
    daily_review = store.load_daily_review()
    resolved_start, resolved_end = _resolve_period_from_actual(actual_trades, start_date, end_date)

    result = generate_period_review(
        actual_trades=actual_trades,
        execution_review=execution_review,
        trade_performance=trade_performance,
        daily_review=daily_review,
        start_date=resolved_start,
        end_date=resolved_end,
    )
    store.save_period_review(result)
    return result


def _resolve_period_from_actual(
    actual_trades: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str | None, str | None]:
    if actual_trades.empty or "trade_date" not in actual_trades.columns:
        return start_date, end_date
    values = actual_trades["trade_date"].dropna().astype(str)
    if values.empty:
        return start_date, end_date
    return start_date or str(values.min()), end_date or str(values.max())


def _filter_by_date(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if df.empty or "trade_date" not in df.columns:
        return df
    result = df.copy()
    dates = result["trade_date"].astype(str)
    if start_date is not None:
        result = result[dates >= str(start_date)]
        dates = result["trade_date"].astype(str)
    if end_date is not None:
        result = result[dates <= str(end_date)]
    return result.copy()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build period execution review from local DuckDB data.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolved_db_path = _resolve_db_path(args.db_path)
    store = StockAgentStore(resolved_db_path)
    actual_trades = store.load_actual_trades()
    resolved_start, resolved_end = _resolve_period_from_actual(actual_trades, args.start_date, args.end_date)
    execution_review = _filter_by_date(store.load_execution_review(), resolved_start, resolved_end)
    trade_performance = _filter_by_date(store.load_actual_trade_performance(), resolved_start, resolved_end)
    daily_review = _filter_by_date(store.load_daily_review(), resolved_start, resolved_end)
    filtered_actual = _filter_by_date(actual_trades, resolved_start, resolved_end)

    result = build_period_review(
        start_date=resolved_start,
        end_date=resolved_end,
        db_path=resolved_db_path,
    )

    print(f"start_date: {resolved_start or ''}")
    print(f"end_date: {resolved_end or ''}")
    print(f"actual_trades 行数: {len(filtered_actual)}")
    print(f"execution_review 行数: {len(execution_review)}")
    print(f"actual_trade_performance 行数: {len(trade_performance)}")
    print(f"daily_review 行数: {len(daily_review)}")
    print(f"period_review 行数: {len(result)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("周期复盘结果:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
