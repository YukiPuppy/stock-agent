from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.trading.daily_review import generate_daily_review


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def build_daily_review(
    trade_date: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)

    actual_trades_all = store.load_actual_trades()
    resolved_trade_date = trade_date or _latest_trade_date(actual_trades_all)
    actual_trades = _filter_by_trade_date(actual_trades_all, resolved_trade_date)
    execution_review = store.load_execution_review(trade_date=resolved_trade_date)
    trade_plan = store.load_trade_plan(trade_date=resolved_trade_date)
    actual_trade_performance = _load_actual_trade_performance(store, resolved_trade_date)

    daily_review = generate_daily_review(
        actual_trades=actual_trades,
        execution_review=execution_review,
        trade_plan=trade_plan,
        trade_date=resolved_trade_date,
        actual_trade_performance=actual_trade_performance,
    )
    store.save_daily_review(daily_review)
    return daily_review


def _latest_trade_date(df: pd.DataFrame) -> str | None:
    if df.empty or "trade_date" not in df.columns:
        return None
    values = df["trade_date"].dropna()
    if values.empty:
        return None
    return str(values.max())


def _filter_by_trade_date(df: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    if df.empty or trade_date is None or "trade_date" not in df.columns:
        return df
    return df[df["trade_date"].astype(str) == str(trade_date)].copy()


def _load_actual_trade_performance(store: StockAgentStore, trade_date: str | None) -> pd.DataFrame:
    try:
        return store.load_actual_trade_performance(trade_date=trade_date)
    except Exception:
        return pd.DataFrame()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local daily post-market execution review.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolved_db_path = _resolve_db_path(args.db_path)
    store = StockAgentStore(resolved_db_path)
    actual_trades_all = store.load_actual_trades()
    resolved_trade_date = args.trade_date or _latest_trade_date(actual_trades_all)
    actual_trades = _filter_by_trade_date(actual_trades_all, resolved_trade_date)
    execution_review = store.load_execution_review(trade_date=resolved_trade_date)

    result = build_daily_review(trade_date=resolved_trade_date, db_path=resolved_db_path)

    print(f"使用交易日期: {resolved_trade_date or '暂无可用交易日期'}")
    print(f"actual_trades 行数: {len(actual_trades)}")
    print(f"execution_review 行数: {len(execution_review)}")
    print(f"daily_review 行数: {len(result)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("复盘结果:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
