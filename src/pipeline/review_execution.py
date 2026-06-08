from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.trading.execution_review import review_execution


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_execution_review(
    trade_date: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    actual_trades = store.load_actual_trades(trade_date=trade_date)
    trade_plan = store.load_trade_plan(trade_date=trade_date)

    result = review_execution(actual_trades, trade_plan)
    store.save_execution_review(result)
    return result


def _load_counts(trade_date: str | None, db_path: str) -> tuple[int, int]:
    store = StockAgentStore(db_path)
    return len(store.load_actual_trades(trade_date=trade_date)), len(store.load_trade_plan(trade_date=trade_date))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review actual trade execution against local trade plans.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolved_db_path = _resolve_db_path(args.db_path)
    actual_count, plan_count = _load_counts(args.trade_date, resolved_db_path)
    result = run_execution_review(trade_date=args.trade_date, db_path=resolved_db_path)

    print(f"actual_trades 行数: {actual_count}")
    print(f"trade_plan 行数: {plan_count}")
    print(f"execution_review 行数: {len(result)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条复盘结果:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
