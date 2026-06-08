from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.trading.positions import build_positions_from_trades, review_positions


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def build_positions(
    as_of_date: str | None = None,
    db_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)

    actual_trades = store.load_actual_trades()
    daily_bars = store.load_daily_bars()
    trade_plan = store.load_trade_plan()

    resolved_as_of_date = as_of_date or _latest_date(actual_trades, "trade_date")
    positions = build_positions_from_trades(
        actual_trades=actual_trades,
        daily_bars=daily_bars,
        as_of_date=resolved_as_of_date,
    )
    position_review = review_positions(
        positions=positions,
        trade_plan=trade_plan,
        as_of_date=resolved_as_of_date,
    )

    store.save_positions(positions)
    store.save_position_review(position_review)
    return positions, position_review


def _latest_date(df: pd.DataFrame, column: str) -> str | None:
    if df.empty or column not in df.columns:
        return None
    values = df[column].dropna()
    return str(values.max()) if not values.empty else None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current positions and T+1 risk review from local DuckDB data.")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolved_db_path = _resolve_db_path(args.db_path)
    store = StockAgentStore(resolved_db_path)
    actual_trades = store.load_actual_trades()
    resolved_as_of_date = args.as_of_date or _latest_date(actual_trades, "trade_date")

    positions, position_review = build_positions(
        as_of_date=resolved_as_of_date,
        db_path=resolved_db_path,
    )

    print(f"as_of_date: {resolved_as_of_date}")
    print(f"actual_trades 行数: {len(actual_trades)}")
    print(f"positions 行数: {len(positions)}")
    print(f"position_review 行数: {len(position_review)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条持仓:")
    print(positions.head(10).to_string(index=False))
    print("前 10 条风险检查结果:")
    print(position_review.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
