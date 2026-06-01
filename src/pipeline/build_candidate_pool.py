"""Build deterministic A-share candidate pools."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.strategy.candidate_selector import select_candidates


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def _build_and_save_candidate_pool(
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
    db_path: str | None = None,
) -> tuple[pd.DataFrame, int, str, str | None]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    stock_basic = store.load_stock_basic()
    candidate_pool = select_candidates(
        daily_factors=daily_factors,
        stock_basic=stock_basic,
        trade_date=trade_date,
        top_n=top_n,
        min_amount_ma5=min_amount_ma5,
    )
    store.save_candidate_pool(candidate_pool)

    used_trade_date = trade_date
    if used_trade_date is None and not daily_factors.empty and "trade_date" in daily_factors.columns:
        trade_dates = daily_factors["trade_date"].dropna()
        if not trade_dates.empty:
            used_trade_date = str(trade_dates.max())

    return candidate_pool, len(daily_factors), resolved_db_path, used_trade_date


def build_candidate_pool(
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Read factors and stock basics, build candidates, persist them, and return the result."""
    candidate_pool, _, _, _ = _build_and_save_candidate_pool(
        trade_date=trade_date,
        top_n=top_n,
        min_amount_ma5=min_amount_ma5,
        db_path=db_path,
    )
    return candidate_pool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic A-share candidate pool.")
    parser.add_argument("--trade-date", default=None, help="Optional trade date, format YYYY-MM-DD.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-amount-ma5", type=float, default=100000000.0)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidate_pool, daily_factors_count, resolved_db_path, used_trade_date = _build_and_save_candidate_pool(
        trade_date=args.trade_date,
        top_n=args.top_n,
        min_amount_ma5=args.min_amount_ma5,
        db_path=args.db_path,
    )

    print(f"使用交易日期: {used_trade_date}")
    print(f"daily_factors 行数: {daily_factors_count}")
    print(f"candidate_pool 行数: {len(candidate_pool)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 20 条候选股:")
    print(candidate_pool.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
