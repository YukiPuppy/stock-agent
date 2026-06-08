"""Build deterministic A-share next-day trade plans."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.strategy.trade_plan_generator import generate_trade_plan


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def _build_and_save_trade_plan(
    trade_date: str | None = None,
    max_items: int = 5,
    db_path: str | None = None,
) -> tuple[pd.DataFrame, int, str, str | None]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    candidate_pool = store.load_candidate_pool()

    used_trade_date = trade_date
    if used_trade_date is None and not candidate_pool.empty and "trade_date" in candidate_pool.columns:
        trade_dates = candidate_pool["trade_date"].dropna()
        if not trade_dates.empty:
            used_trade_date = str(trade_dates.max())

    if used_trade_date is not None:
        selected_candidates = candidate_pool[candidate_pool["trade_date"] == used_trade_date].copy()
    else:
        selected_candidates = candidate_pool.copy()

    trade_plan = generate_trade_plan(selected_candidates, max_items=max_items)
    store.save_trade_plan(trade_plan)
    return trade_plan, len(selected_candidates), resolved_db_path, used_trade_date


def build_trade_plan(
    trade_date: str | None = None,
    max_items: int = 5,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Read candidates, build trade plans, persist them, and return the result."""
    trade_plan, _, _, _ = _build_and_save_trade_plan(
        trade_date=trade_date,
        max_items=max_items,
        db_path=db_path,
    )
    return trade_plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic A-share next-day trade plan.")
    parser.add_argument("--trade-date", default=None, help="Optional trade date, format YYYY-MM-DD.")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trade_plan, candidate_count, resolved_db_path, used_trade_date = _build_and_save_trade_plan(
        trade_date=args.trade_date,
        max_items=args.max_items,
        db_path=args.db_path,
    )

    print(f"使用交易日期: {used_trade_date}")
    print(f"candidate_pool 行数: {candidate_count}")
    print(f"trade_plan 行数: {len(trade_plan)}")
    print(f"包含 strategy_versions: {'是' if 'strategy_versions' in trade_plan.columns else '否'}")
    print(f"包含 recommendations: {'是' if 'recommendations' in trade_plan.columns else '否'}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("交易计划详情:")
    print(trade_plan.to_string(index=False))


if __name__ == "__main__":
    main()
