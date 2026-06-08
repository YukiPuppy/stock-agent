"""Build moneyflow derived factors from local moneyflow table."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.factors.moneyflow_factors import build_moneyflow_factors


def build_and_save_moneyflow_factors(db_path: str | None = None) -> pd.DataFrame:
    resolved_db_path = db_path if db_path is not None else DB_PATH
    store = StockAgentStore(resolved_db_path)
    moneyflow = store.load_moneyflow()
    factors = build_moneyflow_factors(moneyflow)
    store.save_moneyflow_factors(factors)
    return factors


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build moneyflow derived factors.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    resolved_db_path = args.db_path if args.db_path is not None else DB_PATH
    factors = build_and_save_moneyflow_factors(args.db_path)
    print(f"moneyflow_factors 行数: {len(factors)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 20 条资金流因子:")
    print(factors.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
