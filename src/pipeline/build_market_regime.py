"""Build local market regime scores."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.research.market_regime import build_market_regime as build_market_regime_frame


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def build_market_regime(db_path: str | None = None) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    index_daily = store.load_index_daily()
    limit_list_daily = store.load_limit_list_daily()
    result = build_market_regime_frame(index_daily=index_daily, limit_list_daily=limit_list_daily)
    store.save_market_regime(result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build market regime table from local DuckDB data.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    result = build_market_regime(args.db_path)
    print(f"market_regime 行数: {len(result)}")
    print(f"保存数据库路径: {_resolve_db_path(args.db_path)}")
    print("前 20 行:")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
