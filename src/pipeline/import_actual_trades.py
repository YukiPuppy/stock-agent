from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.trading.actual_trades import normalize_actual_trades


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def import_actual_trades(csv_path: str, db_path: str | None = None) -> pd.DataFrame:
    normalized = normalize_actual_trades(pd.read_csv(csv_path))
    store = StockAgentStore(_resolve_db_path(db_path))
    store.save_actual_trades(normalized)
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local actual trade CSV records.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolved_db_path = _resolve_db_path(args.db_path)
    result = import_actual_trades(args.csv_path, db_path=resolved_db_path)

    print(f"csv_path: {args.csv_path}")
    print(f"导入行数: {len(result)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条记录:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
