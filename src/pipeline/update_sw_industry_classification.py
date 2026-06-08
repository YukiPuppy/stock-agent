"""Update SW industry classification and best-effort stock industry map."""

from __future__ import annotations

import argparse

from src.config.settings import DB_PATH
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import StockAgentStore
from src.research.stock_industry_map import build_stock_industry_map


def update_sw_industry_classification(
    level: str = "L1",
    src: str = "SW2021",
    db_path: str | None = None,
):
    resolved_db_path = db_path if db_path is not None else DB_PATH
    provider = get_data_provider("tushare")
    store = StockAgentStore(resolved_db_path)
    sw_classification = provider.get_sw_industry_classification(level=level, src=src)
    store.save_sw_industry_classification(sw_classification)
    stock_basic = store.load_stock_basic()
    stock_industry_map = build_stock_industry_map(stock_basic, sw_classification)
    store.save_stock_industry_map(stock_industry_map)
    return sw_classification, stock_industry_map, resolved_db_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update SW industry classification and stock industry map.")
    parser.add_argument("--level", default="L1")
    parser.add_argument("--src", default="SW2021")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sw_classification, stock_industry_map, resolved_db_path = update_sw_industry_classification(
        level=args.level,
        src=args.src,
        db_path=args.db_path,
    )
    print(f"sw_industry_classification 行数: {len(sw_classification)}")
    print(f"stock_industry_map 行数: {len(stock_industry_map)}")
    print(f"保存数据库路径: {resolved_db_path}")


if __name__ == "__main__":
    main()
