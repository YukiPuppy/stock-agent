"""Build SW industry strength factors."""

from __future__ import annotations

import argparse

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.industry_strength import build_industry_strength


def build_and_save_industry_strength(db_path: str | None = None):
    resolved_db_path = db_path if db_path is not None else DB_PATH
    store = StockAgentStore(resolved_db_path)
    sw_daily = store.load_sw_daily()
    industry_strength = build_industry_strength(sw_daily)
    store.save_industry_strength(industry_strength)
    return industry_strength, len(sw_daily), resolved_db_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SW industry strength factors.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    industry_strength, sw_daily_count, resolved_db_path = build_and_save_industry_strength(args.db_path)
    print(f"sw_daily 行数: {sw_daily_count}")
    print(f"industry_strength 行数: {len(industry_strength)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 20 条行业强度:")
    print(industry_strength.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
