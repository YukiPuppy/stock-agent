"""Update the local A-share stock-basic universe."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import settings
from src.data_providers.factory import get_data_provider
from src.database.duckdb_store import StockAgentStore
from src.filters.universe_filter import filter_tradable_main_board
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _resolve_provider(provider: str | None) -> str:
    return str(provider if provider is not None else settings.DEFAULT_DATA_PROVIDER).strip().lower()


def _fetch_filter_and_save(
    db_path: str | None = None,
    provider: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    resolved_db_path = _resolve_db_path(db_path)
    data_provider = get_data_provider(_resolve_provider(provider))
    raw = data_provider.get_stock_basic()
    filtered = filter_tradable_main_board(raw)

    store = StockAgentStore(resolved_db_path)
    store.save_stock_basic(filtered)

    return raw, filtered, resolved_db_path


def update_stock_basic(db_path: str | None = None, provider: str | None = None) -> pd.DataFrame:
    """Fetch, filter, persist, and return the tradable main-board stock pool."""
    _, filtered, _ = _fetch_filter_and_save(db_path, provider=provider)
    return filtered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the local A-share stock-basic universe.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--provider",
        default=settings.DEFAULT_DATA_PROVIDER,
        help="默认数据源来自 DEFAULT_DATA_PROVIDER，当前推荐使用 tushare。",
    )
    return parser.parse_args()


def main() -> None:
    clear_proxy_env_for_process()
    args = _parse_args()
    raw, filtered, db_path = _fetch_filter_and_save(db_path=args.db_path, provider=args.provider)

    print(f"原始股票数量: {len(raw)}")
    print(f"过滤后股票数量: {len(filtered)}")
    print(f"保存数据库路径: {db_path}")
    print("前 10 条结果:")
    print(filtered.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
