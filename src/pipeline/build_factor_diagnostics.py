"""Build and persist factor diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.research.factor_diagnostics import build_factor_diagnostics


def run_build_factor_diagnostics(db_path: str | None = None) -> pd.DataFrame:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    store = StockAgentStore(resolved_db_path)
    diagnostics = build_factor_diagnostics(
        daily_factors=_safe_load(store.load_daily_factors),
        candidate_pool=_safe_load(store.load_candidate_pool),
        trade_plan=_safe_load(store.load_trade_plan),
    )
    store.save_factor_diagnostics(diagnostics)
    return diagnostics


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factor diagnostics.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    diagnostics = run_build_factor_diagnostics(db_path=args.db_path)
    print(diagnostics.head(30).to_string(index=False))
    print(f"row_count: {len(diagnostics)}")


if __name__ == "__main__":
    main()
