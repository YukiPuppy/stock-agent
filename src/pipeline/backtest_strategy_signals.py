"""Run basic signal backtests from local DuckDB data."""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest.signal_backtester import (
    backtest_strategy_signals,
    evaluate_strategy_performance,
)
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_signal_backtest(
    db_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    strategy_signals = store.load_strategy_signals()
    daily_bars = store.load_daily_bars()

    backtest_results = backtest_strategy_signals(strategy_signals, daily_bars)
    strategy_performance = evaluate_strategy_performance(backtest_results)

    store.save_backtest_results(backtest_results)
    store.save_strategy_performance(strategy_performance)

    return backtest_results, strategy_performance


def _run_and_report(db_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    strategy_signals = store.load_strategy_signals()
    daily_bars = store.load_daily_bars()

    backtest_results = backtest_strategy_signals(strategy_signals, daily_bars)
    strategy_performance = evaluate_strategy_performance(backtest_results)

    store.save_backtest_results(backtest_results)
    store.save_strategy_performance(strategy_performance)

    return backtest_results, strategy_performance, len(strategy_signals), len(daily_bars), resolved_db_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest deterministic strategy signals.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backtest_results, strategy_performance, signal_count, bar_count, resolved_db_path = _run_and_report(args.db_path)

    print(f"strategy_signals 行数: {signal_count}")
    print(f"daily_bars 行数: {bar_count}")
    print(f"backtest_results 行数: {len(backtest_results)}")
    print(f"strategy_performance 行数: {len(strategy_performance)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("各策略表现:")
    print(strategy_performance.to_string(index=False))


if __name__ == "__main__":
    main()
