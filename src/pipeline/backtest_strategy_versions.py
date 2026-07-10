"""Run strategy-version comparison backtests from local DuckDB data."""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest.signal_backtester import backtest_strategy_signals, evaluate_strategy_performance
from src.backtest.strategy_version_runner import generate_historical_signals_for_versions
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.strategy.strategy_versions import iter_strategy_versions, load_strategy_versions


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_strategy_version_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
    limit_strategies: int | None = None,
    run_id: str | None = None,
    return_signals: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors(start_date=start_date, end_date=end_date)
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    config = load_strategy_versions(config_path)
    versions = iter_strategy_versions(config)
    if limit_strategies is not None:
        versions = versions[: int(limit_strategies)]

    strategy_signals = generate_historical_signals_for_versions(
        daily_factors=daily_factors,
        versions=versions,
        start_date=start_date,
        end_date=end_date,
    )
    backtest_results = backtest_strategy_signals(strategy_signals, daily_bars)
    strategy_version_performance = evaluate_strategy_performance(backtest_results)

    if run_id is not None:
        strategy_signals = strategy_signals.assign(run_id=run_id)
        backtest_results = backtest_results.assign(run_id=run_id)
        strategy_version_performance = strategy_version_performance.assign(run_id=run_id)

    store.save_backtest_results(backtest_results)
    store.save_strategy_version_performance(strategy_version_performance)

    if return_signals:
        return backtest_results, strategy_version_performance, strategy_signals
    return backtest_results, strategy_version_performance


def _run_and_report(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
    limit_strategies: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors(start_date=start_date, end_date=end_date)
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    config = load_strategy_versions(config_path)
    versions = iter_strategy_versions(config)
    if limit_strategies is not None:
        versions = versions[: int(limit_strategies)]

    strategy_signals = generate_historical_signals_for_versions(
        daily_factors=daily_factors,
        versions=versions,
        start_date=start_date,
        end_date=end_date,
    )
    backtest_results = backtest_strategy_signals(strategy_signals, daily_bars)
    strategy_version_performance = evaluate_strategy_performance(backtest_results)

    store.save_backtest_results(backtest_results)
    store.save_strategy_version_performance(strategy_version_performance)

    return (
        backtest_results,
        strategy_version_performance,
        len(daily_factors),
        len(daily_bars),
        len(versions),
        len(strategy_signals),
        resolved_db_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest configured strategy versions.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--limit-strategies", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    (
        backtest_results,
        strategy_version_performance,
        daily_factors_count,
        daily_bars_count,
        version_count,
        signal_count,
        resolved_db_path,
    ) = _run_and_report(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config_path,
        db_path=args.db_path,
        limit_strategies=args.limit_strategies,
    )

    print(f"daily_factors 行数: {daily_factors_count}")
    print(f"daily_bars 行数: {daily_bars_count}")
    print(f"策略版本数量: {version_count}")
    print(f"historical signals 行数: {signal_count}")
    print(f"backtest_results 行数: {len(backtest_results)}")
    print(f"strategy_version_performance 行数: {len(strategy_version_performance)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("策略版本表现表:")
    print(strategy_version_performance.to_string(index=False))


if __name__ == "__main__":
    main()
