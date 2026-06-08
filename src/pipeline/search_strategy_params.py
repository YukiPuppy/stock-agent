"""Run local parameter-search backtests without activating strategies."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.backtest.signal_backtester import backtest_strategy_signals, evaluate_strategy_performance
from src.backtest.strategy_version_runner import generate_historical_signals_for_versions
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.parameter_search import generate_search_versions, load_parameter_search_space
from src.research.strategy_version_evaluator import evaluate_strategy_versions


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_parameter_search(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    daily_bars = store.load_daily_bars()
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(config)

    historical_signals = generate_historical_signals_for_versions(
        daily_factors=daily_factors,
        versions=versions,
        start_date=start_date,
        end_date=end_date,
    )
    backtest_results = backtest_strategy_signals(historical_signals, daily_bars)
    performance = evaluate_strategy_performance(backtest_results)
    evaluation = evaluate_strategy_versions(
        performance,
        min_valid_count=min_valid_count,
        min_win_rate_3d=min_win_rate_3d,
        min_avg_return_3d=min_avg_return_3d,
        max_avg_drawdown_3d=max_avg_drawdown_3d,
    )

    store.save_parameter_search_backtest_results(backtest_results)
    store.save_parameter_search_performance(performance)
    store.save_parameter_search_results(evaluation)

    return backtest_results, performance, evaluation


def _run_and_report(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    daily_bars = store.load_daily_bars()
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(config)

    historical_signals = generate_historical_signals_for_versions(
        daily_factors=daily_factors,
        versions=versions,
        start_date=start_date,
        end_date=end_date,
    )
    backtest_results = backtest_strategy_signals(historical_signals, daily_bars)
    performance = evaluate_strategy_performance(backtest_results)
    evaluation = evaluate_strategy_versions(
        performance,
        min_valid_count=min_valid_count,
        min_win_rate_3d=min_win_rate_3d,
        min_avg_return_3d=min_avg_return_3d,
        max_avg_drawdown_3d=max_avg_drawdown_3d,
    )

    store.save_parameter_search_backtest_results(backtest_results)
    store.save_parameter_search_performance(performance)
    store.save_parameter_search_results(evaluation)

    return (
        backtest_results,
        performance,
        evaluation,
        len(daily_factors),
        len(daily_bars),
        len(versions),
        len(historical_signals),
        resolved_db_path,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local strategy parameter search.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--min-valid-count", type=int, default=30)
    parser.add_argument("--min-win-rate-3d", type=float, default=0.50)
    parser.add_argument("--min-avg-return-3d", type=float, default=0.0)
    parser.add_argument("--max-avg-drawdown-3d", type=float, default=-0.08)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    (
        backtest_results,
        performance,
        evaluation,
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
        min_valid_count=args.min_valid_count,
        min_win_rate_3d=args.min_win_rate_3d,
        min_avg_return_3d=args.min_avg_return_3d,
        max_avg_drawdown_3d=args.max_avg_drawdown_3d,
    )

    print(f"daily_factors 行数: {daily_factors_count}")
    print(f"daily_bars 行数: {daily_bars_count}")
    print(f"参数组合数量: {version_count}")
    print(f"historical signals 行数: {signal_count}")
    print(f"backtest_results 行数: {len(backtest_results)}")
    print(f"performance 行数: {len(performance)}")
    print(f"evaluation 行数: {len(evaluation)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("参数搜索评价前 20:")
    if evaluation.empty:
        print("无参数搜索评价结果。")
    else:
        print(evaluation.sort_values("evaluation_score", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
