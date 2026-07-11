"""Run local parameter-search backtests without activating strategies."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence

import pandas as pd

from src.backtest.signal_backtester import backtest_strategy_signals, evaluate_strategy_performance, prepare_bars_by_code
from src.backtest.strategy_version_runner import generate_historical_signals_for_version
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.parameter_search import generate_search_versions, load_parameter_search_space
from src.research.strategy_version_evaluator import evaluate_strategy_versions
from src.pipeline.memory import collect_memory, load_factor_chunk, log_memory
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series


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
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
    run_id: str | None = None,
    materialize_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(
        config,
        limit_strategies=limit_strategies,
        limit_param_combinations=limit_param_combinations,
    )
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    prepared_bars = prepare_bars_by_code(daily_bars)
    del daily_bars
    gc.collect()

    log_memory("parameter_search", "loaded_inputs")
    performance_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    materialized_backtest = pd.DataFrame()
    backtest_frames: list[pd.DataFrame] = []
    backtest_count = 0
    daily_factors: pd.DataFrame | None = None
    loaded_strategy_name: str | None = None
    for index, version in enumerate(versions, start=1):
        name = str(version["strategy_name"])
        version_name = str(version["strategy_version"])
        if name != loaded_strategy_name:
            if daily_factors is not None:
                del daily_factors
                gc.collect()
            daily_factors = load_factor_chunk(store, [version], start_date, end_date)
            daily_factors[TRADE_DATE_KEY_COLUMN] = normalize_trade_date_series(daily_factors["trade_date"])
            loaded_strategy_name = name
            log_memory(f"parameter_search:{name}", "factor_chunk_loaded")
        stage = f"parameter_search:{name}:{version_name}"
        log_memory(stage, "before")
        historical_signals = generate_historical_signals_for_version(
            daily_factors=daily_factors,
            strategy_name=name,
            strategy_version=version_name,
            params=version.get("params", {}),
            start_date=start_date,
            end_date=end_date,
        )
        backtest_results = backtest_strategy_signals(
            historical_signals, pd.DataFrame(), prepared_bars_by_code=prepared_bars
        )
        performance = evaluate_strategy_performance(backtest_results)
        evaluation = evaluate_strategy_versions(
            performance,
            min_valid_count=min_valid_count,
            min_win_rate_3d=min_win_rate_3d,
            min_avg_return_3d=min_avg_return_3d,
            max_avg_drawdown_3d=max_avg_drawdown_3d,
        )
        if run_id is not None:
            backtest_results = backtest_results.assign(run_id=run_id)
            performance = performance.assign(run_id=run_id)
            evaluation = evaluation.assign(run_id=run_id)
        store.save_parameter_search_backtest_results(backtest_results)
        store.save_parameter_search_performance(performance)
        store.save_parameter_search_results(evaluation)
        backtest_count += len(backtest_results)
        performance_frames.append(performance)
        evaluation_frames.append(evaluation)
        if materialize_results:
            backtest_frames.append(backtest_results)
        del historical_signals, backtest_results, performance, evaluation
        gc.collect()
        log_memory(stage, f"after_written_{index}_of_{len(versions)}")

    if materialize_results:
        materialized_backtest = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else pd.DataFrame()
    performance = pd.concat(performance_frames, ignore_index=True) if performance_frames else pd.DataFrame()
    evaluation = pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame()
    if not evaluation.empty and "evaluation_score" in evaluation.columns:
        evaluation = evaluation.sort_values("evaluation_score", ascending=False).reset_index(drop=True)
    materialized_backtest.attrs["row_count"] = backtest_count
    del backtest_frames, performance_frames, evaluation_frames, prepared_bars
    if daily_factors is not None:
        del daily_factors
    collect_memory("parameter_search")
    return materialized_backtest, performance, evaluation


def _run_and_report(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors(start_date=start_date, end_date=end_date)
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(
        config,
        limit_strategies=limit_strategies,
        limit_param_combinations=limit_param_combinations,
    )

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
    parser.add_argument("--limit-strategies", type=int, default=None)
    parser.add_argument("--limit-param-combinations", type=int, default=None)
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
        limit_strategies=args.limit_strategies,
        limit_param_combinations=args.limit_param_combinations,
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
