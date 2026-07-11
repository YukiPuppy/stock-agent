"""Run strategy-version comparison backtests from local DuckDB data."""

from __future__ import annotations

import argparse
import gc

import pandas as pd

from src.backtest.signal_backtester import backtest_strategy_signals, evaluate_strategy_performance, prepare_bars_by_code
from src.backtest.strategy_version_runner import generate_historical_signals_for_version
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.strategy.strategy_versions import iter_strategy_versions, load_strategy_versions
from src.pipeline.memory import collect_memory, load_factor_chunk, log_memory
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series


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
    config = load_strategy_versions(config_path)
    versions = iter_strategy_versions(config)
    if limit_strategies is not None:
        versions = versions[: int(limit_strategies)]
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    prepared_bars = prepare_bars_by_code(daily_bars)
    del daily_bars
    gc.collect()

    log_memory("strategy_version_backtest", "loaded_inputs")
    performance_frames: list[pd.DataFrame] = []
    materialized_backtest = pd.DataFrame()
    materialized_signals = pd.DataFrame()
    backtest_count = 0
    signal_count = 0
    daily_factors: pd.DataFrame | None = None
    loaded_strategy_name: str | None = None
    if not versions:
        daily_factors = load_factor_chunk(store, [], start_date, end_date)
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
            log_memory(f"strategy_version_backtest:{name}", "factor_chunk_loaded")
        stage = f"strategy_version_backtest:{name}:{version_name}"
        log_memory(stage, "before")
        strategy_signals = generate_historical_signals_for_version(
            daily_factors=daily_factors,
            strategy_name=name,
            strategy_version=version_name,
            params=version.get("params", {}),
            start_date=start_date,
            end_date=end_date,
        )
        backtest_results = backtest_strategy_signals(
            strategy_signals, pd.DataFrame(), prepared_bars_by_code=prepared_bars
        )
        performance = evaluate_strategy_performance(backtest_results)
        if run_id is not None:
            backtest_results = backtest_results.assign(run_id=run_id)
            performance = performance.assign(run_id=run_id)
            store.save_research_strategy_signals(strategy_signals, run_id)
        store.save_backtest_results(backtest_results)
        store.save_strategy_version_performance(performance)
        signal_count += len(strategy_signals)
        backtest_count += len(backtest_results)
        performance_frames.append(performance)
        # Preserve the historical direct-call API for the common single-version case only.
        if len(versions) == 1:
            materialized_backtest = backtest_results
            materialized_signals = strategy_signals
        del strategy_signals, backtest_results, performance
        gc.collect()
        log_memory(stage, f"after_written_{index}_of_{len(versions)}")

    strategy_version_performance = (
        pd.concat(performance_frames, ignore_index=True) if performance_frames else pd.DataFrame()
    )
    materialized_backtest.attrs["row_count"] = backtest_count
    materialized_signals.attrs["row_count"] = signal_count
    del performance_frames, prepared_bars
    if daily_factors is not None:
        del daily_factors
    collect_memory("strategy_version_backtest")

    if return_signals:
        return materialized_backtest, strategy_version_performance, materialized_signals
    return materialized_backtest, strategy_version_performance


def _run_and_report(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
    limit_strategies: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    config = load_strategy_versions(config_path)
    versions = iter_strategy_versions(config)
    if limit_strategies is not None:
        versions = versions[: int(limit_strategies)]
    backtest_results, strategy_version_performance, strategy_signals = run_strategy_version_backtest(
        start_date=start_date,
        end_date=end_date,
        config_path=config_path,
        db_path=resolved_db_path,
        limit_strategies=limit_strategies,
        return_signals=True,
    )
    store = StockAgentStore(resolved_db_path)
    factor_conditions, factor_params = [], []
    bar_conditions, bar_params = [], []
    if start_date is not None:
        factor_conditions.append("trade_date >= ?")
        factor_params.append(start_date.replace("-", ""))
        bar_conditions.append("trade_date >= ?")
        bar_params.append(start_date)
    if end_date is not None:
        factor_conditions.append("trade_date <= ?")
        factor_params.append(end_date.replace("-", ""))
        bar_conditions.append("trade_date <= ?")
        bar_params.append(end_date)
    with store._connect() as con:
        store._create_tables(con)
        factor_where = f"WHERE {' AND '.join(factor_conditions)}" if factor_conditions else ""
        bar_where = f"WHERE {' AND '.join(bar_conditions)}" if bar_conditions else ""
        daily_factors_count = int(con.execute(f"SELECT COUNT(*) FROM daily_factors {factor_where}", factor_params).fetchone()[0])
        daily_bars_count = int(con.execute(f"SELECT COUNT(*) FROM daily_bars {bar_where}", bar_params).fetchone()[0])

    return (
        backtest_results,
        strategy_version_performance,
        daily_factors_count,
        daily_bars_count,
        len(versions),
        int(strategy_signals.attrs.get("row_count", len(strategy_signals))),
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
