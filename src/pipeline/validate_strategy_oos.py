"""Run local walk-forward validation for parameter-search versions."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence

import pandas as pd

from src.backtest.signal_backtester import prepare_bars_by_code
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.parameter_search import generate_search_versions, load_parameter_search_space
from src.research.walk_forward_validation import validate_strategy_versions_out_of_sample
from src.pipeline.memory import collect_memory, load_factor_chunk, log_memory
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_oos_validation(
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    config_path: str | None = None,
    db_path: str | None = None,
    min_valid_count_train: int = 30,
    min_valid_count_validation: int = 10,
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(
        config,
        limit_strategies=limit_strategies,
        limit_param_combinations=limit_param_combinations,
    )
    daily_bars = store.load_daily_bars(start_date=train_start_date, end_date=validation_end_date)
    bar_dates = daily_bars["trade_date"].astype(str).str.replace(r"\D", "", regex=True)
    train_start_key, train_end_key = train_start_date.replace("-", ""), train_end_date.replace("-", "")
    validation_start_key = validation_start_date.replace("-", "")
    validation_end_key = validation_end_date.replace("-", "")
    prepared_bars_by_period = {
        "train": prepare_bars_by_code(
            daily_bars[(bar_dates >= train_start_key) & (bar_dates <= train_end_key)].copy()
        ),
        "validation": prepare_bars_by_code(
            daily_bars[(bar_dates >= validation_start_key) & (bar_dates <= validation_end_key)].copy()
        ),
    }
    del daily_bars, bar_dates
    gc.collect()

    log_memory("oos_validation", "loaded_inputs")
    frames: list[pd.DataFrame] = []
    daily_factors: pd.DataFrame | None = None
    loaded_strategy_name: str | None = None
    for index, version in enumerate(versions, start=1):
        strategy_name = str(version["strategy_name"])
        if strategy_name != loaded_strategy_name:
            if daily_factors is not None:
                del daily_factors
                gc.collect()
            daily_factors = load_factor_chunk(store, [version], train_start_date, validation_end_date)
            daily_factors[TRADE_DATE_KEY_COLUMN] = normalize_trade_date_series(daily_factors["trade_date"])
            loaded_strategy_name = strategy_name
            log_memory(f"oos_validation:{strategy_name}", "factor_chunk_loaded")
        stage = f"oos_validation:{version['strategy_name']}:{version['strategy_version']}"
        log_memory(stage, "before")
        validation = validate_strategy_versions_out_of_sample(
            daily_factors=daily_factors,
            daily_bars=pd.DataFrame(),
            versions=[version],
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            min_valid_count_train=min_valid_count_train,
            min_valid_count_validation=min_valid_count_validation,
            prepared_bars_by_period=prepared_bars_by_period,
        )
        if run_id is not None:
            validation = validation.assign(run_id=run_id)
        store.save_walk_forward_validation(validation)
        frames.append(validation)
        del validation
        gc.collect()
        log_memory(stage, f"after_written_{index}_of_{len(versions)}")
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty and "stability_score" in result.columns:
        result = result.sort_values("stability_score", ascending=False).reset_index(drop=True)
    del frames, prepared_bars_by_period
    if daily_factors is not None:
        del daily_factors
    collect_memory("oos_validation")
    return result


def _run_and_report(
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    config_path: str | None = None,
    db_path: str | None = None,
    min_valid_count_train: int = 30,
    min_valid_count_validation: int = 10,
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
) -> tuple[pd.DataFrame, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors(start_date=train_start_date, end_date=validation_end_date)
    daily_bars = store.load_daily_bars(start_date=train_start_date, end_date=validation_end_date)
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(
        config,
        limit_strategies=limit_strategies,
        limit_param_combinations=limit_param_combinations,
    )

    validation = validate_strategy_versions_out_of_sample(
        daily_factors=daily_factors,
        daily_bars=daily_bars,
        versions=versions,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        min_valid_count_train=min_valid_count_train,
        min_valid_count_validation=min_valid_count_validation,
    )
    store.save_walk_forward_validation(validation)
    return validation, len(daily_factors), len(daily_bars), len(versions), resolved_db_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local walk-forward validation.")
    parser.add_argument("--train-start-date", required=True)
    parser.add_argument("--train-end-date", required=True)
    parser.add_argument("--validation-start-date", required=True)
    parser.add_argument("--validation-end-date", required=True)
    parser.add_argument("--config-path", default="configs/parameter_search_space.json")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--min-valid-count-train", type=int, default=30)
    parser.add_argument("--min-valid-count-validation", type=int, default=10)
    parser.add_argument("--limit-strategies", type=int, default=None)
    parser.add_argument("--limit-param-combinations", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    validation, daily_factors_count, daily_bars_count, version_count, resolved_db_path = _run_and_report(
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
        config_path=args.config_path,
        db_path=args.db_path,
        min_valid_count_train=args.min_valid_count_train,
        min_valid_count_validation=args.min_valid_count_validation,
        limit_strategies=args.limit_strategies,
        limit_param_combinations=args.limit_param_combinations,
    )

    print(f"daily_factors 行数: {daily_factors_count}")
    print(f"daily_bars 行数: {daily_bars_count}")
    print(f"参数版本数量: {version_count}")
    print(f"walk_forward_validation 行数: {len(validation)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("样本外验证前 20:")
    if validation.empty:
        print("无样本外验证结果。")
    else:
        print(validation.sort_values("stability_score", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
