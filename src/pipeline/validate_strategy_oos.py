"""Run local walk-forward validation for parameter-search versions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.parameter_search import generate_search_versions, load_parameter_search_space
from src.research.walk_forward_validation import validate_strategy_versions_out_of_sample


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
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    daily_bars = store.load_daily_bars()
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(config)

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
    return validation


def _run_and_report(
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    config_path: str | None = None,
    db_path: str | None = None,
    min_valid_count_train: int = 30,
    min_valid_count_validation: int = 10,
) -> tuple[pd.DataFrame, int, int, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    daily_bars = store.load_daily_bars()
    config = load_parameter_search_space(config_path)
    versions = generate_search_versions(config)

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
