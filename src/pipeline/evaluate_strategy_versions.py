"""Evaluate strategy-version performance from local DuckDB results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.strategy_version_evaluator import (
    build_active_strategy_config,
    evaluate_strategy_versions,
)
from src.pipeline.memory import collect_memory, log_memory


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_strategy_version_evaluation(
    db_path: str | None = None,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
    performance: pd.DataFrame | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    # A workflow run always evaluates the small, persisted aggregate for its run_id;
    # it never retains or consumes the full backtest result frame.
    log_memory("strategy_version_evaluation", "before_read_aggregate")
    if performance is None or run_id is not None:
        performance = store.load_strategy_version_performance(run_id=run_id)
    evaluation = evaluate_strategy_versions(
        performance,
        min_valid_count=min_valid_count,
        min_win_rate_3d=min_win_rate_3d,
        min_avg_return_3d=min_avg_return_3d,
        max_avg_drawdown_3d=max_avg_drawdown_3d,
    )
    if run_id is not None:
        evaluation = evaluation.assign(run_id=run_id)
    store.save_strategy_version_evaluation(evaluation)
    del performance
    collect_memory("strategy_version_evaluation")
    return evaluation


def export_active_strategy_config(evaluation: pd.DataFrame, active_config_path: str) -> dict:
    config = build_active_strategy_config(evaluation)
    path = Path(active_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def _run_and_report(
    db_path: str | None = None,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
) -> tuple[pd.DataFrame, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    performance = store.load_strategy_version_performance()
    evaluation = evaluate_strategy_versions(
        performance,
        min_valid_count=min_valid_count,
        min_win_rate_3d=min_win_rate_3d,
        min_avg_return_3d=min_avg_return_3d,
        max_avg_drawdown_3d=max_avg_drawdown_3d,
    )
    store.save_strategy_version_evaluation(evaluation)
    return evaluation, len(performance), resolved_db_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strategy-version backtest performance.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--min-valid-count", type=int, default=30)
    parser.add_argument("--min-win-rate-3d", type=float, default=0.50)
    parser.add_argument("--min-avg-return-3d", type=float, default=0.0)
    parser.add_argument("--max-avg-drawdown-3d", type=float, default=-0.08)
    parser.add_argument("--export-active-config", action="store_true")
    parser.add_argument("--active-config-path", default="configs/active_strategies.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluation, performance_count, resolved_db_path = _run_and_report(
        db_path=args.db_path,
        min_valid_count=args.min_valid_count,
        min_win_rate_3d=args.min_win_rate_3d,
        min_avg_return_3d=args.min_avg_return_3d,
        max_avg_drawdown_3d=args.max_avg_drawdown_3d,
    )

    print(f"strategy_version_performance 行数: {performance_count}")
    print(f"strategy_version_evaluation 行数: {len(evaluation)}")
    print(f"保存数据库路径: {resolved_db_path}")
    if args.export_active_config:
        export_active_strategy_config(evaluation, args.active_config_path)
        print(f"观察启用建议配置已导出: {args.active_config_path}")
    print("策略版本评估表:")
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
