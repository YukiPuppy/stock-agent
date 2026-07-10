"""Build strategy admission recommendations from local DuckDB research tables."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.research.strategy_admission import build_active_strategy_candidate_config, build_strategy_admission


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_strategy_admission(
    db_path: str | None = None,
    export_candidate_config: bool = False,
    candidate_config_path: str = "configs/active_strategies_candidate.json",
    strategy_evaluation: pd.DataFrame | None = None,
    parameter_search_results: pd.DataFrame | None = None,
    walk_forward_validation: pd.DataFrame | None = None,
    trade_plan_backtest_performance: pd.DataFrame | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)

    if strategy_evaluation is None:
        strategy_evaluation = store.load_strategy_version_evaluation(run_id=run_id)
    if parameter_search_results is None:
        parameter_search_results = store.load_parameter_search_results(run_id=run_id)
    if walk_forward_validation is None:
        walk_forward_validation = store.load_walk_forward_validation(run_id=run_id)
    if trade_plan_backtest_performance is None:
        trade_plan_backtest_performance = store.load_trade_plan_backtest_performance(run_id=run_id)

    admission = build_strategy_admission(
        strategy_evaluation=strategy_evaluation,
        parameter_search_results=parameter_search_results,
        walk_forward_validation=walk_forward_validation,
        trade_plan_backtest_performance=trade_plan_backtest_performance,
    )
    if run_id is not None:
        admission = admission.assign(run_id=run_id)
    store.save_strategy_admission(admission)

    if export_candidate_config:
        config = build_active_strategy_candidate_config(admission)
        output_path = Path(candidate_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return admission


def _run_and_report(
    db_path: str | None = None,
    export_candidate_config: bool = False,
    candidate_config_path: str = "configs/active_strategies_candidate.json",
) -> tuple[pd.DataFrame, dict[str, int], str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    strategy_evaluation = store.load_strategy_version_evaluation()
    parameter_results = store.load_parameter_search_results()
    walk_forward = store.load_walk_forward_validation()
    trade_plan_performance = store.load_trade_plan_backtest_performance()

    admission = build_strategy_admission(
        strategy_evaluation=strategy_evaluation,
        parameter_search_results=parameter_results,
        walk_forward_validation=walk_forward,
        trade_plan_backtest_performance=trade_plan_performance,
    )
    store.save_strategy_admission(admission)

    if export_candidate_config:
        config = build_active_strategy_candidate_config(admission)
        output_path = Path(candidate_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {
        "strategy_version_evaluation": len(strategy_evaluation),
        "parameter_search_results": len(parameter_results),
        "walk_forward_validation": len(walk_forward),
        "trade_plan_backtest_performance": len(trade_plan_performance),
        "strategy_admission": len(admission),
    }
    return admission, counts, resolved_db_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strategy admission recommendations.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--export-candidate-config", action="store_true")
    parser.add_argument("--candidate-config-path", default="configs/active_strategies_candidate.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    admission, counts, resolved_db_path = _run_and_report(
        db_path=args.db_path,
        export_candidate_config=args.export_candidate_config,
        candidate_config_path=args.candidate_config_path,
    )

    print(f"strategy_version_evaluation 行数: {counts['strategy_version_evaluation']}")
    print(f"parameter_search_results 行数: {counts['parameter_search_results']}")
    print(f"walk_forward_validation 行数: {counts['walk_forward_validation']}")
    print(f"trade_plan_backtest_performance 行数: {counts['trade_plan_backtest_performance']}")
    print(f"strategy_admission 行数: {counts['strategy_admission']}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("strategy_admission 前 20:")
    if admission.empty:
        print("无策略准入结果。")
    else:
        print(admission.sort_values("admission_score", ascending=False).head(20).to_string(index=False))
    if args.export_candidate_config:
        print(f"观察候选配置已写入: {args.candidate_config_path}")
        print("说明: active_strategies_candidate.json 只是观察候选配置，不是自动交易配置。")


if __name__ == "__main__":
    main()
