"""Run the local one-command strategy research workflow."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.config import settings
from src.pipeline.backtest_strategy_versions import run_strategy_version_backtest
from src.pipeline.backtest_trade_plans import run_trade_plan_backtest
from src.pipeline.build_strategy_admission import run_strategy_admission
from src.pipeline.evaluate_strategy_versions import run_strategy_version_evaluation
from src.pipeline.export_parameter_search_report import export_parameter_search_report
from src.pipeline.export_strategy_admission_report import export_strategy_admission_report
from src.pipeline.export_strategy_evaluation_report import export_strategy_evaluation_report
from src.pipeline.export_trade_plan_backtest_report import export_trade_plan_backtest_report
from src.pipeline.export_walk_forward_validation_report import export_walk_forward_validation_report
from src.pipeline.search_strategy_params import run_parameter_search
from src.pipeline.validate_strategy_oos import run_oos_validation


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _row_count(value: object) -> int:
    if isinstance(value, pd.DataFrame):
        return len(value)
    return 0


def _has_oos_dates(
    train_start_date: str | None,
    train_end_date: str | None,
    validation_start_date: str | None,
    validation_end_date: str | None,
) -> bool:
    return all(
        [
            train_start_date,
            train_end_date,
            validation_start_date,
            validation_end_date,
        ]
    )


def run_strategy_research_workflow(
    db_path: str | None = None,
    output_dir: str = "reports",
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    validation_start_date: str | None = None,
    validation_end_date: str | None = None,
    parameter_search_start_date: str | None = None,
    parameter_search_end_date: str | None = None,
    strategy_versions_config_path: str | None = None,
    parameter_search_config_path: str | None = None,
    export_reports: bool = True,
    export_candidate_config: bool = True,
    candidate_config_path: str = "configs/active_strategies_candidate.json",
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
) -> dict:
    """Run local research steps and return row counts plus exported artifact paths."""
    resolved_db_path = _resolve_db_path(db_path)
    profile_steps: list[dict[str, Any]] = []
    skipped_oos = not _has_oos_dates(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
    )

    version_backtest_results, version_performance = _profiled(
        profile_steps,
        "run_strategy_version_backtest",
        lambda: run_strategy_version_backtest(
            start_date=train_start_date,
            end_date=train_end_date,
            config_path=strategy_versions_config_path,
            db_path=resolved_db_path,
            limit_strategies=limit_strategies,
        ),
    )
    version_evaluation = _profiled(
        profile_steps,
        "run_strategy_version_evaluation",
        lambda: run_strategy_version_evaluation(db_path=resolved_db_path),
    )

    parameter_backtest_results, parameter_performance, parameter_results = _profiled(
        profile_steps,
        "run_parameter_search",
        lambda: run_parameter_search(
            start_date=parameter_search_start_date,
            end_date=parameter_search_end_date,
            config_path=parameter_search_config_path,
            db_path=resolved_db_path,
            limit_strategies=limit_strategies,
            limit_param_combinations=limit_param_combinations,
        ),
    )

    walk_forward_validation = pd.DataFrame()
    if not skipped_oos:
        walk_forward_validation = _profiled(
            profile_steps,
            "run_oos_validation",
            lambda: run_oos_validation(
                train_start_date=str(train_start_date),
                train_end_date=str(train_end_date),
                validation_start_date=str(validation_start_date),
                validation_end_date=str(validation_end_date),
                config_path=parameter_search_config_path,
                db_path=resolved_db_path,
                limit_strategies=limit_strategies,
                limit_param_combinations=limit_param_combinations,
            ),
        )

    _, trade_plan_backtest_results, trade_plan_performance = _profiled(
        profile_steps,
        "run_trade_plan_backtest",
        lambda: run_trade_plan_backtest(
            db_path=resolved_db_path,
            start_date=train_start_date,
            end_date=train_end_date,
        ),
    )

    admission = _profiled(
        profile_steps,
        "run_strategy_admission",
        lambda: run_strategy_admission(
            db_path=resolved_db_path,
            export_candidate_config=export_candidate_config,
            candidate_config_path=candidate_config_path,
        ),
    )

    strategy_evaluation_report_path = None
    parameter_search_report_path = None
    walk_forward_validation_report_path = None
    trade_plan_backtest_report_path = None
    strategy_admission_report_path = None

    if export_reports:
        strategy_evaluation_report_path = _profiled(
            profile_steps,
            "export_strategy_evaluation_report",
            lambda: export_strategy_evaluation_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
            ),
        )
        parameter_search_report_path = _profiled(
            profile_steps,
            "export_parameter_search_report",
            lambda: export_parameter_search_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
            ),
        )
        if not skipped_oos:
            walk_forward_validation_report_path = _profiled(
                profile_steps,
                "export_walk_forward_validation_report",
                lambda: export_walk_forward_validation_report(
                    db_path=resolved_db_path,
                    output_dir=output_dir,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    validation_start_date=validation_start_date,
                    validation_end_date=validation_end_date,
                ),
            )
        trade_plan_backtest_report_path = _profiled(
            profile_steps,
            "export_trade_plan_backtest_report",
            lambda: export_trade_plan_backtest_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
            ),
        )
        strategy_admission_report_path = _profiled(
            profile_steps,
            "export_strategy_admission_report",
            lambda: export_strategy_admission_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
            ),
        )

    return {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "strategy_version_backtest_results_rows": _row_count(version_backtest_results),
        "strategy_version_performance_rows": _row_count(version_performance),
        "strategy_version_evaluation_rows": _row_count(version_evaluation),
        "parameter_search_backtest_rows": _row_count(parameter_backtest_results),
        "parameter_search_performance_rows": _row_count(parameter_performance),
        "parameter_search_results_rows": _row_count(parameter_results),
        "walk_forward_validation_rows": _row_count(walk_forward_validation),
        "trade_plan_backtest_results_rows": _row_count(trade_plan_backtest_results),
        "trade_plan_backtest_performance_rows": _row_count(trade_plan_performance),
        "strategy_admission_rows": _row_count(admission),
        "skipped_oos": skipped_oos,
        "strategy_evaluation_report_path": strategy_evaluation_report_path,
        "parameter_search_report_path": parameter_search_report_path,
        "walk_forward_validation_report_path": walk_forward_validation_report_path,
        "trade_plan_backtest_report_path": trade_plan_backtest_report_path,
        "strategy_admission_report_path": strategy_admission_report_path,
        "active_candidate_config_path": candidate_config_path if export_candidate_config else None,
        "limit_strategies": limit_strategies,
        "limit_param_combinations": limit_param_combinations,
        "profile_steps": profile_steps,
    }


def _profiled(profile_steps: list[dict[str, Any]], function_name: str, runner):
    started_at = time.perf_counter()
    try:
        result = runner()
    except Exception:
        elapsed = time.perf_counter() - started_at
        profile_steps.append(
            {
                "function_name": function_name,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "rows": 0,
            }
        )
        print(f"[profile] {function_name} failed elapsed={elapsed:.2f}s", flush=True)
        raise

    elapsed = time.perf_counter() - started_at
    rows = _profile_rows(result)
    profile_steps.append(
        {
            "function_name": function_name,
            "status": "success",
            "elapsed_seconds": elapsed,
            "rows": rows,
        }
    )
    print(f"[profile] {function_name} rows={rows} elapsed={elapsed:.2f}s", flush=True)
    return result


def _profile_rows(result: Any) -> int:
    if isinstance(result, pd.DataFrame):
        return len(result)
    if isinstance(result, tuple):
        return int(sum(len(item) for item in result if isinstance(item, pd.DataFrame)))
    if isinstance(result, dict):
        return int(sum(value for key, value in result.items() if key.endswith("_rows") and isinstance(value, int)))
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local strategy research workflow.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--train-start-date", default=None)
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--validation-start-date", default=None)
    parser.add_argument("--validation-end-date", default=None)
    parser.add_argument("--parameter-search-start-date", default=None)
    parser.add_argument("--parameter-search-end-date", default=None)
    parser.add_argument("--strategy-versions-config-path", default=None)
    parser.add_argument("--parameter-search-config-path", default=None)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-candidate-config", action="store_true")
    parser.add_argument("--candidate-config-path", default="configs/active_strategies_candidate.json")
    parser.add_argument("--limit-strategies", type=int, default=None)
    parser.add_argument("--limit-param-combinations", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_strategy_research_workflow(
        db_path=args.db_path,
        output_dir=args.output_dir,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
        parameter_search_start_date=args.parameter_search_start_date,
        parameter_search_end_date=args.parameter_search_end_date,
        strategy_versions_config_path=args.strategy_versions_config_path,
        parameter_search_config_path=args.parameter_search_config_path,
        export_reports=not args.no_report,
        export_candidate_config=not args.no_candidate_config,
        candidate_config_path=args.candidate_config_path,
        limit_strategies=args.limit_strategies,
        limit_param_combinations=args.limit_param_combinations,
    )

    print("Strategy research workflow finished.")
    for key in [
        "strategy_version_evaluation_rows",
        "parameter_search_results_rows",
        "walk_forward_validation_rows",
        "trade_plan_backtest_performance_rows",
        "strategy_admission_rows",
        "active_candidate_config_path",
    ]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
