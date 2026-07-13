"""Run the local one-command strategy research workflow."""

from __future__ import annotations

import argparse
import gc
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline.backtest_strategy_versions import run_strategy_version_backtest
from src.pipeline.backtest_trade_plans import run_trade_plan_backtest
from src.pipeline.build_strategy_admission import run_strategy_admission
from src.pipeline.evaluate_strategy_versions import run_strategy_version_evaluation
from src.pipeline.export_parameter_search_report import export_parameter_search_report
from src.pipeline.export_strategy_admission_report import export_strategy_admission_report
from src.pipeline.export_strategy_evaluation_report import export_strategy_evaluation_report
from src.pipeline.export_trade_plan_backtest_report import export_trade_plan_backtest_report
from src.pipeline.export_walk_forward_validation_report import export_walk_forward_validation_report
from src.pipeline.parallel_strategy_research import (
    PARALLELIZED_STEPS,
    ParallelStageOutput,
    run_oos_validation_parallel,
    run_parameter_search_parallel,
    run_trade_plan_backtest_parallel,
)
from src.pipeline.search_strategy_params import run_parameter_search
from src.pipeline.validate_strategy_oos import run_oos_validation
from src.research.parameter_search import generate_search_versions, load_parameter_search_space
from src.strategy.strategy_versions import iter_strategy_versions, load_strategy_versions
from src.pipeline.memory import log_memory


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _row_count(value: object) -> int:
    if isinstance(value, pd.DataFrame):
        return int(value.attrs.get("row_count", len(value)))
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


def _new_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"strategy-research-{timestamp}-{uuid.uuid4().hex[:8]}"


def _unpack_version_backtest_output(
    output: tuple[pd.DataFrame, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(output) >= 3:
        return output[0], output[1], output[2]
    backtest_results, performance = output
    return backtest_results, performance, _signals_from_backtest_results(backtest_results)


def _signals_from_backtest_results(backtest_results: pd.DataFrame) -> pd.DataFrame:
    required = ["signal_date", "code", "strategy_name", "strategy_version", "signal_strength"]
    if backtest_results.empty or any(column not in backtest_results.columns for column in required):
        return pd.DataFrame(
            columns=[
                "trade_date",
                "code",
                "strategy_name",
                "strategy_version",
                "signal_strength",
                "entry_reason",
                "risk_flags",
            ]
        )
    signals = backtest_results.loc[:, required].rename(columns={"signal_date": "trade_date"}).copy()
    signals["entry_reason"] = ""
    signals["risk_flags"] = ""
    return signals


def _append_chain_profile_steps(profile_steps: list[dict[str, Any]], diagnostics: dict[str, int]) -> None:
    for key in ["historical_signals", "historical_candidates", "historical_trade_plans"]:
        profile_steps.append(
            {
                "function_name": key,
                "status": "success",
                "elapsed_seconds": 0.0,
                "rows": int(diagnostics.get(key, 0)),
            }
        )


def _unpack_trade_plan_backtest_output(
    output: tuple[pd.DataFrame, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if len(output) >= 4:
        return output[0], output[1], output[2], output[3]
    historical_trade_plans, backtest_results, performance = output
    diagnostics = {
        "historical_signals": 0,
        "historical_candidates": 0,
        "historical_trade_plans": int(len(historical_trade_plans)),
        "triggered_trades": (
            int(backtest_results["is_triggered"].fillna(False).astype(bool).sum())
            if "is_triggered" in backtest_results.columns
            else 0
        ),
    }
    return historical_trade_plans, backtest_results, performance, diagnostics


def _research_table_counts(db_path: str) -> dict[str, int]:
    table_names = [
        "strategy_version_evaluation",
        "parameter_search_results",
        "walk_forward_validation",
        "historical_trade_plans",
        "trade_plan_backtest_results",
        "strategy_admission",
    ]
    if not Path(db_path).exists():
        return {table_name: 0 for table_name in table_names}
    store = StockAgentStore(db_path)
    counts: dict[str, int] = {}
    with store._connect() as con:
        store._create_tables(con)
        for table_name in table_names:
            counts[table_name] = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    return counts


def build_dry_run_plan(
    strategy_versions_config_path: str | None = None,
    parameter_search_config_path: str | None = None,
    limit_strategies: int | None = None,
    limit_param_combinations: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    resolved_workers = max(1, int(workers or 1))
    versions = iter_strategy_versions(load_strategy_versions(strategy_versions_config_path))
    if limit_strategies is not None:
        versions = versions[: int(limit_strategies)]

    search_config = load_parameter_search_space(parameter_search_config_path)
    search_versions = generate_search_versions(
        search_config,
        limit_strategies=limit_strategies,
        limit_param_combinations=limit_param_combinations,
    )

    parameter_combinations_by_strategy: dict[str, int] = {}
    for version in search_versions:
        strategy_name = str(version.get("strategy_name", ""))
        parameter_combinations_by_strategy[strategy_name] = parameter_combinations_by_strategy.get(strategy_name, 0) + 1

    strategy_version_rows = [
        {
            "strategy_name": version["strategy_name"],
            "strategy_version": version["strategy_version"],
        }
        for version in versions
    ]
    return {
        "enabled_strategy_versions_count": len(versions),
        "strategy_versions": strategy_version_rows,
        "parameter_search_combinations_count": len(search_versions),
        "parameter_combinations_by_strategy": parameter_combinations_by_strategy,
        "estimated_admission_candidates_count": len(versions) + len(search_versions),
        "market_regime_gating": (
            "candidate_selector reads market_regime when supplied; new strategies also honor "
            "market_regime/risk_level columns if merged into factors"
        ),
        "limit_strategies": limit_strategies,
        "limit_param_combinations": limit_param_combinations,
        "workers": resolved_workers,
        "parallel_enabled": resolved_workers > 1,
        "parallelized_steps": PARALLELIZED_STEPS if resolved_workers > 1 else [],
    }


def _log_current_and_total_rows(current_rows: dict[str, int], table_total_rows: dict[str, int]) -> None:
    for table_name, current_count in current_rows.items():
        print(
            f"[current-run] {table_name} rows={current_count} table_total_rows={table_total_rows.get(table_name, 0)}",
            flush=True,
        )


def _run_parameter_search_streaming(**kwargs):
    """Keep monkeypatched workflow tests compatible while bounding production output."""
    if getattr(run_parameter_search, "__module__", "") == "src.pipeline.search_strategy_params":
        kwargs["materialize_results"] = False
    return run_parameter_search(**kwargs)


def _run_trade_plan_backtest_streaming(**kwargs):
    """Use chunked result persistence and strategy holding grids in production."""
    if getattr(run_trade_plan_backtest, "__module__", "") == "src.pipeline.backtest_trade_plans":
        kwargs["materialize_results"] = False
        kwargs["holding_days_mode"] = "strategy_grid"
    return run_trade_plan_backtest(**kwargs)


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
    run_id: str | None = None,
    workers: int = 1,
) -> dict:
    """Run local research steps and return row counts plus exported artifact paths."""
    resolved_db_path = _resolve_db_path(db_path)
    resolved_run_id = run_id or _new_run_id()
    resolved_workers = max(1, int(workers or 1))
    parallel_enabled = resolved_workers > 1
    parallel_stage_summaries: list[dict[str, Any]] = []
    if parallel_enabled:
        _log_parallel_thread_hint(resolved_workers)
    profile_steps: list[dict[str, Any]] = []
    skipped_oos = not _has_oos_dates(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
    )

    version_backtest_output = _profiled(
        profile_steps,
        "run_strategy_version_backtest",
        lambda: run_strategy_version_backtest(
            start_date=train_start_date,
            end_date=train_end_date,
            config_path=strategy_versions_config_path,
            db_path=resolved_db_path,
            limit_strategies=limit_strategies,
            run_id=resolved_run_id,
            return_signals=True,
        ),
    )
    version_backtest_results, version_performance, historical_signals = _unpack_version_backtest_output(
        version_backtest_output
    )
    version_backtest_results_count = _row_count(version_backtest_results)
    historical_signals_count = _row_count(historical_signals)
    # Real streaming output is persisted by run_id. Do not keep even a single
    # strategy's signals alive until the trade-plan stage.
    signals_for_trade_plan = None if "row_count" in historical_signals.attrs else historical_signals
    del version_backtest_output, version_backtest_results, historical_signals
    gc.collect()
    log_memory("research_workflow:after_version_backtest", "released")
    version_evaluation = _profiled(
        profile_steps,
        "run_strategy_version_evaluation",
        lambda: run_strategy_version_evaluation(
            db_path=resolved_db_path,
            performance=version_performance,
            run_id=resolved_run_id,
        ),
    )

    parameter_versions = (
        generate_search_versions(
            load_parameter_search_space(parameter_search_config_path),
            limit_strategies=limit_strategies,
            limit_param_combinations=limit_param_combinations,
        )
        if parallel_enabled
        else []
    )
    parameter_backtest_results, parameter_performance, parameter_results = _parallel_or_serial_profiled(
        profile_steps,
        parallel_stage_summaries,
        "run_parameter_search",
        parallel_enabled=parallel_enabled,
        requested_workers=resolved_workers,
        serial_runner=lambda: _run_parameter_search_streaming(
            start_date=parameter_search_start_date,
            end_date=parameter_search_end_date,
            config_path=parameter_search_config_path,
            db_path=resolved_db_path,
            limit_strategies=limit_strategies,
            limit_param_combinations=limit_param_combinations,
            run_id=resolved_run_id,
        ),
        parallel_runner=lambda: run_parameter_search_parallel(
            db_path=resolved_db_path,
            versions=parameter_versions,
            start_date=parameter_search_start_date,
            end_date=parameter_search_end_date,
            run_id=resolved_run_id,
            workers=resolved_workers,
        ),
    )
    parameter_backtest_count = _row_count(parameter_backtest_results)
    del parameter_backtest_results
    gc.collect()
    log_memory("research_workflow:after_parameter_search", "released")

    walk_forward_validation = pd.DataFrame()
    if not skipped_oos:
        walk_forward_validation = _parallel_or_serial_profiled(
            profile_steps,
            parallel_stage_summaries,
            "run_oos_validation",
            parallel_enabled=parallel_enabled,
            requested_workers=resolved_workers,
            serial_runner=lambda: run_oos_validation(
                train_start_date=str(train_start_date),
                train_end_date=str(train_end_date),
                validation_start_date=str(validation_start_date),
                validation_end_date=str(validation_end_date),
                config_path=parameter_search_config_path,
                db_path=resolved_db_path,
                limit_strategies=limit_strategies,
                limit_param_combinations=limit_param_combinations,
                run_id=resolved_run_id,
            ),
            parallel_runner=lambda: run_oos_validation_parallel(
                db_path=resolved_db_path,
                versions=parameter_versions,
                train_start_date=str(train_start_date),
                train_end_date=str(train_end_date),
                validation_start_date=str(validation_start_date),
                validation_end_date=str(validation_end_date),
                run_id=resolved_run_id,
                workers=resolved_workers,
            ),
        )

    trade_plan_backtest_output = _parallel_or_serial_profiled(
        profile_steps,
        parallel_stage_summaries,
        "run_trade_plan_backtest",
        parallel_enabled=parallel_enabled,
        requested_workers=resolved_workers,
        serial_runner=lambda: _run_trade_plan_backtest_streaming(
            db_path=resolved_db_path,
            start_date=train_start_date,
            end_date=train_end_date,
            strategy_signals=signals_for_trade_plan,
            strategy_evaluation=version_evaluation,
            run_id=resolved_run_id,
            return_diagnostics=True,
        ),
        parallel_runner=lambda: run_trade_plan_backtest_parallel(
            db_path=resolved_db_path,
            start_date=train_start_date,
            end_date=train_end_date,
            strategy_signals=signals_for_trade_plan,
            strategy_evaluation=version_evaluation,
            run_id=resolved_run_id,
            workers=resolved_workers,
        ),
    )
    historical_trade_plans, trade_plan_backtest_results, trade_plan_performance, trade_plan_diagnostics = (
        _unpack_trade_plan_backtest_output(trade_plan_backtest_output)
    )
    del trade_plan_backtest_output, signals_for_trade_plan
    gc.collect()
    log_memory("research_workflow:after_trade_plan_backtest", "released")
    _append_chain_profile_steps(profile_steps, trade_plan_diagnostics)

    admission = _profiled(
        profile_steps,
        "run_strategy_admission",
        lambda: run_strategy_admission(
            db_path=resolved_db_path,
            export_candidate_config=export_candidate_config,
            candidate_config_path=candidate_config_path,
            strategy_evaluation=version_evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward_validation,
            trade_plan_backtest_performance=trade_plan_performance,
            run_id=resolved_run_id,
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
                evaluation=version_evaluation,
                performance=version_performance,
                run_id=resolved_run_id,
            ),
        )
        parameter_search_report_path = _profiled(
            profile_steps,
            "export_parameter_search_report",
            lambda: export_parameter_search_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
                evaluation=parameter_results,
                performance=parameter_performance,
                run_id=resolved_run_id,
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
                    validation=walk_forward_validation,
                    run_id=resolved_run_id,
                ),
            )
        trade_plan_backtest_report_path = _profiled(
            profile_steps,
            "export_trade_plan_backtest_report",
            lambda: export_trade_plan_backtest_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
                backtest_results=trade_plan_backtest_results,
                performance=trade_plan_performance,
                run_id=resolved_run_id,
            ),
        )
        strategy_admission_report_path = _profiled(
            profile_steps,
            "export_strategy_admission_report",
            lambda: export_strategy_admission_report(
                db_path=resolved_db_path,
                output_dir=output_dir,
                admission=admission,
                run_id=resolved_run_id,
            ),
        )

    table_total_rows = _research_table_counts(resolved_db_path)
    _log_current_and_total_rows(
        {
            "strategy_version_evaluation": len(version_evaluation),
            "parameter_search_results": len(parameter_results),
            "walk_forward_validation": len(walk_forward_validation),
            "historical_trade_plans": len(historical_trade_plans),
            "trade_plan_backtest_results": len(trade_plan_backtest_results),
            "strategy_admission": len(admission),
        },
        table_total_rows,
    )

    return {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "run_id": resolved_run_id,
        "strategy_version_backtest_results_rows": version_backtest_results_count,
        "strategy_version_performance_rows": _row_count(version_performance),
        "strategy_version_evaluation_rows": _row_count(version_evaluation),
        "parameter_search_backtest_rows": parameter_backtest_count,
        "parameter_search_performance_rows": _row_count(parameter_performance),
        "parameter_search_results_rows": _row_count(parameter_results),
        "walk_forward_validation_rows": _row_count(walk_forward_validation),
        "historical_signals_rows": historical_signals_count,
        "historical_candidates_rows": int(trade_plan_diagnostics.get("historical_candidates", 0)),
        "historical_trade_plans_rows": _row_count(historical_trade_plans),
        "triggered_trades_rows": int(trade_plan_diagnostics.get("triggered_trades", 0)),
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
        "workers": resolved_workers,
        "parallel_enabled": parallel_enabled,
        "parallelized_steps": PARALLELIZED_STEPS if parallel_enabled else [],
        "parallel_stage_summaries": parallel_stage_summaries,
        "parallel_worker_errors": [
            error
            for stage in parallel_stage_summaries
            for error in stage.get("worker_errors", [])
        ],
        "profile_steps": profile_steps,
        "table_total_counts": table_total_rows,
    }


def _profiled(profile_steps: list[dict[str, Any]], function_name: str, runner):
    started_at = time.perf_counter()
    rss_before = log_memory(function_name, "before")
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
    rss_after = log_memory(function_name, "after")
    profile_steps[-1]["rss_mb_before"] = rss_before
    profile_steps[-1]["rss_mb_after"] = rss_after
    print(f"[profile] {function_name} rows={rows} elapsed={elapsed:.2f}s", flush=True)
    return result


def _parallel_or_serial_profiled(
    profile_steps: list[dict[str, Any]],
    parallel_stage_summaries: list[dict[str, Any]],
    function_name: str,
    *,
    parallel_enabled: bool,
    requested_workers: int,
    serial_runner,
    parallel_runner,
):
    if not parallel_enabled:
        return _profiled(profile_steps, function_name, serial_runner)
    return _profiled_parallel(
        profile_steps,
        parallel_stage_summaries,
        function_name,
        parallel_runner,
        requested_workers=requested_workers,
    )


def _profiled_parallel(
    profile_steps: list[dict[str, Any]],
    parallel_stage_summaries: list[dict[str, Any]],
    function_name: str,
    runner,
    *,
    requested_workers: int,
):
    started_at = time.perf_counter()
    try:
        output = runner()
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        worker_errors = getattr(exc, "worker_errors", [])
        failure_summary = {
            "stage_name": function_name,
            "status": "failed",
            "requested_workers": requested_workers,
            "workers": min(requested_workers, len(worker_errors)) if worker_errors else 0,
            "elapsed_seconds": elapsed,
            "rows": 0,
            "worker_logs": [error.get("worker_log", {}) for error in worker_errors],
            "worker_errors": worker_errors,
        }
        parallel_stage_summaries.append(failure_summary)
        profile_steps.append(
            {
                "function_name": function_name,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "rows": 0,
            }
        )
        print(f"[profile] {function_name} failed elapsed={elapsed:.2f}s", flush=True)
        setattr(exc, "parallel_stage_summaries", list(parallel_stage_summaries))
        setattr(exc, "profile_steps", list(profile_steps))
        raise

    elapsed = time.perf_counter() - started_at
    if not isinstance(output, ParallelStageOutput):
        raise TypeError(f"{function_name} parallel runner must return ParallelStageOutput")
    result = output.result
    summary = dict(output.summary)
    summary["status"] = "success"
    summary["elapsed_seconds"] = elapsed
    parallel_stage_summaries.append(summary)
    rows = _profile_rows(result)
    profile_steps.append(
        {
            "function_name": function_name,
            "status": "success",
            "elapsed_seconds": elapsed,
            "rows": rows,
        }
    )
    print(
        f"[profile] {function_name} rows={rows} elapsed={elapsed:.2f}s "
        f"workers={summary.get('workers', 0)}",
        flush=True,
    )
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
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _log_parallel_thread_hint(workers: int) -> None:
    print(
        "[parallel] "
        f"workers={workers} enabled=True; suggested env: "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.dry_run_plan:
        plan = build_dry_run_plan(
            strategy_versions_config_path=args.strategy_versions_config_path,
            parameter_search_config_path=args.parameter_search_config_path,
            limit_strategies=args.limit_strategies,
            limit_param_combinations=args.limit_param_combinations,
            workers=args.workers,
        )
        print("Strategy research dry-run plan.")
        print(f"workers: {plan.get('workers', max(1, int(args.workers or 1)))}")
        print(f"parallel enabled: {plan.get('parallel_enabled', max(1, int(args.workers or 1)) > 1)}")
        print(f"enabled strategy versions count: {plan['enabled_strategy_versions_count']}")
        print("strategy versions:")
        for row in plan["strategy_versions"]:
            print(f"- {row['strategy_name']} / {row['strategy_version']}")
        print(f"parameter search combinations count: {plan['parameter_search_combinations_count']}")
        print("parameter combinations by strategy:")
        for strategy_name, count in plan["parameter_combinations_by_strategy"].items():
            print(f"- {strategy_name}: {count}")
        print(f"estimated admission candidates count: {plan['estimated_admission_candidates_count']}")
        print(f"market regime gating: {plan['market_regime_gating']}")
        return

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
        workers=args.workers,
    )

    print("Strategy research workflow finished.")
    for key in [
        "run_id",
        "strategy_version_evaluation_rows",
        "parameter_search_results_rows",
        "walk_forward_validation_rows",
        "historical_signals_rows",
        "historical_candidates_rows",
        "historical_trade_plans_rows",
        "trade_plan_backtest_performance_rows",
        "strategy_admission_rows",
        "active_candidate_config_path",
        "workers",
        "parallel_enabled",
    ]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
