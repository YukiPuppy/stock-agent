"""Process-based helpers for CPU-heavy strategy research stages."""

from __future__ import annotations

import os
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from src.backtest.historical_trade_plan_builder import build_historical_trade_plans
from src.backtest.signal_backtester import backtest_strategy_signals, evaluate_strategy_performance
from src.backtest.strategy_version_runner import generate_historical_signals_for_versions
from src.backtest.trade_plan_backtester import (
    DEFAULT_MAX_HOLDING_DAYS,
    backtest_trade_plans,
    evaluate_trade_plan_backtest,
)
from src.database.duckdb_store import DAILY_FACTOR_COLUMNS, StockAgentStore
from src.research.strategy_version_evaluator import evaluate_strategy_versions
from src.research.walk_forward_validation import validate_strategy_versions_out_of_sample
from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS


PARALLELIZED_STEPS = ["run_parameter_search", "run_oos_validation", "run_trade_plan_backtest"]

_BASE_FACTOR_COLUMNS = {
    "trade_date",
    "code",
    "close",
    "pct_chg_1d",
    "pct_chg_3d",
    "pct_chg_5d",
    "pct_chg_10d",
    "amount_ma5",
    "turnover_rate",
    "is_suspended",
    "is_limit_up_close",
    "is_limit_down_close",
}
_STRATEGY_FACTOR_COLUMNS = {
    "trend_pullback": {"above_ma5", "above_ma10", "close_position_20", "volume_ratio_5"},
    "breakout_volume": {"above_ma5", "close_position_20", "volume_ratio_5"},
    "support_rebound": {"above_ma20", "close_position_20"},
    "industry_rotation": {
        "close_position_20",
        "industry_amount_ratio_5",
        "industry_return_3d",
        "industry_return_5d",
        "industry_strength_level",
        "industry_strength_score",
        "moneyflow_score",
    },
    "moneyflow_accumulation": {
        "big_net_amount",
        "main_net_amount",
        "main_net_amount_ratio",
        "moneyflow_score",
        "net_mf_amount",
    },
    "low_vol_trend": {"above_ma5", "above_ma10", "above_ma20", "close_position_20", "volume_ratio_5"},
    "oversold_rebound": {"close_position_20", "industry_strength_score", "moneyflow_score"},
    "volume_dryup_breakout": {
        "above_ma5",
        "above_ma10",
        "close_position_20",
        "volume_ratio_5",
        "volume_ratio_daily_basic",
    },
    "relative_strength_pullback": {
        "above_ma10",
        "above_ma20",
        "close_position_20",
        "industry_return_5d",
        "industry_strength_score",
        "moneyflow_score",
    },
}


@dataclass(frozen=True)
class ParallelStageOutput:
    result: Any
    summary: dict[str, Any]


class ParallelWorkerError(RuntimeError):
    def __init__(self, stage_name: str, worker_errors: list[dict[str, Any]]):
        self.stage_name = stage_name
        self.worker_errors = worker_errors
        message = "; ".join(
            f"worker={item.get('worker_index')} pid={item.get('pid')} error={item.get('error')}"
            for item in worker_errors
        )
        super().__init__(f"{stage_name} parallel worker failed: {message}")


def effective_worker_count(requested_workers: int, task_count: int) -> int:
    requested = max(1, int(requested_workers or 1))
    if task_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(requested, cpu_count, int(task_count)))


def run_parameter_search_parallel(
    *,
    db_path: str,
    versions: list[dict],
    start_date: str | None,
    end_date: str | None,
    run_id: str,
    workers: int,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
) -> ParallelStageOutput:
    stage_started = time.perf_counter()
    chunks = _chunked(versions, effective_worker_count(workers, len(versions)))
    stage_workers = len(chunks)
    tasks = [
        {
            "worker_index": index,
            "db_path": db_path,
            "versions": chunk,
            "start_date": start_date,
            "end_date": end_date,
            "run_id": run_id,
            "min_valid_count": min_valid_count,
            "min_win_rate_3d": min_win_rate_3d,
            "min_avg_return_3d": min_avg_return_3d,
            "max_avg_drawdown_3d": max_avg_drawdown_3d,
        }
        for index, chunk in enumerate(chunks)
    ]
    results = _execute_parallel("run_parameter_search", tasks, _parameter_search_worker, stage_workers)
    backtest_results = _concat([item["backtest_results"] for item in results])
    performance = _concat([item["performance"] for item in results])
    evaluation = _concat([item["evaluation"] for item in results])
    if not evaluation.empty and "evaluation_score" in evaluation.columns:
        evaluation = evaluation.sort_values("evaluation_score", ascending=False).reset_index(drop=True)
    _validate_run_id([backtest_results, performance, evaluation], run_id, "run_parameter_search")
    store = StockAgentStore(db_path)
    store.save_parameter_search_backtest_results(backtest_results)
    store.save_parameter_search_performance(performance)
    store.save_parameter_search_results(evaluation)
    summary = _stage_summary(
        "run_parameter_search",
        workers,
        stage_workers,
        stage_started,
        rows=len(backtest_results) + len(performance) + len(evaluation),
        worker_logs=[item["worker_log"] for item in results],
    )
    return ParallelStageOutput((backtest_results, performance, evaluation), summary)


def run_oos_validation_parallel(
    *,
    db_path: str,
    versions: list[dict],
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    run_id: str,
    workers: int,
    min_valid_count_train: int = 30,
    min_valid_count_validation: int = 10,
) -> ParallelStageOutput:
    stage_started = time.perf_counter()
    chunks = _chunked(versions, effective_worker_count(workers, len(versions)))
    stage_workers = len(chunks)
    tasks = [
        {
            "worker_index": index,
            "db_path": db_path,
            "versions": chunk,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "validation_start_date": validation_start_date,
            "validation_end_date": validation_end_date,
            "run_id": run_id,
            "min_valid_count_train": min_valid_count_train,
            "min_valid_count_validation": min_valid_count_validation,
        }
        for index, chunk in enumerate(chunks)
    ]
    results = _execute_parallel("run_oos_validation", tasks, _oos_validation_worker, stage_workers)
    validation = _concat([item["validation"] for item in results])
    if not validation.empty and "stability_score" in validation.columns:
        validation = validation.sort_values("stability_score", ascending=False).reset_index(drop=True)
    _validate_run_id([validation], run_id, "run_oos_validation")
    StockAgentStore(db_path).save_walk_forward_validation(validation)
    summary = _stage_summary(
        "run_oos_validation",
        workers,
        stage_workers,
        stage_started,
        rows=len(validation),
        worker_logs=[item["worker_log"] for item in results],
    )
    return ParallelStageOutput(validation, summary)


def run_trade_plan_backtest_parallel(
    *,
    db_path: str,
    start_date: str | None,
    end_date: str | None,
    strategy_signals: pd.DataFrame,
    strategy_evaluation: pd.DataFrame,
    run_id: str,
    workers: int,
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
) -> ParallelStageOutput:
    stage_started = time.perf_counter()
    store = StockAgentStore(db_path)
    daily_factors = store.load_daily_factors(start_date=start_date, end_date=end_date)
    stock_basic = store.load_stock_basic()
    try:
        market_regime = store.load_market_regime()
    except Exception:
        market_regime = pd.DataFrame()
    historical_trade_plans, diagnostics = build_historical_trade_plans(
        strategy_signals=strategy_signals,
        daily_factors=daily_factors,
        stock_basic=stock_basic,
        strategy_evaluation=strategy_evaluation,
        top_n=top_n,
        max_plan_items=max_plan_items,
        min_amount_ma5=min_amount_ma5,
        market_regime=market_regime,
        return_diagnostics=True,
    )
    historical_trade_plans = _with_run_id(historical_trade_plans, run_id)
    _validate_run_id([historical_trade_plans], run_id, "run_trade_plan_backtest")
    store.save_historical_trade_plans(historical_trade_plans)

    plan_count = len(historical_trade_plans)
    stage_workers = effective_worker_count(workers, plan_count)
    chunk_sizes = _chunk_sizes(plan_count, stage_workers)
    offsets: list[int] = []
    current_offset = 0
    for size in chunk_sizes:
        offsets.append(current_offset)
        current_offset += size
    tasks = [
        {
            "worker_index": index,
            "db_path": db_path,
            "run_id": run_id,
            "start_date": start_date,
            "end_date": end_date,
            "limit": size,
            "offset": offsets[index],
            "max_holding_days": max_holding_days,
        }
        for index, size in enumerate(chunk_sizes)
        if size > 0
    ]
    results = _execute_parallel("run_trade_plan_backtest", tasks, _trade_plan_backtest_worker, len(tasks))
    backtest_results = _concat([item["backtest_results"] for item in results])
    performance = evaluate_trade_plan_backtest(backtest_results)
    backtest_results = _with_run_id(backtest_results, run_id)
    performance = _with_run_id(performance, run_id)
    _validate_run_id([backtest_results, performance], run_id, "run_trade_plan_backtest")
    store.save_trade_plan_backtest_results(backtest_results)
    store.save_trade_plan_backtest_performance(performance)
    diagnostics["triggered_trades"] = (
        int(backtest_results["is_triggered"].fillna(False).astype(bool).sum())
        if "is_triggered" in backtest_results.columns
        else 0
    )
    print(
        "[historical_trade_plan_chain] "
        f"historical_signals={diagnostics['historical_signals']} "
        f"historical_candidates={diagnostics['historical_candidates']} "
        f"historical_trade_plans={diagnostics['historical_trade_plans']} "
        f"triggered_trades={diagnostics['triggered_trades']}",
        flush=True,
    )
    summary = _stage_summary(
        "run_trade_plan_backtest",
        workers,
        len(tasks),
        stage_started,
        rows=len(backtest_results) + len(performance),
        worker_logs=[item["worker_log"] for item in results],
    )
    return ParallelStageOutput((historical_trade_plans, backtest_results, performance, diagnostics), summary)


def _parameter_search_worker(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    worker_index = int(task["worker_index"])
    versions = list(task["versions"])
    pid = os.getpid()
    status = "success"
    try:
        with duckdb.connect(str(task["db_path"]), read_only=True) as con:
            daily_factors = _load_daily_factors(
                con,
                task["start_date"],
                task["end_date"],
                versions=versions,
            )
            daily_bars = _load_daily_bars(con, task["start_date"], task["end_date"])
        historical_signals = generate_historical_signals_for_versions(
            daily_factors=daily_factors,
            versions=versions,
            start_date=task["start_date"],
            end_date=task["end_date"],
        )
        backtest_results = backtest_strategy_signals(historical_signals, daily_bars)
        performance = evaluate_strategy_performance(backtest_results)
        evaluation = evaluate_strategy_versions(
            performance,
            min_valid_count=task["min_valid_count"],
            min_win_rate_3d=task["min_win_rate_3d"],
            min_avg_return_3d=task["min_avg_return_3d"],
            max_avg_drawdown_3d=task["max_avg_drawdown_3d"],
        )
        run_id = str(task["run_id"])
        return {
            "backtest_results": _with_run_id(backtest_results, run_id),
            "performance": _with_run_id(performance, run_id),
            "evaluation": _with_run_id(evaluation, run_id),
            "worker_log": _worker_log(
                pid=pid,
                worker_index=worker_index,
                task_count=len(versions),
                versions=versions,
                started=started,
                status=status,
            ),
        }
    except Exception as exc:
        status = "failed"
        return _worker_failure(pid, worker_index, len(versions), versions, started, status, exc)


def _oos_validation_worker(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    worker_index = int(task["worker_index"])
    versions = list(task["versions"])
    pid = os.getpid()
    try:
        with duckdb.connect(str(task["db_path"]), read_only=True) as con:
            daily_factors = _load_daily_factors(
                con,
                task["train_start_date"],
                task["validation_end_date"],
                versions=versions,
            )
            daily_bars = _load_daily_bars(con, task["train_start_date"], task["validation_end_date"])
        validation = validate_strategy_versions_out_of_sample(
            daily_factors=daily_factors,
            daily_bars=daily_bars,
            versions=versions,
            train_start_date=task["train_start_date"],
            train_end_date=task["train_end_date"],
            validation_start_date=task["validation_start_date"],
            validation_end_date=task["validation_end_date"],
            min_valid_count_train=task["min_valid_count_train"],
            min_valid_count_validation=task["min_valid_count_validation"],
        )
        run_id = str(task["run_id"])
        return {
            "validation": _with_run_id(validation, run_id),
            "worker_log": _worker_log(
                pid=pid,
                worker_index=worker_index,
                task_count=len(versions),
                versions=versions,
                started=started,
                status="success",
            ),
        }
    except Exception as exc:
        return _worker_failure(pid, worker_index, len(versions), versions, started, "failed", exc)


def _trade_plan_backtest_worker(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    pid = os.getpid()
    worker_index = int(task["worker_index"])
    task_count = int(task["limit"])
    try:
        with duckdb.connect(str(task["db_path"]), read_only=True) as con:
            trade_plans = _load_historical_trade_plan_chunk(
                con,
                run_id=str(task["run_id"]),
                limit=int(task["limit"]),
                offset=int(task["offset"]),
            )
            codes = sorted(trade_plans["code"].dropna().astype(str).unique().tolist()) if not trade_plans.empty else []
            daily_bars = _load_daily_bars(con, task["start_date"], task["end_date"], codes=codes)
        backtest_results = backtest_trade_plans(
            trade_plans=trade_plans,
            daily_bars=daily_bars,
            max_holding_days=int(task["max_holding_days"]),
        )
        run_id = str(task["run_id"])
        return {
            "backtest_results": _with_run_id(backtest_results, run_id),
            "worker_log": _worker_log(
                pid=pid,
                worker_index=worker_index,
                task_count=task_count,
                chunk_range=f"{task['offset']}:{int(task['offset']) + task_count}",
                started=started,
                status="success",
            ),
        }
    except Exception as exc:
        return _worker_failure(
            pid,
            worker_index,
            task_count,
            [],
            started,
            "failed",
            exc,
            chunk_range=f"{task.get('offset')}:{int(task.get('offset', 0)) + task_count}",
        )


def _execute_parallel(stage_name: str, tasks: list[dict[str, Any]], worker_fn, max_workers: int) -> list[dict[str, Any]]:
    if not tasks:
        return []
    results: list[dict[str, Any]] = []
    worker_errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, max_workers), mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(worker_fn, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = _future_failure(task, exc)
            if item.get("status") == "failed":
                item["stage_name"] = stage_name
                worker_errors.append(item)
            else:
                results.append(item)
    results.sort(key=lambda item: int(item["worker_log"]["worker_index"]))
    worker_errors.sort(key=lambda item: int(item.get("worker_index", -1)))
    for item in [*results, *worker_errors]:
        log = item.get("worker_log", item)
        print(
            "[worker] "
            f"stage={stage_name} pid={log.get('pid')} worker_index={log.get('worker_index')} "
            f"task_count={log.get('task_count')} target={log.get('target')} "
            f"chunk_range={log.get('chunk_range', '')} "
            f"elapsed_seconds={float(log.get('elapsed_seconds', 0.0)):.2f} status={log.get('status')}",
            flush=True,
        )
    if worker_errors:
        raise ParallelWorkerError(stage_name, worker_errors)
    return results


def _load_daily_factors(
    con: duckdb.DuckDBPyConnection,
    start_date: str | None,
    end_date: str | None,
    *,
    versions: list[dict],
) -> pd.DataFrame:
    conditions, params = _date_conditions(start_date, end_date, compact=True)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    desired_columns = _factor_columns_for_versions(versions)
    columns = _existing_columns(con, "daily_factors", desired_columns)
    return con.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM daily_factors
        {where_clause}
        ORDER BY trade_date, code
        """,
        params,
    ).fetchdf()


def _load_daily_bars(
    con: duckdb.DuckDBPyConnection,
    start_date: str | None,
    end_date: str | None,
    *,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    conditions, params = _date_conditions(start_date, end_date, compact=False)
    if codes:
        placeholders = ", ".join(["?"] * len(codes))
        conditions.append(f"code IN ({placeholders})")
        params.extend(codes)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return con.execute(
        f"""
        SELECT trade_date, code, open, high, low, close, volume, amount
        FROM daily_bars
        {where_clause}
        ORDER BY trade_date, code
        """,
        params,
    ).fetchdf()


def _load_historical_trade_plan_chunk(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    limit: int,
    offset: int,
) -> pd.DataFrame:
    columns = _existing_columns(con, "historical_trade_plans", [*TRADE_PLAN_COLUMNS, "run_id"])
    return con.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM historical_trade_plans
        WHERE run_id = ?
        ORDER BY trade_date, rank, code, strategy_names, strategy_versions
        LIMIT ? OFFSET ?
        """,
        [run_id, limit, offset],
    ).fetchdf()


def _date_conditions(
    start_date: str | None,
    end_date: str | None,
    *,
    compact: bool,
) -> tuple[list[str], list[Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    if start_date is not None:
        conditions.append("trade_date >= ?")
        params.append(_date_bound(start_date, compact=compact))
    if end_date is not None:
        conditions.append("trade_date <= ?")
        params.append(_date_bound(end_date, compact=compact))
    return conditions, params


def _date_bound(value: str, *, compact: bool) -> str:
    text = str(value)
    return text.replace("-", "") if compact else text


def _factor_columns_for_versions(versions: list[dict]) -> list[str]:
    required = set(_BASE_FACTOR_COLUMNS)
    for version in versions:
        required.update(_STRATEGY_FACTOR_COLUMNS.get(str(version.get("strategy_name")), DAILY_FACTOR_COLUMNS))
    return [column for column in DAILY_FACTOR_COLUMNS if column in required]


def _existing_columns(con: duckdb.DuckDBPyConnection, table_name: str, desired_columns: list[str]) -> list[str]:
    available = {str(row[1]) for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()}
    return [column for column in desired_columns if column in available]


def _chunked(items: list[dict], chunk_count: int) -> list[list[dict]]:
    if not items:
        return []
    chunk_count = max(1, min(int(chunk_count), len(items)))
    sizes = _chunk_sizes(len(items), chunk_count)
    chunks: list[list[dict]] = []
    start = 0
    for size in sizes:
        chunks.append(items[start : start + size])
        start += size
    return chunks


def _chunk_sizes(item_count: int, chunk_count: int) -> list[int]:
    if item_count <= 0:
        return []
    chunk_count = max(1, min(int(chunk_count), item_count))
    base, remainder = divmod(item_count, chunk_count)
    return [base + (1 if index < remainder else 0) for index in range(chunk_count)]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _with_run_id(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["run_id"] = pd.Series(dtype=str)
        return result
    return df.assign(run_id=run_id)


def _validate_run_id(frames: list[pd.DataFrame], run_id: str, stage_name: str) -> None:
    for frame in frames:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{stage_name} result must be a DataFrame")
        if "run_id" not in frame.columns:
            raise ValueError(f"{stage_name} result missing run_id")
        if frame.empty:
            continue
        invalid = frame["run_id"].fillna("").astype(str).ne(str(run_id))
        if invalid.any():
            raise ValueError(f"{stage_name} result contains rows for a different run_id")


def _worker_log(
    *,
    pid: int,
    worker_index: int,
    task_count: int,
    started: float,
    status: str,
    versions: list[dict] | None = None,
    chunk_range: str = "",
) -> dict[str, Any]:
    target = ""
    if versions:
        first = versions[0]
        last = versions[-1]
        target = (
            f"{first.get('strategy_name')}/{first.get('strategy_version')}"
            if first == last
            else (
                f"{first.get('strategy_name')}/{first.get('strategy_version')}"
                f"..{last.get('strategy_name')}/{last.get('strategy_version')}"
            )
        )
    return {
        "pid": pid,
        "worker_index": worker_index,
        "task_count": int(task_count),
        "target": target,
        "chunk_range": chunk_range,
        "elapsed_seconds": time.perf_counter() - started,
        "status": status,
    }


def _worker_failure(
    pid: int,
    worker_index: int,
    task_count: int,
    versions: list[dict],
    started: float,
    status: str,
    exc: Exception,
    *,
    chunk_range: str = "",
) -> dict[str, Any]:
    log = _worker_log(
        pid=pid,
        worker_index=worker_index,
        task_count=task_count,
        versions=versions,
        chunk_range=chunk_range,
        started=started,
        status=status,
    )
    return {
        "status": "failed",
        "pid": pid,
        "worker_index": worker_index,
        "task_count": int(task_count),
        "target": log.get("target", ""),
        "chunk_range": chunk_range,
        "elapsed_seconds": log["elapsed_seconds"],
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "worker_log": log,
    }


def _future_failure(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    versions = list(task.get("versions") or [])
    worker_index = int(task.get("worker_index", -1))
    task_count = len(versions) if versions else int(task.get("limit", 0) or 0)
    chunk_range = ""
    if "offset" in task:
        chunk_range = f"{task.get('offset')}:{int(task.get('offset', 0)) + task_count}"
    log = _worker_log(
        pid=0,
        worker_index=worker_index,
        task_count=task_count,
        versions=versions,
        chunk_range=chunk_range,
        started=time.perf_counter(),
        status="failed",
    )
    return {
        "status": "failed",
        "pid": 0,
        "worker_index": worker_index,
        "task_count": task_count,
        "target": log["target"],
        "chunk_range": chunk_range,
        "elapsed_seconds": log["elapsed_seconds"],
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
        "worker_log": log,
    }


def _stage_summary(
    stage_name: str,
    requested_workers: int,
    effective_workers: int,
    started: float,
    *,
    rows: int,
    worker_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage_name": stage_name,
        "requested_workers": int(requested_workers),
        "workers": int(effective_workers),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": int(rows),
        "worker_logs": worker_logs,
        "worker_errors": [],
    }
