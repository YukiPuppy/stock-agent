"""Rerun only trade-plan validation and strategy admission for an existing run."""

from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.trade_plan_backtester import (
    DEFAULT_MAX_HOLDING_DAYS,
    backtest_trade_plans,
    expand_trade_plans_for_holding_days,
)
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.pipeline.backtest_trade_plans import _load_daily_bars_for_plans
from src.pipeline.memory import log_memory
from src.pipeline.rebuild_strategy_admission import rebuild_strategy_admission
from src.research.strategy_admission import explode_trade_plan_strategy_dimensions


LOGGER = logging.getLogger(__name__)
_PLAN_CHUNK_SIZE = 500
_RESERVED_FLAGS = (
    "trailing_stop",
    "break_even_after_tp1",
    "extend_holding_on_trend",
    "exit_on_industry_weakness",
    "exit_on_moneyflow_weakness",
    "include_holding_days_in_parameter_search",
)


def rerun_trade_plan_and_admission(
    *,
    run_id: str,
    db_path: str | None = None,
    output_dir: str = "reports",
    holding_days_mode: str = "strategy_grid",
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    low_memory: bool = True,
    replace_current_run: bool = False,
    reserved_features: dict[str, bool] | None = None,
) -> dict:
    resolved_db_path = db_path or DB_PATH
    store = StockAgentStore(resolved_db_path)
    reserved_features = reserved_features or {}
    requested_reserved = sorted(name for name in _RESERVED_FLAGS if reserved_features.get(name))
    if requested_reserved:
        LOGGER.warning(
            "reserved high-risk features are not implemented or enabled; unchanged deterministic exits will be used: %s",
            ", ".join(requested_reserved),
        )

    parameter_results = store.load_parameter_search_results(run_id=run_id)
    walk_forward = store.load_walk_forward_validation(run_id=run_id)
    if parameter_results.empty:
        raise RuntimeError(f"run_id={run_id} has no parameter_search_results; trade-plan-only rerun cannot select candidates")
    if walk_forward.empty:
        LOGGER.warning("run_id=%s has no walk_forward_validation; candidate filtering cannot apply OOS status", run_id)
    eligible_keys = _eligible_candidate_keys(parameter_results, walk_forward)

    with store._connect() as con:
        store._create_tables(con)
        plan_count = int(
            con.execute("SELECT count(*) FROM historical_trade_plans WHERE run_id = ?", [run_id]).fetchone()[0]
        )
        dimension_rows = con.execute(
            """
            SELECT DISTINCT strategy_names, strategy_versions
            FROM historical_trade_plans
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchdf()
    if plan_count == 0:
        raise RuntimeError(
            f"run_id={run_id} has no historical_trade_plans; this entry never reruns parameter_search or OOS"
        )
    filter_by_candidates = _has_candidate_overlap(dimension_rows, eligible_keys)
    if not filter_by_candidates:
        LOGGER.warning(
            "parameter/OOS candidate keys do not overlap persisted historical plan dimensions; "
            "using the existing run-scoped historical plans as the previously selected candidate set"
        )

    if replace_current_run:
        store.delete_run_rows(
            run_id,
            ["trade_plan_backtest_results", "trade_plan_backtest_performance", "strategy_admission"],
        )

    log_memory("rerun_trade_plan_and_admission", "before_chunks")
    result_count = 0
    selected_plan_count = 0
    for offset in range(0, plan_count, _PLAN_CHUNK_SIZE):
        plan_chunk = _load_plan_chunk(store, run_id, _PLAN_CHUNK_SIZE, offset)
        if filter_by_candidates:
            plan_chunk = _filter_candidate_plans(plan_chunk, eligible_keys)
        selected_plan_count += len(plan_chunk)
        expanded = expand_trade_plans_for_holding_days(
            plan_chunk,
            mode=holding_days_mode,
            max_holding_days=max_holding_days,
        )
        chunk_max = int(expanded["max_holding_days"].max()) if not expanded.empty else max_holding_days
        daily_bars = _load_daily_bars_for_plans(store, expanded, None, chunk_max)
        result_chunk = backtest_trade_plans(expanded, daily_bars, max_holding_days=max_holding_days).assign(run_id=run_id)
        store.save_trade_plan_backtest_results(result_chunk)
        result_count += len(result_chunk)
        del plan_chunk, expanded, daily_bars, result_chunk
        gc.collect()
        log_memory("rerun_trade_plan_and_admission", f"chunk_written_{offset // _PLAN_CHUNK_SIZE + 1}")

    performance = store.aggregate_trade_plan_backtest_performance(run_id)
    if replace_current_run:
        store.delete_run_rows(run_id, ["trade_plan_backtest_performance"])
    store.save_trade_plan_backtest_performance(performance)
    trade_plan_report_path = export_trade_plan_backtest_report_low_memory(store, run_id, performance, output_dir)
    admission_summary = rebuild_strategy_admission(
        run_id=run_id,
        db_path=resolved_db_path,
        output_dir=output_dir,
        replace_current_run=True,
    )
    log_memory("rerun_trade_plan_and_admission", "finished")
    return {
        "run_id": run_id,
        "db_path": resolved_db_path,
        "low_memory": bool(low_memory),
        "holding_days_mode": holding_days_mode,
        "historical_trade_plans_rows": plan_count,
        "selected_trade_plans_rows": selected_plan_count,
        "trade_plan_backtest_results_rows": result_count,
        "trade_plan_backtest_performance_rows": len(performance),
        "strategy_admission_rows": admission_summary["strategy_admission_rows"],
        "trade_plan_win_rate_nonnull_rows": admission_summary["trade_plan_win_rate_nonnull_rows"],
        "trade_plan_backtest_report_path": trade_plan_report_path,
        "strategy_admission_report_path": admission_summary["strategy_admission_report_path"],
        "reserved_features_requested_but_not_enabled": requested_reserved,
    }


def _eligible_candidate_keys(parameter_results: pd.DataFrame, walk_forward: pd.DataFrame) -> set[tuple[str, str]]:
    candidates = parameter_results.copy()
    recommendations = candidates.get("recommendation", pd.Series("", index=candidates.index)).fillna("").astype(str)
    candidates = candidates.loc[~recommendations.isin(["pause", "reduce_or_pause"])].copy()
    if not walk_forward.empty:
        oos = walk_forward[[
            column
            for column in ["strategy_name", "strategy_version", "validation_status", "overfit_risk"]
            if column in walk_forward.columns
        ]].copy()
        candidates = candidates.merge(oos, on=["strategy_name", "strategy_version"], how="left")
        status = candidates.get("validation_status", pd.Series("", index=candidates.index)).fillna("")
        risk = candidates.get("overfit_risk", pd.Series("", index=candidates.index)).fillna("")
        candidates = candidates.loc[~status.isin(["failed_oos", "unstable"]) & risk.ne("high")]
    return {
        (str(row["strategy_name"]).strip(), str(row["strategy_version"] or "v1").strip())
        for _, row in candidates.iterrows()
        if pd.notna(row.get("strategy_name"))
    }


def _has_candidate_overlap(dimensions: pd.DataFrame, eligible_keys: set[tuple[str, str]]) -> bool:
    if dimensions.empty or not eligible_keys:
        return False
    exploded = explode_trade_plan_strategy_dimensions(dimensions, valid_strategy_keys=eligible_keys)
    return any(
        (str(row["strategy_name"]), str(row["strategy_version"])) in eligible_keys
        for _, row in exploded.iterrows()
    )


def _filter_candidate_plans(plans: pd.DataFrame, eligible_keys: set[tuple[str, str]]) -> pd.DataFrame:
    keep = []
    for index, row in plans.iterrows():
        exploded = explode_trade_plan_strategy_dimensions(pd.DataFrame([row]), valid_strategy_keys=eligible_keys)
        if any(
            (str(item["strategy_name"]), str(item["strategy_version"])) in eligible_keys
            for _, item in exploded.iterrows()
        ):
            keep.append(index)
    return plans.loc[keep].copy()


def _load_plan_chunk(store: StockAgentStore, run_id: str, limit: int, offset: int) -> pd.DataFrame:
    with store._connect() as con:
        store._create_tables(con)
        return con.execute(
            """
            SELECT * EXCLUDE (run_id)
            FROM historical_trade_plans
            WHERE run_id = ?
            ORDER BY trade_date, rank, code, strategy_names, strategy_versions
            LIMIT ? OFFSET ?
            """,
            [run_id, limit, offset],
        ).fetchdf()


def export_trade_plan_backtest_report_low_memory(
    store: StockAgentStore,
    run_id: str,
    performance: pd.DataFrame,
    output_dir: str,
) -> str:
    with store._connect() as con:
        store._create_tables(con)
        holding_distribution = con.execute(
            """SELECT strategy_names, holding_days, count(*) AS count
               FROM trade_plan_backtest_results WHERE run_id = ?
               GROUP BY strategy_names, holding_days ORDER BY strategy_names, holding_days""",
            [run_id],
        ).fetchdf()
        exit_distribution = con.execute(
            """SELECT strategy_names, exit_reason, count(*) AS count
               FROM trade_plan_backtest_results WHERE run_id = ? AND coalesce(is_valid, false)
               GROUP BY strategy_names, exit_reason ORDER BY strategy_names, exit_reason""",
            [run_id],
        ).fetchdf()
        by_holding = con.execute(
            """SELECT max_holding_days, count(*) AS plan_count,
                      count(*) FILTER (WHERE coalesce(is_valid, false)) AS valid_count,
                      avg(CASE WHEN coalesce(is_valid, false) THEN (return_pct > 0)::INTEGER END) AS win_rate,
                      avg(return_pct) FILTER (WHERE coalesce(is_valid, false)) AS avg_return,
                      avg(max_drawdown) FILTER (WHERE coalesce(is_valid, false)) AS avg_max_drawdown
               FROM trade_plan_backtest_results WHERE run_id = ?
               GROUP BY max_holding_days ORDER BY max_holding_days""",
            [run_id],
        ).fetchdf()
    report_date = date.today().isoformat()
    lines = [
        "# 交易计划多持仓周期回测报告",
        "",
        f"报告日期：{report_date}",
        f"run_id：{run_id}",
        "",
        "本报告由分块读取和 DuckDB 聚合生成；未在内存中拼接全部回测明细。",
        "不构成投资建议，不启用策略，不生成交易指令。",
        "",
        "## holding_days distribution by strategy",
        "",
        *_markdown_table(holding_distribution),
        "",
        "## exit_reason distribution by strategy",
        "",
        *_markdown_table(exit_distribution),
        "",
        "## performance by max_holding_days",
        "",
        *_markdown_table(by_holding),
        "",
        "## performance by strategy_name / strategy_version / max_holding_days",
        "",
        *_markdown_table(performance),
        "",
        "高风险退出增强功能均未启用；max_holding_days 未进入 parameter_search 主搜索。",
    ]
    output_path = Path(output_dir) / f"trade_plan_backtest_{run_id}_{report_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(output_path)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return ["当前没有可用数据。"]
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = ["-" if pd.isna(row.get(column)) else str(row.get(column)).replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun trade-plan backtests and admission for one run without parameter search or OOS."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--holding-days-mode", choices=["fixed", "strategy_grid"], default="strategy_grid")
    parser.add_argument("--max-holding-days", type=int, default=DEFAULT_MAX_HOLDING_DAYS)
    parser.add_argument("--low-memory", action="store_true", help="Compatibility flag; this command is always chunked.")
    parser.add_argument("--replace-current-run", action="store_true")
    parser.add_argument("--trailing-stop", action="store_true", help="Reserved only; not enabled.")
    parser.add_argument("--break-even-after-tp1", action="store_true", help="Reserved only; not enabled.")
    parser.add_argument("--extend-holding-on-trend", action="store_true", help="Reserved only; not enabled.")
    parser.add_argument("--exit-on-industry-weakness", action="store_true", help="Reserved only; not enabled.")
    parser.add_argument("--exit-on-moneyflow-weakness", action="store_true", help="Reserved only; not enabled.")
    parser.add_argument(
        "--include-holding-days-in-parameter-search",
        action="store_true",
        help="Reserved only and default-off; this command never runs parameter search.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args(argv)
    reserved = {name: bool(getattr(args, name)) for name in _RESERVED_FLAGS}
    summary = rerun_trade_plan_and_admission(
        run_id=args.run_id,
        db_path=args.db_path,
        output_dir=args.output_dir,
        holding_days_mode=args.holding_days_mode,
        max_holding_days=args.max_holding_days,
        low_memory=True,
        replace_current_run=args.replace_current_run,
        reserved_features=reserved,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
