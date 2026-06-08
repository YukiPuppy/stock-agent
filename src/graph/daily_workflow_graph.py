"""LangGraph skeleton for the local daily A-share workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.graph.state import DailyWorkflowState
from src.pipeline.build_candidate_pool import build_candidate_pool
from src.pipeline.build_daily_factors import build_daily_factors
from src.pipeline.build_strategy_signals import build_strategy_signals
from src.pipeline.build_trade_plan import build_trade_plan
from src.pipeline.export_daily_report import export_daily_report
from src.pipeline.update_daily_bars import update_daily_bars
from src.pipeline.update_stock_basic import update_stock_basic


def _row_count(result: Any) -> int:
    if result is None:
        return 0
    return len(result)


def _record_error(state: DailyWorkflowState, step: str, exc: Exception) -> None:
    state["errors"] = [*state.get("errors", []), f"{step}: {exc}"]


def _resolve_provider(provider: str | None) -> str:
    return str(provider if provider is not None else settings.DEFAULT_DATA_PROVIDER).strip().lower()


def node_update_stock_basic(state: DailyWorkflowState) -> DailyWorkflowState:
    if not state["update_stock_basic_first"]:
        return {**state, "stock_basic_rows": state.get("stock_basic_rows", -1)}

    try:
        stock_basic = update_stock_basic(db_path=state["db_path"], provider=state["provider"])
    except Exception as exc:
        _record_error(state, "update_stock_basic", exc)
        raise
    return {**state, "stock_basic_rows": _row_count(stock_basic)}


def node_update_daily_bars(state: DailyWorkflowState) -> DailyWorkflowState:
    try:
        daily_bars = update_daily_bars(
            start_date=state["start_date"],
            end_date=state["end_date"],
            db_path=state["db_path"],
            limit=state["limit"],
            sleep_seconds=state["sleep_seconds"],
            provider=state["provider"],
        )
    except Exception as exc:
        _record_error(state, "update_daily_bars", exc)
        raise
    return {**state, "daily_bars_rows": _row_count(daily_bars)}


def node_build_daily_factors(state: DailyWorkflowState) -> DailyWorkflowState:
    try:
        daily_factors = build_daily_factors(db_path=state["db_path"])
    except Exception as exc:
        _record_error(state, "build_daily_factors", exc)
        raise
    return {**state, "daily_factors_rows": _row_count(daily_factors)}


def node_build_strategy_signals(state: DailyWorkflowState) -> DailyWorkflowState:
    try:
        strategy_signals = build_strategy_signals(
            db_path=state["db_path"],
            use_active_candidates=state["use_active_candidates"],
            active_config_path=state["active_config_path"],
        )
    except Exception as exc:
        _record_error(state, "build_strategy_signals", exc)
        raise
    return {**state, "strategy_signals_rows": _row_count(strategy_signals)}


def node_build_candidate_pool(state: DailyWorkflowState) -> DailyWorkflowState:
    if state["use_active_candidates"] and state.get("strategy_signals_rows", 0) == 0:
        return {**state, "candidate_pool_rows": 0}

    try:
        candidate_pool = build_candidate_pool(
            top_n=state["top_n"],
            min_amount_ma5=state["min_amount_ma5"],
            db_path=state["db_path"],
        )
    except Exception as exc:
        _record_error(state, "build_candidate_pool", exc)
        raise
    return {**state, "candidate_pool_rows": _row_count(candidate_pool)}


def node_build_trade_plan(state: DailyWorkflowState) -> DailyWorkflowState:
    if state["use_active_candidates"] and state.get("candidate_pool_rows", 0) == 0:
        return {**state, "trade_plan_rows": 0}

    try:
        trade_plan = build_trade_plan(
            max_items=state["max_plan_items"],
            db_path=state["db_path"],
        )
    except Exception as exc:
        _record_error(state, "build_trade_plan", exc)
        raise
    return {**state, "trade_plan_rows": _row_count(trade_plan)}


def node_export_daily_report(state: DailyWorkflowState) -> DailyWorkflowState:
    if not state["export_report"]:
        return {**state, "report_path": None}

    try:
        report_path = export_daily_report(db_path=state["db_path"], output_dir="reports")
    except Exception as exc:
        _record_error(state, "export_daily_report", exc)
        raise
    return {**state, "report_path": report_path}


def build_daily_workflow_graph():
    graph = StateGraph(DailyWorkflowState)
    graph.add_node("update_stock_basic", node_update_stock_basic)
    graph.add_node("update_daily_bars", node_update_daily_bars)
    graph.add_node("build_daily_factors", node_build_daily_factors)
    graph.add_node("build_strategy_signals", node_build_strategy_signals)
    graph.add_node("build_candidate_pool", node_build_candidate_pool)
    graph.add_node("build_trade_plan", node_build_trade_plan)
    graph.add_node("export_daily_report", node_export_daily_report)

    graph.add_edge(START, "update_stock_basic")
    graph.add_edge("update_stock_basic", "update_daily_bars")
    graph.add_edge("update_daily_bars", "build_daily_factors")
    graph.add_edge("build_daily_factors", "build_strategy_signals")
    graph.add_edge("build_strategy_signals", "build_candidate_pool")
    graph.add_edge("build_candidate_pool", "build_trade_plan")
    graph.add_edge("build_trade_plan", "export_daily_report")
    graph.add_edge("export_daily_report", END)
    return graph


def run_daily_workflow_graph(
    start_date: str,
    end_date: str,
    provider: str | None = None,
    limit: int | None = None,
    sleep_seconds: float = 1.0,
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    db_path: str | None = None,
    update_stock_basic_first: bool = True,
    export_report: bool = True,
    use_active_candidates: bool = False,
    active_config_path: str = "configs/active_strategies_candidate.json",
) -> dict:
    resolved_provider = _resolve_provider(provider)
    initial_state: DailyWorkflowState = {
        "provider": resolved_provider,
        "db_path": db_path,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "sleep_seconds": sleep_seconds,
        "top_n": top_n,
        "max_plan_items": max_plan_items,
        "min_amount_ma5": min_amount_ma5,
        "update_stock_basic_first": update_stock_basic_first,
        "export_report": export_report,
        "use_active_candidates": use_active_candidates,
        "active_config_path": active_config_path,
        "stock_basic_rows": 0,
        "daily_bars_rows": 0,
        "daily_factors_rows": 0,
        "strategy_signals_rows": 0,
        "candidate_pool_rows": 0,
        "trade_plan_rows": 0,
        "report_path": None,
        "errors": [],
    }
    compiled_graph = build_daily_workflow_graph().compile()
    return dict(compiled_graph.invoke(initial_state))
