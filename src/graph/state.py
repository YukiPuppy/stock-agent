"""State definitions for LangGraph workflows."""

from __future__ import annotations

from typing import TypedDict


class DailyWorkflowState(TypedDict):
    provider: str
    db_path: str | None
    start_date: str
    end_date: str
    limit: int | None
    sleep_seconds: float
    top_n: int
    max_plan_items: int
    min_amount_ma5: float
    update_stock_basic_first: bool
    export_report: bool
    use_active_candidates: bool
    active_config_path: str
    stock_basic_rows: int
    daily_bars_rows: int
    daily_factors_rows: int
    strategy_signals_rows: int
    candidate_pool_rows: int
    trade_plan_rows: int
    report_path: str | None
    errors: list[str]
