"""Run the local daily planning workflow for next-day observation and trade plans."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from src.config import settings
from src.pipeline.build_candidate_pool import build_candidate_pool
from src.pipeline.build_daily_factors import build_daily_factors
from src.pipeline.build_strategy_signals import build_strategy_signals
from src.pipeline.build_trade_plan import build_trade_plan
from src.pipeline.export_daily_report import export_daily_report
from src.pipeline.update_daily_bars import update_daily_bars
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _resolve_provider(provider: str | None) -> str:
    return str(provider if provider is not None else settings.DEFAULT_DATA_PROVIDER).strip().lower()


def _row_count(result: Any) -> int:
    if result is None:
        return 0
    return len(result)


def run_daily_planning_workflow(
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str | None = None,
    provider: str | None = None,
    limit: int | None = None,
    sleep_seconds: float = 1.0,
    update_data: bool = False,
    build_factors: bool = True,
    use_active_candidates: bool = True,
    active_config_path: str = "configs/active_strategies_candidate.json",
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    export_report: bool = True,
    output_dir: str = "reports",
) -> dict:
    """Build daily factors, signals, candidate pool, trade plan, and optional report.

    The default path is fully local: it does not update market data or call a data
    provider unless ``update_data`` is explicitly enabled.
    """
    resolved_db_path = _resolve_db_path(db_path)
    resolved_provider = _resolve_provider(provider)

    if update_data:
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required when update_data=True.")
        update_daily_bars(
            start_date=start_date,
            end_date=end_date,
            db_path=resolved_db_path,
            provider=resolved_provider,
            limit=limit,
            sleep_seconds=sleep_seconds,
        )

    daily_factors = None
    if build_factors:
        daily_factors = build_daily_factors(db_path=resolved_db_path)

    strategy_signals = build_strategy_signals(
        db_path=resolved_db_path,
        use_active_candidates=use_active_candidates,
        active_config_path=active_config_path,
    )
    candidate_pool = build_candidate_pool(
        db_path=resolved_db_path,
        top_n=top_n,
        min_amount_ma5=min_amount_ma5,
    )
    trade_plan = build_trade_plan(
        db_path=resolved_db_path,
        max_items=max_plan_items,
    )

    daily_report_path = None
    if export_report:
        daily_report_path = export_daily_report(
            db_path=resolved_db_path,
            output_dir=output_dir,
        )

    return {
        "db_path": resolved_db_path,
        "provider": resolved_provider,
        "start_date": start_date,
        "end_date": end_date,
        "update_data": update_data,
        "build_factors": build_factors,
        "use_active_candidates": use_active_candidates,
        "active_config_path": active_config_path,
        "daily_factors_rows": _row_count(daily_factors),
        "strategy_signals_rows": _row_count(strategy_signals),
        "candidate_pool_rows": _row_count(candidate_pool),
        "trade_plan_rows": _row_count(trade_plan),
        "daily_report_path": daily_report_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local daily planning workflow.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--provider",
        default=settings.DEFAULT_DATA_PROVIDER,
        help="默认数据源来自 DEFAULT_DATA_PROVIDER，当前推荐使用 tushare。",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--update-data", action="store_true", default=False)
    parser.add_argument("--no-build-factors", action="store_true", default=False)
    parser.add_argument("--no-active-candidates", action="store_true", default=False)
    parser.add_argument("--active-config-path", default="configs/active_strategies_candidate.json")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-plan-items", type=int, default=5)
    parser.add_argument(
        "--min-amount-ma5",
        type=float,
        default=0.0,
        help="Minimum amount_ma5 filter, in thousand yuan.",
    )
    parser.add_argument("--no-report", action="store_true", default=False)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def _print_summary(summary: dict) -> None:
    print("Daily planning workflow finished.")
    for key in (
        "update_data",
        "use_active_candidates",
        "daily_factors_rows",
        "strategy_signals_rows",
        "candidate_pool_rows",
        "trade_plan_rows",
        "daily_report_path",
    ):
        print(f"{key}: {summary[key]}")


def main(argv: Sequence[str] | None = None) -> None:
    clear_proxy_env_for_process()
    args = _parse_args(argv)
    summary = run_daily_planning_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        db_path=args.db_path,
        provider=args.provider,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        update_data=args.update_data,
        build_factors=not args.no_build_factors,
        use_active_candidates=not args.no_active_candidates,
        active_config_path=args.active_config_path,
        top_n=args.top_n,
        max_plan_items=args.max_plan_items,
        min_amount_ma5=args.min_amount_ma5,
        export_report=not args.no_report,
        output_dir=args.output_dir,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
