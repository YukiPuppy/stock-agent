"""Run the full local A-share daily workflow."""

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
from src.pipeline.update_stock_basic import update_stock_basic
from src.utils.network import clear_proxy_env_for_process


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else settings.DB_PATH


def _resolve_provider(provider: str | None) -> str:
    return str(provider if provider is not None else settings.DEFAULT_DATA_PROVIDER).strip().lower()


def _row_count(result: Any) -> int:
    if result is None:
        return 0
    return len(result)


def run_daily_workflow(
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
    report_path: str | None = None,
    use_active_candidates: bool = False,
    active_config_path: str = "configs/active_strategies_candidate.json",
) -> dict:
    """Run stock-basic, daily-bar, factor, candidate, and trade-plan steps."""
    resolved_provider = _resolve_provider(provider)
    stock_basic_rows = 0

    if update_stock_basic_first:
        stock_basic = update_stock_basic(db_path=db_path, provider=resolved_provider)
        stock_basic_rows = _row_count(stock_basic)

    daily_bars = update_daily_bars(
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        limit=limit,
        sleep_seconds=sleep_seconds,
        provider=resolved_provider,
    )
    daily_factors = build_daily_factors(db_path=db_path)
    strategy_signals = build_strategy_signals(
        db_path=db_path,
        use_active_candidates=use_active_candidates,
        active_config_path=active_config_path,
    )
    if use_active_candidates and _row_count(strategy_signals) == 0:
        candidate_pool = None
        trade_plan = None
    else:
        candidate_pool = build_candidate_pool(
            top_n=top_n,
            min_amount_ma5=min_amount_ma5,
            db_path=db_path,
        )
        trade_plan = build_trade_plan(
            max_items=max_plan_items,
            db_path=db_path,
        )
    exported_report_path = None
    if export_report:
        exported_report_path = export_daily_report(
            db_path=db_path,
            output_dir=report_path or "reports",
        )

    return {
        "provider": resolved_provider,
        "db_path": _resolve_db_path(db_path),
        "start_date": start_date,
        "end_date": end_date,
        "stock_basic_rows": stock_basic_rows,
        "daily_bars_rows": _row_count(daily_bars),
        "daily_factors_rows": _row_count(daily_factors),
        "strategy_signals_rows": _row_count(strategy_signals),
        "candidate_pool_rows": _row_count(candidate_pool),
        "trade_plan_rows": _row_count(trade_plan),
        "report_path": exported_report_path,
        "use_active_candidates": use_active_candidates,
        "active_config_path": active_config_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full local A-share daily workflow.")
    parser.add_argument("--start-date", required=True, help="Start date, format YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date, format YYYYMMDD.")
    parser.add_argument(
        "--provider",
        default=settings.DEFAULT_DATA_PROVIDER,
        help="默认数据源来自 DEFAULT_DATA_PROVIDER，当前推荐使用 tushare。",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-plan-items", type=int, default=5)
    parser.add_argument(
        "--min-amount-ma5",
        type=float,
        default=0.0,
        help="Minimum amount_ma5 filter, in thousand yuan.",
    )
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--skip-stock-basic",
        action="store_true",
        help="Skip updating the stock-basic universe before daily bars.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip exporting the daily Markdown report.",
    )
    parser.add_argument("--use-active-candidates", action="store_true", default=False)
    parser.add_argument("--active-config-path", default="configs/active_strategies_candidate.json")
    return parser.parse_args(argv)


def _print_summary(summary: dict) -> None:
    print("Daily workflow finished.")
    for key in (
        "provider",
        "db_path",
        "start_date",
        "end_date",
        "stock_basic_rows",
        "daily_bars_rows",
        "daily_factors_rows",
        "strategy_signals_rows",
        "candidate_pool_rows",
        "trade_plan_rows",
        "use_active_candidates",
        "active_config_path",
    ):
        print(f"{key}: {summary[key]}")
    if "report_path" in summary:
        print(f"report_path: {summary['report_path']}")


def main(argv: Sequence[str] | None = None) -> None:
    clear_proxy_env_for_process()
    args = _parse_args(argv)
    summary = run_daily_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        provider=args.provider,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        top_n=args.top_n,
        max_plan_items=args.max_plan_items,
        min_amount_ma5=args.min_amount_ma5,
        db_path=args.db_path,
        update_stock_basic_first=not args.skip_stock_basic,
        export_report=not args.no_report,
        use_active_candidates=args.use_active_candidates,
        active_config_path=args.active_config_path,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
