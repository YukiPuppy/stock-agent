"""Run the LangGraph daily workflow skeleton."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.config import settings
from src.graph.daily_workflow_graph import run_daily_workflow_graph
from src.utils.network import clear_proxy_env_for_process


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph local A-share daily workflow.")
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
    print("LangGraph daily workflow finished.")
    for key in (
        "provider",
        "db_path",
        "stock_basic_rows",
        "daily_bars_rows",
        "daily_factors_rows",
        "strategy_signals_rows",
        "candidate_pool_rows",
        "trade_plan_rows",
        "use_active_candidates",
        "active_config_path",
        "report_path",
    ):
        print(f"{key}: {summary[key]}")


def main(argv: Sequence[str] | None = None) -> None:
    clear_proxy_env_for_process()
    args = _parse_args(argv)
    summary = run_daily_workflow_graph(
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
