"""Unified data update workflow for local market data ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.pipeline.update_daily_bars import update_daily_bars
from src.pipeline.update_daily_basic import update_daily_basic
from src.pipeline.update_index_daily import update_index_daily
from src.pipeline.update_limit_list_daily import update_limit_list_daily
from src.pipeline.update_moneyflow import update_moneyflow
from src.pipeline.update_stock_basic import update_stock_basic
from src.pipeline.update_stock_limits import update_stock_limits
from src.pipeline.update_suspend_daily import update_suspend_daily
from src.pipeline.update_sw_daily import update_sw_daily
from src.pipeline.update_sw_industry_classification import update_sw_industry_classification
from src.pipeline.update_trade_calendar import update_trade_calendar


CRITICAL_STEPS = {"update_stock_basic", "update_daily_bars"}
WORKFLOW_STEPS = [
    "update_stock_basic",
    "update_trade_calendar",
    "update_daily_bars",
    "update_daily_basic",
    "update_stock_limits",
    "update_suspend_daily",
    "update_index_daily",
    "update_limit_list_daily",
    "update_moneyflow",
    "update_sw_industry_classification",
    "update_sw_daily",
]


def run_data_update_workflow(
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    sleep_seconds: float = 0.5,
    limit_stocks: int | None = None,
    limit_days: int | None = None,
    update_stock_basic_enabled: bool = True,
    update_trade_calendar_enabled: bool = True,
    update_daily_bars_enabled: bool = True,
    update_daily_basic_enabled: bool = True,
    update_stock_limits_enabled: bool = True,
    update_suspend_daily_enabled: bool = True,
    update_index_daily_enabled: bool = True,
    update_limit_list_daily_enabled: bool = True,
    update_moneyflow_enabled: bool = True,
    update_sw_industry_classification_enabled: bool = True,
    update_sw_daily_enabled: bool = True,
    mode: str = "test",
    output_dir: str = "reports",
    export_report: bool = True,
) -> dict:
    """Run selected data update steps and return a structured summary."""
    resolved_mode = _resolve_mode(mode)
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    provider = str(getattr(settings, "DEFAULT_DATA_PROVIDER", "tushare") or "tushare").strip().lower()
    resolved_limit_stocks = 50 if resolved_mode == "test" and limit_stocks is None else limit_stocks
    resolved_limit_days = 10 if resolved_mode == "test" and limit_days is None else limit_days
    started_at = datetime.now().isoformat(timespec="seconds")

    summary: dict[str, Any] = {
        "db_path": resolved_db_path,
        "provider": provider,
        "provider_message": _provider_message(provider),
        "start_date": start_date,
        "end_date": end_date,
        "mode": resolved_mode,
        "sleep_seconds": sleep_seconds,
        "limit_stocks": resolved_limit_stocks,
        "limit_days": resolved_limit_days,
        "steps": [],
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "errors": [],
        "critical_failure": False,
        "started_at": started_at,
        "finished_at": "",
    }

    step_specs: list[tuple[str, bool, Callable[[], Any]]] = [
        (
            "update_stock_basic",
            update_stock_basic_enabled,
            lambda: update_stock_basic(db_path=resolved_db_path, provider=provider),
        ),
        (
            "update_trade_calendar",
            update_trade_calendar_enabled,
            lambda: update_trade_calendar(start_date=start_date, end_date=end_date, db_path=resolved_db_path),
        ),
        (
            "update_daily_bars",
            update_daily_bars_enabled,
            lambda: update_daily_bars(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                limit=resolved_limit_stocks,
                sleep_seconds=sleep_seconds,
                provider=provider,
            ),
        ),
        (
            "update_daily_basic",
            update_daily_basic_enabled,
            lambda: update_daily_basic(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
                limit_days=resolved_limit_days,
            ),
        ),
        (
            "update_stock_limits",
            update_stock_limits_enabled,
            lambda: update_stock_limits(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
                limit_days=resolved_limit_days,
            ),
        ),
        (
            "update_suspend_daily",
            update_suspend_daily_enabled,
            lambda: update_suspend_daily(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
                limit_days=resolved_limit_days,
            ),
        ),
        (
            "update_index_daily",
            update_index_daily_enabled,
            lambda: update_index_daily(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
            ),
        ),
        (
            "update_limit_list_daily",
            update_limit_list_daily_enabled,
            lambda: update_limit_list_daily(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
                limit_days=resolved_limit_days,
            ),
        ),
        (
            "update_moneyflow",
            update_moneyflow_enabled,
            lambda: update_moneyflow(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
                limit_days=resolved_limit_days,
            ),
        ),
        (
            "update_sw_industry_classification",
            update_sw_industry_classification_enabled,
            lambda: update_sw_industry_classification(db_path=resolved_db_path),
        ),
        (
            "update_sw_daily",
            update_sw_daily_enabled,
            lambda: update_sw_daily(
                start_date=start_date,
                end_date=end_date,
                db_path=resolved_db_path,
                sleep_seconds=sleep_seconds,
                limit_days=resolved_limit_days,
            ),
        ),
    ]

    for step_name, enabled, runner in step_specs:
        step_result = _run_step(step_name, enabled, runner)
        summary["steps"].append(step_result)
        if step_result["status"] == "success":
            summary["success_count"] += 1
        elif step_result["status"] == "skipped":
            summary["skipped_count"] += 1
        else:
            summary["failed_count"] += 1
            summary["errors"].append({"step_name": step_name, "error": step_result["error"]})
            if step_name in CRITICAL_STEPS:
                summary["critical_failure"] = True

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if export_report:
        summary["data_update_report_path"] = export_data_update_report(summary, output_dir=output_dir)
    return summary


def export_data_update_report(summary: dict, output_dir: str = "reports") -> str:
    output_path = Path(output_dir) / f"data_update_workflow_{datetime.now().date().isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 数据更新工作流报告",
        "",
        f"- 日期区间：{summary.get('start_date')} - {summary.get('end_date')}",
        f"- 数据源：{summary.get('provider')}",
        f"- 模式：{summary.get('mode')}",
        f"- 数据库：{summary.get('db_path')}",
        f"- 成功步骤：{summary.get('success_count')}",
        f"- 失败步骤：{summary.get('failed_count')}",
        f"- 跳过步骤：{summary.get('skipped_count')}",
        "",
        "## 各步骤执行结果",
        "",
        "| step_name | status | rows | message | error |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for step in summary.get("steps", []):
        rows = step.get("rows_written", step.get("rows_loaded", ""))
        lines.append(
            "| {step_name} | {status} | {rows} | {message} | {error} |".format(
                step_name=step.get("step_name", ""),
                status=step.get("status", ""),
                rows="" if rows is None else rows,
                message=_markdown_cell(step.get("message", "")),
                error=_markdown_cell(step.get("error", "")),
            )
        )
    lines.extend(["", "## 失败步骤", ""])
    errors = summary.get("errors", [])
    if errors:
        lines.extend(f"- {item.get('step_name')}: {_markdown_cell(item.get('error', ''))}" for item in errors)
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 后续建议",
            "",
            "- 若 test 模式通过，再显式使用 --mode full 做夜间全量更新。",
            "- 若存在 failed 步骤，优先检查 Tushare 权限、日期范围、频率限制和本地 trade_calendar。",
            "- 本工作流只负责数据拉取与保存，不执行 LLM、策略研究、日度计划或自动下单。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def _run_step(step_name: str, enabled: bool, runner: Callable[[], Any]) -> dict:
    if not enabled:
        return {
            "step_name": step_name,
            "status": "skipped",
            "rows_written": None,
            "message": "step skipped by flag",
            "error": "",
        }
    try:
        result = runner()
    except Exception as exc:
        return {
            "step_name": step_name,
            "status": "failed",
            "rows_written": None,
            "message": "step failed; workflow continues",
            "error": _sanitize(str(exc)),
        }
    rows = _result_rows(result)
    return {
        "step_name": step_name,
        "status": "success",
        "rows_written": rows,
        "message": "step completed",
        "error": "",
    }


def _result_rows(result: Any) -> int | None:
    if isinstance(result, pd.DataFrame):
        return len(result)
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return len(item)
    if hasattr(result, "__len__") and not isinstance(result, (str, bytes, dict)):
        try:
            return len(result)
        except TypeError:
            return None
    return None


def _provider_message(provider: str) -> str:
    if provider == "tushare":
        return "DEFAULT_DATA_PROVIDER=tushare"
    return "DEFAULT_DATA_PROVIDER is not tushare; tushare is recommended for this workflow"


def _resolve_mode(mode: str) -> str:
    normalized = str(mode or "test").strip().lower()
    if normalized not in {"test", "full"}:
        raise ValueError("mode must be 'test' or 'full'")
    return normalized


def _sanitize(value: str) -> str:
    sanitized = value
    for secret in [
        getattr(settings, "TUSHARE_TOKEN", ""),
        getattr(settings, "LLM_API_KEY", ""),
        getattr(settings, "OPENAI_API_KEY", ""),
    ]:
        if secret:
            sanitized = sanitized.replace(str(secret), "***")
    return sanitized


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified local data update workflow.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--limit-stocks", type=int, default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--skip-stock-basic", action="store_true")
    parser.add_argument("--skip-trade-calendar", action="store_true")
    parser.add_argument("--skip-daily-bars", action="store_true")
    parser.add_argument("--skip-daily-basic", action="store_true")
    parser.add_argument("--skip-stock-limits", action="store_true")
    parser.add_argument("--skip-suspend-daily", action="store_true")
    parser.add_argument("--skip-index-daily", action="store_true")
    parser.add_argument("--skip-limit-list-daily", action="store_true")
    parser.add_argument("--skip-moneyflow", action="store_true")
    parser.add_argument("--skip-sw-industry-classification", action="store_true")
    parser.add_argument("--skip-sw-daily", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "full":
        print("full mode may take a long time")
    summary = run_data_update_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        db_path=args.db_path,
        sleep_seconds=args.sleep_seconds,
        limit_stocks=args.limit_stocks,
        limit_days=args.limit_days,
        update_stock_basic_enabled=not args.skip_stock_basic,
        update_trade_calendar_enabled=not args.skip_trade_calendar,
        update_daily_bars_enabled=not args.skip_daily_bars,
        update_daily_basic_enabled=not args.skip_daily_basic,
        update_stock_limits_enabled=not args.skip_stock_limits,
        update_suspend_daily_enabled=not args.skip_suspend_daily,
        update_index_daily_enabled=not args.skip_index_daily,
        update_limit_list_daily_enabled=not args.skip_limit_list_daily,
        update_moneyflow_enabled=not args.skip_moneyflow,
        update_sw_industry_classification_enabled=not args.skip_sw_industry_classification,
        update_sw_daily_enabled=not args.skip_sw_daily,
        mode=args.mode,
        output_dir=args.output_dir,
        export_report=not args.no_report,
    )
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("Data update workflow finished.")
    print(f"provider: {summary['provider']}")
    print(f"date range: {summary['start_date']} - {summary['end_date']}")
    print(f"mode: {summary['mode']}")
    print(f"success_count: {summary['success_count']}")
    print(f"failed_count: {summary['failed_count']}")
    print(f"skipped_count: {summary['skipped_count']}")
    print(f"errors: {summary['errors']}")
    for step in summary["steps"]:
        if step["status"] == "success":
            print(f"[success] {step['step_name']} rows={step.get('rows_written')}")
        elif step["status"] == "failed":
            print(f"[failed] {step['step_name']} error={step.get('error')}")
        else:
            print(f"[skipped] {step['step_name']}")
    if summary.get("data_update_report_path"):
        print(f"data_update_report_path: {summary['data_update_report_path']}")


if __name__ == "__main__":
    main()
