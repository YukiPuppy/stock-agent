"""Run the daily operations workflow across local stock-agent pipelines."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.pipeline.run_data_update_workflow import run_data_update_workflow
from src.pipeline.run_daily_planning_workflow import run_daily_planning_workflow
from src.pipeline.run_factor_build_workflow import run_factor_build_workflow
from src.pipeline.run_llm_agents_workflow import run_llm_agents_workflow
from src.pipeline.run_system_health_check import export_system_health_report
from src.diagnostics.system_health import run_system_health_check


WORKFLOW_STEPS = [
    "run_data_update_workflow",
    "run_factor_build_workflow",
    "run_daily_planning_workflow",
    "run_system_health_check",
    "run_llm_agents_workflow",
]


def run_daily_ops_workflow(
    start_date: str | None = None,
    end_date: str | None = None,
    trade_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
    update_data: bool = False,
    data_update_mode: str = "test",
    sleep_seconds: float = 0.5,
    limit_stocks: int | None = None,
    limit_days: int | None = None,
    build_factors: bool = True,
    run_daily_plan: bool = True,
    run_health_check: bool = True,
    run_llm_agents: bool = True,
) -> dict:
    """Run the daily post-market operations workflow and return a summary.

    The workflow never places orders and does not write strategy configuration
    files. Market data updates are opt-in to avoid accidental long daytime runs.
    """
    if update_data and (not start_date or not end_date):
        raise ValueError("start_date and end_date are required when update_data=True.")

    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    started_at = datetime.now().isoformat(timespec="seconds")
    summary: dict[str, Any] = {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "start_date": start_date,
        "end_date": end_date,
        "trade_date": trade_date,
        "update_data": update_data,
        "data_update_mode": data_update_mode,
        "steps": [],
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "errors": [],
        "started_at": started_at,
        "finished_at": "",
        "daily_ops_report_path": "",
        "data_update_report_path": None,
        "factor_build_report_path": None,
        "daily_report_path": None,
        "system_health_report_path": None,
        "llm_agents_index_path": None,
    }

    _run_or_skip(
        summary,
        step_name="run_data_update_workflow",
        enabled=update_data,
        runner=lambda: run_data_update_workflow(
            start_date=str(start_date),
            end_date=str(end_date),
            db_path=resolved_db_path,
            sleep_seconds=sleep_seconds,
            limit_stocks=limit_stocks,
            limit_days=limit_days,
            mode=data_update_mode,
            output_dir=output_dir,
            export_report=True,
        ),
        path_mappings={"data_update_report_path": "data_update_report_path"},
    )
    _run_or_skip(
        summary,
        step_name="run_factor_build_workflow",
        enabled=build_factors,
        runner=lambda: run_factor_build_workflow(db_path=resolved_db_path, output_dir=output_dir),
        path_mappings={"factor_build_report_path": "factor_build_report_path"},
    )
    _run_or_skip(
        summary,
        step_name="run_daily_planning_workflow",
        enabled=run_daily_plan,
        runner=lambda: run_daily_planning_workflow(
            start_date=start_date,
            end_date=end_date,
            db_path=resolved_db_path,
            sleep_seconds=sleep_seconds,
            update_data=False,
            build_factors=False,
            export_report=True,
            output_dir=output_dir,
        ),
        path_mappings={"daily_report_path": "daily_report_path"},
    )
    _run_or_skip(
        summary,
        step_name="run_system_health_check",
        enabled=run_health_check,
        runner=lambda: _run_health_check_and_export(resolved_db_path, output_dir),
        path_mappings={"system_health_report_path": "system_health_report_path"},
    )
    _run_or_skip(
        summary,
        step_name="run_llm_agents_workflow",
        enabled=run_llm_agents,
        runner=lambda: run_llm_agents_workflow(
            db_path=resolved_db_path,
            output_dir=output_dir,
            trade_date=trade_date,
        ),
        path_mappings={"llm_agents_index_path": "llm_agents_index_path"},
    )

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["daily_ops_report_path"] = export_daily_ops_report(summary, output_dir=output_dir)
    return summary


def _run_or_skip(
    summary: dict[str, Any],
    *,
    step_name: str,
    enabled: bool,
    runner: Callable[[], dict],
    path_mappings: dict[str, str],
) -> None:
    if not enabled:
        step_result = {
            "step_name": step_name,
            "status": "skipped",
            "message": "step skipped by flag",
            "error": "",
        }
        summary["steps"].append(step_result)
        summary["skipped_count"] += 1
        return

    try:
        result = runner()
    except Exception as exc:
        error = _sanitize(str(exc))
        step_result = {
            "step_name": step_name,
            "status": "failed",
            "message": "step failed; workflow continues",
            "error": error,
        }
        summary["steps"].append(step_result)
        summary["failed_count"] += 1
        summary["errors"].append({"step_name": step_name, "error": error})
        return

    for summary_key, result_key in path_mappings.items():
        if isinstance(result, dict) and result.get(result_key):
            summary[summary_key] = result.get(result_key)

    step_result = {
        "step_name": step_name,
        "status": "success",
        "message": "step completed",
        "error": "",
    }
    summary["steps"].append(step_result)
    summary["success_count"] += 1


def _run_health_check_and_export(db_path: str, output_dir: str) -> dict:
    health_summary = run_system_health_check(db_path=db_path, reports_dir=output_dir)
    report_path = export_system_health_report(health_summary, output_dir=output_dir)
    return {
        "system_health_summary": health_summary,
        "system_health_report_path": report_path,
    }


def export_daily_ops_report(summary: dict, output_dir: str = "reports") -> str:
    output_path = Path(output_dir) / f"daily_ops_workflow_{date.today().isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_summary = dict(summary)
    report_summary["daily_ops_report_path"] = str(output_path)
    output_path.write_text(_build_daily_ops_markdown(report_summary), encoding="utf-8")
    return str(output_path)


def _build_daily_ops_markdown(summary: dict) -> str:
    lines = [
        "# 日度总流程运行报告",
        "",
        "## 一、运行说明",
        "",
        "本流程用于盘后自动串联数据更新、因子构建、日度计划、健康检查和 LLM 报告。",
        "本流程不自动下单，不构成投资建议。",
        "",
        "## 二、执行参数",
        "",
        f"- start_date: {summary.get('start_date')}",
        f"- end_date: {summary.get('end_date')}",
        f"- trade_date: {summary.get('trade_date')}",
        f"- update_data: {summary.get('update_data')}",
        f"- data_update_mode: {summary.get('data_update_mode')}",
        f"- db_path: {summary.get('db_path')}",
        "",
        "## 三、步骤执行结果",
        "",
        "| step_name | status | message | error |",
        "| --- | --- | --- | --- |",
    ]
    step_by_name = {step.get("step_name"): step for step in summary.get("steps", [])}
    for step_name in WORKFLOW_STEPS:
        step = step_by_name.get(step_name, {"status": "skipped", "message": "not run", "error": ""})
        lines.append(
            "| {step_name} | {status} | {message} | {error} |".format(
                step_name=step_name,
                status=step.get("status", ""),
                message=_markdown_cell(step.get("message", "")),
                error=_markdown_cell(step.get("error", "")),
            )
        )

    lines.extend(
        [
            "",
            "## 四、生成报告",
            "",
            f"- daily_ops_report_path: {summary.get('daily_ops_report_path') or '未生成'}",
            f"- data_update_report_path: {summary.get('data_update_report_path') or '未生成'}",
            f"- factor_build_report_path: {summary.get('factor_build_report_path') or '未生成'}",
            f"- daily_report_path: {summary.get('daily_report_path') or '未生成'}",
            f"- system_health_report_path: {summary.get('system_health_report_path') or '未生成'}",
            f"- llm_agents_index_path: {summary.get('llm_agents_index_path') or '未生成'}",
            "",
            "## 五、失败步骤",
            "",
        ]
    )
    errors = summary.get("errors", [])
    if errors:
        lines.extend(f"- {item.get('step_name')}: {_markdown_cell(item.get('error', ''))}" for item in errors)
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 六、后续建议",
            "",
            "- 如果数据更新失败，可单独补跑对应 update pipeline。",
            "- 如果因子构建失败，先检查上游数据表。",
            "- 如果日度计划为空，检查 strategy_signals / candidate_pool。",
            "- 如果 LLM 报告失败，检查 LLM_PROVIDER / LLM_API_KEY / LLM_DISABLE_PROXY。",
            "",
        ]
    )
    return "\n".join(lines)


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily stock-agent operations workflow.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--update-data", action="store_true", default=False)
    parser.add_argument("--data-update-mode", choices=["test", "full"], default="test")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--limit-stocks", type=int, default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--skip-factor-build", action="store_true")
    parser.add_argument("--skip-daily-plan", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--skip-llm-agents", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_daily_ops_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        trade_date=args.trade_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
        update_data=args.update_data,
        data_update_mode=args.data_update_mode,
        sleep_seconds=args.sleep_seconds,
        limit_stocks=args.limit_stocks,
        limit_days=args.limit_days,
        build_factors=not args.skip_factor_build,
        run_daily_plan=not args.skip_daily_plan,
        run_health_check=not args.skip_health_check,
        run_llm_agents=not args.skip_llm_agents,
    )
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("Daily ops workflow finished.")
    print(f"success_count: {summary['success_count']}")
    print(f"failed_count: {summary['failed_count']}")
    print(f"skipped_count: {summary['skipped_count']}")
    print(f"daily_ops_report_path: {summary['daily_ops_report_path']}")
    if summary.get("errors"):
        print(f"errors: {summary['errors']}")


if __name__ == "__main__":
    main()
