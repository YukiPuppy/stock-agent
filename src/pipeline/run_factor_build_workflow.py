"""Unified workflow for building local derived factor tables."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.pipeline.build_daily_factors import build_daily_factors
from src.pipeline.build_factor_diagnostics import run_build_factor_diagnostics
from src.pipeline.build_industry_strength import build_and_save_industry_strength
from src.pipeline.build_market_regime import build_market_regime
from src.pipeline.build_moneyflow_factors import build_and_save_moneyflow_factors


WORKFLOW_STEPS = [
    "build_moneyflow_factors",
    "build_market_regime",
    "build_industry_strength",
    "build_daily_factors",
    "build_factor_diagnostics",
]


def run_factor_build_workflow(
    db_path: str | None = None,
    output_dir: str = "reports",
    build_moneyflow_factors_enabled: bool = True,
    build_market_regime_enabled: bool = True,
    build_industry_strength_enabled: bool = True,
    build_daily_factors_enabled: bool = True,
    build_factor_diagnostics_enabled: bool = True,
) -> dict:
    """Build all local derived factor tables and return a structured summary."""
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    started_at = datetime.now().isoformat(timespec="seconds")
    summary: dict[str, Any] = {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "steps": [],
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "errors": [],
        "warnings": [],
        "started_at": started_at,
        "finished_at": "",
        "factor_build_report_path": "",
    }

    step_specs: list[tuple[str, bool, Callable[[], Any]]] = [
        (
            "build_moneyflow_factors",
            build_moneyflow_factors_enabled,
            lambda: build_and_save_moneyflow_factors(db_path=resolved_db_path),
        ),
        (
            "build_market_regime",
            build_market_regime_enabled,
            lambda: build_market_regime(db_path=resolved_db_path),
        ),
        (
            "build_industry_strength",
            build_industry_strength_enabled,
            lambda: build_and_save_industry_strength(db_path=resolved_db_path),
        ),
        (
            "build_daily_factors",
            build_daily_factors_enabled,
            lambda: build_daily_factors(db_path=resolved_db_path),
        ),
        (
            "build_factor_diagnostics",
            build_factor_diagnostics_enabled,
            lambda: run_build_factor_diagnostics(db_path=resolved_db_path),
        ),
    ]

    daily_factors_failed = False
    for step_name, enabled, runner in step_specs:
        step_result = _run_step(step_name, enabled, runner)
        if step_name == "build_factor_diagnostics" and daily_factors_failed:
            warning = "build_daily_factors 失败，factor_diagnostics 已继续尝试，但 daily_factors 可能不是最新。"
            step_result["message"] = f"{step_result['message']}; {warning}"
            summary["warnings"].append(warning)
        summary["steps"].append(step_result)

        if step_result["status"] == "success":
            summary["success_count"] += 1
        elif step_result["status"] == "skipped":
            summary["skipped_count"] += 1
        else:
            summary["failed_count"] += 1
            summary["errors"].append({"step_name": step_name, "error": step_result["error"]})
            if step_name == "build_daily_factors":
                daily_factors_failed = True

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["factor_build_report_path"] = export_factor_build_report(summary, output_dir=output_dir)
    return summary


def export_factor_build_report(summary: dict, output_dir: str = "reports") -> str:
    output_path = Path(output_dir) / f"factor_build_workflow_{datetime.now().date().isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 因子构建工作流报告",
        "",
        "## 一、运行说明",
        "",
        "本流程只构建衍生因子，不拉取数据、不运行策略、不运行日度计划、不调用 LLM、不自动交易。",
        "",
        f"- 数据库：{summary.get('db_path')}",
        f"- 输出目录：{summary.get('output_dir')}",
        f"- 开始时间：{summary.get('started_at')}",
        f"- 结束时间：{summary.get('finished_at')}",
        f"- 成功步骤：{summary.get('success_count')}",
        f"- 失败步骤：{summary.get('failed_count')}",
        f"- 跳过步骤：{summary.get('skipped_count')}",
        "",
        "## 二、执行结果",
        "",
        "| step_name | status | rows | message | error |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for step in summary.get("steps", []):
        rows = step.get("rows_written", "")
        lines.append(
            "| {step_name} | {status} | {rows} | {message} | {error} |".format(
                step_name=step.get("step_name", ""),
                status=step.get("status", ""),
                rows="" if rows is None else rows,
                message=_markdown_cell(step.get("message", "")),
                error=_markdown_cell(step.get("error", "")),
            )
        )

    lines.extend(["", "## 三、失败步骤", ""])
    errors = summary.get("errors", [])
    if errors:
        lines.extend(f"- {item.get('step_name')}: {_markdown_cell(item.get('error', ''))}" for item in errors)
    else:
        lines.append("- 无")

    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## 依赖提示", ""])
        lines.extend(f"- {_markdown_cell(warning)}" for warning in warnings)

    lines.extend(
        [
            "",
            "## 四、后续建议",
            "",
            "- 如果 moneyflow_factors 为空，请先更新 moneyflow。",
            "- 如果 industry_strength 为空，请先更新 sw_daily。",
            "- 如果 market_regime 为空，请先更新 index_daily 和 limit_list_daily。",
            "- 如果 factor_diagnostics 为空，请确认 daily_factors 是否已构建。",
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
    return {
        "step_name": step_name,
        "status": "success",
        "rows_written": _result_rows(result),
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
    parser = argparse.ArgumentParser(description="Run unified local factor build workflow.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--skip-moneyflow-factors", action="store_true")
    parser.add_argument("--skip-market-regime", action="store_true")
    parser.add_argument("--skip-industry-strength", action="store_true")
    parser.add_argument("--skip-daily-factors", action="store_true")
    parser.add_argument("--skip-factor-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_factor_build_workflow(
        db_path=args.db_path,
        output_dir=args.output_dir,
        build_moneyflow_factors_enabled=not args.skip_moneyflow_factors,
        build_market_regime_enabled=not args.skip_market_regime,
        build_industry_strength_enabled=not args.skip_industry_strength,
        build_daily_factors_enabled=not args.skip_daily_factors,
        build_factor_diagnostics_enabled=not args.skip_factor_diagnostics,
    )
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("Factor build workflow finished.")
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
    if summary.get("factor_build_report_path"):
        print(f"factor_build_report_path: {summary['factor_build_report_path']}")


if __name__ == "__main__":
    main()
