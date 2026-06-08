"""Run the staged strategy operations research workflow."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.diagnostics.system_health import run_system_health_check
from src.pipeline.run_backtest_analysis_agent import run_backtest_analysis_agent_pipeline
from src.pipeline.run_factor_build_workflow import run_factor_build_workflow
from src.pipeline.run_parameter_iteration_agent import run_parameter_iteration_agent_pipeline
from src.pipeline.run_strategy_research_agent import run_strategy_research_agent_pipeline
from src.pipeline.run_strategy_research_workflow import run_strategy_research_workflow
from src.pipeline.run_system_health_check import export_system_health_report


WORKFLOW_STEPS = [
    "run_factor_build_workflow",
    "run_strategy_research_workflow",
    "run_backtest_analysis_agent",
    "run_strategy_research_agent",
    "run_parameter_iteration_agent",
    "run_system_health_check",
]


def run_strategy_ops_workflow(
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    validation_start_date: str | None = None,
    validation_end_date: str | None = None,
    db_path: str | None = None,
    output_dir: str = "reports",
    build_factors: bool = True,
    run_strategy_research: bool = True,
    run_backtest_analysis_agent: bool = True,
    run_strategy_research_agent: bool = True,
    run_parameter_iteration_agent: bool = True,
    run_health_check: bool = True,
) -> dict:
    """Run the staged strategy research workflow and return a summary.

    This workflow is research-only. It does not place orders, enable strategies,
    or write formal strategy configuration files.
    """
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    report_date = date.today().isoformat()
    summary: dict[str, Any] = {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "validation_start_date": validation_start_date,
        "validation_end_date": validation_end_date,
        "steps": [],
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "errors": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "strategy_ops_report_path": "",
        "factor_build_report_path": None,
        "strategy_research_report_path": None,
        "strategy_evaluation_report_path": None,
        "parameter_search_report_path": None,
        "walk_forward_validation_report_path": None,
        "trade_plan_backtest_report_path": None,
        "strategy_admission_report_path": None,
        "backtest_analysis_agent_path": None,
        "strategy_research_agent_path": None,
        "strategy_research_suggestions_path": None,
        "parameter_iteration_agent_path": None,
        "parameter_search_space_candidate_path": None,
        "system_health_report_path": None,
    }

    _run_or_skip(
        summary,
        step_name="run_factor_build_workflow",
        enabled=build_factors,
        runner=lambda: run_factor_build_workflow(db_path=resolved_db_path, output_dir=output_dir),
        path_mappings={"factor_build_report_path": "factor_build_report_path"},
    )
    _run_or_skip(
        summary,
        step_name="run_strategy_research_workflow",
        enabled=run_strategy_research,
        runner=lambda: run_strategy_research_workflow(
            db_path=resolved_db_path,
            output_dir=output_dir,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            parameter_search_start_date=train_start_date,
            parameter_search_end_date=train_end_date,
            export_reports=True,
            export_candidate_config=False,
        ),
        path_mappings={
            "strategy_evaluation_report_path": "strategy_evaluation_report_path",
            "parameter_search_report_path": "parameter_search_report_path",
            "walk_forward_validation_report_path": "walk_forward_validation_report_path",
            "trade_plan_backtest_report_path": "trade_plan_backtest_report_path",
            "strategy_admission_report_path": "strategy_admission_report_path",
        },
    )
    _run_or_skip(
        summary,
        step_name="run_backtest_analysis_agent",
        enabled=run_backtest_analysis_agent,
        runner=lambda: run_backtest_analysis_agent_pipeline(
            db_path=resolved_db_path,
            output_dir=output_dir,
            report_date=report_date,
        ),
        path_mappings={"backtest_analysis_agent_path": None},
    )
    _run_or_skip(
        summary,
        step_name="run_strategy_research_agent",
        enabled=run_strategy_research_agent,
        runner=lambda: run_strategy_research_agent_pipeline(
            db_path=resolved_db_path,
            output_dir=output_dir,
            report_date=report_date,
            export_candidate_json=True,
        ),
        path_mappings={
            "strategy_research_agent_path": "strategy_research_report_path",
            "strategy_research_suggestions_path": "strategy_research_suggestions_path",
        },
    )
    _run_or_skip(
        summary,
        step_name="run_parameter_iteration_agent",
        enabled=run_parameter_iteration_agent,
        runner=lambda: run_parameter_iteration_agent_pipeline(
            db_path=resolved_db_path,
            output_dir=output_dir,
            report_date=report_date,
            export_candidate_json=True,
        ),
        path_mappings={
            "parameter_iteration_agent_path": "parameter_iteration_report_path",
            "parameter_search_space_candidate_path": "parameter_search_space_candidate_path",
        },
    )
    _run_or_skip(
        summary,
        step_name="run_system_health_check",
        enabled=run_health_check,
        runner=lambda: _run_health_check_and_export(resolved_db_path, output_dir),
        path_mappings={"system_health_report_path": "system_health_report_path"},
    )

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["strategy_ops_report_path"] = export_strategy_ops_report(summary, output_dir=output_dir)
    return summary


def _run_or_skip(
    summary: dict[str, Any],
    *,
    step_name: str,
    enabled: bool,
    runner: Callable[[], Any],
    path_mappings: dict[str, str | None],
) -> None:
    if not enabled:
        summary["steps"].append(
            {
                "step_name": step_name,
                "status": "skipped",
                "message": "step skipped by flag",
                "error": "",
            }
        )
        summary["skipped_count"] += 1
        return

    try:
        result = runner()
    except Exception as exc:
        error = _sanitize(str(exc))
        summary["steps"].append(
            {
                "step_name": step_name,
                "status": "failed",
                "message": "step failed; workflow continues",
                "error": error,
            }
        )
        summary["failed_count"] += 1
        summary["errors"].append({"step_name": step_name, "error": error})
        return

    for summary_key, result_key in path_mappings.items():
        if result_key is None:
            if result:
                summary[summary_key] = result
        elif isinstance(result, dict) and result.get(result_key):
            summary[summary_key] = result.get(result_key)

    summary["steps"].append(
        {
            "step_name": step_name,
            "status": "success",
            "message": "step completed",
            "error": "",
        }
    )
    summary["success_count"] += 1


def _run_health_check_and_export(db_path: str, output_dir: str) -> dict:
    health_summary = run_system_health_check(db_path=db_path, reports_dir=output_dir)
    report_path = export_system_health_report(health_summary, output_dir=output_dir)
    return {
        "system_health_summary": health_summary,
        "system_health_report_path": report_path,
    }


def export_strategy_ops_report(summary: dict, output_dir: str = "reports") -> str:
    output_path = Path(output_dir) / f"strategy_ops_workflow_{date.today().isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_summary = dict(summary)
    report_summary["strategy_ops_report_path"] = str(output_path)
    output_path.write_text(_build_strategy_ops_markdown(report_summary), encoding="utf-8")
    return str(output_path)


def _build_strategy_ops_markdown(summary: dict) -> str:
    lines = [
        "# 策略研究总流程运行报告",
        "",
        "## 一、运行说明",
        "",
        "本流程用于策略研究、回测分析、策略假设生成和参数候选建议。",
        "本流程不自动下单；不直接启用策略；不直接修改正式策略配置；不构成投资建议。",
        "",
        "## 二、执行参数",
        "",
        f"- train_start_date: {summary.get('train_start_date')}",
        f"- train_end_date: {summary.get('train_end_date')}",
        f"- validation_start_date: {summary.get('validation_start_date')}",
        f"- validation_end_date: {summary.get('validation_end_date')}",
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
            "## 四、生成报告和候选文件",
            "",
            f"- strategy_ops_report_path: {summary.get('strategy_ops_report_path') or '未生成'}",
            f"- factor_build_report_path: {summary.get('factor_build_report_path') or '未生成'}",
            f"- strategy_version_evaluation 报告路径: {summary.get('strategy_evaluation_report_path') or '未生成'}",
            f"- parameter_search 报告路径: {summary.get('parameter_search_report_path') or '未生成'}",
            f"- walk_forward_validation 报告路径: {summary.get('walk_forward_validation_report_path') or '未生成'}",
            f"- trade_plan_backtest 报告路径: {summary.get('trade_plan_backtest_report_path') or '未生成'}",
            f"- strategy_admission 报告路径: {summary.get('strategy_admission_report_path') or '未生成'}",
            f"- llm_backtest_analysis 报告路径: {summary.get('backtest_analysis_agent_path') or '未生成'}",
            f"- llm_strategy_research 报告路径: {summary.get('strategy_research_agent_path') or '未生成'}",
            f"- strategy_research_suggestions JSON 路径: {summary.get('strategy_research_suggestions_path') or '未生成'}",
            f"- llm_parameter_iteration 报告路径: {summary.get('parameter_iteration_agent_path') or '未生成'}",
            f"- parameter_search_space_candidate JSON 路径: {summary.get('parameter_search_space_candidate_path') or '未生成'}",
            f"- system_health 报告路径: {summary.get('system_health_report_path') or '未生成'}",
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
            "## 六、人工确认事项",
            "",
            "- strategy_research_suggestions_*.json 只是研究建议；",
            "- parameter_search_space_candidate_*.json 只是候选参数；",
            "- 不能直接用于实盘；",
            "- 必须经过人工确认、回测、样本外验证和交易计划级回测。",
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
    parser = argparse.ArgumentParser(description="Run staged strategy operations research workflow.")
    parser.add_argument("--train-start-date", default=None)
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--validation-start-date", default=None)
    parser.add_argument("--validation-end-date", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--skip-factor-build", action="store_true")
    parser.add_argument("--skip-strategy-research", action="store_true")
    parser.add_argument("--skip-backtest-analysis-agent", action="store_true")
    parser.add_argument("--skip-strategy-research-agent", action="store_true")
    parser.add_argument("--skip-parameter-iteration-agent", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_strategy_ops_workflow(
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
        db_path=args.db_path,
        output_dir=args.output_dir,
        build_factors=not args.skip_factor_build,
        run_strategy_research=not args.skip_strategy_research,
        run_backtest_analysis_agent=not args.skip_backtest_analysis_agent,
        run_strategy_research_agent=not args.skip_strategy_research_agent,
        run_parameter_iteration_agent=not args.skip_parameter_iteration_agent,
        run_health_check=not args.skip_health_check,
    )
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("Strategy ops workflow finished.")
    print(f"success_count: {summary['success_count']}")
    print(f"failed_count: {summary['failed_count']}")
    print(f"skipped_count: {summary['skipped_count']}")
    print(f"strategy_ops_report_path: {summary['strategy_ops_report_path']}")
    if summary.get("errors"):
        print(f"errors: {summary['errors']}")


if __name__ == "__main__":
    main()
