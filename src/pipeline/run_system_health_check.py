"""Run local system health checks and optionally export a Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.diagnostics.system_health import run_system_health_check
from src.reports.system_health_report import generate_system_health_report


def export_system_health_report(summary: dict, output_dir: str = "reports", report_date: str | None = None) -> str:
    resolved_date = report_date or date.today().isoformat()
    report = generate_system_health_report(summary, report_date=resolved_date)
    output_path = Path(output_dir) / f"system_health_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local stock-agent system health checks.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--export-report", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_system_health_check(
        db_path=args.db_path,
        reports_dir=args.reports_dir,
        configs_dir=args.configs_dir,
    )
    print("System health check finished.")
    print(f"overall_status: {summary['overall_status']}")
    _print_list("blocking_issues", summary["blocking_issues"])
    _print_list("warnings", summary["warnings"])
    _print_list("next_suggestions", summary["next_suggestions"])
    _print_frame("table_health", summary["table_health"])
    _print_frame("config_files", summary["config_files"])
    _print_frame("report_files", summary["report_files"])

    if args.export_report:
        output_path = export_system_health_report(summary, output_dir=args.output_dir)
        print(f"report_path: {output_path}")


def _print_list(title: str, values: list[str]) -> None:
    print(f"{title}:")
    if not values:
        print("- none")
        return
    for value in values:
        print(f"- {value}")


def _print_frame(title: str, df: pd.DataFrame) -> None:
    print(f"{title}:")
    if df.empty:
        print("(empty)")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
