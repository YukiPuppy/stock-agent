"""System acceptance workflow for local stock-agent readiness checks."""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb

from src.config import settings


CORE_TABLES = [
    "stock_basic",
    "trade_calendar",
    "daily_bars",
    "daily_basic",
    "stock_limits",
    "suspend_daily",
    "index_daily",
    "limit_list_daily",
    "market_regime",
    "moneyflow",
    "moneyflow_factors",
    "sw_industry_classification",
    "stock_industry_map",
    "sw_daily",
    "industry_strength",
    "daily_factors",
    "factor_diagnostics",
]

STRATEGY_TABLES = [
    "strategy_signals",
    "candidate_pool",
    "trade_plan",
    "strategy_version_evaluation",
    "parameter_search_results",
    "walk_forward_validation",
    "trade_plan_backtest_performance",
    "strategy_admission",
]

REPORT_PATTERNS = [
    "data_update_workflow_*.md",
    "factor_build_workflow_*.md",
    "daily_ops_workflow_*.md",
    "strategy_ops_workflow_*.md",
    "system_health_*.md",
    "daily_report_*.md",
    "llm_agents_index_*.md",
    "llm_report_summary_*.md",
    "llm_backtest_analysis_*.md",
    "llm_risk_review_*.md",
    "llm_daily_review_*.md",
    "llm_market_regime_*.md",
    "llm_industry_insight_*.md",
    "llm_factor_insight_*.md",
    "llm_strategy_research_*.md",
    "llm_parameter_iteration_*.md",
]

PROTECTED_CONFIG_FILES = [
    "active_strategies.json",
    "active_strategies_candidate.json",
    "strategy_versions.json",
    "parameter_search_space.json",
]


def run_system_acceptance_workflow(
    db_path: str | None = None,
    output_dir: str = "reports",
) -> dict:
    """Run read-only system acceptance checks and write a Markdown report."""

    started_at = datetime.now().isoformat(timespec="seconds")
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    warnings: list[str] = []
    errors: list[str] = []

    config_checks = _check_runtime_config()
    _append_config_findings(config_checks, warnings, errors)

    table_checks = _check_tables(resolved_db_path, warnings, errors)
    report_checks = _check_report_files(output_dir)
    config_file_checks = _check_protected_config_files()
    _append_config_file_findings(config_file_checks, warnings)
    _append_table_findings(table_checks, warnings)

    acceptance_status = _acceptance_status(warnings, errors)
    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "db_path": resolved_db_path,
        "output_dir": output_dir,
        "acceptance_status": acceptance_status,
        "config_checks": config_checks,
        "table_checks": table_checks,
        "report_checks": report_checks,
        "config_file_checks": config_file_checks,
        "warnings": warnings,
        "errors": errors,
        "started_at": started_at,
        "finished_at": finished_at,
        "system_acceptance_report_path": "",
    }
    summary["system_acceptance_report_path"] = _write_acceptance_report(summary, output_dir)
    return summary


def _check_runtime_config() -> dict:
    llm_provider = str(getattr(settings, "LLM_PROVIDER", "none") or "none").strip()
    return {
        "DEFAULT_DATA_PROVIDER": str(getattr(settings, "DEFAULT_DATA_PROVIDER", "") or "").strip(),
        "TUSHARE_TOKEN_configured": bool(str(getattr(settings, "TUSHARE_TOKEN", "") or "").strip()),
        "TUSHARE_API_URL": str(getattr(settings, "TUSHARE_API_URL", "") or "").strip(),
        "DATA_FETCH_DISABLE_PROXY": bool(getattr(settings, "DATA_FETCH_DISABLE_PROXY", False)),
        "LLM_PROVIDER": llm_provider,
        "LLM_MODEL": str(getattr(settings, "LLM_MODEL", "") or "").strip(),
        "LLM_API_KEY_configured": bool(str(getattr(settings, "LLM_API_KEY", "") or "").strip()),
        "LLM_DISABLE_PROXY": bool(getattr(settings, "LLM_DISABLE_PROXY", False)),
        "ENABLE_LLM_REPORT_AGENT": bool(getattr(settings, "ENABLE_LLM_REPORT_AGENT", False)),
    }


def _append_config_findings(config_checks: dict, warnings: list[str], errors: list[str]) -> None:
    default_provider = str(config_checks.get("DEFAULT_DATA_PROVIDER", "")).strip().lower()
    if default_provider != "tushare":
        warnings.append("DEFAULT_DATA_PROVIDER 不是 tushare，系统验收建议使用 tushare。")
    if not bool(config_checks.get("TUSHARE_TOKEN_configured", False)):
        errors.append("TUSHARE_TOKEN 未配置。")

    llm_provider = str(config_checks.get("LLM_PROVIDER", "") or "").strip().lower()
    llm_enabled = bool(config_checks.get("ENABLE_LLM_REPORT_AGENT", False)) or llm_provider not in {"", "none", "disabled"}
    if llm_enabled and not bool(config_checks.get("LLM_API_KEY_configured", False)):
        warnings.append("LLM 已启用或配置了 provider，但 LLM_API_KEY 未配置。")


def _check_tables(db_path: str, warnings: list[str], errors: list[str]) -> dict[str, dict]:
    table_checks: dict[str, dict] = {}
    path = Path(db_path)
    if not path.exists():
        warnings.append(f"数据库文件不存在：{db_path}")
        for table_name in CORE_TABLES:
            table_checks[table_name] = _missing_table_check(table_name, "core")
        for table_name in STRATEGY_TABLES:
            table_checks[table_name] = _missing_table_check(table_name, "strategy")
        return table_checks

    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        errors.append(f"数据库只读连接失败：{exc}")
        for table_name in CORE_TABLES:
            table_checks[table_name] = _missing_table_check(table_name, "core")
        for table_name in STRATEGY_TABLES:
            table_checks[table_name] = _missing_table_check(table_name, "strategy")
        return table_checks

    try:
        existing_tables = _existing_tables(con)
        for table_name in CORE_TABLES:
            table_checks[table_name] = _check_one_table(con, existing_tables, table_name, "core", errors)
        for table_name in STRATEGY_TABLES:
            table_checks[table_name] = _check_one_table(con, existing_tables, table_name, "strategy", errors)
    finally:
        con.close()
    return table_checks


def _existing_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def _check_one_table(
    con: duckdb.DuckDBPyConnection,
    existing_tables: set[str],
    table_name: str,
    category: str,
    errors: list[str],
) -> dict:
    if table_name not in existing_tables:
        return _missing_table_check(table_name, category)
    try:
        quoted = _quote_identifier(table_name)
        row_count = int(con.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        columns = {
            str(row[1])
            for row in con.execute(f"PRAGMA table_info({quoted})").fetchall()
        }
        date_range = ""
        if "trade_date" in columns and row_count > 0:
            min_date, max_date = con.execute(f"SELECT MIN(trade_date), MAX(trade_date) FROM {quoted}").fetchone()
            date_range = f"{min_date} - {max_date}"
        return {
            "table_name": table_name,
            "category": category,
            "exists": True,
            "row_count": row_count,
            "date_range": date_range,
            "status": "ok" if row_count > 0 else "warning",
            "message": "table has data" if row_count > 0 else "table is empty",
        }
    except Exception as exc:
        errors.append(f"{table_name} 核心表读取报错：{exc}" if category == "core" else f"{table_name} 表读取报错：{exc}")
        return {
            "table_name": table_name,
            "category": category,
            "exists": True,
            "row_count": 0,
            "date_range": "",
            "status": "missing" if category != "core" else "missing",
            "message": str(exc),
        }


def _missing_table_check(table_name: str, category: str) -> dict:
    return {
        "table_name": table_name,
        "category": category,
        "exists": False,
        "row_count": 0,
        "date_range": "",
        "status": "missing",
        "message": "table is missing",
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _append_table_findings(table_checks: dict[str, dict], warnings: list[str]) -> None:
    if _row_count(table_checks, "daily_bars") == 0:
        warnings.append("daily_bars 为空。")
    if _row_count(table_checks, "daily_factors") == 0:
        warnings.append("daily_factors 为空。")
    if _row_count(table_checks, "trade_plan") == 0:
        warnings.append("trade_plan 为空。")


def _row_count(table_checks: dict[str, dict], table_name: str) -> int:
    return int(table_checks.get(table_name, {}).get("row_count", 0) or 0)


def _check_report_files(output_dir: str) -> dict[str, dict]:
    base = Path(output_dir)
    checks = {}
    for pattern in REPORT_PATTERNS:
        paths = sorted(path for path in base.glob(pattern) if path.is_file()) if base.exists() else []
        checks[pattern] = {
            "pattern": pattern,
            "exists": bool(paths),
            "file_count": len(paths),
            "latest_file": str(paths[-1]) if paths else "",
            "status": "ok" if paths else "missing",
        }
    return checks


def _check_protected_config_files() -> dict[str, dict]:
    checks = {}
    for file_name in PROTECTED_CONFIG_FILES:
        path = _resolve_config_file(file_name)
        exists = path.is_file()
        checks[file_name] = {
            "file_name": file_name,
            "path": str(path),
            "exists": exists,
            "status": "ok" if exists else "missing",
            "message": "file exists; not modified by this workflow" if exists else "file is missing",
        }
    return checks


def _resolve_config_file(file_name: str) -> Path:
    candidates = [Path("configs") / file_name, Path(file_name)]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _append_config_file_findings(config_file_checks: dict[str, dict], warnings: list[str]) -> None:
    for file_name, check in config_file_checks.items():
        if not bool(check.get("exists", False)):
            warnings.append(f"正式配置文件缺失：{file_name}")


def _acceptance_status(warnings: list[str], errors: list[str]) -> str:
    if errors:
        return "failed"
    if warnings:
        return "warning"
    return "pass"


def _write_acceptance_report(summary: dict, output_dir: str) -> str:
    output_path = Path(output_dir) / f"system_acceptance_{date.today().isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_system_acceptance_report(summary), encoding="utf-8")
    return str(output_path)


def generate_system_acceptance_report(summary: dict) -> str:
    lines = [
        "# 系统总体验收报告",
        "",
        f"started_at: {summary.get('started_at', '')}",
        f"finished_at: {summary.get('finished_at', '')}",
        f"db_path: {summary.get('db_path', '')}",
        f"output_dir: {summary.get('output_dir', '')}",
        "",
        "## 一、验收结论",
        f"acceptance_status: {summary.get('acceptance_status', 'unknown')}",
        "",
        "本流程只做检查和报告，不拉取数据、不构建因子、不运行策略、不运行 LLM、不自动下单。",
        "本流程不会修改 active_strategies.json、active_strategies_candidate.json、strategy_versions.json 或 parameter_search_space.json。",
        "",
        "## 二、配置检查",
    ]
    lines.extend(_config_lines(summary.get("config_checks", {})))
    lines.extend(["", "## 三、核心数据表检查"])
    lines.extend(_table_lines(summary.get("table_checks", {}), "core"))
    lines.extend(["", "## 四、策略研究与日度计划表检查"])
    lines.extend(_table_lines(summary.get("table_checks", {}), "strategy"))
    lines.extend(["", "## 五、报告文件检查"])
    lines.extend(_report_lines(summary.get("report_checks", {})))
    lines.extend(["", "## 六、正式配置文件保护检查"])
    lines.append("本流程仅检查正式配置文件是否存在，没有在本流程中修改正式配置。")
    lines.extend(_config_file_lines(summary.get("config_file_checks", {})))
    lines.extend(["", "## 七、警告与错误", "### warnings"])
    lines.extend(_list_or_empty(summary.get("warnings", []), "暂无 warning。"))
    lines.extend(["", "### errors"])
    lines.extend(_list_or_empty(summary.get("errors", []), "暂无 error。"))
    lines.extend(["", "## 八、下一步建议"])
    lines.extend(_next_suggestion_lines(summary))
    return "\n".join(lines).strip() + "\n"


def _config_lines(config_checks: dict) -> list[str]:
    rows = [
        ("DEFAULT_DATA_PROVIDER", config_checks.get("DEFAULT_DATA_PROVIDER", "")),
        ("TUSHARE_TOKEN configured", config_checks.get("TUSHARE_TOKEN_configured", False)),
        ("TUSHARE_API_URL", config_checks.get("TUSHARE_API_URL", "")),
        ("DATA_FETCH_DISABLE_PROXY", config_checks.get("DATA_FETCH_DISABLE_PROXY", False)),
        ("LLM_PROVIDER", config_checks.get("LLM_PROVIDER", "")),
        ("LLM_MODEL", config_checks.get("LLM_MODEL", "")),
        ("LLM_API_KEY configured", config_checks.get("LLM_API_KEY_configured", False)),
        ("LLM_DISABLE_PROXY", config_checks.get("LLM_DISABLE_PROXY", False)),
    ]
    return _markdown_rows(["item", "value"], rows)


def _table_lines(table_checks: dict[str, dict], category: str) -> list[str]:
    rows = []
    for check in table_checks.values():
        if check.get("category") != category:
            continue
        rows.append(
            (
                check.get("table_name", ""),
                check.get("exists", False),
                check.get("row_count", 0),
                check.get("date_range", ""),
                check.get("status", ""),
            )
        )
    return _markdown_rows(["table", "exists", "row_count", "date_range", "status"], rows)


def _report_lines(report_checks: dict[str, dict]) -> list[str]:
    rows = [
        (check.get("pattern", pattern), check.get("exists", False), check.get("latest_file", ""), check.get("status", ""))
        for pattern, check in report_checks.items()
    ]
    return _markdown_rows(["pattern", "exists", "latest_file", "status"], rows)


def _config_file_lines(config_file_checks: dict[str, dict]) -> list[str]:
    rows = [
        (check.get("file_name", file_name), check.get("exists", False), check.get("path", ""), check.get("status", ""))
        for file_name, check in config_file_checks.items()
    ]
    return _markdown_rows(["file", "exists", "path", "status"], rows)


def _next_suggestion_lines(summary: dict) -> list[str]:
    table_checks = summary.get("table_checks", {})
    report_checks = summary.get("report_checks", {})
    rows = []
    if _row_count(table_checks, "daily_bars") == 0:
        rows.append("daily_bars 为空：先运行 run_data_update_workflow。")
    if _row_count(table_checks, "daily_factors") == 0:
        rows.append("daily_factors 为空：先运行 run_factor_build_workflow。")
    if _row_count(table_checks, "trade_plan") == 0:
        rows.append("trade_plan 为空：运行 run_daily_ops_workflow。")
    if _row_count(table_checks, "strategy_admission") == 0:
        rows.append("strategy_admission 为空：运行 run_strategy_ops_workflow。")
    llm_patterns = [pattern for pattern in REPORT_PATTERNS if pattern.startswith("llm_")]
    if any(not report_checks.get(pattern, {}).get("exists", False) for pattern in llm_patterns):
        rows.append("LLM 报告为空：运行 run_llm_agents_workflow。")
    return _list_or_empty(rows, "暂无下一步建议。")


def _markdown_rows(headers: list[str], rows: Iterable[tuple]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return lines


def _format_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _list_or_empty(items: Iterable[object], empty_message: str) -> list[str]:
    values = [str(item) for item in items if str(item)]
    return [f"- {value}" for value in values] if values else [empty_message]


def read_acceptance_status_from_report(path: str | Path) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""
    matched = re.search(r"acceptance_status:\s*(pass|warning|failed)", content)
    return matched.group(1) if matched else ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run system acceptance checks without side effects.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    summary = run_system_acceptance_workflow(db_path=args.db_path, output_dir=args.output_dir)
    print("System acceptance workflow finished.")
    print(f"acceptance_status: {summary['acceptance_status']}")
    print(f"system_acceptance_report_path: {summary['system_acceptance_report_path']}")
    return summary


if __name__ == "__main__":
    main()
