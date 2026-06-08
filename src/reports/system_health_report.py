"""Markdown rendering for system health checks."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_system_health_report(
    summary: dict,
    report_date: str | None = None,
) -> str:
    resolved_date = report_date or date.today().isoformat()
    lines = [
        "# A股多智能体选股系统健康检查报告",
        "",
        f"报告日期：{resolved_date}",
        "",
        "## 一、总体状态",
        str(summary.get("overall_status", "unknown")),
        _enriched_factors_summary(summary.get("enriched_factors")),
        "",
        "## 二、阻塞问题",
    ]
    lines.extend(_list_or_empty(summary.get("blocking_issues", []), "当前未发现阻塞性问题。"))
    lines.extend(["", "## 三、风险与提醒"])
    lines.extend(_list_or_empty(summary.get("warnings", []), "当前未发现风险提醒。"))
    lines.extend(["", "## 四、核心数据表检查"])
    lines.extend(_markdown_table(summary.get("table_health")))
    lines.extend(["", "## 五、配置文件检查"])
    lines.extend(_markdown_table(summary.get("config_files")))
    lines.extend(["", "## 六、数据源配置检查"])
    lines.extend(_markdown_table(summary.get("data_source_config")))
    lines.extend(["", "## 七、LLM ReportAgent 配置检查"])
    lines.extend(_markdown_table(summary.get("llm_config")))
    lines.extend(["", "## 八、LLM Agent 索引内容检查"])
    lines.extend(_markdown_table(summary.get("llm_agents_index_content")))
    lines.extend(["", "## 九、数据质量状态"])
    lines.extend(_markdown_table(summary.get("data_quality_status")))
    lines.extend(["", "## 十、扩展因子状态"])
    lines.extend(_markdown_table(summary.get("enriched_factors")))
    lines.extend(["", "## 十一、报告文件检查"])
    lines.extend(_markdown_table(summary.get("report_files")))
    lines.extend(["", "## 十二、系统验收报告检查"])
    lines.extend(_markdown_table(summary.get("system_acceptance_report_status")))
    lines.extend(["", "## 十三、下一步建议"])
    lines.extend(_list_or_empty(summary.get("next_suggestions", []), "暂无下一步建议。"))
    lines.extend(
        [
            "",
            "## 十四、说明",
            "- 本报告用于检查系统运行状态；",
            "- 不构成投资建议；",
            "- 系统健康不代表策略未来有效；",
            "- 实盘前仍需人工确认和风险控制。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _list_or_empty(items: Iterable[object], empty_message: str) -> list[str]:
    values = [str(item) for item in items if str(item)]
    if not values:
        return [empty_message]
    return [f"- {value}" for value in values]


def _markdown_table(value: object) -> list[str]:
    if not isinstance(value, pd.DataFrame) or value.empty:
        return ["暂无数据。"]
    columns = [str(column) for column in value.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in value.iterrows():
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in value.columns) + " |")
    return lines


def _enriched_factors_summary(value: object) -> str:
    if not isinstance(value, pd.DataFrame) or value.empty:
        return "enriched_factors=unknown"
    row = value.iloc[0]
    has_fields = bool(row.get("has_enriched_fields", False))
    missing_rate = row.get("daily_basic_missing_rate")
    if pd.isna(missing_rate):
        return f"enriched_factors={has_fields}"
    return f"enriched_factors={has_fields}, daily_basic_missing_rate={float(missing_rate):.1%}"


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")
