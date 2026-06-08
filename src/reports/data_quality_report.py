"""Markdown rendering for data quality checks."""

from __future__ import annotations

from datetime import date

import pandas as pd


FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_data_quality_report(
    quality_report: pd.DataFrame,
    compare_result: pd.DataFrame | None = None,
    compare_summary: pd.DataFrame | None = None,
    report_date: str | None = None,
) -> str:
    resolved_date = report_date or date.today().isoformat()
    lines = [
        "# 数据质量与数据源对齐检查报告",
        "",
        f"报告日期：{resolved_date}",
        "",
        "## 一、报告说明",
        "- 本报告以本地 daily_bars 数据质量为核心检查对象；",
        "- 正式行情源为 Tushare Pro，AKShare 仅作为可选诊断源；",
        "- daily_bars 使用 Tushare Pro 标准单位：volume 为手，amount 为千元；",
        "- daily_factors.amount_ma5 和 min_amount_ma5 单位为千元；",
        "- actual_trades.amount 仍为元；",
        "- positions 金额字段仍为元；",
        "- 不构成投资建议；",
        "- 数据质量问题会影响回测、参数搜索和候选池结果。",
        "",
        "## 二、daily_bars 数据质量检查",
    ]
    lines.extend(_markdown_table(quality_report))
    lines.extend(["", "## 三、可选数据源诊断"])
    if compare_summary is not None and not compare_summary.empty:
        lines.extend(_markdown_table(compare_summary))
    else:
        lines.append("当前未执行或暂无数据源对齐异常。")
    if compare_result is not None and not compare_result.empty:
        lines.extend(["", "### 异常明细（前 50 条）"])
        lines.extend(_markdown_table(compare_result.head(50)))
    else:
        lines.append("当前未执行或暂无数据源对齐异常。")

    lines.extend(["", "## 四、主要风险"])
    lines.extend(_risk_lines(quality_report))
    lines.extend(
        [
            "",
            "## 五、后续建议",
            "- 若 daily_bars 出现 error，应先修复本地行情数据完整性与字段合法性；",
            "- 若 OHLC 差异明显，应检查复权方式；",
            "- 若缺失交易日较多，应补齐数据。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _risk_lines(quality_report: pd.DataFrame) -> list[str]:
    if not isinstance(quality_report, pd.DataFrame) or quality_report.empty:
        return ["- 未发现可用的数据质量检查结果。"]
    error_count = int((quality_report["status"] == "error").sum()) if "status" in quality_report.columns else 0
    warning_count = int((quality_report["status"] == "warning").sum()) if "status" in quality_report.columns else 0
    if error_count == 0 and warning_count == 0:
        return ["- 当前 daily_bars 检查未发现 error 或 warning。"]
    rows = []
    if error_count:
        rows.append(f"- 发现 {error_count} 项 error，相关数据流程应先修复后再用于研究或复盘。")
    if warning_count:
        rows.append(f"- 发现 {warning_count} 项 warning，建议核对异常字段和数据源口径。")
    return rows


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


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")
