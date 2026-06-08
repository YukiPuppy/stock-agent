from __future__ import annotations

from typing import Iterable

import pandas as pd


EXECUTION_REVIEW_TABLE_COLUMNS = [
    "code",
    "name",
    "side",
    "actual_price",
    "plan_match_status",
    "execution_status",
    "execution_flags",
    "execution_comment",
]

ACTUAL_TRADES_TABLE_COLUMNS = [
    "trade_time",
    "code",
    "name",
    "side",
    "price",
    "volume",
    "amount",
    "position_ratio",
    "reason",
    "note",
]

FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_daily_review_report(
    daily_review: pd.DataFrame,
    execution_review: pd.DataFrame | None = None,
    actual_trades: pd.DataFrame | None = None,
    trade_date: str | None = None,
) -> str:
    daily = _as_dataframe(daily_review)
    execution = _as_dataframe(execution_review)
    actual = _as_dataframe(actual_trades)
    row = daily.iloc[0] if not daily.empty else pd.Series(dtype=object)
    report_date = _resolve_trade_date(daily, execution, actual, trade_date)

    lines: list[str] = [
        "# A股盘后执行复盘报告",
        "",
        f"报告日期：{report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于用户实际交易记录和系统交易计划生成；",
        "- 用于复盘执行纪律和计划偏差；",
        "- 不构成投资建议；",
        "- 当前阶段未接入券商接口，交易记录来自手动导入。",
        "",
        "## 二、当日执行总览",
        f"- 实际交易笔数：{_format_cell(row.get('actual_trade_count'))}",
        f"- 买入笔数：{_format_cell(row.get('buy_count'))}",
        f"- 卖出笔数：{_format_cell(row.get('sell_count'))}",
        f"- 计划内匹配笔数：{_format_cell(row.get('matched_plan_count'))}",
        f"- 计划外交易笔数：{_format_cell(row.get('off_plan_count'))}",
        f"- 执行偏差笔数：{_format_cell(row.get('deviation_count'))}",
        f"- 追高偏差笔数：{_format_cell(row.get('chase_count'))}",
        f"- 超仓笔数：{_format_cell(row.get('over_position_count'))}",
        f"- 执行评分：{_format_cell(row.get('execution_score'))}",
        "",
        "## 三、主要问题",
        _format_text(row.get("main_issues"), "未发现明显执行偏差"),
        "",
        "## 四、执行复盘明细",
    ]

    if execution.empty:
        lines.append("当前暂无执行复盘明细。")
    else:
        lines.extend(_markdown_table(execution, EXECUTION_REVIEW_TABLE_COLUMNS))

    lines.extend(["", "## 五、实际交易记录"])
    if actual.empty:
        lines.append("当前暂无实际交易记录。")
    else:
        lines.extend(_markdown_table(actual, ACTUAL_TRADES_TABLE_COLUMNS))

    lines.extend(
        [
            "",
            "## 六、后续建议",
            _format_text(row.get("next_action_suggestion"), "暂无后续建议。"),
            "",
            "## 七、风险提示",
            "- 单日复盘不能直接否定或确认策略有效性；",
            "- 计划外交易应单独统计，避免污染策略评价；",
            "- 追高、超仓、未按止损执行会显著放大风险；",
            "- 策略问题和执行问题应分开复盘。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df


def _resolve_trade_date(
    daily_review: pd.DataFrame,
    execution_review: pd.DataFrame,
    actual_trades: pd.DataFrame,
    trade_date: str | None,
) -> str:
    if trade_date:
        return str(trade_date)
    for df in (daily_review, execution_review, actual_trades):
        if not df.empty and "trade_date" in df.columns:
            values = df["trade_date"].dropna()
            if not values.empty:
                return str(values.max())
    return "暂无可用交易日期"


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    selected_columns = [column for column in columns if column in df.columns]
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join(["---"] * len(selected_columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_cell(row.get(column)) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_text(value: object, fallback: str) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value)


def _format_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
