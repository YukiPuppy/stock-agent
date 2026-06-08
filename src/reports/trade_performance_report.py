from __future__ import annotations

from typing import Iterable

import pandas as pd


TRADE_PERFORMANCE_TABLE_COLUMNS = [
    "code",
    "name",
    "entry_price",
    "execution_status",
    "execution_flags",
    "return_1d",
    "return_3d",
    "return_5d",
    "max_drawdown_3d",
    "max_favorable_3d",
    "performance_comment",
]

FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_trade_performance_report(
    trade_performance: pd.DataFrame,
    daily_review: pd.DataFrame | None = None,
    trade_date: str | None = None,
) -> str:
    performance = _as_dataframe(trade_performance)
    daily = _as_dataframe(daily_review)
    report_date = _resolve_trade_date(performance, daily, trade_date)
    valid = _valid_performance(performance)
    chase = _flagged(valid, "chase_above_entry")

    lines: list[str] = [
        "# A股实盘交易表现复盘报告",
        "",
        f"报告日期：{report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于用户实际交易记录和本地行情数据生成；",
        "- 用于复盘交易执行后的市场表现；",
        "- 不构成投资建议；",
        "- 单日或少量样本不能直接证明策略有效或无效。",
        "",
        "## 二、交易表现总览",
        f"- 有效交易样本数：{len(valid)}",
        f"- 1日平均收益：{_format_percent(_mean(valid, 'return_1d'))}",
        f"- 3日平均收益：{_format_percent(_mean(valid, 'return_3d'))}",
        f"- 5日平均收益：{_format_percent(_mean(valid, 'return_5d'))}",
        f"- 3日平均最大回撤：{_format_percent(_mean(valid, 'max_drawdown_3d'))}",
        f"- 计划内交易 3 日平均收益：{_format_percent(_mean(_planned(valid), 'return_3d'))}",
        f"- 计划外交易 3 日平均收益：{_format_percent(_mean(_off_plan(valid), 'return_3d'))}",
        f"- 追高交易数量：{len(chase)}",
        f"- 追高交易 3 日平均收益：{_format_percent(_mean(chase, 'return_3d'))}",
        "",
        "## 三、交易表现明细",
    ]

    if performance.empty:
        lines.append("当前暂无交易表现数据。")
    else:
        lines.extend(_markdown_table(performance, TRADE_PERFORMANCE_TABLE_COLUMNS))

    lines.extend(["", "## 四、执行与结果关系"])
    lines.extend(_execution_relation_lines(valid))

    lines.extend(
        [
            "",
            "## 五、风险提示",
            "- 少量交易样本不能直接用于策略参数调整；",
            "- 计划外交易应单独统计，避免污染策略评价；",
            "- 追高和超仓可能放大回撤；",
            "- 实盘表现需要结合更长周期持续观察。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df


def _resolve_trade_date(
    trade_performance: pd.DataFrame,
    daily_review: pd.DataFrame,
    trade_date: str | None,
) -> str:
    if trade_date:
        return str(trade_date)
    for df in (trade_performance, daily_review):
        if not df.empty and "trade_date" in df.columns:
            values = df["trade_date"].dropna()
            if not values.empty:
                return str(values.max())
    return "暂无可用交易日期"


def _valid_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_valid" not in df.columns:
        return pd.DataFrame()
    return df[df["is_valid"].fillna(False).astype(bool)].copy()


def _planned(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_status") != "off_plan"]


def _off_plan(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_status") == "off_plan"]


def _flagged(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_flags").str.contains(flag, regex=False)]


def _string_column(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[column].fillna("").astype(str)


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


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


def _execution_relation_lines(valid: pd.DataFrame) -> list[str]:
    plan = _planned(valid)
    off_plan = _off_plan(valid)
    chase = _flagged(valid, "chase_above_entry")
    over_position = _flagged(valid, "over_position")
    return [
        f"- 计划内交易表现：样本 {len(plan)} 笔，3 日平均收益 {_format_percent(_mean(plan, 'return_3d'))}。",
        f"- 计划外交易表现：样本 {len(off_plan)} 笔，3 日平均收益 {_format_percent(_mean(off_plan, 'return_3d'))}。",
        f"- 追高交易表现：样本 {len(chase)} 笔，3 日平均收益 {_format_percent(_mean(chase, 'return_3d'))}，3 日平均最大回撤 {_format_percent(_mean(chase, 'max_drawdown_3d'))}。",
        f"- 超仓交易表现：样本 {len(over_position)} 笔，3 日平均收益 {_format_percent(_mean(over_position, 'return_3d'))}，3 日平均最大回撤 {_format_percent(_mean(over_position, 'max_drawdown_3d'))}。",
    ]


def _format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _format_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
