from __future__ import annotations

from typing import Iterable

import pandas as pd


POSITION_TABLE_COLUMNS = [
    "code",
    "name",
    "holding_volume",
    "available_volume",
    "frozen_volume",
    "cost_price",
    "latest_price",
    "floating_pnl_pct",
    "t_plus_1_status",
    "position_status",
]

POSITION_REVIEW_TABLE_COLUMNS = [
    "code",
    "name",
    "position_risk_level",
    "position_flags",
    "position_comment",
    "next_action_hint",
]

FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_position_review_report(
    positions: pd.DataFrame,
    position_review: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> str:
    position_df = _as_dataframe(positions)
    review_df = _as_dataframe(position_review)
    report_date = _resolve_report_date(position_df, review_df, as_of_date)

    lines: list[str] = [
        "# A股持仓与T+1风险检查报告",
        "",
        f"报告日期：{report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于手动导入的实际交易记录和本地行情数据生成；",
        "- 当前阶段未接入券商接口；",
        "- 持仓结果仅用于复盘和风险检查；",
        "- 不构成投资建议。",
        "",
        "## 二、持仓总览",
        f"- 持仓股票数量：{len(position_df)}",
        f"- 总市值：{_format_number(_sum(position_df, 'market_value'))}",
        f"- 总浮动盈亏：{_format_number(_sum(position_df, 'floating_pnl'))}",
        f"- 高风险持仓数量：{_count_equal(review_df, 'position_risk_level', 'high')}",
        f"- T+1 锁定持仓数量：{_count_equal(position_df, 't_plus_1_status', 'not_sellable_today')}",
        f"- 进入止盈观察区间数量：{_count_contains(review_df, 'position_flags', 'take_profit_zone')}",
        "",
        "## 三、持仓明细",
    ]

    if position_df.empty:
        lines.append("当前暂无持仓数据。")
    else:
        lines.extend(_markdown_table(position_df, POSITION_TABLE_COLUMNS))

    lines.extend(["", "## 四、风险检查明细"])
    if review_df.empty:
        lines.append("当前暂无持仓风险检查数据。")
    else:
        lines.extend(_markdown_table(review_df, POSITION_REVIEW_TABLE_COLUMNS))

    lines.extend(["", "## 五、后续处理提示"])
    lines.extend(_follow_up_lines(position_df, review_df))

    lines.extend(
        [
            "",
            "## 六、风险提示",
            "- T+1 会限制当日买入后的卖出处理；",
            "- 止损和止盈需要结合盘中流动性和市场环境；",
            "- 单票仓位过高会放大波动风险；",
            "- 本报告不构成投资建议。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df


def _resolve_report_date(positions: pd.DataFrame, position_review: pd.DataFrame, as_of_date: str | None) -> str:
    if as_of_date:
        return str(as_of_date)
    for df in (positions, position_review):
        if not df.empty and "as_of_date" in df.columns:
            values = df["as_of_date"].dropna()
            if not values.empty:
                return str(values.max())
    return "暂无可用日期"


def _follow_up_lines(positions: pd.DataFrame, position_review: pd.DataFrame) -> list[str]:
    t1_count = _count_equal(positions, "t_plus_1_status", "not_sellable_today")
    stop_loss_count = _count_contains(position_review, "position_flags", "below_stop_loss")
    take_profit_count = _count_contains(position_review, "position_flags", "take_profit_zone")
    high_ratio_count = _count_contains(position_review, "position_flags", "high_position_ratio")
    return [
        f"- T+1 锁定风险：{t1_count} 只持仓需关注可卖数量变化。",
        f"- 跌破止损风险：{stop_loss_count} 只持仓触发或接近计划止损条件。",
        f"- 止盈观察：{take_profit_count} 只持仓进入第一止盈观察区间。",
        f"- 单票仓位偏高：{high_ratio_count} 只持仓需要关注集中度波动。",
    ]


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    selected_columns = [column for column in columns if column in df.columns]
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join(["---"] * len(selected_columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_cell(row.get(column), column) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _sum(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return None if values.empty else float(values.sum())


def _count_equal(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int((df[column].fillna("").astype(str) == value).sum())


def _count_contains(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.contains(value, regex=False).sum())


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _format_cell(value: object, column: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    if column == "floating_pnl_pct":
        return f"{float(value) * 100:.2f}%"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
