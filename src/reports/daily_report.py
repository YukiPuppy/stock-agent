"""Markdown report rendering for deterministic daily trade plans."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


TRADE_PLAN_TABLE_COLUMNS = [
    "rank",
    "code",
    "name",
    "action",
    "close",
    "entry_low",
    "entry_high",
    "position_low",
    "position_high",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
]

CANDIDATE_POOL_TABLE_COLUMNS = [
    "rank",
    "code",
    "name",
    "close",
    "pct_chg_5d",
    "volume_ratio_5",
    "close_position_20",
    "score",
    "reason",
]

FORBIDDEN_REPORT_PHRASES = ("保证盈利", "稳赚", "满仓")


def generate_daily_report(
    trade_plan: pd.DataFrame,
    candidate_pool: pd.DataFrame | None = None,
    trade_date: str | None = None,
) -> str:
    """Render a daily A-share trade-plan report as Markdown."""
    report_date = _resolve_trade_date(trade_plan, candidate_pool, trade_date)
    trade_plan = _as_dataframe(trade_plan)
    candidate_pool = _as_dataframe(candidate_pool)

    lines: list[str] = [
        "# A股日度交易计划报告",
        "",
        f"报告日期：{report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于日线数据和规则模型生成；",
        "- 不构成投资建议；",
        "- A股 T+1 机制下需控制仓位和隔夜风险；",
        "- 当前阶段未接入 LLM，仅使用确定性规则。",
        "",
        "## 二、次日重点交易计划",
    ]

    if trade_plan.empty:
        lines.append("当前没有生成可执行交易计划。")
    else:
        lines.extend(_markdown_table(trade_plan, TRADE_PLAN_TABLE_COLUMNS))

    lines.extend(["", "## 三、个股计划详情"])
    if trade_plan.empty:
        lines.append("当前没有生成可执行交易计划。")
    else:
        for _, row in trade_plan.iterrows():
            lines.extend(_trade_plan_detail(row))

    lines.extend(["", "## 四、候选股池摘要"])
    if candidate_pool.empty:
        lines.append("当前候选股池为空。")
    else:
        lines.extend(_markdown_table(candidate_pool.head(20), CANDIDATE_POOL_TABLE_COLUMNS))

    lines.extend(
        [
            "",
            "## 五、风险提示",
            "- 不追高开快速回落；",
            "- 不在跌破止损价后补仓摊薄；",
            "- 单票仓位应受总资金规模和市场环境约束；",
            "- 若大盘环境明显走弱，应降低交易频率；",
            "- 本报告只提供交易计划框架，最终操作需结合盘中实际成交、盘口承接和市场情绪。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df


def _resolve_trade_date(
    trade_plan: pd.DataFrame,
    candidate_pool: pd.DataFrame | None,
    trade_date: str | None,
) -> str:
    if trade_date:
        return str(trade_date)

    latest = _latest_trade_date(trade_plan)
    if latest is not None:
        return latest

    latest = _latest_trade_date(candidate_pool)
    if latest is not None:
        return latest

    return "暂无可用交易日期"


def _latest_trade_date(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty or "trade_date" not in df.columns:
        return None
    values = df["trade_date"].dropna()
    if values.empty:
        return None
    return str(values.max())


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    selected_columns = list(columns)
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join(["---"] * len(selected_columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_cell(row.get(column)) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _trade_plan_detail(row: pd.Series) -> list[str]:
    code = _format_cell(row.get("code"))
    name = _format_cell(row.get("name"))
    return [
        "",
        f"### {code} {name}",
        f"- 策略类型：{_format_cell(row.get('strategy_type'))}",
        f"- 操作类型：{_format_cell(row.get('action'))}",
        f"- 买入区间：{_format_range(row.get('entry_low'), row.get('entry_high'))}",
        f"- 仓位区间：{_format_range(row.get('position_low'), row.get('position_high'))}",
        f"- 止损价：{_format_cell(row.get('stop_loss'))}",
        f"- 止盈价：{_format_range(row.get('take_profit_1'), row.get('take_profit_2'))}",
        f"- 失效条件：{_format_cell(row.get('invalid_condition'))}",
        f"- T+1 风险：{_format_cell(row.get('t_plus_1_risk'))}",
        f"- 计划理由：{_format_cell(row.get('plan_reason'))}",
    ]


def _format_range(low: object, high: object) -> str:
    return f"{_format_cell(low)} ~ {_format_cell(high)}"


def _format_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
