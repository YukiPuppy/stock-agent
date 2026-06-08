from __future__ import annotations

import pandas as pd


FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_period_review_report(
    period_review: pd.DataFrame,
    execution_review: pd.DataFrame | None = None,
    trade_performance: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    review = _as_dataframe(period_review)
    execution = _as_dataframe(execution_review)
    performance = _as_dataframe(trade_performance)
    row = review.iloc[0] if not review.empty else pd.Series(dtype=object)
    resolved_start = str(start_date or row.get("start_date") or "")
    resolved_end = str(end_date or row.get("end_date") or "")
    valid = _valid_performance(performance)

    lines = [
        "# A股周期执行复盘报告",
        "",
        "## 一、报告说明",
        "- 本报告基于用户手动导入的实际交易记录、本地执行复盘和交易表现数据生成；",
        "- 用于复盘执行纪律和交易结果；",
        "- 不构成投资建议；",
        "- 少量样本不能直接用于策略参数调整。",
        "",
        "## 二、周期执行总览",
        f"- 统计区间：{resolved_start} 至 {resolved_end}",
        f"- 实际交易笔数：{_format_count(row.get('actual_trade_count'))}",
        f"- 买入笔数：{_format_count(row.get('buy_count'))}",
        f"- 卖出笔数：{_format_count(row.get('sell_count'))}",
        f"- 计划内交易笔数：{_format_count(row.get('follow_plan_count'))}",
        f"- 计划外交易笔数：{_format_count(row.get('off_plan_count'))}",
        f"- 执行偏差笔数：{_format_count(row.get('deviation_count'))}",
        f"- 追高偏差笔数：{_format_count(row.get('chase_count'))}",
        f"- 超出计划仓位笔数：{_format_count(row.get('over_position_count'))}",
        f"- 平均执行评分：{_format_number(row.get('avg_execution_score'))}",
        "",
        "## 三、交易表现总览",
        f"- 有效表现样本数：{_format_count(row.get('valid_performance_count'))}",
        f"- 1日平均收益：{_format_percent(row.get('avg_return_1d'))}",
        f"- 3日平均收益：{_format_percent(row.get('avg_return_3d'))}",
        f"- 5日平均收益：{_format_percent(row.get('avg_return_5d'))}",
        f"- 计划内交易 3日平均收益：{_format_percent(row.get('plan_trade_avg_return_3d'))}",
        f"- 计划外交易 3日平均收益：{_format_percent(row.get('off_plan_avg_return_3d'))}",
        f"- 追高交易 3日平均收益：{_format_percent(row.get('chase_avg_return_3d'))}",
        f"- 超出计划仓位交易 3日平均收益：{_format_percent(row.get('over_position_avg_return_3d'))}",
        f"- 表现最好交易：{_format_text(row.get('best_trade_code'))}",
        f"- 表现最差交易：{_format_text(row.get('worst_trade_code'))}",
        "",
        "## 四、主要问题",
        _format_text(row.get("main_issues")),
        "",
        "## 五、执行与结果关系",
    ]
    lines.extend(_execution_relation_lines(valid, execution))
    if len(valid) < 10:
        lines.append("样本数量较少，暂不宜据此调整策略参数。")

    lines.extend(
        [
            "",
            "## 六、下一周期建议",
            _format_text(row.get("next_period_suggestion")),
            "",
            "## 七、风险提示",
            "- 周期复盘用于改善执行纪律，不应直接替代策略回测；",
            "- 计划外交易应单独统计，避免污染策略评价；",
            "- 单周期结果可能受市场环境影响；",
            "- 策略参数调整应基于更长周期样本、回测和样本外验证。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df


def _valid_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_valid" not in df.columns:
        return pd.DataFrame()
    return df[df["is_valid"].fillna(False).astype(bool)].copy()


def _execution_relation_lines(valid: pd.DataFrame, execution: pd.DataFrame) -> list[str]:
    plan = _status(valid, "follow_plan")
    off_plan = _status(valid, "off_plan")
    chase = _flagged(valid, "chase_above_entry")
    over_position = _flagged(valid, "over_position")
    if valid.empty and not execution.empty:
        return [
            f"- 计划内交易表现：执行记录 {_count_equal(execution, 'execution_status', 'follow_plan')} 笔，暂无有效表现样本。",
            f"- 计划外交易表现：执行记录 {_count_equal(execution, 'execution_status', 'off_plan')} 笔，暂无有效表现样本。",
            f"- 追高交易表现：执行记录 {_count_flag(execution, 'chase_above_entry')} 笔，暂无有效表现样本。",
            f"- 超出计划仓位交易表现：执行记录 {_count_flag(execution, 'over_position')} 笔，暂无有效表现样本。",
        ]
    return [
        f"- 计划内交易表现：样本 {len(plan)} 笔，3日平均收益 {_format_percent(_mean(plan, 'return_3d'))}。",
        f"- 计划外交易表现：样本 {len(off_plan)} 笔，3日平均收益 {_format_percent(_mean(off_plan, 'return_3d'))}。",
        f"- 追高交易表现：样本 {len(chase)} 笔，3日平均收益 {_format_percent(_mean(chase, 'return_3d'))}。",
        f"- 超出计划仓位交易表现：样本 {len(over_position)} 笔，3日平均收益 {_format_percent(_mean(over_position, 'return_3d'))}。",
    ]


def _status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_status") == status]


def _flagged(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_flags").str.contains(flag, regex=False)]


def _count_equal(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int((_string_column(df, column) == value).sum())


def _count_flag(df: pd.DataFrame, flag: str) -> int:
    if df.empty or "execution_flags" not in df.columns:
        return 0
    return int(_string_column(df, "execution_flags").str.contains(flag, regex=False).sum())


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


def _format_count(value: object) -> str:
    if value is None or pd.isna(value):
        return "0"
    return str(int(value))


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _format_text(value: object) -> str:
    if value is None or pd.isna(value) or str(value) == "":
        return "-"
    return str(value)
