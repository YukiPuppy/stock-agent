"""Markdown report rendering for strategy admission recommendations."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


ADMISSION_TABLE_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "source",
    "valid_count",
    "evaluation_recommendation",
    "oos_status",
    "oos_risk",
    "trade_plan_trigger_rate",
    "trade_plan_win_rate",
    "trade_plan_avg_return",
    "admission_score",
    "admission_status",
    "admission_recommendation",
]
PERCENT_COLUMNS = {
    "trade_plan_trigger_rate",
    "trade_plan_win_rate",
    "trade_plan_avg_return",
    "trade_plan_avg_drawdown",
}
SCORE_COLUMNS = {"evaluation_score", "oos_stability_score", "admission_score"}
FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_strategy_admission_report(
    admission: pd.DataFrame,
    report_date: str | None = None,
) -> str:
    admission = admission.copy() if admission is not None else pd.DataFrame()
    resolved_report_date = report_date or date.today().isoformat()

    lines = [
        "# 策略准入与观察候选报告",
        "",
        f"报告日期：{resolved_report_date}",
        "",
        "## 一、报告说明",
        "",
        "- 本报告基于策略评价、参数搜索、样本外验证和交易规则级回测结果生成；",
        "- 用于判断策略版本是否适合进入观察池；",
        "- 不构成投资建议；",
        "- “观察候选”不代表自动实盘启用；",
        "- 实盘前仍需人工确认、模拟盘和小仓位验证。",
        "",
        "## 二、总体结论",
        "",
    ]
    lines.extend(_summary_lines(admission))
    if admission.empty or "trade_plan_win_rate" not in admission.columns or admission["trade_plan_win_rate"].notna().sum() == 0:
        lines.extend(
            [
                "",
                "> 警告：没有可关联的交易计划准入指标。可能原因包括当前 run_id 无交易计划表现、",
                "> strategy_names / strategy_versions 无法映射到候选版本，或没有买入类 action。",
                "> 准入结论不完整，不建议实盘。",
            ]
        )

    lines.extend(["", "## 三、策略准入总表", ""])
    if admission.empty:
        lines.append("当前没有策略准入结果。")
    else:
        lines.extend(_markdown_table(admission, ADMISSION_TABLE_COLUMNS))

    lines.extend(["", "## 四、建议进入观察池的策略版本", ""])
    candidates = _filter_recommendation(admission, ["enable_observation_candidate"])
    if candidates.empty:
        lines.append("当前没有满足观察候选条件的策略版本。")
    else:
        lines.extend(_markdown_table(candidates, ADMISSION_TABLE_COLUMNS + ["admission_reason"]))

    lines.extend(["", "## 五、Top candidates by admission_score", ""])
    top_score = _top_by(admission, "admission_score")
    lines.extend(_markdown_table(top_score, ADMISSION_TABLE_COLUMNS) if not top_score.empty else ["暂无候选。"])

    lines.extend(["", "## 六、Top candidates by trade_plan_avg_return", ""])
    top_return = _top_by(admission, "trade_plan_avg_return")
    lines.extend(
        _markdown_table(top_return, ADMISSION_TABLE_COLUMNS + ["trade_plan_avg_drawdown"])
        if not top_return.empty
        else ["暂无可用的交易计划收益指标。"]
    )

    lines.extend(["", "## 七、需要继续研究的策略版本", ""])
    research = _filter_recommendation(admission, ["continue_research", "observe_more"])
    if research.empty:
        lines.append("当前没有需要继续研究的策略版本。")
    else:
        lines.extend(_markdown_table(research, ADMISSION_TABLE_COLUMNS + ["admission_reason"]))

    lines.extend(["", "## 八、不建议启用的策略版本", ""])
    rejected = _filter_recommendation(admission, ["do_not_enable"])
    if rejected.empty:
        lines.append("当前没有明确不建议启用的策略版本。")
    else:
        for _, row in rejected.iterrows():
            lines.append(
                f"- {_format_cell(row.get('strategy_name'))} / {_format_cell(row.get('strategy_version'))}："
                f"{_format_cell(row.get('admission_reason'))}"
            )

    lines.extend(
        [
            "",
            "## 九、风险提示",
            "",
            "- 策略准入结果依赖历史数据和当前规则，可能存在样本偏差；",
            "- 观察候选不等于实盘启用；",
            "- 策略启用前仍需模拟盘和小仓位验证；",
            "- 回测和样本外验证不能消除未来市场不确定性。",
            "",
        ]
    )
    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _summary_lines(admission: pd.DataFrame) -> list[str]:
    if admission.empty:
        recommendation = pd.Series(dtype=str)
        status = pd.Series(dtype=str)
    else:
        recommendation = _column(admission, "admission_recommendation").fillna("").astype(str)
        status = _column(admission, "admission_status").fillna("").astype(str)
    return [
        f"- 策略版本总数：{len(admission)}",
        f"- enable_observation_candidate 数量：{int((recommendation == 'enable_observation_candidate').sum())}",
        f"- observe_more 数量：{int((recommendation == 'observe_more').sum())}",
        f"- continue_research 数量：{int((recommendation == 'continue_research').sum())}",
        f"- do_not_enable 数量：{int((recommendation == 'do_not_enable').sum())}",
        f"- insufficient_samples 数量：{int((status == 'insufficient_samples').sum())}",
        f"- oos_failed 数量：{int((status == 'oos_failed').sum())}",
        f"- risk_rejected 数量：{int((status == 'risk_rejected').sum())}",
        f"- trade_plan_win_rate 非空数量：{int(_column(admission, 'trade_plan_win_rate').notna().sum())}",
    ]


def _top_by(admission: pd.DataFrame, column: str, limit: int = 10) -> pd.DataFrame:
    if admission.empty or column not in admission.columns:
        return pd.DataFrame()
    values = pd.to_numeric(admission[column], errors="coerce")
    if values.notna().sum() == 0:
        return pd.DataFrame()
    return admission.loc[values.notna()].assign(_rank_value=values[values.notna()]).sort_values(
        "_rank_value", ascending=False
    ).head(limit).drop(columns=["_rank_value"])


def _filter_recommendation(admission: pd.DataFrame, recommendations: list[str]) -> pd.DataFrame:
    if admission.empty or "admission_recommendation" not in admission.columns:
        return pd.DataFrame()
    return admission[admission["admission_recommendation"].isin(recommendations)].copy()


def _column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    selected_columns = list(columns)
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join(["---"] * len(selected_columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_cell(row.get(column), column) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_cell(value: object, column: str | None = None) -> str:
    if pd.isna(value):
        return "-"
    if column in PERCENT_COLUMNS:
        return f"{float(value) * 100:.2f}%"
    if column in SCORE_COLUMNS:
        return f"{float(value):.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
