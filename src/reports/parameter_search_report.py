from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


RESULT_TABLE_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "valid_count",
    "win_rate_3d",
    "avg_return_3d",
    "median_return_3d",
    "avg_max_drawdown_3d",
    "evaluation_score",
    "recommendation",
]
PERCENT_COLUMNS = {
    "win_rate_3d",
    "avg_return_3d",
    "median_return_3d",
    "avg_max_drawdown_3d",
}
FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_parameter_search_report(
    evaluation: pd.DataFrame,
    performance: pd.DataFrame | None = None,
    report_date: str | None = None,
) -> str:
    evaluation = _as_dataframe(evaluation)
    performance = _as_dataframe(performance)
    resolved_report_date = report_date or date.today().isoformat()

    lines: list[str] = [
        "# 策略参数搜索报告",
        "",
        f"报告日期：{resolved_report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于历史数据和参数搜索结果生成；",
        "- 不构成投资建议；",
        "- 参数搜索结果存在过拟合风险；",
        "- 建议进入观察池前，还需样本外验证、模拟盘和小仓位实盘。",
        "",
        "## 二、总体结果",
    ]
    lines.extend(_summary_lines(evaluation))

    lines.extend(["", "## 三、参数搜索结果总表"])
    if evaluation.empty:
        lines.append("当前没有可用的参数搜索结果。")
    else:
        lines.extend(_markdown_table(evaluation, RESULT_TABLE_COLUMNS))

    lines.extend(["", "## 四、各策略最佳候选版本"])
    if evaluation.empty:
        lines.append("当前没有可用的参数搜索结果。")
    else:
        best = (
            evaluation.sort_values(["strategy_name", "evaluation_score"], ascending=[True, False])
            .groupby("strategy_name", sort=True, group_keys=False)
            .head(3)
        )
        _append_strategy_list(lines, best, "当前没有可用的候选版本。")

    lines.extend(["", "## 五、建议启用观察的参数版本"])
    _append_strategy_list(
        lines,
        _filter_equal(evaluation, "recommendation", "enable_observation"),
        "当前没有达到启用观察条件的参数版本。",
    )

    lines.extend(["", "## 六、样本不足或需继续回测的参数版本"])
    continue_backtest = _filter_continue_backtest(evaluation)
    _append_strategy_list(
        lines,
        continue_backtest,
        "当前没有明显样本不足或需继续回测的参数版本。",
    )

    lines.extend(["", "## 七、建议降权或暂停的参数版本"])
    reduce_or_pause = _filter_recommendations(evaluation, ["reduce_or_pause", "pause"])
    _append_strategy_list(
        lines,
        reduce_or_pause,
        "当前没有明确建议降权或暂停的参数版本。",
    )

    lines.extend(
        [
            "",
            "## 八、风险提示",
            "- 参数搜索可能过拟合；",
            "- 样本不足不能直接实盘；",
            "- 不应只选择历史收益最高的参数；",
            "- 需结合样本外验证和实盘复盘。",
        ]
    )

    if not performance.empty:
        lines.extend(["", "附：performance 行数：" + str(len(performance))])

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df


def _summary_lines(evaluation: pd.DataFrame) -> list[str]:
    if evaluation.empty:
        total = enable = observe = continue_backtest = reduce_or_pause = pause = 0
    else:
        recommendation = _column(evaluation, "recommendation").fillna("").astype(str)
        total = len(evaluation)
        enable = int((recommendation == "enable_observation").sum())
        observe = int((recommendation == "observe").sum())
        continue_backtest = int((recommendation == "continue_backtest").sum())
        reduce_or_pause = int((recommendation == "reduce_or_pause").sum())
        pause = int((recommendation == "pause").sum())

    return [
        f"- 参数版本总数：{total}",
        f"- enable_observation 数量：{enable}",
        f"- observe 数量：{observe}",
        f"- continue_backtest 数量：{continue_backtest}",
        f"- reduce_or_pause 数量：{reduce_or_pause}",
        f"- pause 数量：{pause}",
    ]


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


def _append_strategy_list(lines: list[str], strategies: pd.DataFrame, empty_message: str) -> None:
    if strategies.empty:
        lines.append(empty_message)
        return

    for _, row in strategies.iterrows():
        lines.append(
            "- "
            + f"{_format_cell(row.get('strategy_name'))}:{_format_cell(row.get('strategy_version'))}"
            + f" | score={_format_cell(row.get('evaluation_score'), 'evaluation_score')}"
            + f" | valid_count={_format_cell(row.get('valid_count'))}"
            + f" | recommendation={_format_cell(row.get('recommendation'))}"
        )


def _filter_equal(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_column(df, column).fillna("").astype(str) == value]


def _filter_continue_backtest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    recommendation = _column(df, "recommendation").fillna("").astype(str)
    status = _column(df, "evaluation_status").fillna("").astype(str)
    return df[(recommendation == "continue_backtest") | (status == "insufficient_samples")]


def _filter_recommendations(df: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_column(df, "recommendation").fillna("").astype(str).isin(values)]


def _format_cell(value: object, column: str | None = None) -> str:
    if pd.isna(value):
        return ""
    if column in PERCENT_COLUMNS and isinstance(value, (float, int)):
        return f"{value:.2%}"
    if column == "evaluation_score" and isinstance(value, (float, int)):
        return f"{value:.4f}"
    return str(value)
