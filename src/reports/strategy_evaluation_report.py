"""Markdown report rendering for strategy-version evaluation results."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


EVALUATION_TABLE_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "valid_count",
    "win_rate_3d",
    "avg_return_3d",
    "median_return_3d",
    "avg_max_drawdown_3d",
    "evaluation_score",
    "evaluation_status",
    "risk_level",
    "recommendation",
]
PERCENT_COLUMNS = {
    "win_rate_1d",
    "win_rate_3d",
    "win_rate_5d",
    "avg_return_1d",
    "avg_return_3d",
    "avg_return_5d",
    "median_return_3d",
    "avg_max_drawdown_1d",
    "avg_max_drawdown_3d",
    "avg_max_drawdown_5d",
}
FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_strategy_evaluation_report(
    evaluation: pd.DataFrame,
    performance: pd.DataFrame | None = None,
    report_date: str | None = None,
) -> str:
    """Render strategy-version evaluation and performance data as Markdown."""
    evaluation = _as_dataframe(evaluation)
    performance = _as_dataframe(performance)
    resolved_report_date = report_date or date.today().isoformat()

    lines: list[str] = [
        "# 策略版本评价报告",
        "",
        f"报告日期：{resolved_report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于历史回测结果和规则化评价模型生成；",
        "- 当前阶段不构成投资建议；",
        "- 回测表现不代表未来收益；",
        "- “启用观察”只代表进入观察池或模拟验证，不代表自动实盘；",
        "- 策略版本是否用于实盘，需要经过人工确认、模拟盘和小仓位验证。",
        "",
    ]

    if evaluation.empty:
        lines.extend(
            [
                "当前没有可用的策略版本评价结果，请先运行 backtest_strategy_versions 和 evaluate_strategy_versions。",
                "",
            ]
        )

    lines.extend(["## 二、总体结论"])
    lines.extend(_summary_lines(evaluation))

    lines.extend(["", "## 三、策略版本评价总表"])
    if evaluation.empty:
        lines.append("当前没有可用的策略版本评价结果。")
    else:
        lines.extend(_markdown_table(evaluation, EVALUATION_TABLE_COLUMNS))

    lines.extend(["", "## 四、建议启用观察的策略"])
    _append_strategy_list(
        lines,
        evaluation[_column(evaluation, "recommendation") == "enable_observation"]
        if not evaluation.empty
        else evaluation,
        empty_message="当前没有达到启用观察条件的策略版本。",
        detail=True,
    )

    lines.extend(["", "## 五、需要继续回测的策略"])
    if evaluation.empty:
        continue_backtest = evaluation
    else:
        recommendation = _column(evaluation, "recommendation")
        status = _column(evaluation, "evaluation_status")
        continue_backtest = evaluation[
            (recommendation == "continue_backtest") | (status == "insufficient_samples")
        ]
    _append_strategy_list(
        lines,
        continue_backtest,
        empty_message="当前没有明显样本不足的策略版本。",
        detail=False,
    )

    lines.extend(["", "## 六、建议降权或暂停的策略"])
    if evaluation.empty:
        reduce_or_pause = evaluation
    else:
        recommendation = _column(evaluation, "recommendation")
        reduce_or_pause = evaluation[recommendation.isin(["reduce_or_pause", "pause"])]
    _append_strategy_list(
        lines,
        reduce_or_pause,
        empty_message="当前没有明确建议降权或暂停的策略版本。",
        detail=False,
    )

    lines.extend(["", "## 七、策略版本详细评价"])
    if evaluation.empty:
        lines.append("当前没有可用的策略版本评价结果。")
    else:
        for _, row in evaluation.iterrows():
            lines.extend(_strategy_detail(row, performance))

    lines.extend(
        [
            "",
            "## 八、风险提示",
            "- 回测结果可能存在过拟合风险；",
            "- 样本数量不足的策略不能直接用于实盘；",
            "- 高回撤策略应降低权重或暂停观察；",
            "- 策略启用前应经过样本外验证、模拟盘和小仓位实盘；",
            "- 本报告仅用于策略研究和复盘，不构成投资建议。",
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


def _summary_lines(evaluation: pd.DataFrame) -> list[str]:
    if evaluation.empty:
        total = enable = observe = continue_backtest = reduce_or_pause = pause = insufficient = 0
    else:
        recommendation = _column(evaluation, "recommendation")
        status = _column(evaluation, "evaluation_status")
        total = len(evaluation)
        enable = int((recommendation == "enable_observation").sum())
        observe = int((recommendation == "observe").sum())
        continue_backtest = int((recommendation == "continue_backtest").sum())
        reduce_or_pause = int((recommendation == "reduce_or_pause").sum())
        pause = int((recommendation == "pause").sum())
        insufficient = int(
            ((recommendation == "insufficient_samples") | (status == "insufficient_samples")).sum()
        )

    return [
        f"- 策略版本总数：{total}",
        f"- enable_observation 数量：{enable}",
        f"- observe 数量：{observe}",
        f"- continue_backtest 数量：{continue_backtest}",
        f"- reduce_or_pause 数量：{reduce_or_pause}",
        f"- pause 数量：{pause}",
        f"- insufficient_samples 数量：{insufficient}",
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


def _append_strategy_list(
    lines: list[str],
    strategies: pd.DataFrame,
    empty_message: str,
    detail: bool,
) -> None:
    if strategies.empty:
        lines.append(empty_message)
        return

    for _, row in strategies.iterrows():
        name = _format_cell(row.get("strategy_name"))
        version = _format_cell(row.get("strategy_version"))
        lines.append(f"- {name}:{version}")
        if detail:
            lines.append(f"  - 样本数：{_format_cell(row.get('valid_count'))}")
            lines.append(f"  - 3日胜率：{_format_cell(row.get('win_rate_3d'), 'win_rate_3d')}")
            lines.append(f"  - 3日平均收益：{_format_cell(row.get('avg_return_3d'), 'avg_return_3d')}")
            lines.append(f"  - 3日平均回撤：{_format_cell(row.get('avg_max_drawdown_3d'), 'avg_max_drawdown_3d')}")
            lines.append(f"  - 推荐理由：{_format_cell(row.get('evaluation_reason'))}")


def _strategy_detail(row: pd.Series, performance: pd.DataFrame) -> list[str]:
    name = _format_cell(row.get("strategy_name"))
    version = _format_cell(row.get("strategy_version"))
    lines = [
        "",
        f"### {name}:{version}",
        f"- evaluation_status：{_format_cell(row.get('evaluation_status'))}",
        f"- risk_level：{_format_cell(row.get('risk_level'))}",
        f"- recommendation：{_format_cell(row.get('recommendation'))}",
        f"- evaluation_score：{_format_cell(row.get('evaluation_score'), 'evaluation_score')}",
        f"- valid_count：{_format_cell(row.get('valid_count'))}",
        f"- win_rate_3d：{_format_cell(row.get('win_rate_3d'), 'win_rate_3d')}",
        f"- avg_return_3d：{_format_cell(row.get('avg_return_3d'), 'avg_return_3d')}",
        f"- avg_max_drawdown_3d：{_format_cell(row.get('avg_max_drawdown_3d'), 'avg_max_drawdown_3d')}",
        f"- evaluation_reason：{_format_cell(row.get('evaluation_reason'))}",
    ]

    matched = _match_performance(performance, row.get("strategy_name"), row.get("strategy_version"))
    if matched is not None:
        for column in [
            "win_rate_1d",
            "win_rate_5d",
            "avg_return_1d",
            "avg_return_5d",
            "avg_max_drawdown_1d",
            "avg_max_drawdown_5d",
        ]:
            lines.append(f"- {column}：{_format_cell(matched.get(column), column)}")
    return lines


def _match_performance(
    performance: pd.DataFrame,
    strategy_name: object,
    strategy_version: object,
) -> pd.Series | None:
    if performance.empty or not {"strategy_name", "strategy_version"}.issubset(performance.columns):
        return None
    matched = performance[
        (performance["strategy_name"].astype(str) == str(strategy_name))
        & (performance["strategy_version"].astype(str) == str(strategy_version))
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def _format_cell(value: object, column: str | None = None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if column in PERCENT_COLUMNS:
        return f"{float(value) * 100:.2f}%"
    if column == "evaluation_score":
        return f"{float(value):.4f}"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
