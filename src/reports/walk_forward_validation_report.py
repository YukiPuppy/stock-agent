from __future__ import annotations

from datetime import date

import pandas as pd


REPORT_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "train_valid_count",
    "train_win_rate_3d",
    "train_avg_return_3d",
    "validation_valid_count",
    "validation_win_rate_3d",
    "validation_avg_return_3d",
    "return_decay",
    "win_rate_decay",
    "stability_score",
    "overfit_risk",
    "validation_status",
]
PERCENT_COLUMNS = {
    "train_win_rate_3d",
    "train_avg_return_3d",
    "validation_win_rate_3d",
    "validation_avg_return_3d",
    "return_decay",
    "win_rate_decay",
}


def generate_walk_forward_validation_report(
    validation: pd.DataFrame,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    validation_start_date: str | None = None,
    validation_end_date: str | None = None,
    report_date: str | None = None,
) -> str:
    report_date = report_date or date.today().isoformat()
    validation = validation.copy() if validation is not None else pd.DataFrame()

    lines = [
        "# 策略样本外验证报告",
        "",
        f"报告日期：{report_date}",
        "",
        "## 一、报告说明",
        "",
        "- 本报告基于训练区间和样本外验证区间的回测结果生成；",
        "- 用于识别参数搜索中的过拟合风险；",
        "- 不构成投资建议；",
        "- 样本外通过只代表历史验证更稳健，不代表未来表现。",
        "",
        "## 二、验证区间",
        "",
        f"- 训练区间：{_range_text(train_start_date, train_end_date)}",
        f"- 样本外验证区间：{_range_text(validation_start_date, validation_end_date)}",
        "",
        "## 三、总体结论",
        "",
    ]

    counts = _status_counts(validation)
    lines.extend(
        [
            f"- 参数版本总数：{len(validation)}",
            f"- passed_oos 数量：{counts.get('passed_oos', 0)}",
            f"- unstable 数量：{counts.get('unstable', 0)}",
            f"- failed_oos 数量：{counts.get('failed_oos', 0)}",
            f"- insufficient_train_samples 数量：{counts.get('insufficient_train_samples', 0)}",
            f"- insufficient_validation_samples 数量：{counts.get('insufficient_validation_samples', 0)}",
            f"- needs_more_observation 数量：{counts.get('needs_more_observation', 0)}",
            "",
            "## 四、样本外验证总表",
            "",
        ]
    )

    lines.append(_format_table(validation, REPORT_COLUMNS))
    lines.extend(["", "## 五、通过样本外验证的策略版本", ""])
    passed = _filter_status(validation, "passed_oos")
    if passed.empty:
        lines.append("当前没有明确通过样本外验证的参数版本。")
    else:
        lines.append(_format_table(passed, REPORT_COLUMNS))

    lines.extend(["", "## 六、疑似过拟合或不稳定版本", ""])
    unstable = _unstable_versions(validation)
    if unstable.empty:
        lines.append("当前没有高风险或明显不稳定的参数版本。")
    else:
        for record in unstable.to_dict("records"):
            lines.append(
                f"- {record.get('strategy_name')} / {record.get('strategy_version')}："
                f"{record.get('validation_reason', '')}"
            )

    lines.extend(["", "## 七、样本不足版本", ""])
    insufficient = validation[
        validation.get("validation_status", pd.Series(dtype=str)).isin(
            ["insufficient_train_samples", "insufficient_validation_samples"]
        )
    ] if not validation.empty else pd.DataFrame()
    if insufficient.empty:
        lines.append("当前没有样本不足的参数版本。")
    else:
        lines.append("以下版本样本不足，不能用于实盘判断。")
        for record in insufficient.to_dict("records"):
            lines.append(
                f"- {record.get('strategy_name')} / {record.get('strategy_version')}："
                f"{record.get('validation_reason', '')}"
            )

    lines.extend(
        [
            "",
            "## 八、风险提示",
            "",
            "- 样本外验证可以降低过拟合风险，但不能消除未来不确定性；",
            "- 不应只选择历史训练区间表现最好的参数；",
            "- 样本数量不足的版本不能进入实盘；",
            "- 策略启用前仍需模拟盘和小仓位实盘验证。",
            "",
        ]
    )
    return "\n".join(lines)


def _range_text(start_date: str | None, end_date: str | None) -> str:
    return f"{start_date or '未指定'} 至 {end_date or '未指定'}"


def _status_counts(validation: pd.DataFrame) -> dict[str, int]:
    if validation.empty or "validation_status" not in validation.columns:
        return {}
    return validation["validation_status"].fillna("").astype(str).value_counts().to_dict()


def _filter_status(validation: pd.DataFrame, status: str) -> pd.DataFrame:
    if validation.empty or "validation_status" not in validation.columns:
        return pd.DataFrame()
    return validation[validation["validation_status"].astype(str) == status].copy()


def _unstable_versions(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    risk = validation.get("overfit_risk", pd.Series(dtype=str)).isin(["high", "medium"])
    status = validation.get("validation_status", pd.Series(dtype=str)).isin(["failed_oos", "unstable"])
    return validation[risk | status].copy()


def _format_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "当前没有样本外验证结果。"
    table = df.copy()
    selected = [column for column in columns if column in table.columns]
    table = table.loc[:, selected]
    for column in table.columns:
        if column in PERCENT_COLUMNS:
            table[column] = table[column].apply(_format_percent)
        elif column == "stability_score":
            table[column] = table[column].apply(_format_score)
    return table.to_markdown(index=False)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _format_score(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"
