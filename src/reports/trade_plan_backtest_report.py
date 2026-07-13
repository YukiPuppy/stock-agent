from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


PERFORMANCE_TABLE_COLUMNS = [
    "strategy_names",
    "strategy_versions",
    "action",
    "max_holding_days",
    "plan_count",
    "trigger_rate",
    "win_rate",
    "avg_return",
    "avg_max_drawdown",
    "stop_loss_rate",
    "take_profit_rate",
]
DETAIL_COLUMNS = [
    "plan_date",
    "code",
    "name",
    "action",
    "entry_price",
    "exit_price",
    "exit_reason",
    "holding_days",
    "max_holding_days",
    "return_pct",
    "max_drawdown",
    "max_favorable",
    "invalid_reason",
]
PERCENT_COLUMNS = {
    "trigger_rate",
    "win_rate",
    "avg_return",
    "avg_max_drawdown",
    "stop_loss_rate",
    "take_profit_rate",
    "return_pct",
    "max_drawdown",
    "max_favorable",
}
FORBIDDEN_REPORT_PHRASES = ("保证" + "盈利", "稳" + "赚", "满" + "仓")


def generate_trade_plan_backtest_report(
    backtest_results: pd.DataFrame,
    performance: pd.DataFrame,
    report_date: str | None = None,
) -> str:
    results = _as_dataframe(backtest_results)
    performance = _as_dataframe(performance)
    resolved_report_date = report_date or date.today().isoformat()

    lines = [
        "# 交易计划规则回测报告",
        "",
        f"报告日期：{resolved_report_date}",
        "",
        "## 一、报告说明",
        "- 本报告基于历史交易计划和日线行情进行规则级回测；",
        "- 用于评估买入区间、止损、止盈和持有周期设计；",
        "- 不构成投资建议；",
        "- 当前回测暂未完整考虑滑点、涨跌停无法成交等复杂情况。",
        "",
        "## 二、总体结果",
    ]
    lines.extend(_summary_lines(results))

    lines.extend(["", "## 三、策略版本表现"])
    lines.extend(_markdown_table(performance, PERFORMANCE_TABLE_COLUMNS) if not performance.empty else ["当前没有可用的策略版本表现。"])

    lines.extend(["", "## 四、holding_days distribution by strategy"])
    holding_distribution = _distribution(results, ["strategy_names", "holding_days"])
    lines.extend(
        _markdown_table(holding_distribution, ["strategy_names", "holding_days", "count"])
        if not holding_distribution.empty
        else ["当前没有持仓天数分布。"]
    )

    lines.extend(["", "## 五、exit_reason distribution by strategy"])
    exit_distribution = _distribution(_valid_results(results), ["strategy_names", "exit_reason"])
    lines.extend(
        _markdown_table(exit_distribution, ["strategy_names", "exit_reason", "count"])
        if not exit_distribution.empty
        else ["当前没有退出原因分布。"]
    )

    lines.extend(["", "## 六、performance by max_holding_days"])
    holding_performance = _performance_by_holding_days(results)
    lines.extend(
        _markdown_table(
            holding_performance,
            ["max_holding_days", "plan_count", "valid_count", "win_rate", "avg_return", "avg_max_drawdown"],
        )
        if not holding_performance.empty
        else ["当前没有按最大持仓天数汇总的表现。"]
    )

    lines.extend(["", "## 七、performance by strategy_name / strategy_version / max_holding_days"])
    strategy_holding_columns = [
        "strategy_names",
        "strategy_versions",
        "max_holding_days",
        "plan_count",
        "valid_count",
        "win_rate",
        "avg_return",
        "avg_max_drawdown",
    ]
    lines.extend(
        _markdown_table(performance, strategy_holding_columns)
        if not performance.empty
        else ["当前没有策略版本与持仓周期交叉表现。"]
    )

    lines.extend(["", "## 八、回测明细"])
    lines.extend(_markdown_table(results.head(50), DETAIL_COLUMNS) if not results.empty else ["当前没有可用的回测明细。"])

    lines.extend(["", "## 九、问题观察"])
    lines.extend(_observation_lines(results, performance))

    lines.extend(
        [
            "",
            "## 十、风险提示",
            "- 规则回测结果不代表未来收益；",
            "- 当前回测仍可能高估实际可成交性；",
            "- 涨跌停、滑点、成交量约束后续仍需补充；",
            "- 实盘前仍需模拟盘和小仓位验证。",
        ]
    )

    report = "\n".join(lines).strip() + "\n"
    for phrase in FORBIDDEN_REPORT_PHRASES:
        report = report.replace(phrase, "")
    return report


def _summary_lines(results: pd.DataFrame) -> list[str]:
    valid = _valid_results(results)
    returns = _numeric_series(valid, "return_pct")
    drawdowns = _numeric_series(valid, "max_drawdown")
    exit_reason = valid.get("exit_reason", pd.Series(dtype=object)).fillna("").astype(str)
    return [
        f"- 计划数量：{len(results)}",
        f"- 触发交易数量：{_count_bool(results, 'is_triggered')}",
        f"- 有效回测数量：{len(valid)}",
        f"- 平均收益：{_format_percent(returns.mean() if not returns.empty else None)}",
        f"- 胜率：{_format_percent((returns > 0).mean() if not returns.empty else None)}",
        f"- 平均最大回撤：{_format_percent(drawdowns.mean() if not drawdowns.empty else None)}",
        f"- 止损触发比例：{_format_percent((exit_reason == 'stop_loss').mean() if not valid.empty else None)}",
        f"- 止盈触发比例：{_format_percent(exit_reason.isin(['take_profit_1', 'take_profit_2']).mean() if not valid.empty else None)}",
    ]


def _observation_lines(results: pd.DataFrame, performance: pd.DataFrame) -> list[str]:
    not_reach = _count_reason(results, "not_reach_entry_range")
    gap_above = _count_reason(results, "gap_above_entry_range")
    stop_loss_heavy = _top_strategy(performance, "stop_loss_rate")
    time_exit_heavy = _top_strategy(performance, "time_exit_rate")
    return [
        f"- 未触发买入区间的计划数量：{not_reach}",
        f"- 高开越过买入区间的数量：{gap_above}",
        f"- 止损触发较多的策略：{stop_loss_heavy}",
        f"- 时间退出较多的策略：{time_exit_heavy}",
    ]


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    selected = list(columns)
    lines = [
        "| " + " | ".join(selected) + " |",
        "| " + " | ".join(["---"] * len(selected)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_cell(row.get(column), column) for column in selected]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df


def _valid_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "is_valid" not in results.columns:
        return pd.DataFrame()
    return results[results["is_valid"].fillna(False).astype(bool)].copy()


def _count_bool(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def _count_reason(df: pd.DataFrame, reason: str) -> int:
    if df.empty or "invalid_reason" not in df.columns:
        return 0
    return int((df["invalid_reason"].fillna("").astype(str) == reason).sum())


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _distribution(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty or any(column not in df.columns for column in columns):
        return pd.DataFrame(columns=columns + ["count"])
    return df.groupby(columns, dropna=False).size().rename("count").reset_index()


def _performance_by_holding_days(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "max_holding_days" not in results.columns:
        return pd.DataFrame()
    rows = []
    for max_holding_days, group in results.groupby("max_holding_days", dropna=False, sort=True):
        valid = _valid_results(group)
        returns = _numeric_series(valid, "return_pct")
        drawdowns = _numeric_series(valid, "max_drawdown")
        rows.append(
            {
                "max_holding_days": max_holding_days,
                "plan_count": len(group),
                "valid_count": len(valid),
                "win_rate": float((returns > 0).mean()) if not returns.empty else None,
                "avg_return": float(returns.mean()) if not returns.empty else None,
                "avg_max_drawdown": float(drawdowns.mean()) if not drawdowns.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _top_strategy(performance: pd.DataFrame, metric: str) -> str:
    if performance.empty or metric not in performance.columns:
        return "暂无"
    values = pd.to_numeric(performance[metric], errors="coerce")
    if values.dropna().empty or values.max() <= 0:
        return "暂无"
    row = performance.loc[values.idxmax()]
    return f"{row.get('strategy_names', '')}:{row.get('strategy_versions', '')}（{_format_percent(row.get(metric))}）"


def _format_cell(value: object, column: str | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    if column in PERCENT_COLUMNS:
        return _format_percent(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/")


def _format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"
