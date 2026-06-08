from __future__ import annotations

import pandas as pd


PERIOD_REVIEW_COLUMNS = [
    "start_date",
    "end_date",
    "trading_days",
    "actual_trade_count",
    "buy_count",
    "sell_count",
    "follow_plan_count",
    "off_plan_count",
    "deviation_count",
    "chase_count",
    "over_position_count",
    "bought_watch_only_count",
    "avg_execution_score",
    "valid_performance_count",
    "avg_return_1d",
    "avg_return_3d",
    "avg_return_5d",
    "plan_trade_avg_return_3d",
    "off_plan_avg_return_3d",
    "chase_avg_return_3d",
    "over_position_avg_return_3d",
    "best_trade_code",
    "worst_trade_code",
    "main_issues",
    "period_summary",
    "next_period_suggestion",
]


def generate_period_review(
    actual_trades: pd.DataFrame,
    execution_review: pd.DataFrame,
    trade_performance: pd.DataFrame,
    daily_review: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    actual = _filter_by_date(_as_dataframe(actual_trades), start_date, end_date)
    execution = _filter_by_date(_as_dataframe(execution_review), start_date, end_date)
    performance = _filter_by_date(_as_dataframe(trade_performance), start_date, end_date)
    daily = _filter_by_date(_as_dataframe(daily_review), start_date, end_date)
    resolved_start, resolved_end = _resolve_period(actual, execution, performance, daily, start_date, end_date)

    if actual.empty:
        return pd.DataFrame(
            [
                {
                    "start_date": resolved_start,
                    "end_date": resolved_end,
                    "trading_days": 0,
                    "actual_trade_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "follow_plan_count": 0,
                    "off_plan_count": 0,
                    "deviation_count": 0,
                    "chase_count": 0,
                    "over_position_count": 0,
                    "bought_watch_only_count": 0,
                    "avg_execution_score": _mean(daily, "execution_score") if not daily.empty else None,
                    "valid_performance_count": 0,
                    "avg_return_1d": None,
                    "avg_return_3d": None,
                    "avg_return_5d": None,
                    "plan_trade_avg_return_3d": None,
                    "off_plan_avg_return_3d": None,
                    "chase_avg_return_3d": None,
                    "over_position_avg_return_3d": None,
                    "best_trade_code": "",
                    "worst_trade_code": "",
                    "main_issues": "本周期无实际交易记录。",
                    "period_summary": "本周期无实际交易记录。",
                    "next_period_suggestion": "可继续观察系统计划表现，等待更多实盘样本。",
                }
            ],
            columns=PERIOD_REVIEW_COLUMNS,
        )

    valid = _valid_performance(performance)
    follow_plan_count = _count_equal(execution, "execution_status", "follow_plan")
    off_plan_count = _count_equal(execution, "execution_status", "off_plan")
    deviation_count = _count_equal(execution, "execution_status", "deviation")
    chase_count = _count_flag(execution, "chase_above_entry")
    over_position_count = _count_flag(execution, "over_position")
    bought_watch_only_count = _count_flag(execution, "bought_watch_only")

    row = {
        "start_date": resolved_start,
        "end_date": resolved_end,
        "trading_days": _nunique(actual, "trade_date"),
        "actual_trade_count": len(actual),
        "buy_count": _count_equal(actual, "side", "buy"),
        "sell_count": _count_equal(actual, "side", "sell"),
        "follow_plan_count": follow_plan_count,
        "off_plan_count": off_plan_count,
        "deviation_count": deviation_count,
        "chase_count": chase_count,
        "over_position_count": over_position_count,
        "bought_watch_only_count": bought_watch_only_count,
        "avg_execution_score": _mean(daily, "execution_score") if not daily.empty else None,
        "valid_performance_count": len(valid),
        "avg_return_1d": _mean(valid, "return_1d"),
        "avg_return_3d": _mean(valid, "return_3d"),
        "avg_return_5d": _mean(valid, "return_5d"),
        "plan_trade_avg_return_3d": _mean(_status(valid, "follow_plan"), "return_3d"),
        "off_plan_avg_return_3d": _mean(_status(valid, "off_plan"), "return_3d"),
        "chase_avg_return_3d": _mean(_flagged(valid, "chase_above_entry"), "return_3d"),
        "over_position_avg_return_3d": _mean(_flagged(valid, "over_position"), "return_3d"),
        "best_trade_code": _extreme_code(valid, "return_3d", "max"),
        "worst_trade_code": _extreme_code(valid, "return_3d", "min"),
        "main_issues": _build_main_issues(
            off_plan_count,
            deviation_count,
            chase_count,
            over_position_count,
            bought_watch_only_count,
        ),
        "period_summary": _build_period_summary(
            len(actual),
            follow_plan_count,
            off_plan_count,
            deviation_count,
            len(valid),
            _mean(valid, "return_3d"),
        ),
        "next_period_suggestion": _build_next_period_suggestion(
            off_plan_count,
            chase_count,
            over_position_count,
            _mean(_flagged(valid, "chase_above_entry"), "return_3d"),
        ),
    }
    return pd.DataFrame([row], columns=PERIOD_REVIEW_COLUMNS)


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df.copy()


def _filter_by_date(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if df.empty or "trade_date" not in df.columns:
        return df
    result = df.copy()
    dates = result["trade_date"].astype(str)
    if start_date is not None:
        result = result[dates >= str(start_date)]
        dates = result["trade_date"].astype(str)
    if end_date is not None:
        result = result[dates <= str(end_date)]
    return result.copy()


def _resolve_period(
    actual: pd.DataFrame,
    execution: pd.DataFrame,
    performance: pd.DataFrame,
    daily: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    values: list[str] = []
    for df in (actual, execution, performance, daily):
        if not df.empty and "trade_date" in df.columns:
            values.extend(str(value) for value in df["trade_date"].dropna())
    resolved_start = str(start_date) if start_date is not None else (min(values) if values else "")
    resolved_end = str(end_date) if end_date is not None else (max(values) if values else "")
    return resolved_start, resolved_end


def _count_equal(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int((_string_column(df, column) == value).sum())


def _count_flag(df: pd.DataFrame, flag: str) -> int:
    if df.empty or "execution_flags" not in df.columns:
        return 0
    return int(_string_column(df, "execution_flags").str.contains(flag, regex=False).sum())


def _nunique(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].dropna().astype(str).nunique())


def _valid_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_valid" not in df.columns:
        return pd.DataFrame()
    return df[df["is_valid"].fillna(False).astype(bool)].copy()


def _status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_status") == status].copy()


def _flagged(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[_string_column(df, "execution_flags").str.contains(flag, regex=False)].copy()


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


def _extreme_code(df: pd.DataFrame, column: str, method: str) -> str:
    if df.empty or column not in df.columns or "code" not in df.columns:
        return ""
    values = pd.to_numeric(df[column], errors="coerce")
    if values.dropna().empty:
        return ""
    index = values.idxmax() if method == "max" else values.idxmin()
    return str(df.loc[index, "code"])


def _build_main_issues(
    off_plan_count: int,
    deviation_count: int,
    chase_count: int,
    over_position_count: int,
    bought_watch_only_count: int,
) -> str:
    issues: list[str] = []
    if off_plan_count > 0:
        issues.append(f"计划外交易 {off_plan_count} 笔")
    if deviation_count > 0:
        issues.append(f"执行偏差 {deviation_count} 笔")
    if chase_count > 0:
        issues.append(f"追高偏差 {chase_count} 笔")
    if over_position_count > 0:
        issues.append(f"超出计划仓位 {over_position_count} 笔")
    if bought_watch_only_count > 0:
        issues.append(f"仅观察标的实际买入 {bought_watch_only_count} 笔")
    return "；".join(issues) if issues else "未发现明显执行偏差。"


def _build_period_summary(
    actual_trade_count: int,
    follow_plan_count: int,
    off_plan_count: int,
    deviation_count: int,
    valid_performance_count: int,
    avg_return_3d: float | None,
) -> str:
    summary = (
        f"本周期共记录 {actual_trade_count} 笔实际交易，其中 {follow_plan_count} 笔按计划执行，"
        f"{off_plan_count} 笔计划外交易，{deviation_count} 笔存在执行偏差。"
    )
    if valid_performance_count > 0 and avg_return_3d is not None:
        summary += f" 有效表现样本 {valid_performance_count} 笔，3日平均收益为 {avg_return_3d * 100:.2f}%。"
    elif valid_performance_count == 0:
        summary += " 当前暂无有效交易表现样本。"
    return summary


def _build_next_period_suggestion(
    off_plan_count: int,
    chase_count: int,
    over_position_count: int,
    chase_avg_return_3d: float | None,
) -> str:
    suggestions: list[str] = []
    if off_plan_count > 0:
        suggestions.append("计划外交易较多，建议减少计划外交易，并将其与系统计划交易分开统计。")
    if chase_count > 0 and (chase_avg_return_3d is None or chase_avg_return_3d < 0):
        suggestions.append("追高交易表现较差，建议严格执行高开或超过买入区间不追规则。")
    elif chase_count > 0:
        suggestions.append("存在追高交易，建议继续单独跟踪其表现并严格执行买入区间。")
    if over_position_count > 0:
        suggestions.append("超出计划仓位的交易较多，建议控制单票仓位。")
    if not suggestions:
        suggestions.append("执行良好，建议继续保持，并等待更多样本评估策略有效性。")
    return " ".join(suggestions)
