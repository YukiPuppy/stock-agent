from __future__ import annotations

import pandas as pd


DAILY_REVIEW_COLUMNS = [
    "trade_date",
    "actual_trade_count",
    "buy_count",
    "sell_count",
    "planned_trade_count",
    "matched_plan_count",
    "off_plan_count",
    "follow_plan_count",
    "deviation_count",
    "chase_count",
    "over_position_count",
    "bought_watch_only_count",
    "execution_score",
    "main_issues",
    "review_summary",
    "next_action_suggestion",
    "valid_performance_count",
    "avg_return_1d",
    "avg_return_3d",
    "avg_return_5d",
    "plan_trade_avg_return_3d",
    "off_plan_avg_return_3d",
    "chase_trade_count",
    "chase_avg_return_3d",
]


def generate_daily_review(
    actual_trades: pd.DataFrame,
    execution_review: pd.DataFrame,
    trade_plan: pd.DataFrame | None = None,
    trade_date: str | None = None,
    actual_trade_performance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize one day's actual execution quality from local deterministic reviews."""
    actual = _as_dataframe(actual_trades)
    execution = _as_dataframe(execution_review)
    plan = _as_dataframe(trade_plan)
    performance = _as_dataframe(actual_trade_performance)
    resolved_trade_date = _resolve_trade_date(actual, execution, plan, trade_date)
    performance_stats = _build_performance_stats(performance)

    planned_trade_count = len(plan)
    if actual.empty:
        row = {
            "trade_date": resolved_trade_date,
            "actual_trade_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "planned_trade_count": planned_trade_count,
            "matched_plan_count": 0,
            "off_plan_count": 0,
            "follow_plan_count": 0,
            "deviation_count": 0,
            "chase_count": 0,
            "over_position_count": 0,
            "bought_watch_only_count": 0,
            "execution_score": 100,
            "main_issues": "未发现明显执行偏差",
            "review_summary": "当日无实际交易记录。",
            "next_action_suggestion": "无需执行偏差复盘，可继续观察系统计划表现。",
            **performance_stats,
        }
        return pd.DataFrame([row], columns=DAILY_REVIEW_COLUMNS)

    actual_trade_count = len(actual)
    buy_count = int((actual.get("side", pd.Series(dtype=object)) == "buy").sum())
    sell_count = int((actual.get("side", pd.Series(dtype=object)) == "sell").sum())
    matched_plan_count = _count_not_equal(execution, "plan_match_status", "no_plan")
    off_plan_count = _count_equal(execution, "execution_status", "off_plan")
    follow_plan_count = _count_equal(execution, "execution_status", "follow_plan")
    deviation_count = _count_equal(execution, "execution_status", "deviation")
    chase_count = _count_flag(execution, "chase_above_entry")
    over_position_count = _count_flag(execution, "over_position")
    bought_watch_only_count = _count_flag(execution, "bought_watch_only")

    execution_score = max(
        0,
        100
        - off_plan_count * 20
        - deviation_count * 15
        - chase_count * 10
        - over_position_count * 10
        - bought_watch_only_count * 20,
    )
    main_issues = _build_main_issues(
        off_plan_count=off_plan_count,
        chase_count=chase_count,
        over_position_count=over_position_count,
        bought_watch_only_count=bought_watch_only_count,
    )

    row = {
        "trade_date": resolved_trade_date,
        "actual_trade_count": actual_trade_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "planned_trade_count": planned_trade_count,
        "matched_plan_count": matched_plan_count,
        "off_plan_count": off_plan_count,
        "follow_plan_count": follow_plan_count,
        "deviation_count": deviation_count,
        "chase_count": chase_count,
        "over_position_count": over_position_count,
        "bought_watch_only_count": bought_watch_only_count,
        "execution_score": execution_score,
        "main_issues": main_issues,
        "review_summary": _build_review_summary(
            actual_trade_count,
            follow_plan_count,
            off_plan_count,
            deviation_count,
            performance_stats,
            performance,
        ),
        "next_action_suggestion": _build_next_action_suggestion(
            off_plan_count + deviation_count + chase_count + over_position_count + bought_watch_only_count
        ),
        **performance_stats,
    }
    return pd.DataFrame([row], columns=DAILY_REVIEW_COLUMNS)


def _as_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df


def _resolve_trade_date(
    actual_trades: pd.DataFrame,
    execution_review: pd.DataFrame,
    trade_plan: pd.DataFrame,
    trade_date: str | None,
) -> str:
    if trade_date:
        return str(trade_date)
    for df in (actual_trades, execution_review, trade_plan):
        if not df.empty and "trade_date" in df.columns:
            values = df["trade_date"].dropna()
            if not values.empty:
                return str(values.max())
    return ""


def _count_equal(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int((df[column].fillna("").astype(str) == value).sum())


def _count_not_equal(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    series = df[column].fillna("").astype(str)
    return int((series != value).sum())


def _count_flag(df: pd.DataFrame, flag: str) -> int:
    if df.empty or "execution_flags" not in df.columns:
        return 0
    return int(df["execution_flags"].fillna("").astype(str).str.contains(flag, regex=False).sum())


def _build_main_issues(
    off_plan_count: int,
    chase_count: int,
    over_position_count: int,
    bought_watch_only_count: int,
) -> str:
    issues: list[str] = []
    if off_plan_count > 0:
        issues.append("存在计划外交易")
    if chase_count > 0:
        issues.append("存在高于计划买入区间的追高偏差")
    if over_position_count > 0:
        issues.append("存在仓位超过计划上限")
    if bought_watch_only_count > 0:
        issues.append("存在仅观察标的被实际买入")
    return "；".join(issues) if issues else "未发现明显执行偏差"


def _build_review_summary(
    actual_trade_count: int,
    follow_plan_count: int,
    off_plan_count: int,
    deviation_count: int,
    performance_stats: dict | None = None,
    performance: pd.DataFrame | None = None,
) -> str:
    if off_plan_count == 0 and deviation_count == 0:
        summary = f"当日共记录 {actual_trade_count} 笔实际交易，整体执行与计划匹配度较好。"
    else:
        summary = (
            f"当日共记录 {actual_trade_count} 笔实际交易，其中 {follow_plan_count} 笔按计划执行，"
            f"{off_plan_count} 笔计划外交易，{deviation_count} 笔存在执行偏差。"
        )
    notes = _build_performance_summary_notes(performance_stats or {}, _as_dataframe(performance))
    return summary + (" " + "；".join(notes) + "。" if notes else "")


def _build_performance_stats(performance: pd.DataFrame) -> dict:
    valid = _valid_performance(performance)
    status = _string_column(valid, "execution_status")
    plan = valid[status != "off_plan"] if not valid.empty else valid
    off_plan = valid[status == "off_plan"] if not valid.empty else valid
    chase = valid[
        _string_column(valid, "execution_flags").str.contains("chase_above_entry", regex=False)
    ] if not valid.empty else valid
    return {
        "valid_performance_count": len(valid),
        "avg_return_1d": _mean(valid, "return_1d"),
        "avg_return_3d": _mean(valid, "return_3d"),
        "avg_return_5d": _mean(valid, "return_5d"),
        "plan_trade_avg_return_3d": _mean(plan, "return_3d"),
        "off_plan_avg_return_3d": _mean(off_plan, "return_3d"),
        "chase_trade_count": len(chase),
        "chase_avg_return_3d": _mean(chase, "return_3d"),
    }


def _string_column(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[column].fillna("").astype(str)


def _valid_performance(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty or "is_valid" not in performance.columns:
        return pd.DataFrame()
    return performance[performance["is_valid"].fillna(False).astype(bool)].copy()


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _build_performance_summary_notes(performance_stats: dict, performance: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    plan_return = performance_stats.get("plan_trade_avg_return_3d")
    off_plan_return = performance_stats.get("off_plan_avg_return_3d")
    if pd.notna(plan_return) and pd.notna(off_plan_return) and float(off_plan_return) < float(plan_return) - 0.02:
        notes.append("计划外交易表现偏弱")
    valid = _valid_performance(performance)
    if not valid.empty:
        chase = valid[
            _string_column(valid, "execution_flags").str.contains("chase_above_entry", regex=False)
        ]
        drawdown = pd.to_numeric(chase.get("max_drawdown_3d", pd.Series(dtype=float)), errors="coerce")
        if not drawdown.dropna().empty and float(drawdown.mean()) < -0.05:
            notes.append("追高交易风险较高")
    return notes


def _build_next_action_suggestion(issue_count: int) -> str:
    if issue_count >= 2:
        return "问题较多，建议降低交易频率、严格按计划执行，并将计划外交易单独记录。"
    if issue_count == 1:
        return "建议复核当日偏差来源，下一交易日优先按计划价格区间和仓位约束执行。"
    return "执行良好，建议继续保持，后续结合收益结果评估策略有效性。"
