from __future__ import annotations

import pandas as pd

from src.trading.actual_trades import normalize_actual_trades


EXECUTION_REVIEW_COLUMNS = [
    "trade_date",
    "trade_time",
    "code",
    "name",
    "side",
    "actual_price",
    "actual_volume",
    "actual_amount",
    "position_ratio",
    "plan_rank",
    "planned_action",
    "entry_low",
    "entry_high",
    "position_low",
    "position_high",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "plan_match_status",
    "execution_status",
    "execution_flags",
    "execution_comment",
]

PLAN_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "rank",
    "action",
    "entry_low",
    "entry_high",
    "position_low",
    "position_high",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
]


def review_execution(actual_trades: pd.DataFrame, trade_plan: pd.DataFrame) -> pd.DataFrame:
    """Compare actual trades with deterministic trade plans."""
    if actual_trades.empty:
        return pd.DataFrame(columns=EXECUTION_REVIEW_COLUMNS)

    actual = normalize_actual_trades(actual_trades)
    plan = _normalize_trade_plan(trade_plan)
    plan_map = {
        (str(row["trade_date"]), str(row["code"])): row
        for _, row in plan.iterrows()
    }

    rows = [_review_row(row, plan_map.get((str(row["trade_date"]), str(row["code"])))) for _, row in actual.iterrows()]
    return pd.DataFrame(rows, columns=EXECUTION_REVIEW_COLUMNS)


def _normalize_trade_plan(trade_plan: pd.DataFrame) -> pd.DataFrame:
    plan = trade_plan.copy()
    for column in PLAN_COLUMNS:
        if column not in plan.columns:
            plan[column] = None
    plan = plan.loc[:, PLAN_COLUMNS].copy()
    plan["code"] = plan["code"].map(lambda value: normalize_actual_trades(pd.DataFrame({"code": [value]})).loc[0, "code"])
    for column in [
        "rank",
        "entry_low",
        "entry_high",
        "position_low",
        "position_high",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
    ]:
        plan[column] = pd.to_numeric(plan[column], errors="coerce")
    return plan.drop_duplicates(subset=["trade_date", "code"], keep="last")


def _review_row(actual: pd.Series, plan: pd.Series | None) -> dict:
    row = {
        "trade_date": actual.get("trade_date"),
        "trade_time": actual.get("trade_time"),
        "code": actual.get("code"),
        "name": actual.get("name"),
        "side": actual.get("side"),
        "actual_price": actual.get("price"),
        "actual_volume": actual.get("volume"),
        "actual_amount": actual.get("amount"),
        "position_ratio": actual.get("position_ratio"),
        "plan_rank": actual.get("plan_rank"),
        "planned_action": None,
        "entry_low": None,
        "entry_high": None,
        "position_low": None,
        "position_high": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "plan_match_status": "no_plan",
        "execution_status": "off_plan",
        "execution_flags": "plan_not_found",
        "execution_comment": "该交易未匹配到系统交易计划",
    }
    if plan is None:
        return row

    row.update(
        {
            "name": actual.get("name") or plan.get("name"),
            "plan_rank": _coalesce(actual.get("plan_rank"), plan.get("rank")),
            "planned_action": plan.get("action"),
            "entry_low": plan.get("entry_low"),
            "entry_high": plan.get("entry_high"),
            "position_low": plan.get("position_low"),
            "position_high": plan.get("position_high"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit_1": plan.get("take_profit_1"),
            "take_profit_2": plan.get("take_profit_2"),
            "plan_match_status": "matched",
            "execution_status": "recorded",
            "execution_flags": "",
            "execution_comment": "",
        }
    )

    flags: list[str] = []
    comments: list[str] = []
    side = actual.get("side")

    if side == "sell":
        flags.append("sell_recorded")
        row["execution_status"] = "recorded_sell"
    elif side == "buy":
        _review_buy_price(actual, plan, row, flags, comments)
        _review_position(actual, plan, row, flags, comments)
    else:
        flags.append("unknown_side")
        row["execution_status"] = "recorded"

    row["execution_flags"] = ",".join(flags)
    row["execution_comment"] = "；".join(comments)
    return row


def _review_buy_price(
    actual: pd.Series,
    plan: pd.Series,
    row: dict,
    flags: list[str],
    comments: list[str],
) -> None:
    price = actual.get("price")
    entry_low = plan.get("entry_low")
    entry_high = plan.get("entry_high")
    if pd.isna(entry_low) or pd.isna(entry_high):
        flags.append("bought_watch_only")
        row["execution_status"] = "deviation"
        comments.append("该标的计划为仅观察，但实际发生买入")
        return

    if pd.isna(price):
        flags.append("price_missing")
        row["execution_status"] = "recorded"
    elif entry_low <= price <= entry_high:
        flags.append("price_in_range")
        row["execution_status"] = "follow_plan"
    elif price > entry_high:
        flags.append("chase_above_entry")
        row["execution_status"] = "deviation"
        comments.append("买入价格高于计划买入区间上沿，存在追高偏差")
    elif price < entry_low:
        flags.append("below_entry_range")
        row["execution_status"] = "deviation"
        comments.append("买入价格低于计划区间下沿，需确认是否仍符合原计划")


def _review_position(
    actual: pd.Series,
    plan: pd.Series,
    row: dict,
    flags: list[str],
    comments: list[str],
) -> None:
    position_ratio = actual.get("position_ratio")
    position_high = plan.get("position_high")
    if pd.notna(position_ratio) and pd.notna(position_high) and position_ratio > position_high:
        flags.append("over_position")
        row["execution_status"] = "deviation"
        comments.append("实际仓位超过计划仓位上限")


def _coalesce(left: object, right: object) -> object:
    return left if pd.notna(left) else right
