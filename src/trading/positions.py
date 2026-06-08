from __future__ import annotations

import re

import pandas as pd

from src.trading.actual_trades import normalize_actual_trades


POSITION_COLUMNS = [
    "as_of_date",
    "code",
    "name",
    "holding_volume",
    "available_volume",
    "frozen_volume",
    "cost_amount",
    "cost_price",
    "latest_price",
    "market_value",
    "floating_pnl",
    "floating_pnl_pct",
    "position_ratio",
    "first_buy_date",
    "latest_trade_date",
    "strategy_name",
    "plan_rank",
    "t_plus_1_status",
    "position_status",
]

POSITION_REVIEW_COLUMNS = POSITION_COLUMNS + [
    "planned_stop_loss",
    "planned_take_profit_1",
    "planned_take_profit_2",
    "position_risk_level",
    "position_flags",
    "position_comment",
    "next_action_hint",
]


def build_positions_from_trades(
    actual_trades: pd.DataFrame,
    daily_bars: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    if actual_trades.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    trades = normalize_actual_trades(actual_trades)
    trades = trades[trades["side"].isin(["buy", "sell"])].copy()
    if trades.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    if as_of_date is None:
        values = trades["trade_date"].dropna()
        as_of_date = str(values.max()) if not values.empty else None
    if as_of_date is not None:
        trades = trades[_date_key_series(trades["trade_date"]) <= _date_key(as_of_date)].copy()
    if trades.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    prices = _latest_prices(daily_bars, as_of_date)
    rows = [_build_position_row(code, group, prices.get(code), as_of_date) for code, group in trades.groupby("code")]
    rows = [row for row in rows if row["holding_volume"] > 0]
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def review_positions(
    positions: pd.DataFrame,
    trade_plan: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(columns=POSITION_REVIEW_COLUMNS)

    normalized = _normalize_positions(positions)
    plan_map = _latest_plan_by_code(trade_plan, as_of_date)
    rows = [_review_position_row(row, plan_map.get(str(row.get("code")))) for _, row in normalized.iterrows()]
    return pd.DataFrame(rows, columns=POSITION_REVIEW_COLUMNS)


def _build_position_row(
    code: str,
    group: pd.DataFrame,
    latest_price: float | None,
    as_of_date: str | None,
) -> dict:
    group = group.sort_values(["trade_date", "trade_time"]).copy()
    holding_volume = 0.0
    cost_amount = 0.0

    for _, trade in group.iterrows():
        side = trade.get("side")
        volume = _number(trade.get("volume"), default=0.0)
        price = _number(trade.get("price"), default=0.0)
        amount = _number(trade.get("amount"), default=price * volume)
        if side == "buy":
            holding_volume += volume
            cost_amount += amount
        elif side == "sell" and volume > 0:
            avg_cost = cost_amount / holding_volume if holding_volume > 0 else 0.0
            reduce_volume = min(volume, holding_volume)
            cost_amount -= avg_cost * reduce_volume
            holding_volume -= volume
            if holding_volume <= 0:
                holding_volume = 0.0
                cost_amount = 0.0

    buy_trades = group[group["side"] == "buy"]
    same_day_buys = buy_trades[_date_key_series(buy_trades["trade_date"]) == _date_key(as_of_date)]
    frozen_volume = min(float(same_day_buys["volume"].sum()), holding_volume) if as_of_date is not None else 0.0
    available_volume = holding_volume - frozen_volume
    cost_price = cost_amount / holding_volume if holding_volume > 0 else None
    market_value = holding_volume * latest_price if latest_price is not None else None
    floating_pnl = market_value - cost_amount if market_value is not None else None
    floating_pnl_pct = latest_price / cost_price - 1 if latest_price is not None and cost_price else None

    latest_trade = group.iloc[-1]
    latest_buy = buy_trades.iloc[0] if not buy_trades.empty else latest_trade
    position_ratio = _number(latest_trade.get("position_ratio"), default=None)
    return {
        "as_of_date": as_of_date,
        "code": code,
        "name": latest_trade.get("name") or latest_buy.get("name"),
        "holding_volume": holding_volume,
        "available_volume": available_volume,
        "frozen_volume": frozen_volume,
        "cost_amount": cost_amount,
        "cost_price": cost_price,
        "latest_price": latest_price,
        "market_value": market_value,
        "floating_pnl": floating_pnl,
        "floating_pnl_pct": floating_pnl_pct,
        "position_ratio": position_ratio,
        "first_buy_date": latest_buy.get("trade_date"),
        "latest_trade_date": latest_trade.get("trade_date"),
        "strategy_name": latest_trade.get("strategy_name") or latest_buy.get("strategy_name"),
        "plan_rank": _number(latest_trade.get("plan_rank"), default=None),
        "t_plus_1_status": "not_sellable_today" if available_volume <= 0 else "sellable",
        "position_status": _position_status(floating_pnl_pct),
    }


def _review_position_row(position: pd.Series, plan: pd.Series | None) -> dict:
    row = {column: position.get(column) for column in POSITION_COLUMNS}
    row.update(
        {
            "planned_stop_loss": _number(plan.get("stop_loss"), default=None) if plan is not None else None,
            "planned_take_profit_1": _number(plan.get("take_profit_1"), default=None) if plan is not None else None,
            "planned_take_profit_2": _number(plan.get("take_profit_2"), default=None) if plan is not None else None,
            "position_risk_level": "low",
            "position_flags": "",
            "position_comment": "",
            "next_action_hint": "继续按计划观察",
        }
    )

    flags: list[str] = []
    comments: list[str] = []
    latest_price = _number(position.get("latest_price"), default=None)
    position_ratio = _number(position.get("position_ratio"), default=None)

    if latest_price is not None and row["planned_stop_loss"] is not None and latest_price <= row["planned_stop_loss"]:
        flags.append("below_stop_loss")
        comments.append("当前价格低于或接近计划止损价")
        row["position_risk_level"] = _max_risk(row["position_risk_level"], "high")

    if latest_price is not None and row["planned_take_profit_1"] is not None and latest_price >= row["planned_take_profit_1"]:
        flags.append("take_profit_zone")
        comments.append("当前价格进入第一止盈观察区间")

    if position.get("t_plus_1_status") == "not_sellable_today":
        flags.append("t_plus_1_locked")
        comments.append("该持仓可能受 T+1 限制，当日不可卖出")

    if position_ratio is not None and position_ratio > 0.25:
        flags.append("high_position_ratio")
        comments.append("单票仓位占比较高")
        row["position_risk_level"] = _max_risk(row["position_risk_level"], "medium")

    row["position_flags"] = ",".join(flags)
    row["position_comment"] = "；".join(comments)
    row["next_action_hint"] = _next_action_hint(flags, position.get("t_plus_1_status"))
    return row


def _latest_prices(daily_bars: pd.DataFrame | None, as_of_date: str | None) -> dict[str, float]:
    if daily_bars is None or daily_bars.empty or "code" not in daily_bars.columns or "close" not in daily_bars.columns:
        return {}
    bars = daily_bars.copy()
    if "trade_date" not in bars.columns:
        return {}
    if as_of_date is not None:
        bars = bars[_date_key_series(bars["trade_date"]) <= _date_key(as_of_date)].copy()
    if bars.empty:
        return {}
    bars["_date_key"] = _date_key_series(bars["trade_date"])
    latest = bars.sort_values(["code", "_date_key"]).drop_duplicates(subset=["code"], keep="last")
    return {
        str(row["code"]): float(row["close"])
        for _, row in latest.iterrows()
        if pd.notna(row.get("close"))
    }


def _latest_plan_by_code(trade_plan: pd.DataFrame | None, as_of_date: str | None) -> dict[str, pd.Series]:
    if trade_plan is None or trade_plan.empty or "code" not in trade_plan.columns:
        return {}
    plan = trade_plan.copy()
    for column in ["trade_date", "stop_loss", "take_profit_1", "take_profit_2"]:
        if column not in plan.columns:
            plan[column] = None
    if as_of_date is not None:
        plan = plan[_date_key_series(plan["trade_date"]) <= _date_key(as_of_date)].copy()
    if plan.empty:
        return {}
    plan["_date_key"] = _date_key_series(plan["trade_date"])
    latest = plan.sort_values(["code", "_date_key"]).drop_duplicates(subset=["code"], keep="last")
    return {str(row["code"]): row for _, row in latest.iterrows()}


def _normalize_positions(positions: pd.DataFrame) -> pd.DataFrame:
    normalized = positions.copy()
    for column in POSITION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized.loc[:, POSITION_COLUMNS]


def _position_status(floating_pnl_pct: float | None) -> str:
    if floating_pnl_pct is None or pd.isna(floating_pnl_pct):
        return "unknown"
    if floating_pnl_pct <= -0.05:
        return "loss_warning"
    if floating_pnl_pct >= 0.05:
        return "profit_watch"
    return "normal"


def _next_action_hint(flags: list[str], t_plus_1_status: object) -> str:
    if "below_stop_loss" in flags and t_plus_1_status == "not_sellable_today":
        return "受 T+1 限制，需次日优先处理风险"
    if "below_stop_loss" in flags:
        return "若盘中无法收回止损位，应按计划处理"
    if "take_profit_zone" in flags:
        return "可关注止盈或移动止盈条件"
    return "继续按计划观察"


def _max_risk(current: str, incoming: str) -> str:
    priority = {"low": 0, "medium": 1, "high": 2}
    return incoming if priority[incoming] > priority[current] else current


def _date_key(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    digits = re.sub(r"\D", "", str(value))
    return int(digits) if digits else 0


def _date_key_series(series: pd.Series) -> pd.Series:
    return series.map(_date_key)


def _number(value: object, default: float | None = None) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return float(number)
