from __future__ import annotations

import pandas as pd

from src.trading.actual_trades import normalize_actual_trades


ACTUAL_TRADE_PERFORMANCE_COLUMNS = [
    "trade_date",
    "trade_time",
    "code",
    "name",
    "side",
    "entry_price",
    "entry_volume",
    "entry_amount",
    "position_ratio",
    "strategy_name",
    "plan_rank",
    "plan_match_status",
    "execution_status",
    "execution_flags",
    "return_1d",
    "return_3d",
    "return_5d",
    "max_drawdown_1d",
    "max_drawdown_3d",
    "max_drawdown_5d",
    "max_favorable_1d",
    "max_favorable_3d",
    "max_favorable_5d",
    "is_valid",
    "invalid_reason",
    "performance_comment",
]

DEFAULT_HOLDING_DAYS = [1, 3, 5]
SUPPORTED_HOLDING_DAYS = [1, 3, 5]


def calculate_actual_trade_performance(
    actual_trades: pd.DataFrame,
    daily_bars: pd.DataFrame,
    execution_review: pd.DataFrame | None = None,
    holding_days: list[int] | None = None,
) -> pd.DataFrame:
    """Calculate deterministic post-entry market performance for actual trades."""
    if actual_trades is None or actual_trades.empty:
        return pd.DataFrame(columns=ACTUAL_TRADE_PERFORMANCE_COLUMNS)

    periods = holding_days if holding_days is not None else DEFAULT_HOLDING_DAYS
    actual = normalize_actual_trades(actual_trades)
    bars_by_code = _prepare_bars_by_code(daily_bars)
    execution_map = _prepare_execution_map(execution_review)

    rows = [
        _calculate_row(row, bars_by_code, execution_map, periods)
        for _, row in actual.iterrows()
    ]
    return pd.DataFrame(rows, columns=ACTUAL_TRADE_PERFORMANCE_COLUMNS)


def _prepare_bars_by_code(daily_bars: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if daily_bars is None or daily_bars.empty:
        return {}
    bars = daily_bars.copy()
    for column in ["trade_date", "code", "open", "high", "low", "close", "volume", "amount"]:
        if column not in bars.columns:
            bars[column] = None
    bars["trade_date"] = bars["trade_date"].astype(str)
    bars["code"] = bars["code"].map(lambda value: normalize_actual_trades(pd.DataFrame({"code": [value]})).loc[0, "code"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return {
        str(code): group.sort_values("trade_date").reset_index(drop=True)
        for code, group in bars.groupby("code", dropna=False)
    }


def _prepare_execution_map(execution_review: pd.DataFrame | None) -> dict[tuple[str, str, str, str], pd.Series]:
    if execution_review is None or execution_review.empty:
        return {}
    execution = execution_review.copy()
    for column in [
        "trade_date",
        "trade_time",
        "code",
        "side",
        "plan_match_status",
        "execution_status",
        "execution_flags",
    ]:
        if column not in execution.columns:
            execution[column] = ""
    execution["code"] = execution["code"].map(lambda value: normalize_actual_trades(pd.DataFrame({"code": [value]})).loc[0, "code"])
    result: dict[tuple[str, str, str, str], pd.Series] = {}
    for _, row in execution.iterrows():
        result[
            (
                str(row.get("trade_date", "")),
                str(row.get("trade_time", "")),
                str(row.get("code", "")),
                str(row.get("side", "")),
            )
        ] = row
    return result


def _calculate_row(
    trade: pd.Series,
    bars_by_code: dict[str, pd.DataFrame],
    execution_map: dict[tuple[str, str, str, str], pd.Series],
    holding_days: list[int],
) -> dict:
    key = (
        str(trade.get("trade_date", "")),
        str(trade.get("trade_time", "")),
        str(trade.get("code", "")),
        str(trade.get("side", "")),
    )
    execution = execution_map.get(key, pd.Series(dtype=object))
    row = _base_row(trade, execution)

    if row["side"] != "buy":
        row["is_valid"] = False
        row["invalid_reason"] = "sell_trade_not_evaluated"
        row["performance_comment"] = "卖出记录当前不计算买入后表现。"
        return row

    entry_price = row["entry_price"]
    if pd.isna(entry_price) or float(entry_price) <= 0:
        row["is_valid"] = False
        row["invalid_reason"] = "invalid_entry_price"
        row["performance_comment"] = _build_performance_comment(row)
        return row

    code_bars = bars_by_code.get(str(row["code"]))
    if code_bars is None or code_bars.empty:
        row["is_valid"] = False
        row["invalid_reason"] = "no_daily_bars_found"
        row["performance_comment"] = _build_performance_comment(row)
        return row

    trade_date = str(row["trade_date"])
    holding_bars = code_bars[code_bars["trade_date"].astype(str) >= trade_date].reset_index(drop=True)
    if holding_bars.empty:
        row["is_valid"] = False
        row["invalid_reason"] = "insufficient_daily_bars:" + ",".join(f"{days}d" for days in holding_days)
        row["performance_comment"] = _build_performance_comment(row)
        return row

    insufficient_periods: list[str] = []
    for days in SUPPORTED_HOLDING_DAYS:
        if days not in holding_days:
            continue
        if len(holding_bars) < days:
            insufficient_periods.append(f"{days}d")
            continue
        window = holding_bars.iloc[:days]
        exit_close = window.iloc[-1]["close"]
        low = window["low"].min()
        high = window["high"].max()
        row[f"return_{days}d"] = _ratio(exit_close, entry_price)
        row[f"max_drawdown_{days}d"] = _ratio(low, entry_price)
        row[f"max_favorable_{days}d"] = _ratio(high, entry_price)

    if insufficient_periods:
        row["is_valid"] = False
        row["invalid_reason"] = "insufficient_daily_bars:" + ",".join(insufficient_periods)
    else:
        row["is_valid"] = True
        row["invalid_reason"] = ""
    row["performance_comment"] = _build_performance_comment(row)
    return row


def _base_row(trade: pd.Series, execution: pd.Series) -> dict:
    row = {column: None for column in ACTUAL_TRADE_PERFORMANCE_COLUMNS}
    row.update(
        {
            "trade_date": trade.get("trade_date"),
            "trade_time": trade.get("trade_time"),
            "code": trade.get("code"),
            "name": trade.get("name"),
            "side": trade.get("side"),
            "entry_price": trade.get("price"),
            "entry_volume": trade.get("volume"),
            "entry_amount": trade.get("amount"),
            "position_ratio": trade.get("position_ratio"),
            "strategy_name": trade.get("strategy_name"),
            "plan_rank": trade.get("plan_rank"),
            "plan_match_status": execution.get("plan_match_status", ""),
            "execution_status": execution.get("execution_status", ""),
            "execution_flags": execution.get("execution_flags", ""),
            "is_valid": False,
            "invalid_reason": "",
            "performance_comment": "",
        }
    )
    return row


def _ratio(value: object, entry_price: object) -> float | None:
    if pd.isna(value) or pd.isna(entry_price) or float(entry_price) == 0:
        return None
    return float(value) / float(entry_price) - 1


def _build_performance_comment(row: dict) -> str:
    comments: list[str] = []
    return_3d = row.get("return_3d")
    drawdown_3d = row.get("max_drawdown_3d")
    if pd.notna(return_3d) and pd.notna(drawdown_3d):
        if float(return_3d) > 0 and float(drawdown_3d) > -0.03:
            comments.append("短期表现较稳")
        if float(drawdown_3d) < -0.05:
            comments.append("买入后回撤较大")
    flags = str(row.get("execution_flags") or "")
    if "chase_above_entry" in flags:
        comments.append("该交易存在追高偏差，需与后续表现分开复盘")
    if str(row.get("execution_status") or "") == "off_plan":
        comments.append("该交易为计划外交易，应单独统计")
    return "；".join(comments) if comments else "暂无明确表现结论"
