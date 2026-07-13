from __future__ import annotations

import json
import re

import pandas as pd


DEFAULT_MAX_HOLDING_DAYS = 5
STRATEGY_HOLDING_DAY_GRIDS = {
    "oversold_rebound": [3, 5, 8],
    "support_rebound": [3, 5, 8, 10],
    "volume_dryup_breakout": [3, 5, 8, 10, 15],
    "breakout_volume": [3, 5, 8, 10, 15],
    "trend_pullback": [5, 8, 10, 15, 20],
    "relative_strength_pullback": [5, 8, 10, 15, 20],
    "moneyflow_accumulation": [5, 10, 15, 20],
    "low_vol_trend": [10, 15, 20, 30],
    "industry_rotation": [10, 15, 20, 30],
}


BACKTEST_RESULT_COLUMNS = [
    "plan_date",
    "code",
    "name",
    "action",
    "strategy_names",
    "strategy_versions",
    "recommendations",
    "avg_strategy_weight",
    "entry_low",
    "entry_high",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "exit_reason",
    "holding_days",
    "max_holding_days",
    "return_pct",
    "max_drawdown",
    "max_favorable",
    "is_triggered",
    "is_valid",
    "invalid_reason",
    "backtest_comment",
]

PERFORMANCE_COLUMNS = [
    "strategy_names",
    "strategy_versions",
    "action",
    "max_holding_days",
    "plan_count",
    "triggered_count",
    "valid_count",
    "trigger_rate",
    "win_rate",
    "avg_return",
    "median_return",
    "avg_max_drawdown",
    "avg_max_favorable",
    "stop_loss_rate",
    "take_profit_rate",
    "time_exit_rate",
]


def backtest_trade_plans(
    trade_plans: pd.DataFrame,
    daily_bars: pd.DataFrame,
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
) -> pd.DataFrame:
    """Backtest plan-level entry, stop-loss, take-profit, and time-exit rules."""
    if trade_plans.empty:
        return pd.DataFrame(columns=BACKTEST_RESULT_COLUMNS)

    bars_by_code = _prepare_bars_by_code(daily_bars)
    plans = trade_plans.copy()
    plans["_plan_date_key"] = _date_key(plans["trade_date"])
    plans = plans.sort_values(["_plan_date_key", "code"]).reset_index(drop=True)

    rows = []
    for _, plan in plans.iterrows():
        plan_max_holding_days = _positive_holding_days(plan.get("max_holding_days"), max_holding_days)
        row = _base_row(plan)
        row["max_holding_days"] = plan_max_holding_days
        if _is_watch_only_or_no_entry(plan):
            row.update(
                is_triggered=False,
                is_valid=False,
                invalid_reason="watch_only_or_no_entry_range",
                backtest_comment="计划为仅观察或缺少买入区间，未模拟交易。",
            )
            rows.append(row)
            continue

        bars = bars_by_code.get(str(plan.get("code")))
        if bars is None or bars.empty:
            row.update(is_valid=False, invalid_reason="no_next_trading_day")
            rows.append(row)
            continue

        entry_candidates = bars[bars["_trade_date_key"] > plan["_plan_date_key"]]
        if entry_candidates.empty:
            row.update(is_valid=False, invalid_reason="no_next_trading_day")
            rows.append(row)
            continue

        entry_pos = int(entry_candidates.index[0])
        entry_bar = bars.loc[entry_pos]
        trigger = _entry_trigger(plan, entry_bar)
        row["entry_date"] = entry_bar["trade_date"]
        if not trigger["is_triggered"]:
            row.update(
                is_triggered=False,
                is_valid=False,
                invalid_reason=trigger["invalid_reason"],
                backtest_comment=trigger["comment"],
            )
            rows.append(row)
            continue

        row["is_triggered"] = True
        row["entry_price"] = trigger["entry_price"]
        holding_window = bars.loc[entry_pos : entry_pos + plan_max_holding_days - 1].copy()
        if holding_window.empty:
            row.update(is_valid=False, invalid_reason="insufficient_future_bars")
            rows.append(row)
            continue

        exit_data = _simulate_exit(plan, holding_window)
        row.update(exit_data)
        row["holding_days"] = int(len(holding_window.loc[: exit_data["_exit_index"]]))
        row["return_pct"] = row["exit_price"] / row["entry_price"] - 1
        held = holding_window.loc[: exit_data["_exit_index"]]
        row["max_drawdown"] = held["low"].min() / row["entry_price"] - 1
        row["max_favorable"] = held["high"].max() / row["entry_price"] - 1
        row["is_valid"] = True
        row["invalid_reason"] = ""
        row["backtest_comment"] = f"按规则触发买入并以 {row['exit_reason']} 退出。"
        row.pop("_exit_index", None)
        rows.append(row)

    return pd.DataFrame(rows, columns=BACKTEST_RESULT_COLUMNS)


def evaluate_trade_plan_backtest(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize trade-plan backtest results by strategy source, version, and action."""
    if backtest_results.empty:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    results = backtest_results.copy()
    for column in ["strategy_names", "strategy_versions", "action"]:
        if column not in results.columns:
            results[column] = ""
        results[column] = results[column].fillna("").astype(str)
    if "max_holding_days" not in results.columns:
        results["max_holding_days"] = DEFAULT_MAX_HOLDING_DAYS
    results["max_holding_days"] = pd.to_numeric(results["max_holding_days"], errors="coerce").fillna(
        DEFAULT_MAX_HOLDING_DAYS
    ).astype(int)

    rows = []
    group_columns = ["strategy_names", "strategy_versions", "action", "max_holding_days"]
    for keys, group in results.groupby(group_columns, dropna=False, sort=True):
        strategy_names, strategy_versions, action, max_holding_days = keys
        plan_count = int(len(group))
        triggered = group[group["is_triggered"].fillna(False).astype(bool)]
        valid = group[group["is_valid"].fillna(False).astype(bool)]
        valid_count = int(len(valid))
        returns = _numeric_series(valid, "return_pct")
        drawdowns = _numeric_series(valid, "max_drawdown")
        favorable = _numeric_series(valid, "max_favorable")
        rows.append(
            {
                "strategy_names": strategy_names,
                "strategy_versions": strategy_versions,
                "action": action,
                "max_holding_days": int(max_holding_days),
                "plan_count": plan_count,
                "triggered_count": int(len(triggered)),
                "valid_count": valid_count,
                "trigger_rate": len(triggered) / plan_count if plan_count else 0,
                "win_rate": float((returns > 0).mean()) if not returns.empty else None,
                "avg_return": float(returns.mean()) if not returns.empty else None,
                "median_return": float(returns.median()) if not returns.empty else None,
                "avg_max_drawdown": float(drawdowns.mean()) if not drawdowns.empty else None,
                "avg_max_favorable": float(favorable.mean()) if not favorable.empty else None,
                "stop_loss_rate": _exit_rate(valid, "stop_loss"),
                "take_profit_rate": _exit_rate(valid, ["take_profit_1", "take_profit_2"]),
                "time_exit_rate": _exit_rate(valid, "time_exit"),
            }
        )

    return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)


def _base_row(plan: pd.Series) -> dict[str, object]:
    return {
        "plan_date": plan.get("trade_date"),
        "code": plan.get("code"),
        "name": plan.get("name"),
        "action": plan.get("action"),
        "strategy_names": plan.get("strategy_names"),
        "strategy_versions": plan.get("strategy_versions"),
        "recommendations": plan.get("recommendations"),
        "avg_strategy_weight": plan.get("avg_strategy_weight"),
        "entry_low": plan.get("entry_low"),
        "entry_high": plan.get("entry_high"),
        "stop_loss": plan.get("stop_loss"),
        "take_profit_1": plan.get("take_profit_1"),
        "take_profit_2": plan.get("take_profit_2"),
        "entry_date": None,
        "entry_price": None,
        "exit_date": None,
        "exit_price": None,
        "exit_reason": "",
        "holding_days": 0,
        "max_holding_days": _positive_holding_days(
            plan.get("max_holding_days"), DEFAULT_MAX_HOLDING_DAYS
        ),
        "return_pct": None,
        "max_drawdown": None,
        "max_favorable": None,
        "is_triggered": False,
        "is_valid": False,
        "invalid_reason": "",
        "backtest_comment": "",
    }


def _entry_trigger(plan: pd.Series, bar: pd.Series) -> dict[str, object]:
    entry_low = float(plan["entry_low"])
    entry_high = float(plan["entry_high"])
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])

    if low > entry_high:
        return {"is_triggered": False, "invalid_reason": "gap_above_entry_range", "comment": "买入日低点高于区间上沿，按规则不追。"}
    if high < entry_low:
        return {"is_triggered": False, "invalid_reason": "not_reach_entry_range", "comment": "买入日最高价未触及买入区间。"}
    if not (low <= entry_high and high >= entry_low):
        return {"is_triggered": False, "invalid_reason": "not_reach_entry_range", "comment": "买入日未进入买入区间。"}
    if entry_low <= open_price <= entry_high:
        entry_price = open_price
    elif open_price < entry_low and high >= entry_low:
        entry_price = entry_low
    else:
        entry_price = entry_high
    return {"is_triggered": True, "entry_price": entry_price}


def _simulate_exit(plan: pd.Series, holding_window: pd.DataFrame) -> dict[str, object]:
    stop_loss = _to_float(plan.get("stop_loss"))
    take_profit_1 = _to_float(plan.get("take_profit_1"))
    take_profit_2 = _to_float(plan.get("take_profit_2"))

    for index, bar in holding_window.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        if stop_loss is not None and low <= stop_loss:
            return {"exit_date": bar["trade_date"], "exit_price": stop_loss, "exit_reason": "stop_loss", "_exit_index": index}
        if take_profit_2 is not None and high >= take_profit_2:
            return {"exit_date": bar["trade_date"], "exit_price": take_profit_2, "exit_reason": "take_profit_2", "_exit_index": index}
        if take_profit_1 is not None and high >= take_profit_1:
            return {"exit_date": bar["trade_date"], "exit_price": take_profit_1, "exit_reason": "take_profit_1", "_exit_index": index}

    last_index = holding_window.index[-1]
    last_bar = holding_window.loc[last_index]
    return {"exit_date": last_bar["trade_date"], "exit_price": float(last_bar["close"]), "exit_reason": "time_exit", "_exit_index": last_index}


def _prepare_bars_by_code(daily_bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if daily_bars.empty:
        return {}
    bars = daily_bars.copy()
    bars["_trade_date_key"] = _date_key(bars["trade_date"])
    for column in ["open", "high", "low", "close"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["_trade_date_key", "open", "high", "low", "close"])
    bars = bars.sort_values(["code", "_trade_date_key"]).reset_index(drop=True)
    return {str(code): group.reset_index(drop=True) for code, group in bars.groupby("code", sort=False)}


def _date_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), errors="coerce")


def _is_watch_only_or_no_entry(plan: pd.Series) -> bool:
    return str(plan.get("action") or "") == "仅观察" or _to_float(plan.get("entry_low")) is None or _to_float(plan.get("entry_high")) is None


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _exit_rate(valid: pd.DataFrame, reasons: str | list[str]) -> float | None:
    if valid.empty or "exit_reason" not in valid.columns:
        return None
    values = valid["exit_reason"].fillna("").astype(str)
    if isinstance(reasons, str):
        return float((values == reasons).mean())
    return float(values.isin(reasons).mean())


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def holding_day_grid_for_strategy_names(strategy_names: object) -> list[int]:
    """Return the union of configured secondary-validation holding periods."""
    names = _parse_strategy_names(strategy_names)
    grid = {
        holding_day
        for strategy_name in names
        for holding_day in STRATEGY_HOLDING_DAY_GRIDS.get(strategy_name, [DEFAULT_MAX_HOLDING_DAYS])
    }
    return sorted(grid or {DEFAULT_MAX_HOLDING_DAYS})


def expand_trade_plans_for_holding_days(
    trade_plans: pd.DataFrame,
    mode: str = "strategy_grid",
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
) -> pd.DataFrame:
    """Expand a bounded plan chunk for fixed or per-strategy holding validation."""
    if trade_plans is None or trade_plans.empty:
        empty = trade_plans.copy() if trade_plans is not None else pd.DataFrame()
        empty["max_holding_days"] = pd.Series(dtype="int64")
        return empty
    if mode not in {"fixed", "strategy_grid"}:
        raise ValueError(f"unsupported holding-days mode: {mode}")
    fixed = _positive_holding_days(max_holding_days, DEFAULT_MAX_HOLDING_DAYS)
    records: list[dict] = []
    for _, plan in trade_plans.iterrows():
        strategy_names = plan.get("strategy_names", plan.get("strategy_type"))
        grid = holding_day_grid_for_strategy_names(strategy_names) if mode == "strategy_grid" else [fixed]
        for holding_day in grid:
            record = plan.to_dict()
            record["max_holding_days"] = holding_day
            records.append(record)
    return pd.DataFrame(records)


def _parse_strategy_names(value: object) -> list[str]:
    if value is None:
        return []
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    text = str(value).strip()
    if text.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return _parse_strategy_names(payload)
    return list(dict.fromkeys(part.strip() for part in re.split(r"\s*[,|;+]\s*", text) if part.strip()))


def _positive_holding_days(value: object, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = int(fallback)
    if parsed <= 0:
        raise ValueError("max_holding_days must be greater than zero")
    return parsed
