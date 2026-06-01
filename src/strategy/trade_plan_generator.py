from __future__ import annotations

import pandas as pd


TRADE_PLAN_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "rank",
    "close",
    "strategy_type",
    "action",
    "entry_low",
    "entry_high",
    "position_low",
    "position_high",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "invalid_condition",
    "t_plus_1_risk",
    "plan_reason",
]

T_PLUS_1_RISK = "A股T+1机制下，今日买入次日才能卖出，若买入后当日回落无法即时止损，需控制仓位。"

INVALID_CONDITIONS = {
    "trend_pullback": "若次日高开超过5%且开盘30分钟放量回落，不追；若跌破买入区间下沿且无法收回，计划失效。",
    "breakout_watch": "只有放量突破时观察，不追缩量高开；若突破后快速回落至前收盘价下方，计划失效。",
    "support_watch": "只有回踩支撑区间并出现承接时观察；若跌破止损价且无法收回，计划失效。",
    "watch_only": "仅观察，不主动买入；等待新的量价确认。",
}


def generate_trade_plan(
    candidate_pool: pd.DataFrame,
    max_items: int = 5,
) -> pd.DataFrame:
    """Generate deterministic next-day trade plans from candidate rows."""
    if candidate_pool.empty or "rank" not in candidate_pool.columns:
        return _empty_trade_plan()

    candidates = candidate_pool.copy()
    candidates["rank"] = pd.to_numeric(candidates["rank"], errors="coerce")
    candidates = candidates[candidates["rank"] <= max_items].copy()
    if candidates.empty:
        return _empty_trade_plan()

    for column in [
        "close",
        "pct_chg_1d",
        "pct_chg_5d",
        "volume_ratio_5",
        "close_position_20",
        "score",
    ]:
        if column not in candidates.columns:
            candidates[column] = pd.NA
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    for column in ["above_ma5", "above_ma10", "above_ma20"]:
        if column not in candidates.columns:
            candidates[column] = False
        candidates[column] = candidates[column].fillna(False).astype(bool)

    rows = [_build_plan_row(row) for _, row in candidates.sort_values(["rank", "code"]).iterrows()]
    return pd.DataFrame(rows, columns=TRADE_PLAN_COLUMNS).reset_index(drop=True)


def _build_plan_row(row: pd.Series) -> dict:
    strategy_type, action = _classify_strategy(row)
    price_fields = _price_fields(strategy_type, row["close"])

    return {
        "trade_date": row.get("trade_date"),
        "code": row.get("code"),
        "name": row.get("name"),
        "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
        "close": _round_price(row["close"]),
        "strategy_type": strategy_type,
        "action": action,
        **price_fields,
        "invalid_condition": INVALID_CONDITIONS[strategy_type],
        "t_plus_1_risk": T_PLUS_1_RISK,
        "plan_reason": _plan_reason(row),
    }


def _classify_strategy(row: pd.Series) -> tuple[str, str]:
    if row["above_ma5"] and row["above_ma10"] and row["close_position_20"] >= 0.6:
        return "trend_pullback", "回踩低吸"
    if row["pct_chg_5d"] > 0.05 and row["volume_ratio_5"] >= 1.2 and row["close_position_20"] >= 0.7:
        return "breakout_watch", "突破观察"
    if row["pct_chg_1d"] < -0.03 and row["above_ma20"]:
        return "support_watch", "支撑观察"
    return "watch_only", "仅观察"


def _price_fields(strategy_type: str, close: float) -> dict:
    if strategy_type == "trend_pullback":
        return {
            "entry_low": _round_price(close * 0.975),
            "entry_high": _round_price(close * 0.995),
            "position_low": 0.10,
            "position_high": 0.20,
            "stop_loss": _round_price(close * 0.95),
            "take_profit_1": _round_price(close * 1.04),
            "take_profit_2": _round_price(close * 1.08),
        }
    if strategy_type == "breakout_watch":
        return {
            "entry_low": _round_price(close * 1.005),
            "entry_high": _round_price(close * 1.025),
            "position_low": 0.05,
            "position_high": 0.15,
            "stop_loss": _round_price(close * 0.97),
            "take_profit_1": _round_price(close * 1.05),
            "take_profit_2": _round_price(close * 1.10),
        }
    if strategy_type == "support_watch":
        return {
            "entry_low": _round_price(close * 0.96),
            "entry_high": _round_price(close * 0.985),
            "position_low": 0.05,
            "position_high": 0.15,
            "stop_loss": _round_price(close * 0.94),
            "take_profit_1": _round_price(close * 1.035),
            "take_profit_2": _round_price(close * 1.07),
        }
    return {
        "entry_low": None,
        "entry_high": None,
        "position_low": 0,
        "position_high": 0,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
    }


def _round_price(value: float) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _plan_reason(row: pd.Series) -> str:
    reason = row.get("reason") or "候选池入选"
    score = _format_number(row.get("score"))
    pct_chg_5d = _format_percent(row.get("pct_chg_5d"))
    volume_ratio_5 = _format_number(row.get("volume_ratio_5"))
    close_position_20 = _format_number(row.get("close_position_20"))
    return (
        f"{reason}；score={score}，5日涨跌幅={pct_chg_5d}，"
        f"5日量比={volume_ratio_5}，20日位置={close_position_20}。"
    )


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def _empty_trade_plan() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_PLAN_COLUMNS)
