from __future__ import annotations

import pandas as pd


TRADE_PLAN_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "rank",
    "strategy_names",
    "strategy_versions",
    "active_signal_count",
    "avg_strategy_weight",
    "recommendations",
    "risk_flags",
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
        "strategy_names": row.get("strategy_names"),
        "strategy_versions": row.get("strategy_versions"),
        "active_signal_count": _to_int_or_none(row.get("active_signal_count")),
        "avg_strategy_weight": _round_number(row.get("avg_strategy_weight")),
        "recommendations": row.get("recommendations"),
        "risk_flags": row.get("risk_flags"),
        "close": _round_price(row["close"]),
        "strategy_type": strategy_type,
        "action": action,
        **price_fields,
        "invalid_condition": INVALID_CONDITIONS[strategy_type],
        "t_plus_1_risk": T_PLUS_1_RISK,
        "plan_reason": _plan_reason(row),
    }


def _classify_strategy(row: pd.Series) -> tuple[str, str]:
    risk_flags = _risk_flag_set(row.get("risk_flags"))
    if "suspended" in risk_flags:
        return "watch_only", "仅观察"
    if "limit_down_close" in risk_flags:
        return "watch_only", "仅观察"
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
    if _is_missing(value):
        return None
    return round(float(value), 2)


def _round_number(value: object) -> float | None:
    if _is_missing(value):
        return None
    return round(float(value), 4)


def _to_int_or_none(value: object) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _plan_reason(row: pd.Series) -> str:
    reason = row.get("reason") or "候选池入选"
    score = _format_number(row.get("score"))
    pct_chg_5d = _format_percent(row.get("pct_chg_5d"))
    volume_ratio_5 = _format_number(row.get("volume_ratio_5"))
    close_position_20 = _format_number(row.get("close_position_20"))
    base_reason = (
        f"{reason}；score={score}，5日涨跌幅={pct_chg_5d}，"
        f"5日量比={volume_ratio_5}，20日位置={close_position_20}。"
    )
    strategy_reason = _strategy_evaluation_reason(row)
    risk_reason = _risk_reason(row)
    return f"{base_reason}{strategy_reason}{risk_reason}"


def _risk_reason(row: pd.Series) -> str:
    flags = _risk_flag_set(row.get("risk_flags"))
    parts = []
    if "suspended" in flags:
        parts.append("停牌或停牌风险，暂不生成买入计划")
    if "limit_up_close" in flags:
        parts.append("涨停收盘，次日可能存在买入不可执行风险")
    if "limit_down_close" in flags:
        parts.append("跌停收盘，短期流动性和风险较高")
    if "market_high_risk" in flags:
        parts.append("当前市场环境偏弱，计划置信度需降低")
    if "missing_daily_basic" in flags or "missing_market_value" in flags:
        parts.append("部分 daily_basic 扩展指标缺失，需降低置信度")
    if "strong_main_outflow" in flags:
        parts.append("主力资金明显流出，计划置信度降低")
    elif "main_outflow" in flags:
        parts.append("资金流偏弱，需观察承接")
    if "strong_main_inflow" in flags:
        parts.append("资金流相对积极，但仍需结合价格和计划区间执行")
    if "weak_industry" in flags:
        parts.append("所属行业相对弱势，计划置信度降低")
    if "strong_industry" in flags:
        parts.append("所属行业相对强势，存在板块共振加分，但仍需按计划执行")
    if "missing_industry_strength" in flags:
        parts.append("行业强度数据缺失，需降低判断置信度")
    if not parts:
        return ""
    return "；".join(parts) + "。"


def _risk_flag_set(value: object) -> set[str]:
    if _is_missing(value):
        return set()
    return {flag.strip() for flag in str(value).split(",") if flag.strip()}


def _strategy_evaluation_reason(row: pd.Series) -> str:
    parts = []
    strategy_names = _format_optional_text(row.get("strategy_names"))
    strategy_versions = _format_optional_text(row.get("strategy_versions"))
    recommendations = _format_optional_text(row.get("recommendations"))
    avg_strategy_weight = row.get("avg_strategy_weight")

    if strategy_names or strategy_versions:
        source = strategy_names
        if strategy_names and strategy_versions:
            source = f"{strategy_names}:{strategy_versions}"
        elif strategy_versions:
            source = strategy_versions
        parts.append(f"策略来源：{source}")
    if recommendations:
        parts.append(f"策略建议：{recommendations}")
    if not _is_missing(avg_strategy_weight):
        parts.append(f"平均策略权重：{float(avg_strategy_weight):.2f}")

    if not parts:
        return ""
    return "；".join(parts) + "。"


def _format_optional_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return bool(pd.isna(value))


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
