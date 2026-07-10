from __future__ import annotations

import numpy as np
import pandas as pd

from src.filters.universe_filter import filter_tradable_main_board
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series, normalize_trade_date_value


COMMON_FILTER_NUMERIC_COLUMNS = [
    "close",
    "pct_chg_1d",
    "pct_chg_3d",
    "pct_chg_5d",
    "pct_chg_10d",
    "amount_ma5",
    "turnover_rate",
]


def prepare_factors(
    daily_factors: pd.DataFrame,
    trade_date: str | None,
    *,
    numeric_columns: list[str],
    bool_columns: list[str] | None = None,
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_trade_date = normalize_trade_date_value(trade_date) if trade_date is not None else None
    trade_date_keys = (
        daily_factors[TRADE_DATE_KEY_COLUMN]
        if TRADE_DATE_KEY_COLUMN in daily_factors.columns
        else normalize_trade_date_series(daily_factors["trade_date"])
    )
    if selected_trade_date is None:
        trade_dates = trade_date_keys[trade_date_keys.ne("")]
        if trade_dates.empty:
            return pd.DataFrame()
        selected_trade_date = str(trade_dates.max())

    factors = daily_factors.loc[trade_date_keys == selected_trade_date].copy()
    factors = factors.drop(columns=[TRADE_DATE_KEY_COLUMN], errors="ignore")
    for column in dict.fromkeys(COMMON_FILTER_NUMERIC_COLUMNS + numeric_columns):
        if column not in factors.columns:
            factors[column] = pd.NA
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    for column in bool_columns or []:
        if column not in factors.columns:
            factors[column] = False
        factors[column] = to_bool_series(factors[column])
    for column in ["code", "name", "market", "board"] + (text_columns or []):
        if column not in factors.columns:
            factors[column] = None
    return factors


def apply_tradable_universe_filter(
    factors: pd.DataFrame,
    config: dict,
    *,
    exclude_limit_up_close: bool = True,
    exclude_limit_down_close: bool = True,
    exclude_continuous_downtrend: bool = False,
) -> pd.DataFrame:
    if factors.empty or "code" not in factors.columns:
        return pd.DataFrame(columns=factors.columns)

    filtered = filter_tradable_main_board(factors).copy()
    if filtered.empty:
        return filtered

    for column in ["is_suspended", "is_limit_up_close", "is_limit_down_close", "paused"]:
        if column in filtered.columns:
            filtered[column] = to_bool_series(filtered[column])

    min_amount_ma5 = float(config.get("min_amount_ma5", 10000.0))
    min_turnover_rate = config.get("min_turnover_rate")

    mask = filtered["amount_ma5"].fillna(-np.inf) >= min_amount_ma5
    if "close" in filtered.columns:
        mask &= filtered["close"].fillna(0) > 0
    if "is_suspended" in filtered.columns:
        mask &= ~filtered["is_suspended"]
    if "paused" in filtered.columns:
        mask &= ~filtered["paused"]
    if exclude_limit_up_close and "is_limit_up_close" in filtered.columns:
        mask &= ~filtered["is_limit_up_close"]
    if exclude_limit_down_close and "is_limit_down_close" in filtered.columns:
        mask &= ~filtered["is_limit_down_close"]
    if "pct_chg_1d" in filtered.columns:
        if exclude_limit_up_close:
            mask &= filtered["pct_chg_1d"].fillna(0) < 0.095
        if exclude_limit_down_close:
            mask &= filtered["pct_chg_1d"].fillna(0) > -0.095
    if min_turnover_rate is not None and "turnover_rate" in filtered.columns:
        turnover = filtered["turnover_rate"]
        mask &= turnover.isna() | (turnover >= float(min_turnover_rate))
    if exclude_continuous_downtrend:
        continuous_drop = (
            (filtered["pct_chg_3d"].fillna(0) <= float(config.get("max_down_pct_chg_3d", -0.10)))
            & (filtered["pct_chg_5d"].fillna(0) <= float(config.get("max_down_pct_chg_5d", -0.16)))
        ) | (filtered["pct_chg_10d"].fillna(0) <= float(config.get("max_down_pct_chg_10d", -0.25)))
        mask &= ~continuous_drop
    return filtered.loc[mask].copy()


def append_flags(base: pd.Series, flags: list[tuple[pd.Series, str]]) -> pd.Series:
    result = base.fillna("").astype(str).str.strip().copy()
    for mask, flag in flags:
        aligned = mask.reindex(result.index, fill_value=False).fillna(False).astype(bool)
        if not aligned.any():
            continue
        existing = result.loc[aligned]
        needs_flag = ~("," + existing.str.replace(" ", "", regex=False) + ",").str.contains(
            f",{flag},", regex=False, na=False
        )
        target_index = existing.loc[needs_flag].index
        if len(target_index) == 0:
            continue
        current = result.loc[target_index]
        result.loc[target_index] = np.where(current.eq(""), flag, current + "," + flag)
    return result


def apply_market_regime_gating(
    signals: pd.DataFrame,
    *,
    strategy_group: str,
) -> pd.DataFrame:
    if signals.empty:
        return signals
    if "market_regime" not in signals.columns and "risk_level" not in signals.columns:
        return signals

    result = signals.copy()
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""
    regime = result.get("market_regime", pd.Series("", index=result.index)).fillna("").astype(str).str.lower()
    risk_level = result.get("risk_level", pd.Series("", index=result.index)).fillna("").astype(str).str.lower()
    weak = regime.eq("weak") | risk_level.eq("high")
    strong = regime.eq("strong") | risk_level.eq("low")

    aggressive = strategy_group in {"breakout", "rotation", "moneyflow"}
    if aggressive:
        result.loc[weak, "signal_strength"] = result.loc[weak, "signal_strength"] * 0.75
        result.loc[strong, "signal_strength"] = result.loc[strong, "signal_strength"] * 1.05
        result["risk_flags"] = append_flags(result["risk_flags"], [(weak, "weak_market_regime")])
    else:
        result.loc[weak, "signal_strength"] = result.loc[weak, "signal_strength"] * 0.90
        result["risk_flags"] = append_flags(result["risk_flags"], [(weak, "weak_market_regime_conservative")])
    return result


def required_bool(values: pd.Series, required: bool) -> pd.Series:
    return values if required else pd.Series(True, index=values.index)


def to_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    text = values.fillna("").astype(str).str.strip().str.lower()
    false_values = {"", "0", "false", "f", "no", "n", "none", "nan"}
    true_values = {"1", "true", "t", "yes", "y"}
    default_bool = values.fillna(False).astype(bool)
    return pd.Series(
        np.select(
            [text.isin(false_values), text.isin(true_values)],
            [False, True],
            default=default_bool,
        ),
        index=values.index,
    ).astype(bool)
