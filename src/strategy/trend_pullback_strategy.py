from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series, normalize_trade_date_value
from src.strategy.strategy_config import get_strategy_config


class TrendPullbackStrategy(BaseStrategy):
    name = "trend_pullback"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1"))
        self.enabled = bool(self.config.get("enabled", True))

    def generate_signals(
        self,
        daily_factors: pd.DataFrame,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not self.enabled:
            return empty_signals()
        if daily_factors.empty or "trade_date" not in daily_factors.columns:
            return empty_signals()

        factors = _prepare_factors(daily_factors, trade_date)
        if factors.empty:
            return empty_signals()

        mask = (
            _required_bool(factors["above_ma5"], self.config["require_above_ma5"])
            & _required_bool(factors["above_ma10"], self.config["require_above_ma10"])
            & (factors["pct_chg_5d"] > self.config["min_pct_chg_5d"])
            & (factors["pct_chg_1d"] < self.config["max_pct_chg_1d"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["volume_ratio_5"] >= self.config["min_volume_ratio_5"])
        )
        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["pct_chg_5d"] * 100
            + signals["close_position_20"] * 20
            + signals["volume_ratio_5"].clip(upper=3) * 5
        )
        signals["entry_reason"] = "趋势保持较强，站上5日和10日均线，适合关注回踩低吸。"
        signals["risk_flags"] = _risk_flags(signals)
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def _prepare_factors(daily_factors: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
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
    for column in ["pct_chg_5d", "pct_chg_1d", "close_position_20", "volume_ratio_5"]:
        if column not in factors.columns:
            factors[column] = pd.NA
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    for column in ["above_ma5", "above_ma10"]:
        if column not in factors.columns:
            factors[column] = False
        factors[column] = factors[column].fillna(False).astype(bool)
    return factors


def _risk_flags(signals: pd.DataFrame) -> pd.Series:
    near_high = signals["close_position_20"] > 0.9
    chase_risk = signals["pct_chg_1d"] > 0.05
    return pd.Series(
        np.select(
            [near_high & chase_risk, near_high, chase_risk],
            ["near_20d_high,short_term_chase_risk", "near_20d_high", "short_term_chase_risk"],
            default="",
        ),
        index=signals.index,
    )


def _required_bool(values: pd.Series, required: bool) -> pd.Series:
    return values if required else pd.Series(True, index=values.index)


def _normalize_trade_date(value: object) -> str:
    return normalize_trade_date_value(value)
