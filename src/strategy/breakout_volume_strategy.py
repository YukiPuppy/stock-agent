from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series, normalize_trade_date_value
from src.strategy.strategy_config import get_strategy_config


class BreakoutVolumeStrategy(BaseStrategy):
    name = "breakout_volume"

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
            (factors["pct_chg_5d"] > self.config["min_pct_chg_5d"])
            & (factors["volume_ratio_5"] >= self.config["min_volume_ratio_5"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["pct_chg_1d"] < self.config["max_pct_chg_1d"])
            & _required_bool(factors["above_ma5"], self.config["require_above_ma5"])
        )
        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["pct_chg_5d"] * 120
            + signals["volume_ratio_5"].clip(upper=3) * 8
            + signals["close_position_20"] * 20
        )
        signals["entry_reason"] = "短期强度较高且量能放大，适合作为突破观察标的。"
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
    if "above_ma5" not in factors.columns:
        factors["above_ma5"] = False
    factors["above_ma5"] = factors["above_ma5"].fillna(False).astype(bool)
    return factors


def _risk_flags(signals: pd.DataFrame) -> pd.Series:
    chase_risk = signals["pct_chg_1d"] > 0.07
    extended = signals["close_position_20"] > 0.95
    return pd.Series(
        np.select(
            [chase_risk & extended, chase_risk, extended],
            ["near_limit_chase_risk,extended_position", "near_limit_chase_risk", "extended_position"],
            default="",
        ),
        index=signals.index,
    )


def _required_bool(values: pd.Series, required: bool) -> pd.Series:
    return values if required else pd.Series(True, index=values.index)


def _normalize_trade_date(value: object) -> str:
    return normalize_trade_date_value(value)
