from __future__ import annotations

import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
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
        signals["risk_flags"] = signals.apply(_risk_flags, axis=1)
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def _prepare_factors(daily_factors: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    factors = daily_factors.copy()
    selected_trade_date = trade_date
    if selected_trade_date is None:
        trade_dates = factors["trade_date"].dropna()
        if trade_dates.empty:
            return pd.DataFrame()
        selected_trade_date = str(trade_dates.max())

    factors = factors[factors["trade_date"] == selected_trade_date].copy()
    for column in ["pct_chg_5d", "pct_chg_1d", "close_position_20", "volume_ratio_5"]:
        if column not in factors.columns:
            factors[column] = pd.NA
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    for column in ["above_ma5", "above_ma10"]:
        if column not in factors.columns:
            factors[column] = False
        factors[column] = factors[column].fillna(False).astype(bool)
    return factors


def _risk_flags(row: pd.Series) -> str:
    flags = []
    if row["close_position_20"] > 0.9:
        flags.append("near_20d_high")
    if row["pct_chg_1d"] > 0.05:
        flags.append("short_term_chase_risk")
    return ",".join(flags)


def _required_bool(values: pd.Series, required: bool) -> pd.Series:
    return values if required else pd.Series(True, index=values.index)
