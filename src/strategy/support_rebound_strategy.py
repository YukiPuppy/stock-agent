from __future__ import annotations

import re

import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
from src.strategy.strategy_config import get_strategy_config


class SupportReboundStrategy(BaseStrategy):
    name = "support_rebound"

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
            (factors["pct_chg_1d"] < self.config["max_pct_chg_1d"])
            & (factors["pct_chg_1d"] > self.config["min_pct_chg_1d"])
            & _required_bool(factors["above_ma20"], self.config["require_above_ma20"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["close_position_20"] <= self.config["max_close_position_20"])
            # amount_ma5 is in thousand yuan.
            & (factors["amount_ma5"] > self.config["min_amount_ma5"])
        )
        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["pct_chg_1d"].abs() * 80
            + signals["close_position_20"] * 10
            + signals["volume_ratio_5"].clip(upper=2) * 5
        )
        signals["entry_reason"] = "短线回落但仍在20日均线上方，适合作为支撑观察标的。"
        signals["risk_flags"] = signals.apply(_risk_flags, axis=1)
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def _prepare_factors(daily_factors: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    factors = daily_factors.copy()
    selected_trade_date = _normalize_trade_date(trade_date) if trade_date is not None else None
    factors["_trade_date_key"] = factors["trade_date"].map(_normalize_trade_date)
    if selected_trade_date is None:
        trade_dates = factors["_trade_date_key"].dropna()
        if trade_dates.empty:
            return pd.DataFrame()
        selected_trade_date = str(trade_dates.max())

    factors = factors[factors["_trade_date_key"] == selected_trade_date].copy()
    factors = factors.drop(columns=["_trade_date_key"], errors="ignore")
    for column in ["pct_chg_1d", "pct_chg_3d", "close_position_20", "amount_ma5", "volume_ratio_5"]:
        if column not in factors.columns:
            factors[column] = pd.NA
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    if "above_ma20" not in factors.columns:
        factors["above_ma20"] = False
    factors["above_ma20"] = factors["above_ma20"].fillna(False).astype(bool)
    return factors


def _risk_flags(row: pd.Series) -> str:
    flags = []
    if row["pct_chg_3d"] < -0.06:
        flags.append("short_term_weakness")
    if row["volume_ratio_5"] > 2.5:
        flags.append("panic_volume_possible")
    return ",".join(flags)


def _required_bool(values: pd.Series, required: bool) -> pd.Series:
    return values if required else pd.Series(True, index=values.index)


def _normalize_trade_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return digits
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y%m%d")
