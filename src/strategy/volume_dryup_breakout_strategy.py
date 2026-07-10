from __future__ import annotations

import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
from src.strategy.strategy_config import get_strategy_config
from src.strategy.strategy_utils import (
    append_flags,
    apply_market_regime_gating,
    apply_tradable_universe_filter,
    prepare_factors,
    required_bool,
)


class VolumeDryupBreakoutStrategy(BaseStrategy):
    name = "volume_dryup_breakout"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1_dryup_recover"))
        self.enabled = bool(self.config.get("enabled", True))

    def generate_signals(
        self,
        daily_factors: pd.DataFrame,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not self.enabled or daily_factors.empty or "trade_date" not in daily_factors.columns:
            return empty_signals()

        factors = prepare_factors(
            daily_factors,
            trade_date,
            numeric_columns=[
                "volume_ratio_5",
                "volume_ratio_daily_basic",
                "amount_ma5",
                "pct_chg_1d",
                "pct_chg_3d",
                "close_position_20",
            ],
            bool_columns=["above_ma5", "above_ma10"],
            text_columns=["market_regime", "risk_level"],
        )
        factors = apply_tradable_universe_filter(factors, self.config)
        if factors.empty:
            return empty_signals()

        daily_volume_ratio = factors["volume_ratio_daily_basic"].fillna(factors["volume_ratio_5"])
        mask = (
            required_bool(factors["above_ma5"], self.config["require_above_ma5"])
            & required_bool(factors["above_ma10"], self.config["require_above_ma10"])
            & (factors["volume_ratio_5"] >= self.config["min_volume_ratio_5"])
            & (factors["volume_ratio_5"] <= self.config["max_volume_ratio_5"])
            & (daily_volume_ratio >= self.config["min_volume_ratio_daily_basic"])
            & (daily_volume_ratio <= self.config["max_volume_ratio_daily_basic"])
            & (factors["pct_chg_1d"] >= self.config["min_pct_chg_1d"])
            & (factors["pct_chg_1d"] <= self.config["max_pct_chg_1d"])
            & (factors["pct_chg_3d"] <= self.config["max_pct_chg_3d"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
        )
        if self.version == "v2_breakout_confirm":
            mask &= factors["close_position_20"] >= self.config["confirm_min_close_position_20"]

        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["pct_chg_1d"].fillna(0) * 150
            + signals["close_position_20"].fillna(0) * 20
            + signals["volume_ratio_5"].fillna(0).clip(upper=2.5) * 8
            + signals[["above_ma5", "above_ma10"]].astype(int).sum(axis=1) * 4
        )
        signals["entry_reason"] = "缩量整理后温和放量并向上突破，适合作为突破确认观察标的。"
        signals["risk_flags"] = append_flags(
            pd.Series("", index=signals.index),
            [
                (signals["pct_chg_1d"] > 0.06, "short_term_chase_risk"),
                (signals["volume_ratio_5"] > 2.0, "volume_spike"),
                (signals["close_position_20"] > 0.92, "near_20d_high"),
            ],
        )
        signals = apply_market_regime_gating(signals, strategy_group="breakout")
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
