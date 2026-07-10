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


class LowVolTrendStrategy(BaseStrategy):
    name = "low_vol_trend"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1_ma_alignment"))
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
                "pct_chg_1d",
                "pct_chg_3d",
                "pct_chg_5d",
                "close_position_20",
                "volume_ratio_5",
                "turnover_rate",
            ],
            bool_columns=["above_ma5", "above_ma10", "above_ma20"],
            text_columns=["market_regime", "risk_level"],
        )
        factors = apply_tradable_universe_filter(factors, self.config)
        if factors.empty:
            return empty_signals()

        mask = (
            required_bool(factors["above_ma5"], self.config["require_above_ma5"])
            & required_bool(factors["above_ma10"], self.config["require_above_ma10"])
            & required_bool(factors["above_ma20"], self.config["require_above_ma20"])
            & (factors["pct_chg_5d"] >= self.config["min_pct_chg_5d"])
            & (factors["pct_chg_5d"] <= self.config["max_pct_chg_5d"])
            & (factors["pct_chg_3d"].abs() <= self.config["max_abs_pct_chg_3d"])
            & (factors["pct_chg_1d"].abs() <= self.config["max_abs_pct_chg_1d"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["close_position_20"] <= self.config["max_close_position_20"])
            & (factors["volume_ratio_5"] >= self.config["min_volume_ratio_5"])
            & (factors["volume_ratio_5"] <= self.config["max_volume_ratio_5"])
        )
        if self.version == "v2_low_chase":
            mask &= factors["pct_chg_1d"] <= self.config["max_pct_chg_1d"]
        elif self.version == "v3_steady_volume":
            mask &= factors["turnover_rate"].isna() | (
                (factors["turnover_rate"] >= self.config["min_turnover_rate"])
                & (factors["turnover_rate"] <= self.config["max_turnover_rate"])
            )

        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["pct_chg_5d"].fillna(0) * 100
            + signals["close_position_20"].fillna(0) * 18
            + (2 - (signals["volume_ratio_5"].fillna(1) - 1).abs()).clip(lower=0) * 6
            + signals[["above_ma5", "above_ma10", "above_ma20"]].astype(int).sum(axis=1) * 3
        )
        signals["entry_reason"] = "均线多头排列，短期波动和量能较稳定，适合作为低波趋势观察标的。"
        signals["risk_flags"] = append_flags(
            pd.Series("", index=signals.index),
            [
                (signals["close_position_20"] > 0.85, "near_20d_high"),
                (signals["volume_ratio_5"] > 1.8, "volume_expansion"),
            ],
        )
        signals = apply_market_regime_gating(signals, strategy_group="trend")
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
