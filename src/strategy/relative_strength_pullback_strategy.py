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


class RelativeStrengthPullbackStrategy(BaseStrategy):
    name = "relative_strength_pullback"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1_rs_pullback"))
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
                "pct_chg_5d",
                "pct_chg_10d",
                "industry_return_5d",
                "industry_strength_score",
                "close_position_20",
                "moneyflow_score",
            ],
            bool_columns=["above_ma10", "above_ma20"],
            text_columns=["market_regime", "risk_level"],
        )
        factors = apply_tradable_universe_filter(factors, self.config)
        if factors.empty:
            return empty_signals()

        relative_strength = factors["pct_chg_5d"] - factors["industry_return_5d"]
        mask = (
            required_bool(factors["above_ma10"], self.config["require_above_ma10"])
            & required_bool(factors["above_ma20"], self.config["require_above_ma20"])
            & (relative_strength >= self.config["min_relative_strength_5d"])
            & (factors["pct_chg_10d"] >= self.config["min_pct_chg_10d"])
            & (factors["pct_chg_5d"] >= self.config["min_pct_chg_5d"])
            & (factors["pct_chg_5d"] <= self.config["max_pct_chg_5d"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["close_position_20"] <= self.config["max_close_position_20"])
        )
        if self.version == "v2_industry_strong_pullback":
            mask &= factors["industry_strength_score"].fillna(-999) >= self.config["min_industry_strength_score"]
        elif self.version == "v3_moneyflow_confirm":
            mask &= factors["moneyflow_score"].fillna(-999) >= self.config["min_moneyflow_score"]

        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        relative_strength = (signals["pct_chg_5d"] - signals["industry_return_5d"]).fillna(0)
        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            relative_strength.clip(lower=0) * 180
            + signals["pct_chg_10d"].fillna(0).clip(lower=0) * 90
            + signals["industry_strength_score"].fillna(0).clip(upper=100) * 0.25
            + signals["moneyflow_score"].fillna(0).clip(lower=-10, upper=30) * 0.4
        )
        signals["entry_reason"] = "个股相对行业保持强势，短期温和回调，适合作为强势低吸观察标的。"
        signals["risk_flags"] = append_flags(
            pd.Series("", index=signals.index),
            [
                (signals["close_position_20"] > 0.85, "near_20d_high"),
                (signals["moneyflow_score"].fillna(0) < 0, "moneyflow_not_confirmed"),
            ],
        )
        signals = apply_market_regime_gating(signals, strategy_group="pullback")
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
