from __future__ import annotations

import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
from src.strategy.strategy_config import get_strategy_config
from src.strategy.strategy_utils import (
    append_flags,
    apply_market_regime_gating,
    apply_tradable_universe_filter,
    prepare_factors,
)


class IndustryRotationStrategy(BaseStrategy):
    name = "industry_rotation"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1_strength_follow"))
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
                "industry_strength_score",
                "industry_return_3d",
                "industry_return_5d",
                "industry_amount_ratio_5",
                "pct_chg_3d",
                "pct_chg_5d",
                "close_position_20",
                "moneyflow_score",
            ],
            text_columns=["industry_strength_level", "market_regime", "risk_level"],
        )
        factors = apply_tradable_universe_filter(factors, self.config)
        if factors.empty:
            return empty_signals()

        level = factors["industry_strength_level"].fillna("").astype(str).str.lower()
        strong_industry = (factors["industry_strength_score"] >= self.config["min_industry_strength_score"]) | level.eq(
            "strong"
        )
        mask = (
            strong_industry
            & (factors["industry_return_3d"] >= self.config["min_industry_return_3d"])
            & (factors["industry_return_5d"] >= self.config["min_industry_return_5d"])
            & (factors["industry_amount_ratio_5"] >= self.config["min_industry_amount_ratio_5"])
            & (factors["pct_chg_5d"] >= self.config["min_pct_chg_5d"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["moneyflow_score"].fillna(0) >= self.config["min_moneyflow_score"])
        )

        if self.version == "v2_no_overheat":
            mask &= (
                (factors["pct_chg_3d"] <= self.config["max_pct_chg_3d"])
                & (factors["pct_chg_5d"] <= self.config["max_pct_chg_5d"])
                & (factors["close_position_20"] <= self.config["max_close_position_20"])
            )
        elif self.version == "v3_industry_leader":
            relative_strength = factors["pct_chg_5d"] - factors["industry_return_5d"]
            mask &= relative_strength >= self.config["min_relative_strength_5d"]

        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        relative_strength = (signals["pct_chg_5d"] - signals["industry_return_5d"]).fillna(0)
        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["industry_strength_score"].fillna(0).clip(upper=100) * 0.45
            + signals["industry_return_5d"].fillna(0) * 300
            + signals["pct_chg_5d"].fillna(0) * 120
            + signals["moneyflow_score"].fillna(0).clip(lower=-10, upper=30) * 0.5
            + relative_strength.clip(lower=0) * 120
        )
        signals["entry_reason"] = "行业强度占优，个股短期强度跟随或领先行业，适合作为轮动观察标的。"
        signals["risk_flags"] = append_flags(
            pd.Series("", index=signals.index),
            [
                (signals["close_position_20"] > 0.9, "near_20d_high"),
                (signals["pct_chg_3d"] > 0.10, "short_term_overheat"),
                (signals["moneyflow_score"].fillna(0) < 0, "moneyflow_not_confirmed"),
            ],
        )
        signals = apply_market_regime_gating(signals, strategy_group="rotation")
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
