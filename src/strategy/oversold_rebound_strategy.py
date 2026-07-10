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


class OversoldReboundStrategy(BaseStrategy):
    name = "oversold_rebound"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1_mild_oversold"))
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
                "pct_chg_3d",
                "pct_chg_5d",
                "pct_chg_10d",
                "moneyflow_score",
                "industry_strength_score",
                "close_position_20",
            ],
            bool_columns=["is_limit_down_close"],
            text_columns=["market_regime", "risk_level"],
        )
        factors = apply_tradable_universe_filter(
            factors,
            self.config,
            exclude_limit_up_close=True,
            exclude_limit_down_close=True,
            exclude_continuous_downtrend=True,
        )
        if factors.empty:
            return empty_signals()

        mask = (
            (factors["pct_chg_3d"] >= self.config["min_pct_chg_3d"])
            & (factors["pct_chg_3d"] <= self.config["max_pct_chg_3d"])
            & (factors["pct_chg_5d"] >= self.config["min_pct_chg_5d"])
            & (factors["pct_chg_10d"] >= self.config["min_pct_chg_10d"])
            & (factors["close_position_20"] >= self.config["min_close_position_20"])
            & (factors["close_position_20"] <= self.config["max_close_position_20"])
            & ~factors["is_limit_down_close"]
        )
        if self.version == "v2_moneyflow_repair":
            mask &= factors["moneyflow_score"].fillna(-999) >= self.config["min_moneyflow_score"]
        elif self.version == "v3_industry_repair":
            mask &= factors["industry_strength_score"].fillna(-999) >= self.config["min_industry_strength_score"]

        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        rebound_support = (
            signals["moneyflow_score"].fillna(0).clip(lower=-10, upper=25) * 0.7
            + signals["industry_strength_score"].fillna(0).clip(upper=100) * 0.2
        )
        signals["signal_strength"] = (
            signals["pct_chg_3d"].abs().fillna(0) * 120
            + (1 - signals["close_position_20"].fillna(0)).clip(lower=0) * 12
            + rebound_support
        )
        signals["entry_reason"] = "短期温和超跌后资金或行业出现修复，适合作为反弹观察标的。"
        signals["risk_flags"] = append_flags(
            pd.Series("", index=signals.index),
            [
                (signals["pct_chg_5d"] < -0.12, "short_term_weakness"),
                (signals["pct_chg_10d"] < -0.20, "medium_term_weakness"),
                (signals["moneyflow_score"].fillna(0) < 0, "moneyflow_not_confirmed"),
            ],
        )
        signals = apply_market_regime_gating(signals, strategy_group="rebound")
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
