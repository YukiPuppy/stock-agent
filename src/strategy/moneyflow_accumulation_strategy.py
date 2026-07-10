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


class MoneyflowAccumulationStrategy(BaseStrategy):
    name = "moneyflow_accumulation"

    def __init__(self, config: dict | None = None):
        self.config = get_strategy_config(self.name, {self.name: config or {}})
        self.version = str(self.config.get("version", "v1_main_inflow"))
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
                "moneyflow_score",
                "main_net_amount",
                "main_net_amount_ratio",
                "big_net_amount",
                "net_mf_amount",
                "pct_chg_1d",
                "pct_chg_3d",
                "pct_chg_5d",
                "turnover_rate",
            ],
            text_columns=["market_regime", "risk_level"],
        )
        factors = apply_tradable_universe_filter(factors, self.config)
        if factors.empty:
            return empty_signals()

        mask = (
            (factors["moneyflow_score"] >= self.config["min_moneyflow_score"])
            & (factors["pct_chg_1d"] <= self.config["max_pct_chg_1d"])
            & (factors["pct_chg_3d"] <= self.config["max_pct_chg_3d"])
            & (factors["pct_chg_5d"] <= self.config["max_pct_chg_5d"])
            & (factors["pct_chg_5d"] >= self.config["min_pct_chg_5d"])
        )
        if self.version == "v1_main_inflow":
            mask &= (factors["main_net_amount"] > 0) & (
                factors["main_net_amount_ratio"] >= self.config["min_main_net_amount_ratio"]
            )
        elif self.version == "v2_big_order_confirm":
            mask &= (
                (factors["big_net_amount"] > 0)
                & (factors["net_mf_amount"] > 0)
                & (factors["main_net_amount"] >= 0)
            )
        elif self.version == "v3_price_not_chased":
            mask &= (
                (factors["main_net_amount_ratio"] >= self.config["min_main_net_amount_ratio"])
                & (factors["turnover_rate"].isna() | (factors["turnover_rate"] <= self.config["max_turnover_rate"]))
                & (factors["pct_chg_1d"] >= self.config["min_pct_chg_1d"])
            )

        signals = factors.loc[mask].copy()
        if signals.empty:
            return empty_signals()

        signals["strategy_name"] = self.name
        signals["strategy_version"] = self.version
        signals["signal_strength"] = (
            signals["moneyflow_score"].fillna(0).clip(upper=40) * 1.2
            + signals["main_net_amount_ratio"].fillna(0).clip(lower=0, upper=12) * 2.0
            + (signals["big_net_amount"].fillna(0) > 0).astype(int) * 6
            - signals["pct_chg_5d"].fillna(0).clip(lower=0) * 80
        )
        signals["entry_reason"] = "主力资金持续流入且短期涨幅未明显透支，适合作为资金积累观察标的。"
        signals["risk_flags"] = append_flags(
            pd.Series("", index=signals.index),
            [
                (signals["pct_chg_5d"] > 0.08, "price_chase_risk"),
                (signals["turnover_rate"].fillna(0) > self.config["max_turnover_rate"], "high_turnover"),
                (signals["main_net_amount_ratio"].fillna(0) < 0, "main_outflow"),
            ],
        )
        signals = apply_market_regime_gating(signals, strategy_group="moneyflow")
        return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
