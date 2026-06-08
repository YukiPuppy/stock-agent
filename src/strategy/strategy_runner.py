from __future__ import annotations

import pandas as pd

from src.strategy.base_strategy import BaseStrategy, SIGNAL_COLUMNS, empty_signals
from src.strategy.breakout_volume_strategy import BreakoutVolumeStrategy
from src.strategy.strategy_config import get_strategy_config, is_strategy_enabled, load_strategy_config
from src.strategy.support_rebound_strategy import SupportReboundStrategy
from src.strategy.trend_pullback_strategy import TrendPullbackStrategy


def run_strategies(
    daily_factors: pd.DataFrame,
    trade_date: str | None = None,
    strategies: list[BaseStrategy] | None = None,
    config_path: str | None = None,
) -> pd.DataFrame:
    if strategies is None:
        config = load_strategy_config(config_path)
        active_strategies = [
            strategy_class(get_strategy_config(strategy_class.name, config))
            for strategy_class in [TrendPullbackStrategy, BreakoutVolumeStrategy, SupportReboundStrategy]
            if is_strategy_enabled(strategy_class.name, config)
        ]
    else:
        active_strategies = strategies
    frames = [strategy.generate_signals(daily_factors, trade_date) for strategy in active_strategies]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_signals()

    signals = pd.concat(frames, ignore_index=True)
    for column in SIGNAL_COLUMNS:
        if column not in signals.columns:
            signals[column] = "v1" if column == "strategy_version" else None
    signals["signal_strength"] = pd.to_numeric(signals["signal_strength"], errors="coerce").fillna(0)
    signals = signals.sort_values("signal_strength", ascending=False).drop_duplicates(
        subset=["trade_date", "code", "strategy_name", "strategy_version"],
        keep="first",
    )
    return signals.sort_values(["trade_date", "signal_strength"], ascending=[True, False]).loc[
        :, SIGNAL_COLUMNS
    ].reset_index(drop=True)
