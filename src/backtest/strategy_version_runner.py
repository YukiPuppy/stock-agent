from __future__ import annotations

import re

import pandas as pd

from src.strategy.base_strategy import SIGNAL_COLUMNS, empty_signals
from src.strategy.breakout_volume_strategy import BreakoutVolumeStrategy
from src.strategy.support_rebound_strategy import SupportReboundStrategy
from src.strategy.trend_pullback_strategy import TrendPullbackStrategy


STRATEGY_CLASSES = {
    "trend_pullback": TrendPullbackStrategy,
    "breakout_volume": BreakoutVolumeStrategy,
    "support_rebound": SupportReboundStrategy,
}


def generate_historical_signals_for_version(
    daily_factors: pd.DataFrame,
    strategy_name: str,
    strategy_version: str,
    params: dict,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    strategy_class = STRATEGY_CLASSES.get(strategy_name)
    if strategy_class is None:
        raise ValueError(f"Unsupported strategy_name: {strategy_name}")
    if daily_factors.empty or "trade_date" not in daily_factors.columns:
        return empty_signals()

    config = {
        "enabled": True,
        "version": strategy_version,
        "params": params,
        **params,
    }
    strategy = strategy_class(config)
    trade_dates = _trade_dates(daily_factors, start_date, end_date)
    frames = [strategy.generate_signals(daily_factors, trade_date=trade_date) for trade_date in trade_dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_signals()

    signals = pd.concat(frames, ignore_index=True)
    signals["strategy_version"] = strategy_version
    return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def generate_historical_signals_for_versions(
    daily_factors: pd.DataFrame,
    versions: list[dict],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    frames = []
    for version in versions:
        if not version.get("enabled", True):
            continue
        frames.append(
            generate_historical_signals_for_version(
                daily_factors=daily_factors,
                strategy_name=version["strategy_name"],
                strategy_version=version["strategy_version"],
                params=version.get("params", {}),
                start_date=start_date,
                end_date=end_date,
            )
        )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_signals()

    signals = pd.concat(frames, ignore_index=True)
    signals = signals.sort_values(
        ["trade_date", "strategy_name", "strategy_version", "signal_strength"],
        ascending=[True, True, True, False],
    )
    return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def _trade_dates(daily_factors: pd.DataFrame, start_date: str | None, end_date: str | None) -> list[str]:
    trade_dates = daily_factors["trade_date"].dropna().astype(str).drop_duplicates().sort_values()
    normalized_dates = trade_dates.map(_normalize_trade_date)
    if start_date is not None:
        normalized_start = _normalize_trade_date(start_date)
        trade_dates = trade_dates[normalized_dates >= normalized_start]
        normalized_dates = normalized_dates[normalized_dates >= normalized_start]
    if end_date is not None:
        normalized_end = _normalize_trade_date(end_date)
        trade_dates = trade_dates[normalized_dates <= normalized_end]
        normalized_dates = normalized_dates[normalized_dates <= normalized_end]
    return trade_dates.tolist()


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
