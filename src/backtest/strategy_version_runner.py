from __future__ import annotations

import pandas as pd

from src.strategy.base_strategy import SIGNAL_COLUMNS, empty_signals
from src.strategy.breakout_volume_strategy import BreakoutVolumeStrategy
from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series, normalize_trade_date_value
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
    prepared_factors = _prepare_historical_factors(daily_factors)
    daily_groups = _daily_factor_groups(prepared_factors, start_date, end_date)
    frames = [
        strategy.generate_signals(day_factors, trade_date=trade_date)
        for trade_date, day_factors in daily_groups
    ]
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
    if daily_factors.empty or "trade_date" not in daily_factors.columns:
        return empty_signals()

    prepared_factors = _prepare_historical_factors(daily_factors)
    daily_groups = _daily_factor_groups(prepared_factors, start_date, end_date)
    if not daily_groups:
        return empty_signals()

    frames = []
    for version in versions:
        if not version.get("enabled", True):
            continue
        frames.append(_generate_signals_for_version_groups(version, daily_groups))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_signals()

    signals = pd.concat(frames, ignore_index=True)
    signals = signals.sort_values(
        ["trade_date", "strategy_name", "strategy_version", "signal_strength"],
        ascending=[True, True, True, False],
    )
    return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def _generate_signals_for_version_groups(
    version: dict,
    daily_groups: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    strategy_class = STRATEGY_CLASSES.get(version["strategy_name"])
    if strategy_class is None:
        raise ValueError(f"Unsupported strategy_name: {version['strategy_name']}")
    strategy_version = version["strategy_version"]
    config = {
        "enabled": True,
        "version": strategy_version,
        "params": version.get("params", {}),
        **version.get("params", {}),
    }
    strategy = strategy_class(config)
    frames = [
        strategy.generate_signals(day_factors, trade_date=trade_date)
        for trade_date, day_factors in daily_groups
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_signals()

    signals = pd.concat(frames, ignore_index=True)
    signals["strategy_version"] = strategy_version
    return signals.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)


def _prepare_historical_factors(daily_factors: pd.DataFrame) -> pd.DataFrame:
    if TRADE_DATE_KEY_COLUMN in daily_factors.columns:
        return daily_factors
    prepared = daily_factors.copy()
    prepared[TRADE_DATE_KEY_COLUMN] = normalize_trade_date_series(prepared["trade_date"])
    return prepared


def _daily_factor_groups(
    daily_factors: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, pd.DataFrame]]:
    keys = daily_factors[TRADE_DATE_KEY_COLUMN]
    mask = keys.ne("")
    if start_date is not None:
        mask &= keys >= normalize_trade_date_value(start_date)
    if end_date is not None:
        mask &= keys <= normalize_trade_date_value(end_date)
    selected = daily_factors.loc[mask]
    if selected.empty:
        return []
    return [
        (str(trade_date), group)
        for trade_date, group in selected.groupby(TRADE_DATE_KEY_COLUMN, sort=True)
    ]


def _trade_dates(daily_factors: pd.DataFrame, start_date: str | None, end_date: str | None) -> list[str]:
    normalized_dates = normalize_trade_date_series(daily_factors["trade_date"])
    trade_dates = normalized_dates[normalized_dates.ne("")].drop_duplicates().sort_values()
    if start_date is not None:
        trade_dates = trade_dates[trade_dates >= normalize_trade_date_value(start_date)]
    if end_date is not None:
        trade_dates = trade_dates[trade_dates <= normalize_trade_date_value(end_date)]
    return trade_dates.tolist()


def _normalize_trade_date(value: object) -> str:
    return normalize_trade_date_value(value)
