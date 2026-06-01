from __future__ import annotations

import pandas as pd


DAILY_FACTOR_COLUMNS = [
    "trade_date",
    "code",
    "close",
    "pct_chg_1d",
    "pct_chg_3d",
    "pct_chg_5d",
    "pct_chg_10d",
    "ma5",
    "ma10",
    "ma20",
    "volume_ma5",
    "amount_ma5",
    "volume_ratio_5",
    "high_20",
    "low_20",
    "close_position_20",
    "above_ma5",
    "above_ma10",
    "above_ma20",
]


def compute_daily_factors(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical factors from daily bars."""
    if daily_bars.empty:
        return pd.DataFrame(columns=DAILY_FACTOR_COLUMNS)

    bars = daily_bars.copy()
    bars = bars.sort_values(["code", "trade_date"]).reset_index(drop=True)
    grouped = bars.groupby("code", sort=False)

    factors = bars.loc[:, ["trade_date", "code", "close"]].copy()
    factors["pct_chg_1d"] = bars["close"] / grouped["close"].shift(1) - 1
    factors["pct_chg_3d"] = bars["close"] / grouped["close"].shift(3) - 1
    factors["pct_chg_5d"] = bars["close"] / grouped["close"].shift(5) - 1
    factors["pct_chg_10d"] = bars["close"] / grouped["close"].shift(10) - 1

    factors["ma5"] = _rolling_mean(grouped["close"], 5)
    factors["ma10"] = _rolling_mean(grouped["close"], 10)
    factors["ma20"] = _rolling_mean(grouped["close"], 20)
    factors["volume_ma5"] = _rolling_mean(grouped["volume"], 5)
    factors["amount_ma5"] = _rolling_mean(grouped["amount"], 5)
    factors["volume_ratio_5"] = _safe_divide(bars["volume"], factors["volume_ma5"])
    factors["high_20"] = _rolling_max(grouped["high"], 20)
    factors["low_20"] = _rolling_min(grouped["low"], 20)
    factors["close_position_20"] = _safe_divide(
        bars["close"] - factors["low_20"],
        factors["high_20"] - factors["low_20"],
    )

    factors["above_ma5"] = bars["close"] > factors["ma5"]
    factors["above_ma10"] = bars["close"] > factors["ma10"]
    factors["above_ma20"] = bars["close"] > factors["ma20"]

    numeric_columns = [
        column
        for column in DAILY_FACTOR_COLUMNS
        if column not in {"trade_date", "code", "above_ma5", "above_ma10", "above_ma20"}
    ]
    factors[numeric_columns] = factors[numeric_columns].replace([float("inf"), float("-inf")], pd.NA)

    return factors.loc[:, DAILY_FACTOR_COLUMNS].sort_values(["trade_date", "code"]).reset_index(drop=True)


def _rolling_mean(series_group: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    return series_group.transform(lambda series: series.rolling(window=window, min_periods=1).mean())


def _rolling_max(series_group: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    return series_group.transform(lambda series: series.rolling(window=window, min_periods=1).max())


def _rolling_min(series_group: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    return series_group.transform(lambda series: series.rolling(window=window, min_periods=1).min())


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    return result.mask(denominator == 0, 0).replace([float("inf"), float("-inf")], pd.NA)
