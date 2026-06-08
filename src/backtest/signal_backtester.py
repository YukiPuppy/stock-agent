"""Basic historical evaluation for deterministic strategy signals."""

from __future__ import annotations

import pandas as pd


DEFAULT_HOLDING_DAYS = [1, 3, 5]
BASE_RESULT_COLUMNS = [
    "signal_date",
    "code",
    "strategy_name",
    "strategy_version",
    "signal_strength",
    "entry_date",
    "entry_open",
    "return_1d",
    "return_3d",
    "return_5d",
    "max_drawdown_1d",
    "max_drawdown_3d",
    "max_drawdown_5d",
    "is_valid",
    "invalid_reason",
]
PERFORMANCE_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "sample_count",
    "valid_count",
    "win_rate_1d",
    "win_rate_3d",
    "win_rate_5d",
    "avg_return_1d",
    "avg_return_3d",
    "avg_return_5d",
    "median_return_1d",
    "median_return_3d",
    "median_return_5d",
    "avg_max_drawdown_1d",
    "avg_max_drawdown_3d",
    "avg_max_drawdown_5d",
]


def backtest_strategy_signals(
    strategy_signals: pd.DataFrame,
    daily_bars: pd.DataFrame,
    holding_days: list[int] | None = None,
) -> pd.DataFrame:
    """Backtest signals by buying next trading day's open and exiting later closes."""
    periods = holding_days or DEFAULT_HOLDING_DAYS
    result_columns = _result_columns(periods)
    if strategy_signals.empty:
        return pd.DataFrame(columns=result_columns)

    bars_by_code = _prepare_bars_by_code(daily_bars)
    signals = strategy_signals.copy()
    if "strategy_version" not in signals.columns:
        signals["strategy_version"] = "v1"
    signals["strategy_version"] = signals["strategy_version"].fillna("v1")
    signals["_trade_date_key"] = _date_key(signals["trade_date"])
    signals = signals.sort_values(["_trade_date_key", "code", "strategy_name", "strategy_version"]).reset_index(drop=True)

    rows = []
    for _, signal in signals.iterrows():
        row = _base_row(signal, periods)
        code = signal["code"]
        bars = bars_by_code.get(code)
        if bars is None or bars.empty:
            row["invalid_reason"] = "no_daily_bars_for_code"
            rows.append(row)
            continue

        signal_key = signal["_trade_date_key"]
        entry_candidates = bars[bars["_trade_date_key"] > signal_key]
        if entry_candidates.empty:
            row["invalid_reason"] = "no_next_trading_day"
            rows.append(row)
            continue

        entry_pos = int(entry_candidates.index[0])
        entry_bar = bars.loc[entry_pos]
        entry_open = entry_bar["open"]
        row["entry_date"] = entry_bar["trade_date"]
        row["entry_open"] = entry_open

        if pd.isna(entry_open) or entry_open == 0:
            row["invalid_reason"] = "invalid_entry_open"
            rows.append(row)
            continue

        missing_periods = []
        for days in periods:
            exit_pos = entry_pos + days
            if exit_pos >= len(bars):
                missing_periods.append(days)
                continue

            exit_bar = bars.loc[exit_pos]
            holding_window = bars.loc[entry_pos:exit_pos]
            row[f"exit_date_{days}d"] = exit_bar["trade_date"]
            row[f"exit_close_{days}d"] = exit_bar["close"]
            row[f"return_{days}d"] = exit_bar["close"] / entry_open - 1
            row[f"max_drawdown_{days}d"] = holding_window["low"].min() / entry_open - 1

        if missing_periods:
            row["invalid_reason"] = "insufficient_future_bars"
        else:
            row["is_valid"] = True
        rows.append(row)

    return pd.DataFrame(rows, columns=result_columns)


def evaluate_strategy_performance(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize backtest results by strategy_name and strategy_version using valid samples only."""
    if backtest_results.empty or "strategy_name" not in backtest_results.columns:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    results = backtest_results.copy()
    if "strategy_version" not in results.columns:
        results["strategy_version"] = "v1"
    results["strategy_version"] = results["strategy_version"].fillna("v1")

    group_columns = ["strategy_name", "strategy_version"]
    sample_counts = results.groupby(group_columns, dropna=False).size()
    valid_results = results[results["is_valid"] == True].copy()
    if valid_results.empty:
        rows = [
            {
                "strategy_name": strategy_name,
                "strategy_version": strategy_version,
                "sample_count": int(sample_count),
                "valid_count": 0,
                **_empty_performance_metrics(),
            }
            for (strategy_name, strategy_version), sample_count in sample_counts.items()
        ]
        return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)

    rows = []
    for (strategy_name, strategy_version), sample_count in sample_counts.items():
        group = valid_results[
            (valid_results["strategy_name"] == strategy_name)
            & (valid_results["strategy_version"] == strategy_version)
        ]
        row = {
            "strategy_name": strategy_name,
            "strategy_version": strategy_version,
            "sample_count": int(sample_count),
            "valid_count": int(len(group)),
        }
        if group.empty:
            row.update(_empty_performance_metrics())
        else:
            for days in DEFAULT_HOLDING_DAYS:
                returns = group[f"return_{days}d"].dropna()
                drawdowns = group[f"max_drawdown_{days}d"].dropna()
                row[f"win_rate_{days}d"] = (returns > 0).mean() if not returns.empty else None
                row[f"avg_return_{days}d"] = returns.mean() if not returns.empty else None
                row[f"median_return_{days}d"] = returns.median() if not returns.empty else None
                row[f"avg_max_drawdown_{days}d"] = drawdowns.mean() if not drawdowns.empty else None
        rows.append(row)

    return (
        pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)
        .sort_values(["strategy_name", "strategy_version"])
        .reset_index(drop=True)
    )


def _prepare_bars_by_code(daily_bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if daily_bars.empty:
        return {}

    bars = daily_bars.copy()
    bars["_trade_date_key"] = _date_key(bars["trade_date"])
    bars = bars.sort_values(["code", "_trade_date_key"]).reset_index(drop=True)
    return {
        code: group.reset_index(drop=True)
        for code, group in bars.groupby("code", sort=False)
    }


def _date_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), errors="coerce")


def _base_row(signal: pd.Series, periods: list[int]) -> dict[str, object]:
    row: dict[str, object] = {
        "signal_date": signal["trade_date"],
        "code": signal["code"],
        "strategy_name": signal["strategy_name"],
        "strategy_version": signal.get("strategy_version", "v1"),
        "signal_strength": signal.get("signal_strength"),
        "entry_date": None,
        "entry_open": None,
        "is_valid": False,
        "invalid_reason": "",
    }
    for days in periods:
        row[f"exit_date_{days}d"] = None
        row[f"exit_close_{days}d"] = None
        row[f"return_{days}d"] = None
        row[f"max_drawdown_{days}d"] = None
    return row


def _result_columns(periods: list[int]) -> list[str]:
    columns = [
        "signal_date",
        "code",
        "strategy_name",
        "strategy_version",
        "signal_strength",
        "entry_date",
        "entry_open",
    ]
    for days in periods:
        columns.extend([f"exit_date_{days}d", f"exit_close_{days}d", f"return_{days}d"])
    for days in periods:
        columns.append(f"max_drawdown_{days}d")
    columns.extend(["is_valid", "invalid_reason"])
    if periods == DEFAULT_HOLDING_DAYS:
        return columns
    return columns


def _empty_performance_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    for days in DEFAULT_HOLDING_DAYS:
        metrics[f"win_rate_{days}d"] = None
        metrics[f"avg_return_{days}d"] = None
        metrics[f"median_return_{days}d"] = None
        metrics[f"avg_max_drawdown_{days}d"] = None
    return metrics
