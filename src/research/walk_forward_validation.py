from __future__ import annotations

import pandas as pd

from src.backtest.signal_backtester import backtest_strategy_signals, evaluate_strategy_performance
from src.backtest.strategy_version_runner import generate_historical_signals_for_versions


WALK_FORWARD_VALIDATION_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "train_valid_count",
    "train_win_rate_3d",
    "train_avg_return_3d",
    "train_avg_drawdown_3d",
    "validation_valid_count",
    "validation_win_rate_3d",
    "validation_avg_return_3d",
    "validation_avg_drawdown_3d",
    "return_decay",
    "win_rate_decay",
    "drawdown_worsening",
    "stability_score",
    "overfit_risk",
    "validation_status",
    "validation_reason",
]


def validate_strategy_versions_out_of_sample(
    daily_factors: pd.DataFrame,
    daily_bars: pd.DataFrame,
    versions: list[dict],
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    min_valid_count_train: int = 30,
    min_valid_count_validation: int = 10,
) -> pd.DataFrame:
    enabled_versions = [version for version in versions if version.get("enabled", True)]
    if daily_factors.empty or daily_bars.empty or not enabled_versions:
        return pd.DataFrame(columns=WALK_FORWARD_VALIDATION_COLUMNS)

    train_performance = _evaluate_period(
        daily_factors=daily_factors,
        daily_bars=daily_bars,
        versions=enabled_versions,
        start_date=train_start_date,
        end_date=train_end_date,
    )
    validation_performance = _evaluate_period(
        daily_factors=daily_factors,
        daily_bars=daily_bars,
        versions=enabled_versions,
        start_date=validation_start_date,
        end_date=validation_end_date,
    )

    train_by_version = _performance_by_version(train_performance)
    validation_by_version = _performance_by_version(validation_performance)

    rows = []
    for version in enabled_versions:
        key = (version.get("strategy_name"), version.get("strategy_version"))
        train = train_by_version.get(key, {})
        validation = validation_by_version.get(key, {})

        train_valid_count = int(_number(train.get("valid_count")))
        train_win_rate_3d = _number(train.get("win_rate_3d"))
        train_avg_return_3d = _number(train.get("avg_return_3d"))
        train_avg_drawdown_3d = _number(train.get("avg_max_drawdown_3d"))
        validation_valid_count = int(_number(validation.get("valid_count")))
        validation_win_rate_3d = _number(validation.get("win_rate_3d"))
        validation_avg_return_3d = _number(validation.get("avg_return_3d"))
        validation_avg_drawdown_3d = _number(validation.get("avg_max_drawdown_3d"))

        return_decay = validation_avg_return_3d - train_avg_return_3d
        win_rate_decay = validation_win_rate_3d - train_win_rate_3d
        drawdown_worsening = validation_avg_drawdown_3d - train_avg_drawdown_3d
        stability_score = round(
            validation_win_rate_3d * 40
            + validation_avg_return_3d * 100
            + validation_avg_drawdown_3d * 50
            + min(validation_valid_count, 100) / 100 * 10
            - abs(return_decay) * 30
            - abs(win_rate_decay) * 10,
            4,
        )

        overfit_risk, validation_status, validation_reason = _classify_validation(
            train_valid_count=train_valid_count,
            train_avg_return_3d=train_avg_return_3d,
            validation_valid_count=validation_valid_count,
            validation_win_rate_3d=validation_win_rate_3d,
            validation_avg_return_3d=validation_avg_return_3d,
            return_decay=return_decay,
            win_rate_decay=win_rate_decay,
            min_valid_count_train=min_valid_count_train,
            min_valid_count_validation=min_valid_count_validation,
        )

        rows.append(
            {
                "strategy_name": version.get("strategy_name"),
                "strategy_version": version.get("strategy_version"),
                "train_valid_count": train_valid_count,
                "train_win_rate_3d": train_win_rate_3d,
                "train_avg_return_3d": train_avg_return_3d,
                "train_avg_drawdown_3d": train_avg_drawdown_3d,
                "validation_valid_count": validation_valid_count,
                "validation_win_rate_3d": validation_win_rate_3d,
                "validation_avg_return_3d": validation_avg_return_3d,
                "validation_avg_drawdown_3d": validation_avg_drawdown_3d,
                "return_decay": return_decay,
                "win_rate_decay": win_rate_decay,
                "drawdown_worsening": drawdown_worsening,
                "stability_score": stability_score,
                "overfit_risk": overfit_risk,
                "validation_status": validation_status,
                "validation_reason": validation_reason,
            }
        )

    return (
        pd.DataFrame(rows, columns=WALK_FORWARD_VALIDATION_COLUMNS)
        .sort_values("stability_score", ascending=False)
        .reset_index(drop=True)
    )


def _evaluate_period(
    daily_factors: pd.DataFrame,
    daily_bars: pd.DataFrame,
    versions: list[dict],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    signals = generate_historical_signals_for_versions(
        daily_factors=daily_factors,
        versions=versions,
        start_date=start_date,
        end_date=end_date,
    )
    bars = _filter_bars(daily_bars, start_date, end_date)
    backtest_results = backtest_strategy_signals(signals, bars)
    return evaluate_strategy_performance(backtest_results)


def _filter_bars(daily_bars: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if daily_bars.empty or "trade_date" not in daily_bars.columns:
        return daily_bars
    trade_dates = daily_bars["trade_date"].astype(str)
    return daily_bars[(trade_dates >= start_date) & (trade_dates <= end_date)].copy()


def _performance_by_version(performance: pd.DataFrame) -> dict[tuple[object, object], dict]:
    if performance.empty:
        return {}
    return {
        (record.get("strategy_name"), record.get("strategy_version")): record
        for record in performance.to_dict("records")
    }


def _classify_validation(
    train_valid_count: int,
    train_avg_return_3d: float,
    validation_valid_count: int,
    validation_win_rate_3d: float,
    validation_avg_return_3d: float,
    return_decay: float,
    win_rate_decay: float,
    min_valid_count_train: int,
    min_valid_count_validation: int,
) -> tuple[str, str, str]:
    if train_valid_count < min_valid_count_train:
        return "unknown", "insufficient_train_samples", "训练区间有效样本不足，不能判断样本外稳定性。"
    if validation_valid_count < min_valid_count_validation:
        return "unknown", "insufficient_validation_samples", "验证区间有效样本不足，不能判断样本外稳定性。"
    if train_avg_return_3d > 0 and validation_avg_return_3d < 0:
        return "high", "failed_oos", "训练区间为正但样本外转负，存在较高过拟合风险。"
    if return_decay < -0.03 or win_rate_decay < -0.15:
        return "medium", "unstable", "样本外表现明显衰减，建议降低参数可信度。"
    if validation_avg_return_3d >= 0 and validation_win_rate_3d >= 0.50:
        return "low", "passed_oos", "样本外表现基本稳定，可作为研究建议继续跟踪。"
    return "medium", "needs_more_observation", "样本外表现一般，建议继续观察。"


def _number(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)
