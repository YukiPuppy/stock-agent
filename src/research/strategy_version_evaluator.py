from __future__ import annotations

import pandas as pd


EVALUATION_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "sample_count",
    "valid_count",
    "win_rate_3d",
    "avg_return_3d",
    "median_return_3d",
    "avg_max_drawdown_3d",
    "evaluation_score",
    "evaluation_status",
    "risk_level",
    "recommendation",
    "evaluation_reason",
]


def evaluate_strategy_versions(
    performance: pd.DataFrame,
    min_valid_count: int = 30,
    min_win_rate_3d: float = 0.50,
    min_avg_return_3d: float = 0.0,
    max_avg_drawdown_3d: float = -0.08,
) -> pd.DataFrame:
    if performance.empty:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    rows = []
    for record in performance.to_dict("records"):
        valid_count = _number(record.get("valid_count"))
        win_rate_3d = _number(record.get("win_rate_3d"))
        avg_return_3d = _number(record.get("avg_return_3d"))
        median_return_3d = _number(record.get("median_return_3d"))
        avg_max_drawdown_3d = _number(record.get("avg_max_drawdown_3d"))

        if valid_count < min_valid_count:
            status = "insufficient_samples"
            risk_level = "unknown"
            recommendation = "continue_backtest"
            reason = f"有效样本不足：valid_count={valid_count:g}，低于阈值 {min_valid_count}。"
        elif avg_return_3d < min_avg_return_3d and win_rate_3d < min_win_rate_3d:
            status = "weak"
            risk_level = "medium"
            recommendation = "pause"
            reason = "3日收益和胜率均未达标。"
        elif avg_max_drawdown_3d < max_avg_drawdown_3d:
            status = "high_drawdown"
            risk_level = "high"
            recommendation = "reduce_or_pause"
            reason = "平均回撤偏大。"
        elif (
            valid_count >= min_valid_count
            and win_rate_3d >= min_win_rate_3d
            and avg_return_3d >= min_avg_return_3d
            and avg_max_drawdown_3d >= max_avg_drawdown_3d
        ):
            status = "qualified"
            risk_level = "low"
            recommendation = "enable_observation"
            reason = "满足观察启用条件。"
        else:
            status = "neutral"
            risk_level = "medium"
            recommendation = "observe"
            reason = "表现中性，建议继续观察。"

        score = (
            win_rate_3d * 40
            + avg_return_3d * 100
            + median_return_3d * 50
            + avg_max_drawdown_3d * 50
            + min(valid_count, 200) / 200 * 10
        )

        rows.append(
            {
                "strategy_name": record.get("strategy_name"),
                "strategy_version": record.get("strategy_version"),
                "sample_count": record.get("sample_count"),
                "valid_count": record.get("valid_count"),
                "win_rate_3d": record.get("win_rate_3d"),
                "avg_return_3d": record.get("avg_return_3d"),
                "median_return_3d": record.get("median_return_3d"),
                "avg_max_drawdown_3d": record.get("avg_max_drawdown_3d"),
                "evaluation_score": round(score, 4),
                "evaluation_status": status,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "evaluation_reason": reason,
            }
        )

    return (
        pd.DataFrame(rows, columns=EVALUATION_COLUMNS)
        .sort_values("evaluation_score", ascending=False)
        .reset_index(drop=True)
    )


def build_active_strategy_config(
    evaluation: pd.DataFrame,
    min_status: str = "qualified",
) -> dict:
    _ = min_status
    if evaluation.empty:
        return {"active_strategies": []}

    active = evaluation[evaluation["recommendation"] == "enable_observation"]
    strategies = []
    for record in active.to_dict("records"):
        strategies.append(
            {
                "strategy_name": record.get("strategy_name"),
                "strategy_version": record.get("strategy_version"),
                "recommendation": record.get("recommendation"),
                "evaluation_score": record.get("evaluation_score"),
            }
        )
    return {"active_strategies": strategies}


def _number(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)
