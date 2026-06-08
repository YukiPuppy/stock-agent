"""Factor coverage and usage diagnostics for local research."""

from __future__ import annotations

import pandas as pd


DEFAULT_FACTOR_COLUMNS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio_daily_basic",
    "amount_ma5",
    "total_mv",
    "circ_mv",
    "up_limit",
    "down_limit",
    "limit_up_distance",
    "limit_down_distance",
    "moneyflow_score",
    "main_net_amount",
    "main_net_amount_ratio",
    "industry_strength_score",
    "industry_return_3d",
    "industry_return_5d",
    "industry_amount_ratio_5",
]

FACTOR_DIAGNOSTIC_COLUMNS = [
    "factor_name",
    "total_count",
    "non_null_count",
    "missing_count",
    "missing_rate",
    "mean",
    "std",
    "min",
    "p25",
    "median",
    "p75",
    "max",
    "candidate_non_null_count",
    "candidate_mean",
    "trade_plan_non_null_count",
    "trade_plan_mean",
    "diagnostic_status",
    "diagnostic_message",
]


def build_factor_diagnostics(
    daily_factors: pd.DataFrame,
    candidate_pool: pd.DataFrame | None = None,
    trade_plan: pd.DataFrame | None = None,
    factor_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build per-factor coverage, distribution, and downstream usage diagnostics."""
    factors = daily_factors.copy() if isinstance(daily_factors, pd.DataFrame) else pd.DataFrame()
    columns = factor_columns if factor_columns is not None else DEFAULT_FACTOR_COLUMNS
    if factors.empty:
        return pd.DataFrame(columns=FACTOR_DIAGNOSTIC_COLUMNS)

    candidate_scope = _merge_scope(factors, candidate_pool)
    trade_plan_scope = _merge_scope(factors, trade_plan)
    rows = []
    total_count = int(len(factors))

    for factor_name in columns:
        if factor_name not in factors.columns:
            rows.append(
                {
                    "factor_name": factor_name,
                    "total_count": total_count,
                    "non_null_count": 0,
                    "missing_count": total_count,
                    "missing_rate": 1.0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "max": None,
                    "candidate_non_null_count": 0,
                    "candidate_mean": None,
                    "trade_plan_non_null_count": 0,
                    "trade_plan_mean": None,
                    "diagnostic_status": "missing_column",
                    "diagnostic_message": "factor column not found",
                }
            )
            continue

        series = pd.to_numeric(factors[factor_name], errors="coerce")
        non_null_count = int(series.notna().sum())
        missing_count = total_count - non_null_count
        missing_rate = float(missing_count / total_count) if total_count else 0.0
        stats = _distribution_stats(series)
        status = _status_from_missing_rate(missing_rate)

        candidate_series = _scope_series(candidate_scope, factor_name)
        trade_plan_series = _scope_series(trade_plan_scope, factor_name)
        rows.append(
            {
                "factor_name": factor_name,
                "total_count": total_count,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                **stats,
                "candidate_non_null_count": int(candidate_series.notna().sum()),
                "candidate_mean": _mean_or_none(candidate_series),
                "trade_plan_non_null_count": int(trade_plan_series.notna().sum()),
                "trade_plan_mean": _mean_or_none(trade_plan_series),
                "diagnostic_status": status,
                "diagnostic_message": _message_from_status(status),
            }
        )

    return pd.DataFrame(rows, columns=FACTOR_DIAGNOSTIC_COLUMNS)


def _merge_scope(factors: pd.DataFrame, scope: pd.DataFrame | None) -> pd.DataFrame:
    if scope is None or not isinstance(scope, pd.DataFrame) or scope.empty:
        return pd.DataFrame()
    if not {"trade_date", "code"} <= set(scope.columns) or not {"trade_date", "code"} <= set(factors.columns):
        return pd.DataFrame()
    keys = scope.loc[:, ["trade_date", "code"]].dropna().drop_duplicates()
    if keys.empty:
        return pd.DataFrame()
    left = factors.copy()
    for column in ["trade_date", "code"]:
        left[column] = left[column].astype(str)
        keys[column] = keys[column].astype(str)
    return keys.merge(left, on=["trade_date", "code"], how="left")


def _scope_series(scope: pd.DataFrame, factor_name: str) -> pd.Series:
    if scope.empty or factor_name not in scope.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(scope[factor_name], errors="coerce")


def _distribution_stats(series: pd.Series) -> dict[str, float | None]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return {key: None for key in ["mean", "std", "min", "p25", "median", "p75", "max"]}
    quantiles = valid.quantile([0.25, 0.5, 0.75])
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()) if len(valid) > 1 else 0.0,
        "min": float(valid.min()),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "max": float(valid.max()),
    }


def _mean_or_none(series: pd.Series) -> float | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return None if valid.empty else float(valid.mean())


def _status_from_missing_rate(missing_rate: float) -> str:
    if missing_rate > 0.8:
        return "high_missing"
    if missing_rate > 0.3:
        return "medium_missing"
    return "ok"


def _message_from_status(status: str) -> str:
    if status == "high_missing":
        return "missing rate is high"
    if status == "medium_missing":
        return "missing rate is medium"
    return "factor coverage is acceptable for diagnostics"
