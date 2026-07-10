from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.date_utils import TRADE_DATE_KEY_COLUMN, normalize_trade_date_series, normalize_trade_date_value


CANDIDATE_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "market",
    "board",
    "close",
    "pct_chg_1d",
    "pct_chg_3d",
    "pct_chg_5d",
    "pct_chg_10d",
    "volume_ratio_5",
    "close_position_20",
    "above_ma5",
    "above_ma10",
    "above_ma20",
    "amount_ma5",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio_daily_basic",
    "total_mv",
    "circ_mv",
    "up_limit",
    "down_limit",
    "is_suspended",
    "is_limit_up_close",
    "is_limit_down_close",
    "net_mf_amount",
    "main_net_amount",
    "main_net_amount_ratio",
    "big_net_amount",
    "small_net_amount",
    "moneyflow_score",
    "moneyflow_risk_flags",
    "industry_code",
    "industry_name",
    "industry_strength_score",
    "industry_strength_level",
    "industry_return_3d",
    "industry_return_5d",
    "industry_amount_ratio_5",
    "industry_risk_flags",
    "score",
    "rank",
    "reason",
    "strategy_names",
    "strategy_versions",
    "signal_count",
    "active_signal_count",
    "max_signal_strength",
    "total_signal_strength",
    "total_weighted_signal_strength",
    "avg_strategy_weight",
    "recommendations",
    "risk_flags",
]


def select_candidates(
    daily_factors: pd.DataFrame,
    stock_basic: pd.DataFrame | None = None,
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
    exclude_suspended: bool = True,
    exclude_limit_up_close: bool = False,
    exclude_limit_down_close: bool = True,
    min_turnover_rate: float | None = None,
    max_turnover_rate: float | None = None,
    min_circ_mv: float | None = None,
    max_circ_mv: float | None = None,
    market_regime: pd.DataFrame | pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Select and score deterministic A-share candidate stocks.

    ``min_amount_ma5`` is compared against amount_ma5 in thousand yuan.
    """
    if daily_factors.empty or "trade_date" not in daily_factors.columns:
        return _empty_candidates()

    selected_trade_date = normalize_trade_date_value(trade_date) if trade_date is not None else None
    trade_date_keys = (
        daily_factors[TRADE_DATE_KEY_COLUMN]
        if TRADE_DATE_KEY_COLUMN in daily_factors.columns
        else normalize_trade_date_series(daily_factors["trade_date"])
    )
    if selected_trade_date is None:
        trade_dates = trade_date_keys[trade_date_keys.ne("")]
        if trade_dates.empty:
            return _empty_candidates()
        selected_trade_date = str(trade_dates.max())

    candidates = daily_factors.loc[trade_date_keys == selected_trade_date].copy()
    candidates = candidates.drop(columns=[TRADE_DATE_KEY_COLUMN], errors="ignore")
    if candidates.empty:
        return _empty_candidates()

    candidates = _merge_stock_basic(candidates, stock_basic)
    available_factor_columns = set(candidates.columns)

    numeric_columns = [
        "close",
        "pct_chg_1d",
        "pct_chg_3d",
        "pct_chg_5d",
        "pct_chg_10d",
        "volume_ratio_5",
        "close_position_20",
        "amount_ma5",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio_daily_basic",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
        "moneyflow_score",
        "net_mf_amount",
        "main_net_amount",
        "main_net_amount_ratio",
        "big_net_amount",
        "small_net_amount",
        "industry_strength_score",
        "industry_return_3d",
        "industry_return_5d",
        "industry_amount_ratio_5",
    ]
    for column in numeric_columns:
        if column not in candidates.columns:
            candidates[column] = pd.NA
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    for column in ["above_ma5", "above_ma10", "above_ma20"]:
        if column not in candidates.columns:
            candidates[column] = False
        candidates[column] = candidates[column].fillna(False).astype(bool)

    mask = (
        (candidates["close"] > 0)
        & (candidates["amount_ma5"] >= min_amount_ma5)
        & (candidates["pct_chg_1d"] < 0.095)
        & (candidates["pct_chg_1d"] > -0.095)
        & candidates["close_position_20"].notna()
        & (candidates["above_ma5"] | candidates["above_ma10"])
    )
    candidates = candidates.loc[mask].copy()
    candidates = _apply_extension_filters_and_flags(
        candidates,
        exclude_suspended=exclude_suspended,
        exclude_limit_up_close=exclude_limit_up_close,
        exclude_limit_down_close=exclude_limit_down_close,
        min_turnover_rate=min_turnover_rate,
        max_turnover_rate=max_turnover_rate,
        min_circ_mv=min_circ_mv,
        max_circ_mv=max_circ_mv,
        available_columns=available_factor_columns,
    )
    if candidates.empty:
        return _empty_candidates()

    candidates["score"] = (
        candidates["pct_chg_5d"].fillna(0) * 100
        + candidates["pct_chg_10d"].fillna(0) * 50
        + candidates["close_position_20"].fillna(0) * 20
        + candidates["volume_ratio_5"].fillna(0).clip(upper=3) * 5
        + candidates["above_ma5"].astype(int) * 5
        + candidates["above_ma10"].astype(int) * 5
        + candidates["above_ma20"].astype(int) * 5
    )
    candidates["reason"] = "趋势较强，位于20日区间高位，量能相对活跃"
    candidates = _apply_market_regime_adjustment(candidates, market_regime)
    candidates = _apply_moneyflow_adjustment(candidates, available_factor_columns)
    candidates = _apply_industry_adjustment(candidates, available_factor_columns)

    result = candidates.sort_values(["score", "code"], ascending=[False, True]).head(top_n).copy()
    result["rank"] = range(1, len(result) + 1)

    for column in CANDIDATE_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result = _normalize_candidate_dtypes(result)
    return result.loc[:, CANDIDATE_COLUMNS].reset_index(drop=True)


def _merge_stock_basic(daily_factors: pd.DataFrame, stock_basic: pd.DataFrame | None) -> pd.DataFrame:
    if stock_basic is None or stock_basic.empty:
        merged = daily_factors.copy()
        merged["name"] = None
        return merged

    basic_columns = ["code", "name", "market", "board"]
    basic = stock_basic.copy()
    for column in basic_columns:
        if column not in basic.columns:
            basic[column] = None

    return daily_factors.merge(
        basic.loc[:, basic_columns].drop_duplicates(subset=["code"], keep="last"),
        on="code",
        how="left",
    )


def select_candidates_from_signals(
    strategy_signals: pd.DataFrame,
    daily_factors: pd.DataFrame,
    stock_basic: pd.DataFrame | None = None,
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
    strategy_evaluation: pd.DataFrame | None = None,
    exclude_suspended: bool = True,
    exclude_limit_up_close: bool = False,
    exclude_limit_down_close: bool = True,
    min_turnover_rate: float | None = None,
    max_turnover_rate: float | None = None,
    min_circ_mv: float | None = None,
    max_circ_mv: float | None = None,
    market_regime: pd.DataFrame | pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Select candidates from strategy signals.

    ``min_amount_ma5`` is compared against amount_ma5 in thousand yuan.
    """
    if strategy_signals.empty or "trade_date" not in strategy_signals.columns:
        return _empty_candidates()

    selected_trade_date = normalize_trade_date_value(trade_date) if trade_date is not None else None
    trade_date_keys = (
        strategy_signals[TRADE_DATE_KEY_COLUMN]
        if TRADE_DATE_KEY_COLUMN in strategy_signals.columns
        else normalize_trade_date_series(strategy_signals["trade_date"])
    )
    if selected_trade_date is None:
        trade_dates = trade_date_keys[trade_date_keys.ne("")]
        if trade_dates.empty:
            return _empty_candidates()
        selected_trade_date = str(trade_dates.max())

    signals = strategy_signals.loc[trade_date_keys == selected_trade_date].copy()
    signals = signals.drop(columns=[TRADE_DATE_KEY_COLUMN], errors="ignore")
    if signals.empty:
        return _empty_candidates()

    for column in ["signal_strength"]:
        if column not in signals.columns:
            signals[column] = 0
        signals[column] = pd.to_numeric(signals[column], errors="coerce").fillna(0)
    for column in ["strategy_name", "strategy_version", "entry_reason", "risk_flags"]:
        if column not in signals.columns:
            signals[column] = "v1" if column == "strategy_version" else ""
        signals[column] = signals[column].fillna("").astype(str)
    signals.loc[signals["strategy_version"] == "", "strategy_version"] = "v1"

    has_evaluation = strategy_evaluation is not None and not strategy_evaluation.empty
    if has_evaluation:
        evaluation = strategy_evaluation.copy()
        for column in ["strategy_name", "strategy_version", "recommendation", "risk_level"]:
            if column not in evaluation.columns:
                evaluation[column] = None
            evaluation[column] = evaluation[column].fillna("").astype(str)
        evaluation.loc[evaluation["strategy_version"] == "", "strategy_version"] = "v1"
        evaluation = evaluation.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")
        signals = signals.merge(
            evaluation.loc[:, ["strategy_name", "strategy_version", "recommendation", "risk_level"]],
            on=["strategy_name", "strategy_version"],
            how="left",
        )
        signals["strategy_weight"] = _strategy_weights_from_evaluation(signals)
    else:
        signals["recommendation"] = ""
        signals["risk_level"] = ""
        signals["strategy_weight"] = 1.0

    signals["weighted_signal_strength"] = signals["signal_strength"] * signals["strategy_weight"]
    signals["active_signal"] = signals["strategy_weight"] > 0
    signals["active_strategy_name"] = signals["strategy_name"].where(signals["active_signal"], "")
    signals["active_strategy_version"] = signals["strategy_version"].where(signals["active_signal"], "")
    signals["active_recommendation"] = signals["recommendation"].where(signals["active_signal"], "")
    signals["active_entry_reason"] = signals["entry_reason"].where(signals["active_signal"], "")

    grouped = (
        signals.groupby(["trade_date", "code"], as_index=False)
        .agg(
            signal_count=("strategy_name", "count"),
            active_signal_count=("active_signal", "sum"),
            max_signal_strength=("signal_strength", "max"),
            total_signal_strength=("signal_strength", "sum"),
            total_weighted_signal_strength=("weighted_signal_strength", "sum"),
            avg_strategy_weight=("strategy_weight", "mean"),
            strategy_names=("active_strategy_name", _join_unique_values),
            strategy_versions=("active_strategy_version", _join_unique_values),
            recommendations=("active_recommendation", _join_unique_values),
            entry_reasons=("active_entry_reason", _join_unique_values),
            risk_flags=("risk_flags", _join_unique_flags),
        )
        .rename(columns={"entry_reasons": "reason"})
    )
    grouped = grouped[grouped["active_signal_count"] > 0].copy()
    if grouped.empty:
        return _empty_candidates()
    grouped["reason"] = _build_signal_candidate_reasons(grouped)

    factors = _factors_for_trade_date(daily_factors, selected_trade_date)
    if not factors.empty:
        grouped = grouped.merge(factors, on=["trade_date", "code"], how="left")
    else:
        for column in [
            "close",
            "pct_chg_1d",
            "pct_chg_3d",
            "pct_chg_5d",
            "pct_chg_10d",
            "volume_ratio_5",
            "close_position_20",
            "above_ma5",
            "above_ma10",
            "above_ma20",
            "amount_ma5",
        ]:
            grouped[column] = None

    grouped = _merge_stock_basic(grouped, stock_basic)
    available_factor_columns = set(grouped.columns)
    for column in [
        "close",
        "pct_chg_1d",
        "pct_chg_3d",
        "pct_chg_5d",
        "pct_chg_10d",
        "volume_ratio_5",
        "close_position_20",
        "amount_ma5",
        "total_signal_strength",
        "total_weighted_signal_strength",
        "avg_strategy_weight",
        "max_signal_strength",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio_daily_basic",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
        "moneyflow_score",
        "net_mf_amount",
        "main_net_amount",
        "main_net_amount_ratio",
        "big_net_amount",
        "small_net_amount",
        "industry_strength_score",
        "industry_return_3d",
        "industry_return_5d",
        "industry_amount_ratio_5",
    ]:
        if column not in grouped.columns:
            grouped[column] = pd.NA
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce")
    for column in ["above_ma5", "above_ma10", "above_ma20"]:
        if column not in grouped.columns:
            grouped[column] = False
        grouped[column] = grouped[column].fillna(False).astype(bool)

    grouped = grouped[grouped["amount_ma5"] >= min_amount_ma5].copy()
    grouped = _apply_extension_filters_and_flags(
        grouped,
        exclude_suspended=exclude_suspended,
        exclude_limit_up_close=exclude_limit_up_close,
        exclude_limit_down_close=exclude_limit_down_close,
        min_turnover_rate=min_turnover_rate,
        max_turnover_rate=max_turnover_rate,
        min_circ_mv=min_circ_mv,
        max_circ_mv=max_circ_mv,
        available_columns=available_factor_columns,
    )
    if grouped.empty:
        return _empty_candidates()

    grouped["score"] = (
        grouped["total_weighted_signal_strength"].fillna(0)
        + grouped["active_signal_count"].fillna(0) * 10
    )
    grouped = _apply_market_regime_adjustment(grouped, market_regime)
    grouped = _apply_moneyflow_adjustment(grouped, available_factor_columns)
    grouped = _apply_industry_adjustment(grouped, available_factor_columns)
    result = grouped.sort_values(["score", "code"], ascending=[False, True]).head(top_n).copy()
    result["rank"] = range(1, len(result) + 1)

    for column in CANDIDATE_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result = _normalize_candidate_dtypes(result)
    return result.loc[:, CANDIDATE_COLUMNS].reset_index(drop=True)


def _factors_for_trade_date(daily_factors: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if daily_factors.empty or "trade_date" not in daily_factors.columns:
        return pd.DataFrame()
    base_factor_columns = [
        "trade_date",
        "code",
        "close",
        "pct_chg_1d",
        "pct_chg_3d",
        "pct_chg_5d",
        "pct_chg_10d",
        "volume_ratio_5",
        "close_position_20",
        "above_ma5",
        "above_ma10",
        "above_ma20",
        "amount_ma5",
    ]
    extension_factor_columns = [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio_daily_basic",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
        "is_suspended",
        "is_limit_up_close",
        "is_limit_down_close",
        "net_mf_amount",
        "main_net_amount",
        "main_net_amount_ratio",
        "big_net_amount",
        "small_net_amount",
        "moneyflow_score",
        "moneyflow_risk_flags",
        "industry_name",
        "industry_code",
        "industry_strength_score",
        "industry_strength_level",
        "industry_return_3d",
        "industry_return_5d",
        "industry_amount_ratio_5",
        "industry_risk_flags",
    ]
    selected_trade_date = normalize_trade_date_value(trade_date)
    factor_dates = (
        daily_factors[TRADE_DATE_KEY_COLUMN]
        if TRADE_DATE_KEY_COLUMN in daily_factors.columns
        else normalize_trade_date_series(daily_factors["trade_date"])
    )
    factors = daily_factors.loc[factor_dates == selected_trade_date].copy()
    for column in base_factor_columns:
        if column not in factors.columns:
            factors[column] = None
    factor_columns = base_factor_columns + [
        column for column in extension_factor_columns if column in factors.columns
    ]
    return factors.loc[:, factor_columns].drop_duplicates(subset=["trade_date", "code"], keep="last")


def _normalize_trade_date(value: object) -> str:
    return normalize_trade_date_value(value)


def _join_unique_values(values: pd.Series) -> str:
    return ",".join(dict.fromkeys(value for value in values.astype(str) if value))


def _join_unique_flags(values: pd.Series) -> str:
    flags: list[str] = []
    for value in values.astype(str):
        for flag in value.split(","):
            flag = flag.strip()
            if flag and flag not in flags:
                flags.append(flag)
    return ",".join(flags)


def _apply_extension_filters_and_flags(
    candidates: pd.DataFrame,
    *,
    exclude_suspended: bool,
    exclude_limit_up_close: bool,
    exclude_limit_down_close: bool,
    min_turnover_rate: float | None,
    max_turnover_rate: float | None,
    min_circ_mv: float | None,
    max_circ_mv: float | None,
    available_columns: set[str] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    result = candidates.copy()
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""

    original_columns = set(result.columns) if available_columns is None else set(available_columns)

    for column in ["is_suspended", "is_limit_up_close", "is_limit_down_close"]:
        if column in result.columns:
            result[column] = _to_bool_series(result[column])
    for column in [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio_daily_basic",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    daily_basic_key_columns = [
        column
        for column in ["volume_ratio_daily_basic", "total_mv", "circ_mv"]
        if column in original_columns
    ]
    if daily_basic_key_columns:
        missing_daily_basic = result[daily_basic_key_columns].isna().any(axis=1)
        result = _append_flag_where(result, missing_daily_basic, "missing_daily_basic")

    market_value_columns = [column for column in ["total_mv", "circ_mv"] if column in original_columns]
    if market_value_columns:
        missing_market_value = result[market_value_columns].isna().any(axis=1)
        result = _append_flag_where(result, missing_market_value, "missing_market_value")

    if "volume_ratio_daily_basic" in original_columns:
        result = _append_flag_where(
            result,
            result["volume_ratio_daily_basic"].isna(),
            "missing_volume_ratio_daily_basic",
        )

    if "turnover_rate" in original_columns:
        result = _append_flag_where(result, result["turnover_rate"].isna(), "missing_turnover_rate")

    limit_columns = [column for column in ["up_limit", "down_limit"] if column in original_columns]
    if limit_columns:
        missing_limit_data = result[limit_columns].isna().any(axis=1)
        result = _append_flag_where(result, missing_limit_data, "missing_limit_data")

    if "is_suspended" in result.columns:
        result = _append_flag_where(result, result["is_suspended"], "suspended")
        if exclude_suspended:
            result = result[~result["is_suspended"]].copy()
    if "is_limit_up_close" in result.columns:
        result = _append_flag_where(result, result["is_limit_up_close"], "limit_up_close")
        if exclude_limit_up_close:
            result = result[~result["is_limit_up_close"]].copy()
    if "is_limit_down_close" in result.columns:
        result = _append_flag_where(result, result["is_limit_down_close"], "limit_down_close")
        if exclude_limit_down_close:
            result = result[~result["is_limit_down_close"]].copy()

    if "turnover_rate" in result.columns:
        if min_turnover_rate is not None:
            low_turnover = result["turnover_rate"].notna() & (result["turnover_rate"] < min_turnover_rate)
            result = _append_flag_where(result, low_turnover, "low_turnover")
            result = result[~low_turnover].copy()
        if max_turnover_rate is not None:
            high_turnover = result["turnover_rate"].notna() & (result["turnover_rate"] > max_turnover_rate)
            result = _append_flag_where(result, high_turnover, "high_turnover")
            result = result[~high_turnover].copy()

    if "circ_mv" in result.columns:
        if min_circ_mv is not None:
            small_circ_mv = result["circ_mv"].notna() & (result["circ_mv"] < min_circ_mv)
            result = _append_flag_where(result, small_circ_mv, "small_circ_mv")
            result = result[~small_circ_mv].copy()
        if max_circ_mv is not None:
            large_circ_mv = result["circ_mv"].notna() & (result["circ_mv"] > max_circ_mv)
            result = _append_flag_where(result, large_circ_mv, "large_circ_mv")
            result = result[~large_circ_mv].copy()

    return result


def _apply_industry_adjustment(candidates: pd.DataFrame, available_columns: set[str] | None = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    result = candidates.copy()
    original_columns = set(result.columns) if available_columns is None else set(available_columns)
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""
    if "score" not in result.columns:
        result["score"] = 0
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)

    if "industry_strength_score" not in original_columns and "industry_strength_level" not in original_columns:
        result = _append_flag_where(result, pd.Series(True, index=result.index), "missing_industry_strength")
        return result

    if "industry_strength_score" in result.columns:
        result["industry_strength_score"] = pd.to_numeric(result["industry_strength_score"], errors="coerce")
    else:
        result["industry_strength_score"] = pd.NA
    if "industry_strength_level" not in result.columns:
        result["industry_strength_level"] = None
    levels = result["industry_strength_level"].fillna("").astype(str).str.strip().str.lower()
    missing = result["industry_strength_score"].isna() & levels.eq("")
    result = _append_flag_where(result, missing, "missing_industry_strength")
    strong = levels.eq("strong")
    weak = levels.eq("weak")
    result.loc[strong, "score"] = result.loc[strong, "score"] + 5
    result.loc[weak, "score"] = result.loc[weak, "score"] - 5
    result = _append_flag_where(result, strong, "strong_industry")
    result = _append_flag_where(result, weak, "weak_industry")

    if "industry_risk_flags" in result.columns:
        industry_flags = result["industry_risk_flags"].fillna("").astype(str)
        for flag in ["strong_industry", "weak_industry", "shrinking_amount", "high_activity"]:
            result = _append_flag_where(result, _contains_csv_flag(industry_flags, flag), flag)
    return result


def _apply_moneyflow_adjustment(candidates: pd.DataFrame, available_columns: set[str] | None = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    result = candidates.copy()
    original_columns = set(result.columns) if available_columns is None else set(available_columns)
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""
    if "score" not in result.columns:
        result["score"] = 0
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)

    if "moneyflow_score" in original_columns:
        moneyflow_score = pd.to_numeric(result.get("moneyflow_score"), errors="coerce")
        result["moneyflow_score"] = moneyflow_score
        result["score"] = result["score"] + moneyflow_score.fillna(0).clip(lower=-20, upper=30)
        result = _append_flag_where(result, moneyflow_score.isna(), "missing_moneyflow")
        result = _append_flag_where(result, moneyflow_score > 20, "strong_main_inflow")
    else:
        result = _append_flag_where(result, pd.Series(True, index=result.index), "missing_moneyflow")

    if "moneyflow_risk_flags" in result.columns:
        moneyflow_flags = result["moneyflow_risk_flags"].fillna("").astype(str)
        for flag in ["main_outflow", "strong_main_outflow", "strong_main_inflow", "missing_moneyflow"]:
            result = _append_flag_where(result, _contains_csv_flag(moneyflow_flags, flag), flag)
    return result


def _append_flag_where(df: pd.DataFrame, mask: pd.Series, flag: str) -> pd.DataFrame:
    if "risk_flags" not in df.columns:
        df["risk_flags"] = ""
    aligned_mask = mask.reindex(df.index, fill_value=False).fillna(False).astype(bool)
    if not aligned_mask.any():
        return df

    risk_flags = df["risk_flags"].fillna("").astype(str).str.strip()
    needs_flag = aligned_mask & ~_contains_csv_flag(risk_flags, flag)
    if not needs_flag.any():
        return df

    existing = risk_flags.loc[needs_flag]
    df.loc[needs_flag, "risk_flags"] = np.where(existing.eq(""), flag, existing + "," + flag)
    return df


def _contains_csv_flag(values: pd.Series, flag: str) -> pd.Series:
    normalized = values.fillna("").astype(str).str.replace(" ", "", regex=False)
    return ("," + normalized + ",").str.contains(f",{flag},", regex=False, na=False)


def _append_flag(value: object, flag: str) -> str:
    flags = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item and item not in flags:
            flags.append(item)
    if flag not in flags:
        flags.append(flag)
    return ",".join(flags)


def _apply_market_regime_adjustment(
    candidates: pd.DataFrame,
    market_regime: pd.DataFrame | pd.Series | dict | None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    row = _market_regime_row(market_regime)
    if row is None:
        return candidates
    result = candidates.copy()
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""
    if "score" not in result.columns:
        result["score"] = 0
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)

    risk_level = str(row.get("risk_level", "") or "").strip().lower()
    regime = str(row.get("market_regime", "") or "").strip().lower()
    if risk_level == "high":
        result["score"] = result["score"] - 5
        result = _append_flag_where(result, pd.Series(True, index=result.index), "market_high_risk")
    elif regime == "strong":
        result["score"] = result["score"] + 3
    return result


def _market_regime_row(market_regime: pd.DataFrame | pd.Series | dict | None) -> dict | None:
    if market_regime is None:
        return None
    if isinstance(market_regime, pd.DataFrame):
        if market_regime.empty:
            return None
        frame = market_regime.copy()
        if "trade_date" in frame.columns:
            frame = frame.sort_values("trade_date")
        return frame.iloc[-1].to_dict()
    if isinstance(market_regime, pd.Series):
        if market_regime.empty:
            return None
        return market_regime.to_dict()
    if isinstance(market_regime, dict):
        return market_regime
    return None


def _to_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "f", "no", "n", "none", "nan"}:
            return False
        if text in {"1", "true", "t", "yes", "y"}:
            return True
    return bool(value)


def _to_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    text = values.fillna("").astype(str).str.strip().str.lower()
    false_values = {"", "0", "false", "f", "no", "n", "none", "nan"}
    true_values = {"1", "true", "t", "yes", "y"}
    default_bool = values.fillna(False).astype(bool)
    return pd.Series(
        np.select(
            [text.isin(false_values), text.isin(true_values)],
            [False, True],
            default=default_bool,
        ),
        index=values.index,
    ).astype(bool)


def _strategy_weights_from_evaluation(signals: pd.DataFrame) -> pd.Series:
    recommendation_weights = {
        "enable_observation": 1.20,
        "observe": 1.00,
        "continue_backtest": 0.80,
        "reduce_or_pause": 0.50,
        "pause": 0.00,
    }
    risk_multipliers = {
        "low": 1.00,
        "medium": 0.85,
        "high": 0.60,
        "unknown": 0.80,
    }
    recommendation = signals["recommendation"].fillna("").astype(str).str.strip()
    risk_level = signals["risk_level"].fillna("").astype(str).str.strip()
    base_weight = recommendation.replace(recommendation_weights)
    risk_multiplier = risk_level.replace(risk_multipliers)
    return (
        pd.to_numeric(base_weight, errors="coerce").fillna(0.80)
        * pd.to_numeric(risk_multiplier, errors="coerce").fillna(0.80)
    )


def _strategy_weight_from_evaluation(row: pd.Series) -> float:
    recommendation_weights = {
        "enable_observation": 1.20,
        "observe": 1.00,
        "continue_backtest": 0.80,
        "reduce_or_pause": 0.50,
        "pause": 0.00,
    }
    risk_multipliers = {
        "low": 1.00,
        "medium": 0.85,
        "high": 0.60,
        "unknown": 0.80,
    }
    recommendation = str(row.get("recommendation") or "").strip()
    risk_level = str(row.get("risk_level") or "unknown").strip() or "unknown"
    base_weight = recommendation_weights.get(recommendation, 0.80)
    risk_multiplier = risk_multipliers.get(risk_level, 0.80)
    return base_weight * risk_multiplier


def _build_signal_candidate_reasons(rows: pd.DataFrame) -> pd.Series:
    versions = rows["strategy_versions"].fillna("").astype(str)
    names = rows["strategy_names"].fillna("").astype(str)
    recommendations = rows["recommendations"].fillna("").astype(str)
    entry_reason = rows["reason"].fillna("").astype(str)

    reason = pd.Series("", index=rows.index, dtype=object)
    strategy_part = pd.Series(
        np.where(names.ne("") | versions.ne(""), "策略版本: " + names + " / " + versions, ""),
        index=rows.index,
    )
    recommendation_part = pd.Series(
        np.where(recommendations.ne(""), "评估建议: " + recommendations, ""),
        index=rows.index,
    )
    for part in [strategy_part, recommendation_part, entry_reason]:
        has_part = part.ne("")
        reason = reason.where(~has_part, np.where(reason.eq(""), part, reason + "；" + part))
    return reason


def _build_signal_candidate_reason(row: pd.Series) -> str:
    parts = []
    versions = str(row.get("strategy_versions") or "")
    names = str(row.get("strategy_names") or "")
    recommendations = str(row.get("recommendations") or "")
    entry_reason = str(row.get("reason") or "")
    if names or versions:
        parts.append(f"策略版本: {names} / {versions}")
    if recommendations:
        parts.append(f"评估建议: {recommendations}")
    if entry_reason:
        parts.append(entry_reason)
    return "；".join(parts)


def _normalize_candidate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "signal_count" in result.columns:
        result["signal_count"] = pd.to_numeric(result["signal_count"], errors="coerce").astype("Int64")
    if "active_signal_count" in result.columns:
        result["active_signal_count"] = pd.to_numeric(result["active_signal_count"], errors="coerce").astype("Int64")
    for column in [
        "max_signal_strength",
        "total_signal_strength",
        "total_weighted_signal_strength",
        "avg_strategy_weight",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)
