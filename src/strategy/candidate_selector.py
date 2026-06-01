from __future__ import annotations

import pandas as pd


CANDIDATE_COLUMNS = [
    "trade_date",
    "code",
    "name",
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
    "score",
    "rank",
    "reason",
]


def select_candidates(
    daily_factors: pd.DataFrame,
    stock_basic: pd.DataFrame | None = None,
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
) -> pd.DataFrame:
    """Select and score deterministic A-share candidate stocks."""
    if daily_factors.empty or "trade_date" not in daily_factors.columns:
        return _empty_candidates()

    factors = daily_factors.copy()
    selected_trade_date = trade_date
    if selected_trade_date is None:
        trade_dates = factors["trade_date"].dropna()
        if trade_dates.empty:
            return _empty_candidates()
        selected_trade_date = str(trade_dates.max())

    candidates = factors[factors["trade_date"] == selected_trade_date].copy()
    if candidates.empty:
        return _empty_candidates()

    candidates = _merge_stock_basic(candidates, stock_basic)

    numeric_columns = [
        "close",
        "pct_chg_1d",
        "pct_chg_3d",
        "pct_chg_5d",
        "pct_chg_10d",
        "volume_ratio_5",
        "close_position_20",
        "amount_ma5",
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

    result = candidates.sort_values(["score", "code"], ascending=[False, True]).head(top_n).copy()
    result["rank"] = range(1, len(result) + 1)

    for column in CANDIDATE_COLUMNS:
        if column not in result.columns:
            result[column] = None

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


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)
