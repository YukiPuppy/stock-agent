from __future__ import annotations

import numpy as np
import pandas as pd


INDUSTRY_STRENGTH_COLUMNS = [
    "trade_date",
    "industry_code",
    "industry_name",
    "close",
    "pct_change",
    "amount",
    "industry_return_3d",
    "industry_return_5d",
    "industry_amount_ratio_5",
    "industry_above_ma5",
    "industry_above_ma10",
    "industry_rank_pct_change",
    "industry_rank_return_5d",
    "industry_rank_amount",
    "industry_strength_score",
    "industry_strength_level",
    "industry_risk_flags",
]


def build_industry_strength(sw_daily: pd.DataFrame) -> pd.DataFrame:
    """Build SW industry strength factors from normalized sw_daily data."""
    if sw_daily.empty:
        return pd.DataFrame(columns=INDUSTRY_STRENGTH_COLUMNS)

    data = sw_daily.copy()
    for column in ["trade_date", "industry_code", "industry_name"]:
        if column not in data.columns:
            raise ValueError(f"sw_daily missing required column: {column}")
    for column in ["close", "pct_change", "amount"]:
        if column not in data.columns:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    data = data.sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    grouped = data.groupby("industry_code", group_keys=False)
    data["industry_return_3d"] = grouped["close"].pct_change(periods=3)
    data["industry_return_5d"] = grouped["close"].pct_change(periods=5)
    amount_ma5 = grouped["amount"].transform(lambda series: series.rolling(5, min_periods=1).mean())
    ma5 = grouped["close"].transform(lambda series: series.rolling(5, min_periods=1).mean())
    ma10 = grouped["close"].transform(lambda series: series.rolling(10, min_periods=1).mean())
    data["industry_amount_ratio_5"] = data["amount"] / amount_ma5
    data["industry_above_ma5"] = (data["close"] > ma5).fillna(False)
    data["industry_above_ma10"] = (data["close"] > ma10).fillna(False)

    data["industry_rank_pct_change"] = data.groupby("trade_date")["pct_change"].rank(pct=True, ascending=False)
    data["industry_rank_return_5d"] = data.groupby("trade_date")["industry_return_5d"].rank(pct=True, ascending=False)
    data["industry_rank_amount"] = data.groupby("trade_date")["amount"].rank(pct=True, ascending=False)

    score = pd.Series(0, index=data.index, dtype="float")
    score += (data["pct_change"] > 0).astype(int) * 10
    score += (data["industry_return_3d"] > 0).astype(int) * 10
    score += (data["industry_return_5d"] > 0).astype(int) * 15
    score += data["industry_above_ma5"].astype(int) * 10
    score += data["industry_above_ma10"].astype(int) * 10
    score += (data["industry_amount_ratio_5"] > 1.2).astype(int) * 10
    score += (data["industry_rank_pct_change"] <= 0.20).astype(int) * 15
    score += (data["industry_rank_return_5d"] <= 0.20).astype(int) * 15
    score -= (data["industry_return_5d"] < -0.03).astype(int) * 15
    score -= (~data["industry_above_ma5"] & ~data["industry_above_ma10"]).astype(int) * 10
    data["industry_strength_score"] = score
    data["industry_strength_level"] = "weak"
    data.loc[data["industry_strength_score"] >= 30, "industry_strength_level"] = "neutral"
    data.loc[data["industry_strength_score"] >= 60, "industry_strength_level"] = "strong"
    data["industry_risk_flags"] = ""
    data = _append_flag_where(data, data["industry_strength_level"].eq("weak"), "weak_industry")
    data = _append_flag_where(data, data["industry_strength_level"].eq("strong"), "strong_industry")
    amount_ratio = pd.to_numeric(data["industry_amount_ratio_5"], errors="coerce")
    data = _append_flag_where(data, amount_ratio < 0.8, "shrinking_amount")
    data = _append_flag_where(data, amount_ratio > 1.2, "high_activity")

    return data.loc[:, INDUSTRY_STRENGTH_COLUMNS].sort_values(["trade_date", "industry_code"]).reset_index(drop=True)


def _append_flag_where(df: pd.DataFrame, mask: pd.Series, flag: str) -> pd.DataFrame:
    aligned_mask = mask.reindex(df.index, fill_value=False).fillna(False).astype(bool)
    if not aligned_mask.any():
        return df
    existing = df.loc[aligned_mask, "industry_risk_flags"].fillna("").astype(str).str.strip()
    df.loc[aligned_mask, "industry_risk_flags"] = np.where(existing.eq(""), flag, existing + "," + flag)
    return df


def _risk_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if row.get("industry_strength_level") == "weak":
        flags.append("weak_industry")
    if row.get("industry_strength_level") == "strong":
        flags.append("strong_industry")
    amount_ratio = row.get("industry_amount_ratio_5")
    if pd.notna(amount_ratio) and float(amount_ratio) < 0.8:
        flags.append("shrinking_amount")
    if pd.notna(amount_ratio) and float(amount_ratio) > 1.2:
        flags.append("high_activity")
    return ",".join(flags)
