from __future__ import annotations

import pandas as pd


STOCK_INDUSTRY_MAP_COLUMNS = [
    "code",
    "name",
    "industry_name",
    "industry_code",
    "industry_level",
    "source",
]


def build_stock_industry_map(
    stock_basic: pd.DataFrame,
    sw_industry_classification: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a best-effort stock to SW industry map from local stock_basic fields."""
    if stock_basic.empty:
        return pd.DataFrame(columns=STOCK_INDUSTRY_MAP_COLUMNS)

    basic = stock_basic.copy()
    for column in ["code", "name"]:
        if column not in basic.columns:
            basic[column] = ""

    industry_column = _first_existing_column(basic, ("industry_name", "industry", "board"))
    result = pd.DataFrame()
    result["code"] = basic["code"].fillna("").astype(str).str.strip()
    result["name"] = basic["name"].fillna("").astype(str).str.strip()
    result["industry_name"] = (
        basic[industry_column].fillna("").astype(str).str.strip() if industry_column is not None else ""
    )
    result["industry_code"] = ""
    result["industry_level"] = ""
    result["source"] = "stock_basic_only"

    if sw_industry_classification is not None and not sw_industry_classification.empty:
        sw = sw_industry_classification.copy()
        for column in ["industry_name", "industry_code", "level"]:
            if column not in sw.columns:
                sw[column] = ""
        sw = sw.drop_duplicates(subset=["industry_name"], keep="last")
        result = result.merge(
            sw.loc[:, ["industry_name", "industry_code", "level"]],
            on="industry_name",
            how="left",
            suffixes=("", "_sw"),
        )
        matched = result["industry_code_sw"].fillna("").astype(str).str.strip() != ""
        result["industry_code"] = result["industry_code_sw"].fillna("").astype(str).str.strip()
        result["industry_level"] = result["level"].fillna("").astype(str).str.strip()
        result["source"] = "stock_basic_sw_name_match"
        result.loc[~matched, "source"] = "stock_basic_only"
        result = result.drop(columns=["industry_code_sw", "level"])

    return result.loc[:, STOCK_INDUSTRY_MAP_COLUMNS].drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None
