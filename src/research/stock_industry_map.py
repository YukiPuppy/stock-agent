from __future__ import annotations

import re

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
    sw_industry_components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a best-effort stock to SW industry map from local stock_basic fields."""
    if stock_basic.empty:
        return pd.DataFrame(columns=STOCK_INDUSTRY_MAP_COLUMNS)

    basic = stock_basic.copy()
    for column in ["code", "name"]:
        if column not in basic.columns:
            basic[column] = ""

    industry_column = _first_non_empty_column(basic, ("industry", "industry_name", "sw_industry", "board"))
    result = pd.DataFrame()
    result["code"] = basic["code"].fillna("").astype(str).str.strip().map(_normalize_stock_code)
    result["name"] = basic["name"].fillna("").astype(str).str.strip()
    result["industry_name"] = (
        basic[industry_column].fillna("").astype(str).str.strip() if industry_column is not None else ""
    )
    result["industry_code"] = ""
    result["industry_level"] = ""
    result["source"] = "stock_basic_only"

    if sw_industry_components is not None and not sw_industry_components.empty:
        components = _normalize_sw_industry_components(sw_industry_components)
        if not components.empty:
            result = result.merge(
                components,
                on="code",
                how="left",
                suffixes=("", "_component"),
            )
            matched = result["industry_code_component"].fillna("").astype(str).str.strip() != ""
            result.loc[matched, "industry_name"] = result.loc[matched, "industry_name_component"]
            result.loc[matched, "industry_code"] = result.loc[matched, "industry_code_component"]
            result.loc[matched, "industry_level"] = result.loc[matched, "industry_level_component"]
            result.loc[matched, "source"] = "sw_component_code_match"
            result = result.drop(
                columns=["industry_name_component", "industry_code_component", "industry_level_component"]
            )

    if sw_industry_classification is not None and not sw_industry_classification.empty:
        sw = _normalize_sw_industry_classification(sw_industry_classification)
        sw = sw.drop_duplicates(subset=["industry_name"], keep="last")
        unmatched = result["industry_code"].fillna("").astype(str).str.strip() == ""
        name_candidates = result.loc[unmatched].copy()
        name_candidates["_row_id"] = name_candidates.index
        name_matched = name_candidates.merge(
            sw.loc[:, ["industry_name", "industry_code", "level"]],
            on="industry_name",
            how="left",
            suffixes=("", "_sw"),
        )
        matched = name_matched["industry_code_sw"].fillna("").astype(str).str.strip() != ""
        matched_index = name_matched.loc[matched, "_row_id"]
        result.loc[matched_index, "industry_code"] = name_matched.loc[matched, "industry_code_sw"].astype(str).str.strip().to_numpy()
        result.loc[matched_index, "industry_level"] = name_matched.loc[matched, "level"].fillna("").astype(str).str.strip().to_numpy()
        result.loc[matched_index, "source"] = "stock_basic_sw_name_match"

    return result.loc[:, STOCK_INDUSTRY_MAP_COLUMNS].drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)


def _first_non_empty_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns and df[column].fillna("").astype(str).str.strip().ne("").any():
            return column
    return None


def _normalize_stock_code(value: str) -> str:
    match = re.search(r"\d{6}", value)
    if match:
        return match.group(0)
    digits = re.sub(r"\D", "", value)
    if digits:
        return digits.zfill(6)[-6:]
    return ""


def _normalize_sw_industry_components(df: pd.DataFrame) -> pd.DataFrame:
    components = df.copy()
    for column in ["code", "industry_code", "industry_name", "industry_level"]:
        if column not in components.columns:
            components[column] = ""
    result = components.loc[:, ["code", "industry_code", "industry_name", "industry_level"]].copy()
    result["code"] = result["code"].fillna("").astype(str).str.strip().map(_normalize_stock_code)
    result["industry_code"] = result["industry_code"].fillna("").astype(str).str.strip()
    result["industry_name"] = result["industry_name"].fillna("").astype(str).str.strip()
    result["industry_level"] = result["industry_level"].fillna("").astype(str).str.strip()
    result = result[(result["code"] != "") & (result["industry_code"] != "")]
    return result.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)


def _normalize_sw_industry_classification(df: pd.DataFrame) -> pd.DataFrame:
    classification = df.copy()
    code_column = _first_non_empty_column(classification, ("industry_code", "index_code", "sw_code"))
    if "industry_name" not in classification.columns:
        classification["industry_name"] = ""
    if "level" not in classification.columns:
        classification["level"] = ""
    result = pd.DataFrame()
    result["industry_name"] = classification["industry_name"].fillna("").astype(str).str.strip()
    result["industry_code"] = (
        classification[code_column].fillna("").astype(str).str.strip() if code_column is not None else ""
    )
    result["level"] = classification["level"].fillna("").astype(str).str.strip()
    return result[(result["industry_name"] != "") & (result["industry_code"] != "")]
