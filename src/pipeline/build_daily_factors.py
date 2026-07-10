"""Build local A-share daily technical factors."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import DAILY_FACTOR_COLUMNS, StockAgentStore
from src.factors.technical_factors import compute_daily_factors


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def _build_and_save_daily_factors(db_path: str | None = None) -> tuple[pd.DataFrame, int, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_bars = store.load_daily_bars()
    daily_factors = compute_daily_factors(daily_bars)
    daily_factors = enrich_daily_factors_with_extension_data(
        daily_factors,
        daily_basic=store.load_daily_basic(),
        stock_limits=store.load_stock_limits(),
        suspend_daily=store.load_suspend_daily(),
        moneyflow_factors=store.load_moneyflow_factors(),
        stock_industry_map=store.load_stock_industry_map(),
        industry_strength=store.load_industry_strength(),
    )
    store.save_daily_factors(daily_factors)
    return daily_factors, len(daily_bars), resolved_db_path


def build_daily_factors(db_path: str | None = None) -> pd.DataFrame:
    """Read daily bars, compute technical factors, persist them, and return the result."""
    daily_factors, _, _ = _build_and_save_daily_factors(db_path)
    return daily_factors


def enrich_daily_factors_with_extension_data(
    daily_factors: pd.DataFrame,
    daily_basic: pd.DataFrame | None = None,
    stock_limits: pd.DataFrame | None = None,
    suspend_daily: pd.DataFrame | None = None,
    moneyflow_factors: pd.DataFrame | None = None,
    stock_industry_map: pd.DataFrame | None = None,
    industry_strength: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge optional Tushare extension data into daily factors."""
    if daily_factors.empty:
        result = daily_factors.copy()
        result = _append_extension_columns(result)
        return _ensure_columns(result, DAILY_FACTOR_COLUMNS).loc[:, DAILY_FACTOR_COLUMNS]

    result = daily_factors.copy()
    result = _normalize_merge_keys(result, trade_date=True, code=True)
    if daily_basic is not None and not daily_basic.empty:
        basic_columns = [
            "trade_date",
            "code",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe_ttm",
            "pb",
            "total_mv",
            "circ_mv",
        ]
        basic = _ensure_columns(daily_basic.copy(), basic_columns)
        basic = basic.loc[:, basic_columns].rename(columns={"volume_ratio": "volume_ratio_daily_basic"})
        basic = _normalize_merge_keys(basic, trade_date=True, code=True)
        result = result.merge(
            basic.drop_duplicates(subset=["trade_date", "code"], keep="last"),
            on=["trade_date", "code"],
            how="left",
        )

    if stock_limits is not None and not stock_limits.empty:
        limit_columns = ["trade_date", "code", "up_limit", "down_limit"]
        limits = _ensure_columns(stock_limits.copy(), limit_columns).loc[:, limit_columns]
        limits = _normalize_merge_keys(limits, trade_date=True, code=True)
        result = result.merge(
            limits.drop_duplicates(subset=["trade_date", "code"], keep="last"),
            on=["trade_date", "code"],
            how="left",
        )

    if suspend_daily is not None and not suspend_daily.empty:
        suspend = _ensure_columns(suspend_daily.copy(), ["trade_date", "code", "suspend_type"])
        suspend = _normalize_merge_keys(suspend, trade_date=True, code=True)
        suspend["is_suspended"] = suspend["suspend_type"].fillna("").astype(str).eq("S")
        suspend_flag = (
            suspend.groupby(["trade_date", "code"], as_index=False)["is_suspended"]
            .max()
            .loc[:, ["trade_date", "code", "is_suspended"]]
        )
        result = result.merge(suspend_flag, on=["trade_date", "code"], how="left")

    if moneyflow_factors is not None and not moneyflow_factors.empty:
        moneyflow_columns = [
            "trade_date",
            "code",
            "net_mf_amount",
            "main_net_amount",
            "main_net_amount_ratio",
            "big_net_amount",
            "small_net_amount",
            "moneyflow_score",
            "moneyflow_risk_flags",
        ]
        moneyflow = _ensure_columns(moneyflow_factors.copy(), moneyflow_columns).loc[:, moneyflow_columns]
        moneyflow = _normalize_merge_keys(moneyflow, trade_date=True, code=True)
        result = result.merge(
            moneyflow.drop_duplicates(subset=["trade_date", "code"], keep="last"),
            on=["trade_date", "code"],
            how="left",
        )

    if stock_industry_map is not None and not stock_industry_map.empty:
        map_columns = ["code", "industry_code", "industry_name"]
        mapping = _ensure_columns(stock_industry_map.copy(), map_columns).loc[:, map_columns]
        mapping = _normalize_merge_keys(mapping, code=True, industry_code=True)
        result = result.merge(
            mapping.drop_duplicates(subset=["code"], keep="last"),
            on="code",
            how="left",
        )

    if industry_strength is not None and not industry_strength.empty:
        strength_columns = [
            "trade_date",
            "industry_code",
            "industry_strength_score",
            "industry_strength_level",
            "industry_return_3d",
            "industry_return_5d",
            "industry_amount_ratio_5",
            "industry_risk_flags",
        ]
        strength = _ensure_columns(industry_strength.copy(), strength_columns).loc[:, strength_columns]
        strength = _normalize_merge_keys(strength, trade_date=True, industry_code=True)
        result = result.merge(
            strength.drop_duplicates(subset=["trade_date", "industry_code"], keep="last"),
            on=["trade_date", "industry_code"],
            how="left",
        )

    result = _append_extension_columns(result)

    if "is_suspended" not in result.columns:
        result["is_suspended"] = False
    result["is_suspended"] = result["is_suspended"].fillna(False).astype(bool)

    result["is_limit_up_close"] = (result["close"] >= result["up_limit"] * 0.999).fillna(False)
    result["is_limit_down_close"] = (result["close"] <= result["down_limit"] * 1.001).fillna(False)
    result["limit_up_distance"] = result["up_limit"] / result["close"] - 1
    result["limit_down_distance"] = result["close"] / result["down_limit"] - 1
    result.loc[result["close"] <= 0, ["limit_up_distance", "limit_down_distance"]] = pd.NA
    result.loc[result["down_limit"] <= 0, "limit_down_distance"] = pd.NA
    return _ensure_columns(result, DAILY_FACTOR_COLUMNS).loc[:, DAILY_FACTOR_COLUMNS]


def _append_extension_columns(result: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio_daily_basic",
        "pe_ttm",
        "pb",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
    ]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ["is_suspended", "is_limit_up_close", "is_limit_down_close"]:
        if column not in result.columns:
            result[column] = False
    for column in ["limit_up_distance", "limit_down_distance"]:
        if column not in result.columns:
            result[column] = pd.NA
    for column in [
        "net_mf_amount",
        "main_net_amount",
        "main_net_amount_ratio",
        "big_net_amount",
        "small_net_amount",
        "moneyflow_score",
    ]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "moneyflow_risk_flags" not in result.columns:
        result["moneyflow_risk_flags"] = None
    else:
        result["moneyflow_risk_flags"] = result["moneyflow_risk_flags"].where(
            result["moneyflow_risk_flags"].notna(),
            None,
        )
    for column in ["industry_code", "industry_name", "industry_strength_level", "industry_risk_flags"]:
        if column not in result.columns:
            result[column] = None
        result[column] = result[column].where(result[column].notna(), None)
    for column in [
        "industry_strength_score",
        "industry_return_3d",
        "industry_return_5d",
        "industry_amount_ratio_5",
    ]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df


def _normalize_merge_keys(
    df: pd.DataFrame,
    *,
    trade_date: bool = False,
    code: bool = False,
    industry_code: bool = False,
) -> pd.DataFrame:
    normalized = df.copy()
    if trade_date and "trade_date" in normalized.columns:
        normalized["trade_date"] = _normalize_trade_date_series(normalized["trade_date"])
    if code and "code" in normalized.columns:
        normalized["code"] = _normalize_stock_code_series(normalized["code"])
    if industry_code and "industry_code" in normalized.columns:
        normalized["industry_code"] = normalized["industry_code"].fillna("").astype(str).str.strip()
    return normalized


def _normalize_trade_date_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    digits = values.str.replace(r"\D", "", regex=True)
    normalized = digits.where(digits.str.len() == 8, "")
    needs_parse = normalized.eq("") & values.ne("")
    if needs_parse.any():
        parsed = pd.to_datetime(values[needs_parse], errors="coerce")
        normalized.loc[needs_parse] = parsed.dt.strftime("%Y%m%d").fillna("")
    return normalized


def _normalize_stock_code_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    six_digits = values.str.extract(r"(\d{6})", expand=False)
    digits = values.str.replace(r"\D", "", regex=True)
    normalized = six_digits.where(six_digits.notna(), digits.str.zfill(6).str[-6:])
    return normalized.fillna("").where(digits.ne("") | six_digits.notna(), "")


def _normalize_stock_code(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    if digits:
        return digits.zfill(6)[-6:]
    return ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local A-share daily technical factors.")
    parser.add_argument("--db-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    daily_factors, daily_bars_count, resolved_db_path = _build_and_save_daily_factors(args.db_path)

    print(f"daily_bars 行数: {daily_bars_count}")
    print(f"daily_factors 行数: {len(daily_factors)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 10 条因子数据:")
    print(daily_factors.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
