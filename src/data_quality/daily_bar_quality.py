"""Daily bar data quality checks."""

from __future__ import annotations

import pandas as pd


QUALITY_COLUMNS = ["check_name", "status", "issue_count", "message"]
REQUIRED_COLUMNS = ["trade_date", "code", "open", "high", "low", "close", "volume", "amount"]
KEY_COLUMNS = REQUIRED_COLUMNS
PRICE_COLUMNS = ["open", "high", "low", "close"]
ENRICHED_DAILY_FACTOR_COLUMNS = ["volume_ratio_daily_basic", "total_mv", "circ_mv"]
MONEYFLOW_FACTOR_COLUMNS = ["moneyflow_score", "main_net_amount", "main_net_amount_ratio"]
INDUSTRY_STRENGTH_COLUMNS = ["industry_strength_score", "industry_strength_level", "industry_return_5d"]


def check_daily_bars_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    rows.append(_row("empty_data", "error" if data.empty else "ok", int(data.empty), "daily_bars is empty" if data.empty else "daily_bars has data"))

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    rows.append(
        _row(
            "required_columns",
            "error" if missing else "ok",
            len(missing),
            f"missing required columns: {', '.join(missing)}" if missing else "all required columns exist",
        )
    )

    if missing or data.empty:
        for check_name in [
            "null_values",
            "duplicated_rows",
            "invalid_price_relation",
            "non_positive_price",
            "negative_volume_amount",
            "trade_date_format",
            "code_format",
        ]:
            rows.append(_row(check_name, "ok", 0, "skipped because data is empty or required columns are missing"))
        return pd.DataFrame(rows, columns=QUALITY_COLUMNS)

    null_count = int(data[KEY_COLUMNS].isna().sum().sum())
    rows.append(_row("null_values", "warning" if null_count else "ok", null_count, f"{null_count} null values found" if null_count else "no null values"))

    duplicated_count = int(data.duplicated(subset=["trade_date", "code"], keep=False).sum())
    rows.append(
        _row(
            "duplicated_rows",
            "warning" if duplicated_count else "ok",
            duplicated_count,
            f"{duplicated_count} duplicated trade_date/code rows found" if duplicated_count else "no duplicated trade_date/code rows",
        )
    )

    prices = data[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_relation = (
        (prices["high"] < prices["low"])
        | (prices["open"] < prices["low"])
        | (prices["open"] > prices["high"])
        | (prices["close"] < prices["low"])
        | (prices["close"] > prices["high"])
    )
    invalid_relation_count = int(invalid_relation.fillna(False).sum())
    rows.append(
        _row(
            "invalid_price_relation",
            "warning" if invalid_relation_count else "ok",
            invalid_relation_count,
            f"{invalid_relation_count} rows with invalid OHLC relation" if invalid_relation_count else "OHLC relations are valid",
        )
    )

    non_positive_count = int((prices <= 0).sum().sum())
    rows.append(
        _row(
            "non_positive_price",
            "warning" if non_positive_count else "ok",
            non_positive_count,
            f"{non_positive_count} non-positive price values found" if non_positive_count else "all price values are positive",
        )
    )

    volume_amount = data[["volume", "amount"]].apply(pd.to_numeric, errors="coerce")
    negative_count = int((volume_amount < 0).sum().sum())
    rows.append(
        _row(
            "negative_volume_amount",
            "warning" if negative_count else "ok",
            negative_count,
            f"{negative_count} negative volume/amount values found" if negative_count else "volume and amount are non-negative",
        )
    )

    parsed_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    date_invalid_count = int(parsed_dates.isna().sum())
    rows.append(
        _row(
            "trade_date_format",
            "warning" if date_invalid_count else "ok",
            date_invalid_count,
            f"{date_invalid_count} invalid trade_date values found" if date_invalid_count else "trade_date values are parseable",
        )
    )

    code_invalid = ~data["code"].astype(str).str.fullmatch(r"\d{6}", na=False)
    code_invalid_count = int(code_invalid.sum())
    rows.append(
        _row(
            "code_format",
            "warning" if code_invalid_count else "ok",
            code_invalid_count,
            f"{code_invalid_count} invalid code values found" if code_invalid_count else "code values are 6-digit strings",
        )
    )

    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def check_enriched_daily_factors_quality(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    present_columns = [column for column in ENRICHED_DAILY_FACTOR_COLUMNS if column in data.columns]
    if data.empty:
        return pd.DataFrame(
            [
                _row(
                    "enriched_daily_factors_missing_rate",
                    "info",
                    0,
                    "daily_factors is empty; enriched missing rate check skipped",
                )
            ],
            columns=QUALITY_COLUMNS,
        )
    if not present_columns:
        return pd.DataFrame(
            [
                _row(
                    "enriched_daily_factors_missing_rate",
                    "info",
                    0,
                    "daily_factors has no daily_basic extension columns; missing rate check skipped",
                )
            ],
            columns=QUALITY_COLUMNS,
        )

    missing_count = int(data[present_columns].isna().any(axis=1).sum())
    missing_rate = missing_count / len(data) if len(data) else 0.0
    status = "warning" if missing_rate > 0.30 else "ok"
    message = (
        "daily_basic extension missing rate "
        f"{missing_rate:.1%} ({missing_count}/{len(data)} rows) "
        f"for columns: {', '.join(present_columns)}"
    )
    return pd.DataFrame(
        [_row("enriched_daily_factors_missing_rate", status, missing_count, message)],
        columns=QUALITY_COLUMNS,
    )


def check_moneyflow_quality(moneyflow: pd.DataFrame, daily_factors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    moneyflow_data = moneyflow.copy() if isinstance(moneyflow, pd.DataFrame) else pd.DataFrame()
    factors = daily_factors.copy() if isinstance(daily_factors, pd.DataFrame) else pd.DataFrame()
    rows.append(
        _row(
            "moneyflow_table_status",
            "warning" if moneyflow_data.empty else "ok",
            int(moneyflow_data.empty),
            "moneyflow is empty; moneyflow factors are optional at this stage"
            if moneyflow_data.empty
            else "moneyflow has data",
        )
    )
    present_columns = [column for column in MONEYFLOW_FACTOR_COLUMNS if column in factors.columns]
    if factors.empty:
        rows.append(
            _row(
                "moneyflow_factors_missing_rate",
                "info",
                0,
                "daily_factors is empty; moneyflow missing rate check skipped",
            )
        )
    elif not present_columns:
        rows.append(
            _row(
                "moneyflow_factors_missing_rate",
                "warning",
                len(factors),
                "daily_factors has no moneyflow factor columns",
            )
        )
    else:
        missing_count = int(factors[present_columns].isna().any(axis=1).sum())
        missing_rate = missing_count / len(factors) if len(factors) else 0.0
        rows.append(
            _row(
                "moneyflow_factors_missing_rate",
                "warning" if missing_rate > 0.50 else "ok",
                missing_count,
                f"moneyflow factor missing rate {missing_rate:.1%} ({missing_count}/{len(factors)} rows)",
            )
        )
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def check_industry_strength_quality(stock_industry_map: pd.DataFrame, daily_factors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stock_map = stock_industry_map.copy() if isinstance(stock_industry_map, pd.DataFrame) else pd.DataFrame()
    factors = daily_factors.copy() if isinstance(daily_factors, pd.DataFrame) else pd.DataFrame()
    rows.append(
        _row(
            "stock_industry_map_status",
            "warning" if stock_map.empty else "ok",
            int(stock_map.empty),
            "stock_industry_map is empty; industry strength is optional at this stage"
            if stock_map.empty
            else "stock_industry_map has data",
        )
    )
    present_columns = [column for column in INDUSTRY_STRENGTH_COLUMNS if column in factors.columns]
    if factors.empty:
        rows.append(
            _row(
                "industry_strength_missing_rate",
                "info",
                0,
                "daily_factors is empty; industry strength missing rate check skipped",
            )
        )
    elif not present_columns:
        rows.append(
            _row(
                "industry_strength_missing_rate",
                "warning",
                len(factors),
                "daily_factors has no industry strength columns",
            )
        )
    else:
        missing_count = int(factors[present_columns].isna().any(axis=1).sum())
        missing_rate = missing_count / len(factors) if len(factors) else 0.0
        rows.append(
            _row(
                "industry_strength_missing_rate",
                "warning" if missing_rate > 0.50 else "ok",
                missing_count,
                f"industry strength missing rate {missing_rate:.1%} ({missing_count}/{len(factors)} rows)",
            )
        )
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def check_factor_diagnostics_quality(factor_diagnostics: pd.DataFrame, daily_factors: pd.DataFrame) -> pd.DataFrame:
    diagnostics = factor_diagnostics.copy() if isinstance(factor_diagnostics, pd.DataFrame) else pd.DataFrame()
    factors = daily_factors.copy() if isinstance(daily_factors, pd.DataFrame) else pd.DataFrame()
    if factors.empty:
        return pd.DataFrame(
            [
                _row(
                    "factor_diagnostics_high_missing",
                    "info",
                    0,
                    "daily_factors is empty; factor diagnostics check skipped",
                )
            ],
            columns=QUALITY_COLUMNS,
        )
    if diagnostics.empty or "diagnostic_status" not in diagnostics.columns:
        return pd.DataFrame(
            [
                _row(
                    "factor_diagnostics_high_missing",
                    "warning",
                    0,
                    "factor_diagnostics is empty; run build_factor_diagnostics for factor coverage checks",
                )
            ],
            columns=QUALITY_COLUMNS,
        )
    high_missing_count = int((diagnostics["diagnostic_status"].fillna("").astype(str) == "high_missing").sum())
    status = "warning" if high_missing_count >= 3 else "ok"
    return pd.DataFrame(
        [
            _row(
                "factor_diagnostics_high_missing",
                status,
                high_missing_count,
                f"factor_diagnostics high_missing count={high_missing_count}",
            )
        ],
        columns=QUALITY_COLUMNS,
    )


def _row(check_name: str, status: str, issue_count: int, message: str) -> dict[str, object]:
    return {
        "check_name": check_name,
        "status": status,
        "issue_count": int(issue_count),
        "message": message,
    }
