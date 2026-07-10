from __future__ import annotations

import pandas as pd


TRADE_DATE_KEY_COLUMN = "_trade_date_key"


def normalize_trade_date_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    digits = values.str.replace(r"\D", "", regex=True)
    normalized = digits.where(digits.str.len() == 8, "")
    needs_parse = normalized.eq("") & values.ne("")
    if needs_parse.any():
        parsed = pd.to_datetime(values[needs_parse], errors="coerce")
        normalized.loc[needs_parse] = parsed.dt.strftime("%Y%m%d").fillna("")
    return normalized


def normalize_trade_date_value(value: object) -> str:
    return normalize_trade_date_series(pd.Series([value])).iloc[0]
