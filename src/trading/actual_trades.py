from __future__ import annotations

import re

import pandas as pd


ACTUAL_TRADE_COLUMNS = [
    "trade_date",
    "trade_time",
    "code",
    "name",
    "side",
    "price",
    "volume",
    "amount",
    "position_ratio",
    "strategy_name",
    "plan_rank",
    "is_follow_plan",
    "reason",
    "note",
]

NUMERIC_COLUMNS = ["price", "volume", "amount", "position_ratio", "plan_rank"]


def normalize_actual_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize user-entered actual trade records into a stable schema."""
    if df.empty:
        return pd.DataFrame(columns=ACTUAL_TRADE_COLUMNS)

    normalized = df.copy()
    for column in ACTUAL_TRADE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized.loc[:, ACTUAL_TRADE_COLUMNS].copy()
    normalized["trade_date"] = normalized["trade_date"].map(_normalize_text)
    normalized["trade_time"] = normalized["trade_time"].map(_normalize_optional_text)
    normalized["code"] = normalized["code"].map(_normalize_code)
    normalized["name"] = normalized["name"].map(_normalize_optional_text)
    normalized["side"] = normalized["side"].map(_normalize_side)

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    missing_amount = normalized["amount"].isna()
    calculated_amount = normalized["price"] * normalized["volume"]
    normalized.loc[missing_amount, "amount"] = calculated_amount.loc[missing_amount]

    for column in ["strategy_name", "reason", "note"]:
        normalized[column] = normalized[column].map(_normalize_optional_text)

    return normalized


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):06d}"
    text = str(value).strip()
    if not text:
        return ""
    digits = re.findall(r"\d+", text)
    if not digits:
        return ""
    joined = "".join(digits)
    if len(joined) >= 6:
        return joined[:6]
    return joined.zfill(6)


def _normalize_side(value: object) -> str:
    text = _normalize_optional_text(value).lower()
    return text if text in {"buy", "sell"} else "unknown"


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_optional_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
