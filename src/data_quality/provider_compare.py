"""Compare daily bars from two data providers."""

from __future__ import annotations

import pandas as pd


COMPARE_COLUMNS = [
    "trade_date",
    "code",
    "field",
    "left_value",
    "right_value",
    "relative_diff",
    "status",
    "message",
]
SUMMARY_COLUMNS = ["field", "issue_count", "max_relative_diff", "avg_relative_diff", "status"]
PRICE_FIELDS = ["open", "high", "low", "close"]
VOLUME_AMOUNT_FIELDS = ["volume", "amount"]


def compare_daily_bars(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str = "akshare",
    right_name: str = "tushare",
    price_tolerance: float = 0.01,
    volume_tolerance: float = 0.20,
    amount_tolerance: float = 0.20,
) -> pd.DataFrame:
    """Compare daily bars that are already normalized to system standard units."""
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame) or left.empty or right.empty:
        return pd.DataFrame(columns=COMPARE_COLUMNS)

    required = ["trade_date", "code"]
    if any(column not in left.columns for column in required) or any(column not in right.columns for column in required):
        return pd.DataFrame(columns=COMPARE_COLUMNS)

    fields = [field for field in PRICE_FIELDS + VOLUME_AMOUNT_FIELDS if field in left.columns or field in right.columns]
    left_normalized = _normalize(left, fields)
    right_normalized = _normalize(right, fields)
    merged = left_normalized.merge(
        right_normalized,
        on=["trade_date", "code"],
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )

    rows = []
    for _, row in merged.iterrows():
        merge_status = str(row["_merge"])
        if merge_status == "left_only":
            rows.append(_issue(row, "row", None, None, None, "missing_right", f"missing in {right_name}"))
            continue
        if merge_status == "right_only":
            rows.append(_issue(row, "row", None, None, None, "missing_left", f"missing in {left_name}"))
            continue

        for field in fields:
            left_value = row.get(f"{field}_left")
            right_value = row.get(f"{field}_right")
            diff = _relative_diff(left_value, right_value)
            if pd.isna(diff):
                continue
            tolerance = _tolerance(field, price_tolerance, volume_tolerance, amount_tolerance)
            if float(diff) > tolerance:
                rows.append(
                    _issue(
                        row,
                        field,
                        left_value,
                        right_value,
                        float(diff),
                        "warning",
                        f"{field} relative diff exceeds tolerance between {left_name} and {right_name}",
                    )
                )

    return pd.DataFrame(rows, columns=COMPARE_COLUMNS)


def summarize_provider_compare(compare_result: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(compare_result, pd.DataFrame) or compare_result.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = []
    for field, group in compare_result.groupby("field", dropna=False):
        diffs = pd.to_numeric(group.get("relative_diff"), errors="coerce").dropna()
        issue_count = len(group)
        rows.append(
            {
                "field": field,
                "issue_count": int(issue_count),
                "max_relative_diff": float(diffs.max()) if not diffs.empty else None,
                "avg_relative_diff": float(diffs.mean()) if not diffs.empty else None,
                "status": "warning" if issue_count else "ok",
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(["issue_count", "field"], ascending=[False, True]).reset_index(drop=True)


def _normalize(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in ["trade_date", "code"]:
        normalized[column] = normalized[column].astype(str)
    for field in fields:
        if field not in normalized.columns:
            normalized[field] = None
    return normalized.loc[:, ["trade_date", "code"] + fields].drop_duplicates(subset=["trade_date", "code"], keep="last")


def _relative_diff(left_value: object, right_value: object) -> float:
    left_float = pd.to_numeric(pd.Series([left_value]), errors="coerce").iloc[0]
    right_float = pd.to_numeric(pd.Series([right_value]), errors="coerce").iloc[0]
    if pd.isna(left_float) or pd.isna(right_float):
        return float("nan")
    denominator = max(abs(float(left_float)), abs(float(right_float)), 1e-12)
    return abs(float(left_float) - float(right_float)) / denominator


def _tolerance(field: str, price_tolerance: float, volume_tolerance: float, amount_tolerance: float) -> float:
    if field in PRICE_FIELDS:
        return price_tolerance
    if field == "volume":
        return volume_tolerance
    return amount_tolerance


def _issue(
    row: pd.Series,
    field: str,
    left_value: object,
    right_value: object,
    relative_diff: float | None,
    status: str,
    message: str,
) -> dict[str, object]:
    return {
        "trade_date": row.get("trade_date"),
        "code": row.get("code"),
        "field": field,
        "left_value": left_value,
        "right_value": right_value,
        "relative_diff": relative_diff,
        "status": status,
        "message": message,
    }
