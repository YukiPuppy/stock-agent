from __future__ import annotations

import pandas as pd


MONEYFLOW_FACTOR_COLUMNS = [
    "trade_date",
    "code",
    "net_mf_amount",
    "net_mf_vol",
    "main_net_amount",
    "main_net_vol",
    "big_net_amount",
    "big_net_vol",
    "small_net_amount",
    "small_net_vol",
    "main_net_amount_ratio",
    "big_net_amount_ratio",
    "small_net_amount_ratio",
    "moneyflow_score",
    "moneyflow_risk_flags",
]

MONEYFLOW_REQUIRED_COLUMNS = [
    "trade_date",
    "code",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount",
]


def build_moneyflow_factors(moneyflow: pd.DataFrame) -> pd.DataFrame:
    if moneyflow is None or moneyflow.empty:
        return pd.DataFrame(columns=MONEYFLOW_FACTOR_COLUMNS)
    if "trade_date" not in moneyflow.columns or "code" not in moneyflow.columns:
        return pd.DataFrame(columns=MONEYFLOW_FACTOR_COLUMNS)

    data = moneyflow.copy()
    missing_input_columns = [column for column in MONEYFLOW_REQUIRED_COLUMNS if column not in data.columns]
    for column in MONEYFLOW_REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    numeric_columns = [column for column in MONEYFLOW_REQUIRED_COLUMNS if column not in {"trade_date", "code"}]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    result = pd.DataFrame()
    result["trade_date"] = data["trade_date"].fillna("").astype(str)
    result["code"] = data["code"].fillna("").astype(str).str.zfill(6)
    result["net_mf_amount"] = data["net_mf_amount"]
    result["net_mf_vol"] = data["net_mf_vol"]
    result["main_net_amount"] = (
        data["buy_lg_amount"].fillna(0)
        + data["buy_elg_amount"].fillna(0)
        - data["sell_lg_amount"].fillna(0)
        - data["sell_elg_amount"].fillna(0)
    )
    result["main_net_vol"] = (
        data["buy_lg_vol"].fillna(0)
        + data["buy_elg_vol"].fillna(0)
        - data["sell_lg_vol"].fillna(0)
        - data["sell_elg_vol"].fillna(0)
    )
    result["big_net_amount"] = data["buy_lg_amount"].fillna(0) - data["sell_lg_amount"].fillna(0)
    result["big_net_vol"] = data["buy_lg_vol"].fillna(0) - data["sell_lg_vol"].fillna(0)
    result["small_net_amount"] = data["buy_sm_amount"].fillna(0) - data["sell_sm_amount"].fillna(0)
    result["small_net_vol"] = data["buy_sm_vol"].fillna(0) - data["sell_sm_vol"].fillna(0)

    total_buy_amount = (
        data["buy_sm_amount"].fillna(0)
        + data["buy_md_amount"].fillna(0)
        + data["buy_lg_amount"].fillna(0)
        + data["buy_elg_amount"].fillna(0)
    )
    denominator = total_buy_amount.clip(lower=1)
    result["main_net_amount_ratio"] = result["main_net_amount"] / denominator
    result["big_net_amount_ratio"] = result["big_net_amount"] / denominator
    result["small_net_amount_ratio"] = result["small_net_amount"] / denominator

    score = pd.Series(0, index=result.index, dtype="float64")
    score += (result["main_net_amount"] > 0).astype(int) * 10
    score += (result["main_net_amount_ratio"] > 0.05).astype(int) * 10
    score += (result["main_net_amount_ratio"] > 0.10).astype(int) * 10
    score += (result["big_net_amount"] > 0).astype(int) * 5
    score += (result["net_mf_amount"] > 0).astype(int) * 5
    score -= (result["main_net_amount"] < 0).astype(int) * 10
    score -= (result["main_net_amount_ratio"] < -0.05).astype(int) * 10
    score -= (result["main_net_amount_ratio"] < -0.10).astype(int) * 10
    score -= ((result["small_net_amount"] > 0) & (result["main_net_amount"] < 0)).astype(int) * 5
    result["moneyflow_score"] = score

    missing_moneyflow = data[MONEYFLOW_REQUIRED_COLUMNS].isna().any(axis=1) | bool(missing_input_columns)
    result["moneyflow_risk_flags"] = [
        _risk_flags(row, missing)
        for (_, row), missing in zip(result.iterrows(), missing_moneyflow)
    ]
    return result.loc[:, MONEYFLOW_FACTOR_COLUMNS].sort_values(["trade_date", "code"]).reset_index(drop=True)


def _risk_flags(row: pd.Series, missing_moneyflow: bool) -> str:
    flags = []
    if row["main_net_amount"] < 0:
        flags.append("main_outflow")
    if row["main_net_amount_ratio"] < -0.10:
        flags.append("strong_main_outflow")
    if row["moneyflow_score"] < 0:
        flags.append("weak_moneyflow")
    if row["main_net_amount_ratio"] > 0.10:
        flags.append("strong_main_inflow")
    if missing_moneyflow:
        flags.append("missing_moneyflow")
    return ",".join(flags)
