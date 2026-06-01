"""Tradable universe filters for A-share stocks."""

from __future__ import annotations

import re

import pandas as pd


MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
CHINEXT_PREFIXES = ("300", "301")
STAR_MARKET_PREFIXES = ("688", "689")
BEIJING_PREFIXES = ("8", "4", "9")


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    match = re.search(r"\d{1,6}", text)
    if not match:
        return ""

    return match.group(0).zfill(6)


def filter_tradable_main_board(df: pd.DataFrame) -> pd.DataFrame:
    """Filter an input stock table down to tradable Shanghai/Shenzhen main-board names."""
    if "code" not in df.columns:
        raise ValueError("df must contain a 'code' column")

    codes = df["code"].map(_normalize_code)

    is_main_board = codes.str.startswith(MAIN_BOARD_PREFIXES, na=False)
    is_chinext = codes.str.startswith(CHINEXT_PREFIXES, na=False)
    is_star_market = codes.str.startswith(STAR_MARKET_PREFIXES, na=False)
    is_beijing = codes.str.startswith(BEIJING_PREFIXES, na=False)

    if "market" in df.columns:
        market = df["market"].fillna("").astype(str).str.upper()
        is_beijing |= market.str.contains("BJ", regex=False, na=False)

    if "board" in df.columns:
        board = df["board"].fillna("").astype(str).str.upper()
        is_beijing |= board.str.contains("BJ", regex=False, na=False)
        is_beijing |= board.str.contains("北交所", regex=False, na=False)

    mask = is_main_board & ~is_chinext & ~is_star_market & ~is_beijing

    if "name" in df.columns:
        name = df["name"].fillna("").astype(str).str.upper()
        mask &= ~name.str.contains("ST", regex=False, na=False)
        mask &= ~name.str.contains("退", regex=False, na=False)

    if "list_status" in df.columns:
        list_status = df["list_status"].fillna("").astype(str).str.upper().str.strip()
        mask &= list_status.eq("L")

    if "paused" in df.columns:
        paused = df["paused"].fillna(False).astype(bool)
        mask &= ~paused

    return df.loc[mask].copy()
