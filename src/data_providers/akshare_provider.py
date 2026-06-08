"""AKShare-backed data provider for A-share market data.

AKShare is kept as a fallback/diagnostic provider. The production data source is Tushare.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import akshare as ak
import pandas as pd

from src.config import settings
from src.data_providers.base import BaseDataProvider
from src.utils.proxy import no_proxy_context


STOCK_BASIC_COLUMNS = ["code", "name", "market", "board", "list_status"]
DAILY_BAR_COLUMNS = [
    "trade_date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
BJ_PREFIXES = ("8", "4", "9")
STAR_PREFIXES = ("688", "689")
CHINEXT_PREFIXES = ("300", "301")
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    match = re.search(r"\d{1,6}", text)
    if not match:
        return ""

    return match.group(0).zfill(6)


def _normalize_akshare_date(value: str) -> str:
    return re.sub(r"\D", "", str(value))


def _to_output_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _infer_market(code: str) -> str:
    if code.startswith(SH_PREFIXES):
        return "SH"
    if code.startswith(SZ_PREFIXES):
        return "SZ"
    if code.startswith(BJ_PREFIXES):
        return "BJ"
    return "UNKNOWN"


def _infer_board(code: str) -> str:
    if code.startswith(STAR_PREFIXES):
        return "科创板"
    if code.startswith(CHINEXT_PREFIXES):
        return "创业板"
    if code.startswith(BJ_PREFIXES):
        return "北交所"
    if code.startswith(MAIN_BOARD_PREFIXES):
        return "主板"
    return "未知"


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _empty_daily_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_BAR_COLUMNS)


def _daily_symbol_for_akshare_daily(code: str) -> str:
    if code.startswith(SH_PREFIXES):
        return f"sh{code}"
    if code.startswith(SZ_PREFIXES):
        return f"sz{code}"
    return code


def _normalize_daily_bars(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw.empty:
        return _empty_daily_bars()

    rename_map = {
        "日期": "trade_date",
        "股票代码": "code",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
    }
    df = raw.rename(columns=rename_map).copy()

    missing_columns = [
        column for column in DAILY_BAR_COLUMNS if column != "code" and column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"AKShare daily bar data missing columns: {missing_columns}")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_to_output_date)
    result["code"] = code

    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(df[column], errors="coerce")

    # AKShare is retained for fallback/diagnostic use; production daily_bars units follow Tushare Pro.
    # Keep fallback output compatible with daily_bars: volume is 手 and amount is 千元.
    result["amount"] = result["amount"] / 1000

    return result.loc[:, DAILY_BAR_COLUMNS]


def _call_with_retries(fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
                return fetch()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1)
    assert last_error is not None
    raise last_error


class AkshareProvider(BaseDataProvider):
    """Fetch and normalize A-share data from AKShare."""

    def get_stock_basic(self) -> pd.DataFrame:
        """Return A-share code/name metadata with normalized provider fields."""
        try:
            with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
                raw = ak.stock_info_a_code_name()
        except Exception as exc:
            raise RuntimeError(f"AKShare stock basic fetch failed in direct/no-proxy mode: {exc}") from exc
        df = raw.copy()

        code_column = _first_existing_column(df, ("code", "代码", "证券代码", "股票代码"))
        name_column = _first_existing_column(df, ("name", "名称", "证券简称", "股票简称"))
        if code_column is None or name_column is None:
            raise ValueError("AKShare stock basic data must contain code and name columns")

        result = pd.DataFrame(
            {
                "code": df[code_column].map(_normalize_code),
                "name": df[name_column],
            }
        )
        result["market"] = result["code"].map(_infer_market)
        result["board"] = result["code"].map(_infer_board)
        result["list_status"] = "L"

        return result.loc[:, STOCK_BASIC_COLUMNS]

    def get_daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized daily bars for a single A-share stock."""
        normalized_code = _normalize_code(code)
        ak_start_date = _normalize_akshare_date(start_date)
        ak_end_date = _normalize_akshare_date(end_date)
        primary_error: str | None = None
        fallback_error: str | None = None
        primary_empty = False
        fallback_empty = False

        try:
            primary = _call_with_retries(
                lambda: ak.stock_zh_a_hist(
                    symbol=normalized_code,
                    period="daily",
                    start_date=ak_start_date,
                    end_date=ak_end_date,
                    adjust="",
                )
            )
            primary_result = _normalize_daily_bars(primary, normalized_code)
            if not primary_result.empty:
                return primary_result
            primary_empty = True
        except Exception as exc:
            primary_error = str(exc)

        fallback_func = getattr(ak, "stock_zh_a_daily", None)
        if fallback_func is None:
            fallback_error = "ak.stock_zh_a_daily is unavailable"
        else:
            try:
                fallback_symbol = _daily_symbol_for_akshare_daily(normalized_code)
                fallback = _call_with_retries(
                    lambda: fallback_func(
                        symbol=fallback_symbol,
                        start_date=ak_start_date,
                        end_date=ak_end_date,
                        adjust="",
                    )
                )
                fallback_result = _normalize_daily_bars(fallback, normalized_code)
                if not fallback_result.empty:
                    return fallback_result
                fallback_empty = True
            except Exception as exc:
                fallback_error = str(exc)

        if primary_error is not None and fallback_error is not None:
            raise RuntimeError(
                "AKShare daily bar fetch failed "
                f"code={normalized_code} start_date={start_date} end_date={end_date} "
                f"primary_error={primary_error} fallback_error={fallback_error}"
            )

        if primary_empty or fallback_empty:
            return _empty_daily_bars()

        raise RuntimeError(
            "AKShare daily bar fetch failed "
            f"code={normalized_code} start_date={start_date} end_date={end_date} "
            f"primary_error={primary_error} fallback_error={fallback_error}"
        )
