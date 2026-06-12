"""Tushare Pro-backed data provider skeleton."""

from __future__ import annotations

import re

import pandas as pd

from src.config import settings
from src.data_providers.base import BaseDataProvider
from src.data_providers.akshare_provider import DAILY_BAR_COLUMNS
from src.utils.proxy import no_proxy_context


STOCK_BASIC_COLUMNS = ["code", "name", "market", "board", "industry", "list_date", "status", "list_status"]
TUSHARE_DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount"
TUSHARE_DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
    "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
    "free_share,total_mv,circ_mv"
)
TRADE_CALENDAR_COLUMNS = ["trade_date", "exchange", "is_open", "pretrade_date"]
DAILY_BASIC_COLUMNS = [
    "trade_date",
    "code",
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]
STOCK_LIMIT_COLUMNS = ["trade_date", "code", "pre_close", "up_limit", "down_limit"]
SUSPEND_DAILY_COLUMNS = ["trade_date", "code", "suspend_type", "suspend_timing"]
INDEX_DAILY_COLUMNS = [
    "trade_date",
    "index_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "volume",
    "amount",
]
LIMIT_LIST_DAILY_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "close",
    "pct_chg",
    "amp",
    "fc_ratio",
    "fl_ratio",
    "fd_amount",
    "first_time",
    "last_time",
    "open_times",
    "strth",
    "limit_type",
    "status",
]
MONEYFLOW_COLUMNS = [
    "trade_date",
    "code",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
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
SW_INDUSTRY_CLASSIFICATION_COLUMNS = [
    "industry_code",
    "industry_name",
    "level",
    "src",
    "parent_code",
    "index_code",
    "is_pub",
    "sort_code",
]
SW_DAILY_COLUMNS = [
    "trade_date",
    "industry_code",
    "industry_name",
    "open",
    "high",
    "low",
    "close",
    "change",
    "pct_change",
    "volume",
    "amount",
    "pe",
    "pb",
    "float_mv",
    "total_mv",
]
TUSHARE_INDEX_DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
TUSHARE_LIMIT_LIST_DAILY_FIELDS = (
    "trade_date,ts_code,name,close,pct_chg,amp,fc_ratio,fl_ratio,fd_amount,"
    "first_time,last_time,open_times,strth,limit_type,status"
)
TUSHARE_MONEYFLOW_FIELDS = "ts_code," + ",".join(column for column in MONEYFLOW_COLUMNS if column != "code")
TUSHARE_SW_DAILY_FIELDS = (
    "ts_code,trade_date,name,open,high,low,close,change,pct_change,vol,amount,pe,pb,float_mv,total_mv"
)
OFFICIAL_TUSHARE_API_URLS = {"http://api.tushare.pro", "https://api.tushare.pro"}
SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
BJ_PREFIXES = ("8", "4", "9")


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"\d{1,6}", str(value).strip().upper())
    if not match:
        return ""
    return match.group(0).zfill(6)


def code_to_ts_code(code: str) -> str:
    """Convert a six-digit A-share code to a Tushare ts_code."""
    normalized = _normalize_code(code)
    if not normalized:
        raise ValueError(f"Invalid stock code: {code}")
    if normalized.startswith(SH_PREFIXES):
        return f"{normalized}.SH"
    if normalized.startswith(SZ_PREFIXES):
        return f"{normalized}.SZ"
    if normalized.startswith(BJ_PREFIXES):
        return f"{normalized}.BJ"
    raise ValueError(f"Unsupported stock code market: {code}")


def is_official_tushare_api_url(url: str) -> bool:
    """Return whether the URL is one of the official Tushare Pro API endpoints."""
    return str(url or "").strip().rstrip("/") in OFFICIAL_TUSHARE_API_URLS


def validate_tushare_api_url(url: str, allow_non_official: bool) -> None:
    """Validate whether a configured Tushare API URL is allowed."""
    if is_official_tushare_api_url(url):
        return
    if not allow_non_official:
        raise ValueError(
            "Non-official Tushare API URL is not allowed unless "
            "TUSHARE_ALLOW_NON_OFFICIAL_API_URL=true"
        )


def ts_code_to_code(ts_code: str) -> str:
    """Convert a Tushare ts_code such as 000001.SZ to a six-digit code."""
    code = _normalize_code(ts_code)
    if not code:
        raise ValueError(f"Invalid ts_code: {ts_code}")
    return code


def _empty_stock_basic() -> pd.DataFrame:
    return pd.DataFrame(columns=STOCK_BASIC_COLUMNS)


def _empty_daily_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_BAR_COLUMNS)


def _empty_with_columns(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _format_trade_date(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d").strftime("%Y-%m-%d")
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _require_columns(df: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Tushare {source_name} data missing columns: {missing}")


def normalize_tushare_stock_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare stock_basic output to stock-agent stock-basic fields."""
    if df.empty:
        return _empty_stock_basic()

    code_column = _first_existing_column(df, ("ts_code", "symbol", "code"))
    name_column = _first_existing_column(df, ("name", "股票简称", "证券简称"))
    if code_column is None:
        raise ValueError("Tushare stock_basic data missing required code column")
    if name_column is None:
        raise ValueError("Tushare stock_basic data missing required name column")

    result = pd.DataFrame()
    result["code"] = df[code_column].map(ts_code_to_code)
    result["name"] = df[name_column]
    if code_column == "ts_code":
        result["market"] = df[code_column].astype(str).str.split(".").str[-1].str.upper()
    else:
        result["market"] = df["market"] if "market" in df.columns else ""
    result["board"] = df["board"] if "board" in df.columns else (df["market"] if "market" in df.columns else "")
    result["industry"] = df["industry"] if "industry" in df.columns else ""
    result["list_date"] = df["list_date"] if "list_date" in df.columns else ""

    status_column = _first_existing_column(df, ("status", "list_status"))
    result["status"] = df[status_column] if status_column is not None else ""
    result["list_status"] = result["status"]

    return result.loc[:, STOCK_BASIC_COLUMNS]


def normalize_tushare_daily_bars(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Normalize Tushare daily/pro_bar output to stock-agent daily-bar fields."""
    if df.empty:
        return _empty_daily_bars()

    normalized_code = _normalize_code(code)
    if not normalized_code:
        raise ValueError(f"Invalid stock code: {code}")

    raw = df.copy()
    volume_column = _first_existing_column(raw, ("volume", "vol"))
    required_columns = ["trade_date", "open", "high", "low", "close", "amount"]
    missing_columns = [column for column in required_columns if column not in raw.columns]
    if volume_column is None:
        missing_columns.append("volume/vol")
    if missing_columns:
        raise ValueError(f"Tushare daily bar data missing columns: {missing_columns}")

    result = pd.DataFrame()
    result["trade_date"] = raw["trade_date"].map(_format_trade_date)
    result["code"] = normalized_code
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(raw[column], errors="coerce")
    result["volume"] = pd.to_numeric(raw[volume_column], errors="coerce")
    result["amount"] = pd.to_numeric(raw["amount"], errors="coerce")

    result = result.loc[:, DAILY_BAR_COLUMNS]
    return result.sort_values("trade_date").reset_index(drop=True)


def normalize_tushare_trade_calendar(df: pd.DataFrame, exchange: str = "SSE") -> pd.DataFrame:
    """Normalize Tushare trade_cal output."""
    if df.empty:
        return _empty_with_columns(TRADE_CALENDAR_COLUMNS)
    _require_columns(df, ["cal_date", "is_open"], "trade_cal")

    result = pd.DataFrame()
    result["trade_date"] = df["cal_date"].map(_format_trade_date)
    result["exchange"] = df["exchange"] if "exchange" in df.columns else exchange
    result["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(int)
    result["pretrade_date"] = df["pretrade_date"].map(_format_trade_date) if "pretrade_date" in df.columns else ""
    return result.loc[:, TRADE_CALENDAR_COLUMNS].sort_values(["trade_date", "exchange"]).reset_index(drop=True)


def normalize_tushare_daily_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare daily_basic output."""
    if df.empty:
        return _empty_with_columns(DAILY_BASIC_COLUMNS)
    _require_columns(df, ["ts_code", "trade_date"], "daily_basic")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["code"] = df["ts_code"].map(ts_code_to_code)
    numeric_columns = [column for column in DAILY_BASIC_COLUMNS if column not in {"trade_date", "code"}]
    for column in numeric_columns:
        result[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    return result.loc[:, DAILY_BASIC_COLUMNS].sort_values(["trade_date", "code"]).reset_index(drop=True)


def normalize_tushare_stock_limits(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare stk_limit output."""
    if df.empty:
        return _empty_with_columns(STOCK_LIMIT_COLUMNS)
    _require_columns(df, ["ts_code", "trade_date", "up_limit", "down_limit"], "stk_limit")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["code"] = df["ts_code"].map(ts_code_to_code)
    for column in ["pre_close", "up_limit", "down_limit"]:
        result[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    return result.loc[:, STOCK_LIMIT_COLUMNS].sort_values(["trade_date", "code"]).reset_index(drop=True)


def normalize_tushare_suspend_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare suspend_d output."""
    if df.empty:
        return _empty_with_columns(SUSPEND_DAILY_COLUMNS)
    _require_columns(df, ["ts_code", "trade_date", "suspend_type"], "suspend_d")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["code"] = df["ts_code"].map(ts_code_to_code)
    result["suspend_type"] = df["suspend_type"].fillna("").astype(str)
    result["suspend_timing"] = df["suspend_timing"].fillna("").astype(str) if "suspend_timing" in df.columns else ""
    return result.loc[:, SUSPEND_DAILY_COLUMNS].sort_values(["trade_date", "code"]).reset_index(drop=True)


def normalize_tushare_index_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare index_daily output."""
    if df.empty:
        return _empty_with_columns(INDEX_DAILY_COLUMNS)

    code_column = _first_existing_column(df, ("ts_code", "index_code"))
    if code_column is None:
        raise ValueError("Tushare index_daily data missing required index code column")
    _require_columns(df, ["trade_date", "open", "high", "low", "close"], "index_daily")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["index_code"] = df[code_column].fillna("").astype(str).str.strip().str.upper()
    for column in ["open", "high", "low", "close", "pre_close", "pct_chg", "amount"]:
        result[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    volume_column = _first_existing_column(df, ("volume", "vol"))
    result["volume"] = pd.to_numeric(df[volume_column], errors="coerce") if volume_column else pd.NA
    return result.loc[:, INDEX_DAILY_COLUMNS].sort_values(["trade_date", "index_code"]).reset_index(drop=True)


def normalize_tushare_limit_list_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare limit_list_d output."""
    if df.empty:
        return _empty_with_columns(LIMIT_LIST_DAILY_COLUMNS)

    code_column = _first_existing_column(df, ("ts_code", "code"))
    if code_column is None:
        raise ValueError("Tushare limit_list_d data missing required code column")
    _require_columns(df, ["trade_date"], "limit_list_d")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["code"] = df[code_column].map(ts_code_to_code)
    result["name"] = df["name"].fillna("").astype(str) if "name" in df.columns else ""
    for column in ["close", "pct_chg", "amp", "fc_ratio", "fl_ratio", "fd_amount", "open_times", "strth"]:
        result[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    for column in ["first_time", "last_time", "limit_type", "status"]:
        result[column] = df[column].fillna("").astype(str) if column in df.columns else ""
    return result.loc[:, LIMIT_LIST_DAILY_COLUMNS].sort_values(["trade_date", "code", "limit_type"]).reset_index(drop=True)


def normalize_tushare_moneyflow(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare moneyflow output to stock-agent moneyflow fields."""
    if df.empty:
        return _empty_with_columns(MONEYFLOW_COLUMNS)

    code_column = _first_existing_column(df, ("ts_code", "code"))
    if code_column is None:
        raise ValueError("Tushare moneyflow data missing required code column")
    if "trade_date" not in df.columns:
        raise ValueError("Tushare moneyflow data missing required trade_date column")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["code"] = df[code_column].map(ts_code_to_code)
    for column in MONEYFLOW_COLUMNS:
        if column in {"trade_date", "code"}:
            continue
        result[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    return result.loc[:, MONEYFLOW_COLUMNS].sort_values(["trade_date", "code"]).reset_index(drop=True)


def normalize_tushare_sw_industry_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare index_classify output for SW industry classifications."""
    if df.empty:
        return _empty_with_columns(SW_INDUSTRY_CLASSIFICATION_COLUMNS)

    code_column = _first_existing_column(df, ("industry_code", "index_code", "ts_code", "code"))
    name_column = _first_existing_column(df, ("industry_name", "name", "index_name"))
    if code_column is None:
        raise ValueError("Tushare index_classify data missing required industry code column")
    if name_column is None:
        raise ValueError("Tushare index_classify data missing required industry name column")

    result = pd.DataFrame()
    industry_code = df[code_column].fillna("").astype(str).str.strip().str.upper()
    result["industry_code"] = industry_code
    result["industry_name"] = df[name_column].fillna("").astype(str).str.strip()
    result["level"] = df["level"].fillna("").astype(str).str.strip() if "level" in df.columns else ""
    result["src"] = df["src"].fillna("").astype(str).str.strip() if "src" in df.columns else ""
    result["parent_code"] = (
        df["parent_code"].fillna("").astype(str).str.strip().str.upper() if "parent_code" in df.columns else ""
    )
    index_code_column = _first_existing_column(df, ("index_code", "ts_code", "industry_code", "code"))
    result["index_code"] = df[index_code_column].fillna("").astype(str).str.strip().str.upper()
    result["is_pub"] = df["is_pub"].fillna("").astype(str).str.strip() if "is_pub" in df.columns else ""
    sort_column = _first_existing_column(df, ("sort_code", "sort"))
    result["sort_code"] = df[sort_column].fillna("").astype(str).str.strip() if sort_column else ""
    return (
        result.loc[:, SW_INDUSTRY_CLASSIFICATION_COLUMNS]
        .sort_values(["level", "industry_code"])
        .reset_index(drop=True)
    )


def normalize_tushare_sw_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare sw_daily output."""
    if df.empty:
        return _empty_with_columns(SW_DAILY_COLUMNS)

    code_column = _first_existing_column(df, ("industry_code", "ts_code", "index_code", "code"))
    if code_column is None:
        raise ValueError("Tushare sw_daily data missing required industry code column")
    _require_columns(df, ["trade_date", "open", "high", "low", "close"], "sw_daily")

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"].map(_format_trade_date)
    result["industry_code"] = df[code_column].fillna("").astype(str).str.strip().str.upper()
    name_column = _first_existing_column(df, ("industry_name", "name", "index_name"))
    result["industry_name"] = df[name_column].fillna("").astype(str).str.strip() if name_column else ""
    rename_map = {"pct_change": ("pct_change", "pct_chg"), "volume": ("volume", "vol")}
    for column in ["open", "high", "low", "close", "change", "amount", "pe", "pb", "float_mv", "total_mv"]:
        result[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    for target, candidates in rename_map.items():
        source = _first_existing_column(df, candidates)
        result[target] = pd.to_numeric(df[source], errors="coerce") if source else pd.NA
    return result.loc[:, SW_DAILY_COLUMNS].sort_values(["trade_date", "industry_code"]).reset_index(drop=True)


class TushareProvider(BaseDataProvider):
    """Fetch and normalize A-share data from Tushare Pro."""

    def __init__(self) -> None:
        token = str(getattr(settings, "TUSHARE_TOKEN", "") or "").strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not configured")

        api_url = str(getattr(settings, "TUSHARE_API_URL", "http://api.tushare.pro") or "").strip()
        if not api_url:
            api_url = "http://api.tushare.pro"
        allow_non_official = bool(getattr(settings, "TUSHARE_ALLOW_NON_OFFICIAL_API_URL", False))
        validate_tushare_api_url(api_url, allow_non_official)

        import tushare as ts

        self.api_url = api_url
        self._pro = ts.pro_api(token)
        if not is_official_tushare_api_url(api_url):
            self._pro._DataApi__http_url = api_url

    def get_stock_basic(self) -> pd.DataFrame:
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date,list_status",
            )
        return normalize_tushare_stock_basic(raw)

    def get_daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str | None = None,
    ) -> pd.DataFrame:
        import tushare as ts

        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = ts.pro_bar(
                ts_code=code_to_ts_code(code),
                api=self._pro,
                start_date=re.sub(r"\D", "", str(start_date)),
                end_date=re.sub(r"\D", "", str(end_date)),
                adj=adjust if adjust is not None else settings.TUSHARE_ADJ,
                fields=TUSHARE_DAILY_FIELDS,
            )
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_daily_bars(raw, code)

    def get_trade_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
    ) -> pd.DataFrame:
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.trade_cal(
                exchange=exchange,
                start_date=re.sub(r"\D", "", str(start_date)),
                end_date=re.sub(r"\D", "", str(end_date)),
            )
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_trade_calendar(raw, exchange=exchange)

    def get_daily_basic(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        code: str | None = None,
    ) -> pd.DataFrame:
        params = _date_range_params(trade_date, start_date, end_date)
        if code is not None:
            params["ts_code"] = code_to_ts_code(code)
        params["fields"] = TUSHARE_DAILY_BASIC_FIELDS
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.daily_basic(**params)
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_daily_basic(raw)

    def get_stock_limits(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        code: str | None = None,
    ) -> pd.DataFrame:
        params = _date_range_params(trade_date, start_date, end_date)
        if code is not None:
            params["ts_code"] = code_to_ts_code(code)
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.stk_limit(**params)
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_stock_limits(raw)

    def get_suspend_daily(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        code: str | None = None,
        suspend_type: str | None = None,
    ) -> pd.DataFrame:
        params = _date_range_params(trade_date, start_date, end_date)
        if code is not None:
            params["ts_code"] = code_to_ts_code(code)
        if suspend_type is not None:
            params["suspend_type"] = suspend_type
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.suspend_d(**params)
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_suspend_daily(raw)

    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.index_daily(
                ts_code=str(index_code).strip().upper(),
                start_date=re.sub(r"\D", "", str(start_date)),
                end_date=re.sub(r"\D", "", str(end_date)),
                fields=TUSHARE_INDEX_DAILY_FIELDS,
            )
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_index_daily(raw)

    def get_limit_list_daily(self, trade_date: str) -> pd.DataFrame:
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.limit_list_d(
                trade_date=re.sub(r"\D", "", str(trade_date)),
                fields=TUSHARE_LIMIT_LIST_DAILY_FIELDS,
            )
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_limit_list_daily(raw)

    def get_moneyflow(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        code: str | None = None,
    ) -> pd.DataFrame:
        params = _date_range_params(trade_date, start_date, end_date)
        if code is not None:
            params["ts_code"] = code_to_ts_code(code)
        params["fields"] = TUSHARE_MONEYFLOW_FIELDS
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.moneyflow(**params)
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_moneyflow(raw)

    def get_sw_industry_classification(
        self,
        level: str = "L1",
        src: str = "SW2021",
    ) -> pd.DataFrame:
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.index_classify(level=level, src=src)
        if raw is None:
            raw = pd.DataFrame()
        result = normalize_tushare_sw_industry_classification(raw)
        if not result.empty:
            result["level"] = result["level"].where(result["level"].astype(str).str.strip() != "", str(level))
            result["src"] = result["src"].where(result["src"].astype(str).str.strip() != "", str(src))
        return result

    def get_sw_daily(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        industry_code: str | None = None,
    ) -> pd.DataFrame:
        params = _date_range_params(trade_date, start_date, end_date)
        if industry_code is not None:
            params["ts_code"] = str(industry_code).strip().upper()
        params["fields"] = TUSHARE_SW_DAILY_FIELDS
        with no_proxy_context(settings.DATA_FETCH_DISABLE_PROXY):
            raw = self._pro.sw_daily(**params)
        if raw is None:
            raw = pd.DataFrame()
        return normalize_tushare_sw_daily(raw)


def _date_range_params(
    trade_date: str | None,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, str]:
    params: dict[str, str] = {}
    if trade_date is not None:
        params["trade_date"] = re.sub(r"\D", "", str(trade_date))
    if start_date is not None:
        params["start_date"] = re.sub(r"\D", "", str(start_date))
    if end_date is not None:
        params["end_date"] = re.sub(r"\D", "", str(end_date))
    return params
