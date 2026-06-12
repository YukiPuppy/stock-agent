import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_providers.tushare_provider import (
    DAILY_BASIC_COLUMNS,
    TUSHARE_DAILY_BASIC_FIELDS,
    TushareProvider,
    code_to_ts_code,
    is_official_tushare_api_url,
    normalize_tushare_daily_bars,
    normalize_tushare_daily_basic,
    normalize_tushare_index_daily,
    normalize_tushare_limit_list_daily,
    normalize_tushare_moneyflow,
    normalize_tushare_sw_daily,
    normalize_tushare_sw_industry_classification,
    normalize_tushare_stock_limits,
    normalize_tushare_suspend_daily,
    normalize_tushare_stock_basic,
    normalize_tushare_trade_calendar,
    ts_code_to_code,
    validate_tushare_api_url,
)


def test_is_official_tushare_api_url_recognizes_official_urls():
    assert is_official_tushare_api_url("http://api.tushare.pro")
    assert is_official_tushare_api_url("https://api.tushare.pro")
    assert not is_official_tushare_api_url("https://example.test/tushare")


def test_validate_tushare_api_url_rejects_non_official_without_allow_flag():
    with pytest.raises(
        ValueError,
        match="Non-official Tushare API URL is not allowed unless TUSHARE_ALLOW_NON_OFFICIAL_API_URL=true",
    ):
        validate_tushare_api_url("https://example.test/tushare", allow_non_official=False)


def test_validate_tushare_api_url_allows_non_official_with_allow_flag():
    validate_tushare_api_url("https://example.test/tushare", allow_non_official=True)


def test_code_to_ts_code_converts_shenzhen_code():
    assert code_to_ts_code("000001") == "000001.SZ"


def test_code_to_ts_code_converts_shanghai_code():
    assert code_to_ts_code("600000") == "600000.SH"


def test_code_to_ts_code_converts_beijing_code():
    assert code_to_ts_code("430000") == "430000.BJ"
    assert code_to_ts_code("830000") == "830000.BJ"
    assert code_to_ts_code("920000") == "920000.BJ"


def test_ts_code_to_code_converts_tushare_code():
    assert ts_code_to_code("000001.SZ") == "000001"


def test_normalize_tushare_stock_basic_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["平安银行"],
            "industry": ["银行"],
            "market": ["主板"],
            "list_date": ["19910403"],
            "list_status": ["L"],
        }
    )

    result = normalize_tushare_stock_basic(raw)

    assert result.to_dict("records") == [
        {
            "code": "000001",
            "name": "平安银行",
            "market": "SZ",
            "board": "主板",
            "industry": "银行",
            "list_date": "19910403",
            "status": "L",
            "list_status": "L",
        }
    ]


def test_normalize_tushare_daily_bars_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "trade_date": ["20250103", "20250102"],
            "open": [10.2, 10.0],
            "high": [10.8, 10.5],
            "low": [10.1, 9.8],
            "close": [10.6, 10.2],
            "vol": [2000, 1000],
            "amount": [21000, 10200],
        }
    )

    result = normalize_tushare_daily_bars(raw, "000001")

    assert result.columns.tolist() == [
        "trade_date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert result["trade_date"].tolist() == ["2025-01-02", "2025-01-03"]
    assert result["code"].tolist() == ["000001", "000001"]
    assert result["volume"].tolist() == [1000, 2000]
    assert result["amount"].tolist() == [10200, 21000]


def test_normalize_tushare_trade_calendar_outputs_standard_fields():
    raw = pd.DataFrame({"cal_date": ["20250102"], "is_open": [1], "pretrade_date": ["20241231"]})

    result = normalize_tushare_trade_calendar(raw, exchange="SSE")

    assert result.to_dict("records") == [
        {"trade_date": "2025-01-02", "exchange": "SSE", "is_open": 1, "pretrade_date": "2024-12-31"}
    ]


def test_normalize_tushare_daily_basic_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20250102"],
            "close": [10.2],
            "turnover_rate": [1.2],
            "turnover_rate_f": [0.8],
            "volume_ratio": [1.5],
            "pe": [9.0],
            "pe_ttm": [10.0],
            "pb": [1.1],
            "ps": [2.0],
            "ps_ttm": [2.1],
            "dv_ratio": [0.5],
            "dv_ttm": [0.6],
            "total_share": [100000.0],
            "float_share": [90000.0],
            "free_share": [80000.0],
            "total_mv": [23456.7],
            "circ_mv": [12345.6],
        }
    )

    result = normalize_tushare_daily_basic(raw)

    assert result.columns.tolist() == DAILY_BASIC_COLUMNS
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "trade_date"] == "2025-01-02"
    assert result.loc[0, "turnover_rate"] == 1.2
    assert result.loc[0, "volume_ratio"] == 1.5
    assert result.loc[0, "total_mv"] == 23456.7
    assert result.loc[0, "circ_mv"] == 12345.6


def test_normalize_tushare_stock_limits_outputs_standard_fields():
    raw = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20250102"], "pre_close": [10.0], "up_limit": [11.0], "down_limit": [9.0]}
    )

    result = normalize_tushare_stock_limits(raw)

    assert result.to_dict("records") == [
        {"trade_date": "2025-01-02", "code": "000001", "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0}
    ]


def test_normalize_tushare_suspend_daily_outputs_standard_fields():
    raw = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20250102"], "suspend_type": ["S"], "suspend_timing": ["09:30"]}
    )

    result = normalize_tushare_suspend_daily(raw)

    assert result.to_dict("records") == [
        {"trade_date": "2025-01-02", "code": "000001", "suspend_type": "S", "suspend_timing": "09:30"}
    ]


def test_normalize_tushare_index_daily_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SH", "000001.SH"],
            "trade_date": ["20250103", "20250102"],
            "open": [3200, 3100],
            "high": [3210, 3120],
            "low": [3180, 3090],
            "close": [3190, 3110],
            "pre_close": [3110, 3100],
            "pct_chg": [2.89, 0.32],
            "vol": [100, 90],
            "amount": [200, 180],
        }
    )

    result = normalize_tushare_index_daily(raw)

    assert result["trade_date"].tolist() == ["2025-01-02", "2025-01-03"]
    assert result["index_code"].tolist() == ["000001.SH", "000001.SH"]
    assert "volume" in result.columns
    assert result.loc[0, "volume"] == 90


def test_normalize_tushare_sw_industry_classification_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "ts_code": ["801010.SI"],
            "name": ["农林牧渔"],
            "level": ["L1"],
            "src": ["SW2021"],
            "parent_code": [""],
            "is_pub": ["1"],
            "sort_code": ["1"],
        }
    )

    result = normalize_tushare_sw_industry_classification(raw)

    assert result.columns.tolist() == [
        "industry_code",
        "industry_name",
        "level",
        "src",
        "parent_code",
        "index_code",
        "is_pub",
        "sort_code",
    ]
    assert result.loc[0, "industry_code"] == "801010.SI"
    assert result.loc[0, "index_code"] == "801010.SI"
    assert result.loc[0, "industry_name"] == "农林牧渔"


def test_normalize_tushare_sw_daily_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "ts_code": ["801010.SI"],
            "trade_date": ["20250102"],
            "name": ["农林牧渔"],
            "open": [1000],
            "high": [1010],
            "low": [990],
            "close": [1005],
            "change": [5],
            "pct_change": [0.5],
            "vol": [123],
            "amount": [456],
            "pe": [20],
            "pb": [2],
            "float_mv": [10000],
            "total_mv": [12000],
        }
    )

    result = normalize_tushare_sw_daily(raw)

    assert result["trade_date"].tolist() == ["2025-01-02"]
    assert result.loc[0, "industry_code"] == "801010.SI"
    assert result.loc[0, "industry_name"] == "农林牧渔"
    assert result.loc[0, "volume"] == 123


def test_normalize_tushare_limit_list_daily_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "trade_date": ["20250102"],
            "ts_code": ["000001.SZ"],
            "name": ["平安银行"],
            "close": [11.0],
            "pct_chg": [10.0],
            "amp": [12.0],
            "fc_ratio": [1.2],
            "fl_ratio": [2.3],
            "fd_amount": [10000],
            "first_time": ["093100"],
            "last_time": ["145700"],
            "open_times": [1],
            "strth": [2],
            "limit_type": ["U"],
            "status": ["炸板回封"],
        }
    )

    result = normalize_tushare_limit_list_daily(raw)

    assert result.loc[0, "trade_date"] == "2025-01-02"
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "limit_type"] == "U"
    assert result.loc[0, "open_times"] == 1


def test_normalize_tushare_moneyflow_outputs_standard_fields():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20250102"],
            "buy_sm_vol": [1],
            "buy_sm_amount": [10],
            "sell_sm_vol": [2],
            "sell_sm_amount": [20],
            "buy_md_vol": [3],
            "buy_md_amount": [30],
            "sell_md_vol": [4],
            "sell_md_amount": [40],
            "buy_lg_vol": [5],
            "buy_lg_amount": [50],
            "sell_lg_vol": [6],
            "sell_lg_amount": [60],
            "buy_elg_vol": [7],
            "buy_elg_amount": [70],
            "sell_elg_vol": [8],
            "sell_elg_amount": [80],
            "net_mf_vol": [-4],
            "net_mf_amount": [-40],
        }
    )

    result = normalize_tushare_moneyflow(raw)

    assert result.loc[0, "trade_date"] == "2025-01-02"
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "buy_lg_amount"] == 50


def test_tushare_provider_get_moneyflow_uses_mock_pro(monkeypatch):
    class FakePro:
        def __init__(self):
            self.kwargs = None

        def moneyflow(self, **kwargs):
            self.kwargs = kwargs
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20250102"],
                    "buy_lg_amount": [100.0],
                    "sell_lg_amount": [20.0],
                }
            )

    provider = object.__new__(TushareProvider)
    provider._pro = FakePro()

    result = provider.get_moneyflow(trade_date="2025-01-02", code="000001")

    assert provider._pro.kwargs["trade_date"] == "20250102"
    assert provider._pro.kwargs["ts_code"] == "000001.SZ"
    assert result.loc[0, "code"] == "000001"


def test_tushare_provider_raises_clear_error_without_token(monkeypatch):
    monkeypatch.setattr("src.data_providers.tushare_provider.settings.TUSHARE_TOKEN", "")

    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN is not configured"):
        TushareProvider()


def test_tushare_provider_sets_non_official_api_url_when_allowed(monkeypatch, capsys):
    created = []
    dummy_token = "test-" + "token"

    class FakePro:
        def __init__(self):
            self._DataApi__http_url = "http://api.tushare.pro"

    def fake_pro_api(token):
        created.append((token, FakePro()))
        return created[-1][1]

    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=fake_pro_api))
    monkeypatch.setattr("src.data_providers.tushare_provider.settings.TUSHARE_TOKEN", dummy_token)
    monkeypatch.setattr(
        "src.data_providers.tushare_provider.settings.TUSHARE_API_URL",
        "https://example.test/tushare",
    )
    monkeypatch.setattr(
        "src.data_providers.tushare_provider.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        True,
    )

    provider = TushareProvider()
    output = capsys.readouterr()

    assert provider.api_url == "https://example.test/tushare"
    assert created[0][1]._DataApi__http_url == "https://example.test/tushare"
    assert dummy_token not in output.out
    assert dummy_token not in output.err


def test_tushare_provider_get_stock_basic_uses_configured_no_proxy_context(monkeypatch):
    calls = []

    @contextmanager
    def fake_no_proxy_context(disable_proxy=False):
        calls.append(disable_proxy)
        yield

    class FakePro:
        def stock_basic(self, exchange, list_status, fields):
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["平安银行"],
                    "market": ["主板"],
                    "list_date": ["19910403"],
                    "list_status": ["L"],
                }
            )

    monkeypatch.setattr("src.data_providers.tushare_provider.no_proxy_context", fake_no_proxy_context)
    monkeypatch.setattr("src.data_providers.tushare_provider.settings.DATA_FETCH_DISABLE_PROXY", True)

    provider = object.__new__(TushareProvider)
    provider._pro = FakePro()

    result = provider.get_stock_basic()

    assert calls == [True]
    assert result["code"].tolist() == ["000001"]


def test_tushare_provider_get_daily_bars_uses_configured_no_proxy_context(monkeypatch):
    calls = []

    @contextmanager
    def fake_no_proxy_context(disable_proxy=False):
        calls.append(disable_proxy)
        yield

    def fake_pro_bar(**kwargs):
        return pd.DataFrame(
            {
                "trade_date": ["20250102"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "vol": [1000],
                "amount": [10200],
            }
        )

    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_bar=fake_pro_bar))
    monkeypatch.setattr("src.data_providers.tushare_provider.no_proxy_context", fake_no_proxy_context)
    monkeypatch.setattr("src.data_providers.tushare_provider.settings.DATA_FETCH_DISABLE_PROXY", True)

    provider = object.__new__(TushareProvider)
    provider._pro = object()

    result = provider.get_daily_bars("000001", "2025-01-01", "2025-01-31")

    assert calls == [True]
    assert result["code"].tolist() == ["000001"]


def test_tushare_provider_get_daily_basic_requests_expected_fields(monkeypatch):
    calls = []

    @contextmanager
    def fake_no_proxy_context(disable_proxy=False):
        calls.append(disable_proxy)
        yield

    class FakePro:
        def __init__(self):
            self.kwargs = None

        def daily_basic(self, **kwargs):
            self.kwargs = kwargs
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20250102"],
                    "volume_ratio": [1.5],
                    "total_mv": [23456.7],
                    "circ_mv": [12345.6],
                }
            )

    monkeypatch.setattr("src.data_providers.tushare_provider.no_proxy_context", fake_no_proxy_context)
    monkeypatch.setattr("src.data_providers.tushare_provider.settings.DATA_FETCH_DISABLE_PROXY", True)

    provider = object.__new__(TushareProvider)
    provider._pro = FakePro()

    result = provider.get_daily_basic(trade_date="2025-01-02", code="000001")

    assert calls == [True]
    assert provider._pro.kwargs["trade_date"] == "20250102"
    assert provider._pro.kwargs["ts_code"] == "000001.SZ"
    assert provider._pro.kwargs["fields"] == TUSHARE_DAILY_BASIC_FIELDS
    assert result.loc[0, "volume_ratio"] == 1.5
    assert result.loc[0, "total_mv"] == 23456.7
    assert result.loc[0, "circ_mv"] == 12345.6
