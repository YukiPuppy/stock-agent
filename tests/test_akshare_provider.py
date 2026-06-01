import os

import pandas as pd
import pytest

from src.data_providers.akshare_provider import AkshareProvider


def test_get_stock_basic_converts_fields(monkeypatch):
    def fake_stock_info_a_code_name():
        return pd.DataFrame(
            {
                "code": ["600000", "000001"],
                "name": ["浦发银行", "平安银行"],
            }
        )

    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_info_a_code_name",
        fake_stock_info_a_code_name,
    )

    result = AkshareProvider().get_stock_basic()

    assert result.columns.tolist() == ["code", "name", "market", "board", "list_status"]
    assert result.to_dict("records") == [
        {
            "code": "600000",
            "name": "浦发银行",
            "market": "SH",
            "board": "主板",
            "list_status": "L",
        },
        {
            "code": "000001",
            "name": "平安银行",
            "market": "SZ",
            "board": "主板",
            "list_status": "L",
        },
    ]


def test_get_stock_basic_infers_market_and_board(monkeypatch):
    def fake_stock_info_a_code_name():
        return pd.DataFrame(
            {
                "代码": ["688001", "300001", "830001", "123456"],
                "名称": ["科创样本", "创业样本", "北交样本", "未知样本"],
            }
        )

    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_info_a_code_name",
        fake_stock_info_a_code_name,
    )

    result = AkshareProvider().get_stock_basic()

    assert result["code"].tolist() == ["688001", "300001", "830001", "123456"]
    assert result["market"].tolist() == ["SH", "SZ", "BJ", "UNKNOWN"]
    assert result["board"].tolist() == ["科创板", "创业板", "北交所", "未知"]
    assert result["list_status"].tolist() == ["L", "L", "L", "L"]


def test_get_daily_bars_converts_fields(monkeypatch):
    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [10.1, 10.2],
                "最高": [10.5, 10.7],
                "最低": [9.9, 10.0],
                "收盘": [10.3, 10.6],
                "成交量": [1000, 1200],
                "成交额": [100000.5, 130000.0],
            }
        )

    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_hist",
        fake_stock_zh_a_hist,
    )

    result = AkshareProvider().get_daily_bars("600000", "2024-01-01", "2024-01-31")

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
    assert result.to_dict("records") == [
        {
            "trade_date": "2024-01-02",
            "code": "600000",
            "open": 10.1,
            "high": 10.5,
            "low": 9.9,
            "close": 10.3,
            "volume": 1000,
            "amount": 100000.5,
        },
        {
            "trade_date": "2024-01-03",
            "code": "600000",
            "open": 10.2,
            "high": 10.7,
            "low": 10.0,
            "close": 10.6,
            "volume": 1200,
            "amount": 130000.0,
        },
    ]


def test_get_daily_bars_normalizes_suffixed_code_for_akshare_and_output(monkeypatch):
    calls = {}

    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        calls["symbol"] = symbol
        calls["period"] = period
        calls["start_date"] = start_date
        calls["end_date"] = end_date
        calls["adjust"] = adjust
        return pd.DataFrame(
            {
                "日期": ["2024-02-01"],
                "开盘": ["1.1"],
                "最高": ["1.2"],
                "最低": ["1.0"],
                "收盘": ["1.15"],
                "成交量": ["10"],
                "成交额": ["100"],
            }
        )

    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_hist",
        fake_stock_zh_a_hist,
    )

    result = AkshareProvider().get_daily_bars("SH600000", "2024-02-01", "2024-02-29")

    assert calls == {
        "symbol": "600000",
        "period": "daily",
        "start_date": "20240201",
        "end_date": "20240229",
        "adjust": "",
    }
    assert result["code"].tolist() == ["600000"]


def test_get_daily_bars_falls_back_to_stock_zh_a_daily(monkeypatch):
    calls = []

    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        calls.append(("hist", symbol, period, start_date, end_date, adjust))
        raise RuntimeError("hist failed")

    def fake_stock_zh_a_daily(symbol, start_date, end_date, adjust):
        calls.append(("daily", symbol, start_date, end_date, adjust))
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": ["10.1"],
                "high": ["10.5"],
                "low": ["9.9"],
                "close": ["10.3"],
                "volume": ["1000"],
                "amount": ["100000.5"],
            }
        )

    monkeypatch.setattr("src.data_providers.akshare_provider.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_hist",
        fake_stock_zh_a_hist,
    )
    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_daily",
        fake_stock_zh_a_daily,
    )

    result = AkshareProvider().get_daily_bars("000001.SZ", "2024-01-01", "2024-01-31")

    assert calls[:3] == [
        ("hist", "000001", "daily", "20240101", "20240131", ""),
        ("hist", "000001", "daily", "20240101", "20240131", ""),
        ("hist", "000001", "daily", "20240101", "20240131", ""),
    ]
    assert calls[3] == ("daily", "sz000001", "20240101", "20240131", "")
    assert result.to_dict("records") == [
        {
            "trade_date": "2024-01-02",
            "code": "000001",
            "open": 10.1,
            "high": 10.5,
            "low": 9.9,
            "close": 10.3,
            "volume": 1000,
            "amount": 100000.5,
        }
    ]


def test_get_daily_bars_raises_runtime_error_when_both_interfaces_fail(monkeypatch):
    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        raise RuntimeError("hist failed")

    def fake_stock_zh_a_daily(symbol, start_date, end_date, adjust):
        raise RuntimeError("daily failed")

    monkeypatch.setattr("src.data_providers.akshare_provider.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_hist",
        fake_stock_zh_a_hist,
    )
    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_daily",
        fake_stock_zh_a_daily,
    )

    with pytest.raises(RuntimeError) as exc_info:
        AkshareProvider().get_daily_bars("SH600000", "2024-01-01", "2024-01-31")

    message = str(exc_info.value)
    assert "code=600000" in message
    assert "start_date=2024-01-01" in message
    assert "end_date=2024-01-31" in message
    assert "primary_error=hist failed" in message
    assert "fallback_error=daily failed" in message


def test_get_daily_bars_returns_empty_standard_dataframe_when_both_interfaces_empty(monkeypatch):
    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        return pd.DataFrame()

    def fake_stock_zh_a_daily(symbol, start_date, end_date, adjust):
        return pd.DataFrame()

    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_hist",
        fake_stock_zh_a_hist,
    )
    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_daily",
        fake_stock_zh_a_daily,
    )

    result = AkshareProvider().get_daily_bars("600000.SH", "2024-01-01", "2024-01-31")

    assert result.empty
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


@pytest.mark.parametrize(
    ("input_code", "expected_code"),
    [
        ("000001.SZ", "000001"),
        ("SZ000001", "000001"),
        ("600000.SH", "600000"),
        ("SH600000", "600000"),
    ],
)
def test_get_daily_bars_normalizes_supported_code_formats(monkeypatch, input_code, expected_code):
    seen_symbols = []

    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        seen_symbols.append(symbol)
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [10.1],
                "最高": [10.5],
                "最低": [9.9],
                "收盘": [10.3],
                "成交量": [1000],
                "成交额": [100000.5],
            }
        )

    monkeypatch.setattr(
        "src.data_providers.akshare_provider.ak.stock_zh_a_hist",
        fake_stock_zh_a_hist,
    )

    result = AkshareProvider().get_daily_bars(input_code, "2024-01-01", "2024-01-31")

    assert seen_symbols == [expected_code]
    assert result["code"].tolist() == [expected_code]


@pytest.mark.parametrize(
    ("method_name", "fake_attr", "call_provider"),
    [
        (
            "stock_basic",
            "stock_info_a_code_name",
            lambda provider: provider.get_stock_basic(),
        ),
        (
            "daily_bars",
            "stock_zh_a_hist",
            lambda provider: provider.get_daily_bars("600000", "2024-01-01", "2024-01-31"),
        ),
    ],
)
def test_akshare_calls_run_without_proxy_and_restore_environment(
    monkeypatch, method_name, fake_attr, call_provider
):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    seen_env = {}

    def fake_stock_info_a_code_name():
        seen_env[method_name] = {
            "http_proxy": "http_proxy" in os.environ,
            "https_proxy": "https_proxy" in os.environ,
        }
        return pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]})

    def fake_stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        seen_env[method_name] = {
            "http_proxy": "http_proxy" in os.environ,
            "https_proxy": "https_proxy" in os.environ,
        }
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [10.1],
                "最高": [10.5],
                "最低": [9.9],
                "收盘": [10.3],
                "成交量": [1000],
                "成交额": [100000.5],
            }
        )

    fake = fake_stock_info_a_code_name if fake_attr == "stock_info_a_code_name" else fake_stock_zh_a_hist
    monkeypatch.setattr(f"src.data_providers.akshare_provider.ak.{fake_attr}", fake)

    call_provider(AkshareProvider())

    assert seen_env[method_name] == {"http_proxy": False, "https_proxy": False}
    assert os.environ["http_proxy"] == "http://127.0.0.1:7890"
    assert os.environ["https_proxy"] == "http://127.0.0.1:7890"
