import pandas as pd

from src.data_quality.provider_compare import compare_daily_bars, summarize_provider_compare


def _bars(code="600000", trade_date="20260102", close=10.0):
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "code": [code],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [close],
            "volume": [1000],
            "amount": [10000],
        }
    )


def test_compare_daily_bars_identifies_price_difference():
    result = compare_daily_bars(_bars(close=10.0), _bars(close=10.5), price_tolerance=0.01)

    row = result[result["field"] == "close"].iloc[0]
    assert row["status"] == "warning"
    assert row["relative_diff"] > 0.01


def test_compare_daily_bars_identifies_missing_left_and_missing_right():
    left = pd.concat([_bars(code="600000"), _bars(code="000001")], ignore_index=True)
    right = pd.concat([_bars(code="600000"), _bars(code="000002")], ignore_index=True)

    result = compare_daily_bars(left, right)

    assert {"missing_left", "missing_right"}.issubset(set(result["status"]))


def test_compare_daily_bars_does_not_warn_for_matching_standard_volume_amount():
    left = _bars()
    right = _bars()

    result = compare_daily_bars(left, right)

    assert result.empty


def test_compare_daily_bars_does_not_warn_when_akshare_volume_matches_tushare_volume():
    akshare = _bars()
    tushare = _bars()
    akshare["volume"] = [181596.99]
    tushare["volume"] = [181596.99]
    akshare["amount"] = [210292.078]
    tushare["amount"] = [210292.078]

    result = compare_daily_bars(akshare, tushare, left_name="akshare", right_name="tushare")

    assert result.empty


def test_compare_daily_bars_warns_for_unconverted_volume_amount_difference():
    left = _bars()
    right = _bars()
    right["volume"] = right["volume"] * 100
    right["amount"] = right["amount"] * 1000

    result = compare_daily_bars(left, right)

    assert {"volume", "amount"}.issubset(set(result["field"]))
    assert set(result["status"]) == {"warning"}


def test_summarize_provider_compare_counts_issues():
    compare_result = pd.DataFrame(
        [
            {"field": "close", "relative_diff": 0.02, "status": "warning"},
            {"field": "close", "relative_diff": 0.04, "status": "warning"},
            {"field": "volume", "relative_diff": 0.3, "status": "warning"},
        ]
    )

    result = summarize_provider_compare(compare_result).set_index("field")

    assert result.loc["close", "issue_count"] == 2
    assert result.loc["close", "max_relative_diff"] == 0.04
