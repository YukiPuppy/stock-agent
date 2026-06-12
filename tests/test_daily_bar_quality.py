import pandas as pd

from src.data_quality.daily_bar_quality import (
    check_daily_bars_quality,
    check_enriched_daily_factors_quality,
    check_industry_strength_quality,
)


def _row(**overrides):
    row = {
        "trade_date": "20260102",
        "code": "600000",
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "volume": 1000,
        "amount": 10200,
    }
    row.update(overrides)
    return row


def test_check_daily_bars_quality_identifies_empty_data():
    result = check_daily_bars_quality(pd.DataFrame()).set_index("check_name")

    assert result.loc["empty_data", "status"] == "error"


def test_check_daily_bars_quality_identifies_missing_columns():
    result = check_daily_bars_quality(pd.DataFrame({"trade_date": ["20260102"]})).set_index("check_name")

    assert result.loc["required_columns", "status"] == "error"
    assert result.loc["required_columns", "issue_count"] > 0


def test_check_daily_bars_quality_identifies_duplicates():
    result = check_daily_bars_quality(pd.DataFrame([_row(), _row(close=10.3)])).set_index("check_name")

    assert result.loc["duplicated_rows", "status"] == "warning"
    assert result.loc["duplicated_rows", "issue_count"] == 2


def test_check_daily_bars_quality_identifies_invalid_price_relations():
    df = pd.DataFrame(
        [
            _row(high=9.0, low=10.0),
            _row(trade_date="20260103", open=11.0),
            _row(trade_date="20260104", close=9.0),
        ]
    )

    result = check_daily_bars_quality(df).set_index("check_name")

    assert result.loc["invalid_price_relation", "status"] == "warning"
    assert result.loc["invalid_price_relation", "issue_count"] == 3


def test_check_daily_bars_quality_identifies_non_positive_price_and_negative_volume_amount():
    result = check_daily_bars_quality(
        pd.DataFrame([_row(open=0.0, volume=-1, amount=-2)])
    ).set_index("check_name")

    assert result.loc["non_positive_price", "status"] == "warning"
    assert result.loc["negative_volume_amount", "status"] == "warning"


def test_check_enriched_daily_factors_missing_rate_allows_moderate_missing_values():
    df = pd.DataFrame(
        {
            "volume_ratio_daily_basic": [pd.NA] * 100 + [1.2] * 250,
            "total_mv": [1000.0] * 350,
            "circ_mv": [800.0] * 350,
        }
    )

    result = check_enriched_daily_factors_quality(df).set_index("check_name")

    assert result.loc["enriched_daily_factors_missing_rate", "status"] == "ok"
    assert "28.6%" in result.loc["enriched_daily_factors_missing_rate", "message"]


def test_check_enriched_daily_factors_missing_rate_warns_when_high():
    df = pd.DataFrame(
        {
            "volume_ratio_daily_basic": [pd.NA] * 4,
            "total_mv": [1000.0, pd.NA, pd.NA, pd.NA],
            "circ_mv": [800.0, 800.0, 800.0, 800.0],
        }
    )

    result = check_enriched_daily_factors_quality(df).set_index("check_name")

    assert result.loc["enriched_daily_factors_missing_rate", "status"] == "warning"


def test_check_industry_strength_quality_warns_on_empty_map_and_high_missing_rate():
    factors = pd.DataFrame(
        {
            "industry_strength_score": [pd.NA, pd.NA, 50],
            "industry_strength_level": [pd.NA, pd.NA, "neutral"],
            "industry_return_5d": [pd.NA, pd.NA, 0.01],
        }
    )

    result = check_industry_strength_quality(pd.DataFrame(), factors).set_index("check_name")

    assert result.loc["stock_industry_map_status", "status"] == "warning"
    assert result.loc["industry_strength_missing_rate", "status"] == "warning"
    assert "66.7%" in result.loc["industry_strength_missing_rate", "message"]


def test_check_industry_strength_quality_ignores_return_window_warmup_missing_values():
    stock_map = pd.DataFrame({"code": ["000001"], "industry_code": ["801780.SI"]})
    factors = pd.DataFrame(
        {
            "industry_strength_score": [50, 65],
            "industry_strength_level": ["neutral", "strong"],
            "industry_return_5d": [pd.NA, pd.NA],
        }
    )

    result = check_industry_strength_quality(stock_map, factors).set_index("check_name")

    assert result.loc["industry_strength_missing_rate", "status"] == "ok"
    assert "0.0%" in result.loc["industry_strength_missing_rate", "message"]
