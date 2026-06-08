import pandas as pd

from src.ui.labels import (
    get_field_label,
    get_table_label,
    translate_dataframe_columns,
    translate_risk_flags,
    translate_value,
)


def test_get_table_label_translates_daily_factors():
    assert get_table_label("daily_factors") == "个股综合因子"


def test_get_field_label_translates_trade_date():
    assert get_field_label("trade_date") == "交易日期"


def test_translate_dataframe_columns_uses_chinese_column_names_and_values():
    df = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "market_regime": ["strong"],
            "risk_level": ["low"],
            "is_suspended": [False],
            "risk_flags": ["limit_up_close"],
        }
    )

    result = translate_dataframe_columns(df)

    assert list(result.columns) == ["交易日期", "市场环境", "风险等级", "是否停牌", "风险标记"]
    assert result.loc[0, "市场环境"] == "强势"
    assert result.loc[0, "风险等级"] == "低"
    assert result.loc[0, "是否停牌"] == "否"
    assert result.loc[0, "风险标记"] == "涨停收盘，次日可能买入困难"


def test_translate_risk_flags_translates_limit_up_close():
    assert translate_risk_flags("limit_up_close") == "涨停收盘，次日可能买入困难"


def test_translate_value_translates_market_levels():
    assert translate_value("strong") == "强势"
    assert translate_value("neutral") == "中性"
    assert translate_value("weak") == "弱势"


def test_unknown_labels_and_values_keep_original_text():
    assert get_table_label("unknown_table") == "unknown_table"
    assert get_field_label("unknown_field") == "unknown_field"
    assert translate_value("custom_value") == "custom_value"
    assert translate_risk_flags("custom_risk") == "custom_risk"


def test_translate_dataframe_columns_does_not_mutate_original_dataframe():
    df = pd.DataFrame({"trade_date": ["2026-01-02"], "risk_flags": ["limit_up_close"]})

    result = translate_dataframe_columns(df)

    assert list(df.columns) == ["trade_date", "risk_flags"]
    assert df.loc[0, "risk_flags"] == "limit_up_close"
    assert list(result.columns) == ["交易日期", "风险标记"]
    assert result.loc[0, "风险标记"] == "涨停收盘，次日可能买入困难"
