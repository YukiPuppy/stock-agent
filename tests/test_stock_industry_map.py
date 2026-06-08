import pandas as pd

from src.research.stock_industry_map import build_stock_industry_map


def test_build_stock_industry_map_matches_sw_classification_by_name():
    stock_basic = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "name": ["A", "B"],
            "industry": ["银行", "地产"],
        }
    )
    sw = pd.DataFrame(
        {
            "industry_code": ["801780.SI"],
            "industry_name": ["银行"],
            "level": ["L1"],
        }
    )

    result = build_stock_industry_map(stock_basic, sw).set_index("code")

    assert result.loc["000001", "industry_code"] == "801780.SI"
    assert result.loc["000001", "source"] == "stock_basic_sw_name_match"
    assert result.loc["000002", "industry_code"] == ""
    assert result.loc["000002", "source"] == "stock_basic_only"
