import pandas as pd

from src.research.stock_industry_map import build_stock_industry_map


def test_build_stock_industry_map_matches_sw_classification_by_name():
    stock_basic = pd.DataFrame(
        {
            "code": ["000001.SZ", "2"],
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


def test_build_stock_industry_map_prefers_industry_over_board():
    stock_basic = pd.DataFrame(
        {
            "code": ["000001"],
            "name": ["平安银行"],
            "board": ["主板"],
            "industry": ["银行"],
        }
    )
    sw = pd.DataFrame(
        {
            "industry_code": ["801780.SI"],
            "industry_name": ["银行"],
            "level": ["L1"],
        }
    )

    result = build_stock_industry_map(stock_basic, sw)

    assert result.loc[0, "industry_name"] == "银行"
    assert result.loc[0, "industry_code"] == "801780.SI"


def test_build_stock_industry_map_matches_sw_components_by_code():
    stock_basic = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "name": ["平安银行", "万科A"],
            "board": ["主板", "主板"],
        }
    )
    components = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "industry_name": ["银行"],
            "industry_code": ["801780.SI"],
            "industry_level": ["L1"],
        }
    )

    result = build_stock_industry_map(stock_basic, sw_industry_components=components).set_index("code")

    assert result.loc["000001", "industry_name"] == "银行"
    assert result.loc["000001", "industry_code"] == "801780.SI"
    assert result.loc["000001", "source"] == "sw_component_code_match"
    assert result.loc["000002", "industry_code"] == ""


def test_build_stock_industry_map_uses_component_source_when_present():
    stock_basic = pd.DataFrame({"code": ["000001"], "name": ["平安银行"]})
    components = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "industry_name": ["银行"],
            "industry_code": ["801780.SI"],
            "industry_level": ["L1"],
            "source": ["sw2021_member"],
        }
    )

    result = build_stock_industry_map(stock_basic, sw_industry_components=components)

    assert result.loc[0, "source"] == "sw2021_member"
