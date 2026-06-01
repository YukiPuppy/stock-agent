import pandas as pd

from src.filters.universe_filter import filter_tradable_main_board


def test_keeps_shanghai_and_shenzhen_main_board_codes():
    df = pd.DataFrame(
        {
            "code": ["600000", "SZ000001", 2594],
            "name": ["浦发银行", "平安银行", "比亚迪"],
        }
    )

    result = filter_tradable_main_board(df)

    assert result["code"].tolist() == ["600000", "SZ000001", 2594]


def test_excludes_chinext_star_market_and_beijing_exchange_codes():
    df = pd.DataFrame(
        {
            "code": ["300750", "688981", "BJ430047", "430047", "600000"],
            "name": ["宁德时代", "中芯国际", "北交样本1", "北交样本2", "浦发银行"],
        }
    )

    result = filter_tradable_main_board(df)

    assert result["code"].tolist() == ["600000"]


def test_excludes_names_containing_st_star_st_or_delisting_marker():
    df = pd.DataFrame(
        {
            "code": ["600000", "600001", "600002", "600003"],
            "name": ["浦发银行", "ST样本", "*ST样本", "退市样本"],
        }
    )

    result = filter_tradable_main_board(df)

    assert result["code"].tolist() == ["600000"]


def test_excludes_paused_stocks():
    df = pd.DataFrame(
        {
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "paused": [False, True],
        }
    )

    result = filter_tradable_main_board(df)

    assert result["code"].tolist() == ["600000"]


def test_excludes_non_listed_stocks_when_list_status_exists():
    df = pd.DataFrame(
        {
            "code": ["600000", "000001", "002594"],
            "name": ["浦发银行", "平安银行", "比亚迪"],
            "list_status": ["L", "D", None],
        }
    )

    result = filter_tradable_main_board(df)

    assert result["code"].tolist() == ["600000"]
