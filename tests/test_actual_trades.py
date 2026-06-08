import pandas as pd

from src.trading.actual_trades import ACTUAL_TRADE_COLUMNS, normalize_actual_trades


def test_normalize_actual_trades_standardizes_code_side_and_amount():
    df = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "code": [1],
            "side": ["BUY"],
            "price": ["10.5"],
            "volume": ["200"],
            "amount": [None],
        }
    )

    result = normalize_actual_trades(df)

    assert result.columns.tolist() == ACTUAL_TRADE_COLUMNS
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "side"] == "buy"
    assert result.loc[0, "amount"] == 2100.0
    assert result.loc[0, "trade_time"] == ""


def test_normalize_actual_trades_empty_returns_standard_columns():
    result = normalize_actual_trades(pd.DataFrame())

    assert result.empty
    assert result.columns.tolist() == ACTUAL_TRADE_COLUMNS
