import pandas as pd

from src.strategy.base_strategy import SIGNAL_COLUMNS, empty_signals


def test_empty_signals_returns_standard_columns():
    result = empty_signals()

    assert result.empty
    assert result.columns.tolist() == SIGNAL_COLUMNS
    assert isinstance(result, pd.DataFrame)
