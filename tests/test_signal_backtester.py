import pandas as pd
import pytest

from src.backtest.signal_backtester import (
    BASE_RESULT_COLUMNS,
    PERFORMANCE_COLUMNS,
    backtest_strategy_signals,
    evaluate_strategy_performance,
)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260101"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "signal_strength": [30.0],
            "entry_reason": ["test"],
            "risk_flags": [""],
        }
    )


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260101", "20260102", "20260105", "20260106", "20260107", "20260108", "20260109"],
            "code": ["600000"] * 7,
            "open": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "low": [8.5, 9.0, 9.5, 8.0, 10.0, 7.0, 13.0],
            "close": [9.2, 10.5, 11.0, 12.0, 13.0, 15.0, 16.0],
            "volume": [1000.0] * 7,
            "amount": [10000.0] * 7,
        }
    )


def test_single_signal_uses_next_trading_day_open_as_entry_open():
    result = backtest_strategy_signals(_signals(), _bars())

    assert len(result) == 1
    assert result.loc[0, "entry_date"] == "20260102"
    assert result.loc[0, "entry_open"] == 10.0
    assert result.loc[0, "is_valid"] == True


def test_backtest_calculates_returns_for_default_holding_days():
    result = backtest_strategy_signals(_signals(), _bars())

    assert result.loc[0, "return_1d"] == pytest.approx(11.0 / 10.0 - 1)
    assert result.loc[0, "return_3d"] == pytest.approx(13.0 / 10.0 - 1)
    assert result.loc[0, "return_5d"] == pytest.approx(16.0 / 10.0 - 1)


def test_backtest_calculates_max_drawdown_from_entry_through_exit():
    result = backtest_strategy_signals(_signals(), _bars())

    assert result.loc[0, "max_drawdown_1d"] == pytest.approx(9.0 / 10.0 - 1)
    assert result.loc[0, "max_drawdown_3d"] == pytest.approx(8.0 / 10.0 - 1)
    assert result.loc[0, "max_drawdown_5d"] == pytest.approx(7.0 / 10.0 - 1)


def test_backtest_marks_signal_invalid_when_future_bars_are_insufficient():
    result = backtest_strategy_signals(_signals(), _bars().head(3))

    assert result.loc[0, "entry_date"] == "20260102"
    assert result.loc[0, "return_1d"] == pytest.approx(11.0 / 10.0 - 1)
    assert result.loc[0, "is_valid"] == False
    assert result.loc[0, "invalid_reason"] == "insufficient_future_bars"


def test_empty_strategy_signals_returns_standard_empty_dataframe():
    result = backtest_strategy_signals(pd.DataFrame(), _bars())

    assert result.empty
    assert list(result.columns) == [
        "signal_date",
        "code",
        "strategy_name",
        "strategy_version",
        "signal_strength",
        "entry_date",
        "entry_open",
        "exit_date_1d",
        "exit_close_1d",
        "return_1d",
        "exit_date_3d",
        "exit_close_3d",
        "return_3d",
        "exit_date_5d",
        "exit_close_5d",
        "return_5d",
        "max_drawdown_1d",
        "max_drawdown_3d",
        "max_drawdown_5d",
        "is_valid",
        "invalid_reason",
    ]
    assert set(BASE_RESULT_COLUMNS) <= set(result.columns)


def test_backtest_strategy_signals_preserves_strategy_version():
    signals = _signals()
    signals["strategy_version"] = "v2"

    result = backtest_strategy_signals(signals, _bars())

    assert result.loc[0, "strategy_version"] == "v2"


def test_backtest_strategy_signals_defaults_missing_strategy_version_to_v1():
    result = backtest_strategy_signals(_signals(), _bars())

    assert result.loc[0, "strategy_version"] == "v1"


def test_evaluate_strategy_performance_groups_multiple_strategies():
    backtest_results = pd.DataFrame(
        {
            "strategy_name": ["trend", "trend", "rebound", "rebound"],
            "strategy_version": ["v1", "v1", "v2", "v2"],
            "is_valid": [True, True, True, False],
            "return_1d": [0.10, -0.05, 0.20, 0.50],
            "return_3d": [0.20, 0.00, -0.10, 0.50],
            "return_5d": [0.30, 0.10, 0.00, 0.50],
            "max_drawdown_1d": [-0.03, -0.08, -0.01, -0.10],
            "max_drawdown_3d": [-0.05, -0.10, -0.02, -0.10],
            "max_drawdown_5d": [-0.07, -0.12, -0.03, -0.10],
        }
    )

    result = evaluate_strategy_performance(backtest_results)
    trend = result[result["strategy_name"] == "trend"].iloc[0]
    rebound = result[result["strategy_name"] == "rebound"].iloc[0]

    assert list(result.columns) == PERFORMANCE_COLUMNS
    assert trend["sample_count"] == 2
    assert trend["valid_count"] == 2
    assert trend["win_rate_1d"] == pytest.approx(0.5)
    assert trend["avg_return_1d"] == pytest.approx(0.025)
    assert trend["median_return_1d"] == pytest.approx(0.025)
    assert trend["avg_max_drawdown_3d"] == pytest.approx(-0.075)
    assert rebound["sample_count"] == 2
    assert rebound["valid_count"] == 1
    assert rebound["win_rate_1d"] == pytest.approx(1.0)
    assert rebound["win_rate_5d"] == pytest.approx(0.0)


def test_evaluate_strategy_performance_groups_by_strategy_name_and_version():
    backtest_results = pd.DataFrame(
        {
            "strategy_name": ["trend", "trend", "trend"],
            "strategy_version": ["v1", "v2", "v2"],
            "is_valid": [True, True, True],
            "return_1d": [0.10, -0.05, 0.20],
            "return_3d": [0.10, -0.05, 0.20],
            "return_5d": [0.10, -0.05, 0.20],
            "max_drawdown_1d": [-0.01, -0.02, -0.03],
            "max_drawdown_3d": [-0.01, -0.02, -0.03],
            "max_drawdown_5d": [-0.01, -0.02, -0.03],
        }
    )

    result = evaluate_strategy_performance(backtest_results)

    assert result[["strategy_name", "strategy_version", "sample_count"]].to_dict("records") == [
        {"strategy_name": "trend", "strategy_version": "v1", "sample_count": 1},
        {"strategy_name": "trend", "strategy_version": "v2", "sample_count": 2},
    ]
