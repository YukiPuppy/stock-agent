import pandas as pd
import pytest

from src.trading.trade_performance import (
    ACTUAL_TRADE_PERFORMANCE_COLUMNS,
    calculate_actual_trade_performance,
)


def _actual_trades(side: str = "buy") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "trade_time": ["10:00:00"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "side": [side],
            "price": [10.0],
            "volume": [100],
            "amount": [1000.0],
            "position_ratio": [0.1],
            "strategy_name": ["trend_pullback"],
            "plan_rank": [1],
            "reason": ["按计划"],
            "note": [""],
        }
    )


def _daily_bars(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16"][:rows],
            "code": ["600000"] * rows,
            "open": [10.0, 10.4, 10.8, 11.0, 11.2][:rows],
            "high": [10.8, 11.0, 11.5, 11.6, 12.0][:rows],
            "low": [9.8, 9.7, 9.9, 10.8, 11.0][:rows],
            "close": [10.5, 10.8, 11.2, 11.4, 11.8][:rows],
            "volume": [1000] * rows,
            "amount": [10000] * rows,
        }
    )


def test_buy_trade_calculates_returns_drawdown_and_favorable():
    result = calculate_actual_trade_performance(_actual_trades(), _daily_bars())

    row = result.iloc[0]
    assert result.columns.tolist() == ACTUAL_TRADE_PERFORMANCE_COLUMNS
    assert row["is_valid"] == True
    assert row["return_1d"] == pytest.approx(0.05)
    assert row["return_3d"] == pytest.approx(0.12)
    assert row["return_5d"] == pytest.approx(0.18)
    assert row["max_drawdown_3d"] == pytest.approx(-0.03)
    assert row["max_favorable_5d"] == pytest.approx(0.2)


def test_insufficient_bars_marks_invalid_and_keeps_available_periods():
    result = calculate_actual_trade_performance(_actual_trades(), _daily_bars(rows=3))

    row = result.iloc[0]
    assert row["is_valid"] == False
    assert row["invalid_reason"] == "insufficient_daily_bars:5d"
    assert row["return_3d"] == pytest.approx(0.12)
    assert pd.isna(row["return_5d"])


def test_sell_trade_is_not_evaluated():
    result = calculate_actual_trade_performance(_actual_trades(side="sell"), _daily_bars())

    row = result.iloc[0]
    assert row["is_valid"] == False
    assert row["invalid_reason"] == "sell_trade_not_evaluated"
    assert pd.isna(row["return_1d"])


def test_execution_review_fields_are_merged_and_comment_mentions_flags():
    execution = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "trade_time": ["10:00:00"],
            "code": ["600000"],
            "side": ["buy"],
            "plan_match_status": ["matched"],
            "execution_status": ["off_plan"],
            "execution_flags": ["chase_above_entry,over_position"],
        }
    )

    result = calculate_actual_trade_performance(_actual_trades(), _daily_bars(), execution)

    row = result.iloc[0]
    assert row["plan_match_status"] == "matched"
    assert row["execution_status"] == "off_plan"
    assert "chase_above_entry" in row["execution_flags"]
    assert "追高偏差" in row["performance_comment"]
    assert "计划外交易" in row["performance_comment"]


def test_no_daily_bars_found_marks_invalid():
    result = calculate_actual_trade_performance(_actual_trades(), pd.DataFrame())

    assert result.loc[0, "is_valid"] == False
    assert result.loc[0, "invalid_reason"] == "no_daily_bars_found"
