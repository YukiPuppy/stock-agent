import pandas as pd

from src.trading.execution_review import EXECUTION_REVIEW_COLUMNS, review_execution


def _plan(entry_low=10.0, entry_high=11.0, position_high=0.2):
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "rank": [1],
            "action": ["回踩低吸"],
            "entry_low": [entry_low],
            "entry_high": [entry_high],
            "position_low": [0.1],
            "position_high": [position_high],
            "stop_loss": [9.5],
            "take_profit_1": [11.5],
            "take_profit_2": [12.0],
        }
    )


def _actual(price=10.5, position_ratio=0.15, code="600000", side="buy"):
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "trade_time": ["09:45:00"],
            "code": [code],
            "name": ["浦发银行"],
            "side": [side],
            "price": [price],
            "volume": [100],
            "position_ratio": [position_ratio],
        }
    )


def test_review_execution_empty_actual_returns_standard_columns():
    result = review_execution(pd.DataFrame(), _plan())

    assert result.empty
    assert result.columns.tolist() == EXECUTION_REVIEW_COLUMNS


def test_review_execution_identifies_follow_plan_buy():
    result = review_execution(_actual(), _plan())

    assert result.loc[0, "execution_status"] == "follow_plan"
    assert result.loc[0, "execution_flags"] == "price_in_range"
    assert result.loc[0, "planned_action"] == "回踩低吸"


def test_review_execution_identifies_chase_above_entry():
    result = review_execution(_actual(price=11.2), _plan())

    assert result.loc[0, "execution_status"] == "deviation"
    assert "chase_above_entry" in result.loc[0, "execution_flags"]
    assert "买入价格高于计划买入区间上沿，存在追高偏差" in result.loc[0, "execution_comment"]


def test_review_execution_identifies_over_position():
    result = review_execution(_actual(position_ratio=0.3), _plan())

    assert result.loc[0, "execution_status"] == "deviation"
    assert "over_position" in result.loc[0, "execution_flags"]
    assert "实际仓位超过计划仓位上限" in result.loc[0, "execution_comment"]


def test_review_execution_identifies_no_plan_trade():
    result = review_execution(_actual(code="000001"), _plan())

    assert result.loc[0, "plan_match_status"] == "no_plan"
    assert result.loc[0, "execution_status"] == "off_plan"
    assert result.loc[0, "execution_flags"] == "plan_not_found"


def test_review_execution_identifies_watch_only_buy():
    result = review_execution(_actual(), _plan(entry_low=None, entry_high=None, position_high=0.0))

    assert result.loc[0, "execution_status"] == "deviation"
    assert "bought_watch_only" in result.loc[0, "execution_flags"]
    assert "该标的计划为仅观察，但实际发生买入" in result.loc[0, "execution_comment"]
