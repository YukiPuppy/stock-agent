import pandas as pd
import pytest

from src.trading.positions import build_positions_from_trades, review_positions


def test_buy_trade_builds_position_and_t_plus_1_lock():
    positions = build_positions_from_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "price": [10.0],
                "volume": [100],
                "amount": [1000.0],
                "position_ratio": [0.1],
                "strategy_name": ["trend"],
                "plan_rank": [1],
            }
        ),
        as_of_date="2025-01-10",
    )

    assert len(positions) == 1
    assert positions.loc[0, "holding_volume"] == 100
    assert positions.loc[0, "frozen_volume"] == 100
    assert positions.loc[0, "available_volume"] == 0
    assert positions.loc[0, "t_plus_1_status"] == "not_sellable_today"


def test_sell_trade_reduces_position_and_keeps_weighted_cost():
    positions = build_positions_from_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-09", "2025-01-09", "2025-01-10"],
                "trade_time": ["10:00:00", "11:00:00", "13:00:00"],
                "code": ["600000", "600000", "600000"],
                "name": ["浦发银行", "浦发银行", "浦发银行"],
                "side": ["buy", "buy", "sell"],
                "price": [10.0, 12.0, 13.0],
                "volume": [100, 100, 50],
                "amount": [1000.0, 1200.0, 650.0],
                "position_ratio": [0.1, 0.2, 0.15],
                "strategy_name": ["trend", "trend", "trend"],
                "plan_rank": [1, 1, 1],
            }
        ),
        as_of_date="2025-01-10",
    )

    assert positions.loc[0, "holding_volume"] == 150
    assert positions.loc[0, "cost_amount"] == 1650
    assert positions.loc[0, "cost_price"] == 11


def test_fully_sold_position_is_not_output():
    positions = build_positions_from_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-09", "2025-01-10"],
                "trade_time": ["10:00:00", "13:00:00"],
                "code": ["600000", "600000"],
                "name": ["浦发银行", "浦发银行"],
                "side": ["buy", "sell"],
                "price": [10.0, 11.0],
                "volume": [100, 100],
                "amount": [1000.0, 1100.0],
                "position_ratio": [0.1, 0.0],
                "strategy_name": ["trend", "trend"],
                "plan_rank": [1, 1],
            }
        ),
        as_of_date="2025-01-10",
    )

    assert positions.empty


def test_daily_bars_provide_latest_price_and_floating_pnl_pct():
    positions = build_positions_from_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-09"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "price": [10.0],
                "volume": [100],
                "amount": [1000.0],
                "position_ratio": [0.1],
                "strategy_name": ["trend"],
                "plan_rank": [1],
            }
        ),
        daily_bars=pd.DataFrame(
            {
                "trade_date": ["2025-01-09", "2025-01-10"],
                "code": ["600000", "600000"],
                "close": [10.5, 11.0],
            }
        ),
        as_of_date="2025-01-10",
    )

    assert positions.loc[0, "latest_price"] == 11
    assert positions.loc[0, "floating_pnl"] == 100
    assert positions.loc[0, "floating_pnl_pct"] == pytest.approx(0.1)
    assert positions.loc[0, "position_status"] == "profit_watch"


def test_review_positions_flags_stop_loss_take_profit_and_position_ratio():
    positions = pd.DataFrame(
        {
            "as_of_date": ["2025-01-10", "2025-01-10", "2025-01-10"],
            "code": ["600000", "000001", "300001"],
            "name": ["浦发银行", "平安银行", "特锐德"],
            "holding_volume": [100, 100, 100],
            "available_volume": [0, 100, 100],
            "frozen_volume": [100, 0, 0],
            "cost_amount": [1000.0, 1000.0, 1000.0],
            "cost_price": [10.0, 10.0, 10.0],
            "latest_price": [9.0, 12.0, 10.0],
            "market_value": [900.0, 1200.0, 1000.0],
            "floating_pnl": [-100.0, 200.0, 0.0],
            "floating_pnl_pct": [-0.1, 0.2, 0.0],
            "position_ratio": [0.1, 0.1, 0.3],
            "first_buy_date": ["2025-01-10", "2025-01-09", "2025-01-09"],
            "latest_trade_date": ["2025-01-10", "2025-01-09", "2025-01-09"],
            "strategy_name": ["trend", "trend", "trend"],
            "plan_rank": [1, 2, 3],
            "t_plus_1_status": ["not_sellable_today", "sellable", "sellable"],
            "position_status": ["loss_warning", "profit_watch", "normal"],
        }
    )
    plan = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10", "2025-01-10"],
            "code": ["600000", "000001", "300001"],
            "stop_loss": [9.5, 9.0, 9.0],
            "take_profit_1": [12.0, 11.0, 12.0],
            "take_profit_2": [13.0, 13.0, 14.0],
        }
    )

    review = review_positions(positions, plan, as_of_date="2025-01-10")

    flags = dict(zip(review["code"], review["position_flags"], strict=True))
    assert "below_stop_loss" in flags["600000"]
    assert "t_plus_1_locked" in flags["600000"]
    assert "take_profit_zone" in flags["000001"]
    assert "high_position_ratio" in flags["300001"]
    assert review.loc[review["code"] == "600000", "position_risk_level"].iloc[0] == "high"
    assert review.loc[review["code"] == "300001", "position_risk_level"].iloc[0] == "medium"
