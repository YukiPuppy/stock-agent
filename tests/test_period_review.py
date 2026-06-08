import pandas as pd
import pytest

from src.trading.period_review import generate_period_review


def _sample_actual_trades():
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10", "2025-01-13", "2025-01-14"],
            "code": ["600000", "000001", "300001", "600001"],
            "side": ["buy", "buy", "sell", "buy"],
        }
    )


def _sample_execution_review():
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10", "2025-01-13", "2025-01-14"],
            "code": ["600000", "000001", "300001", "600001"],
            "side": ["buy", "buy", "sell", "buy"],
            "execution_status": ["follow_plan", "off_plan", "deviation", "deviation"],
            "execution_flags": ["price_in_range", "plan_not_found", "chase_above_entry,over_position", "bought_watch_only"],
        }
    )


def _sample_performance():
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10", "2025-01-13", "2025-01-14"],
            "code": ["600000", "000001", "300001", "600001"],
            "execution_status": ["follow_plan", "off_plan", "deviation", "deviation"],
            "execution_flags": ["price_in_range", "plan_not_found", "chase_above_entry,over_position", "bought_watch_only"],
            "return_1d": [0.01, -0.02, 0.03, 0.00],
            "return_3d": [0.03, -0.04, 0.06, -0.01],
            "return_5d": [0.05, -0.06, 0.08, -0.02],
            "is_valid": [True, True, True, False],
        }
    )


def test_period_review_empty_actual_trades_returns_standard_row():
    result = generate_period_review(
        actual_trades=pd.DataFrame(),
        execution_review=pd.DataFrame(),
        trade_performance=pd.DataFrame(),
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert len(result) == 1
    assert result.loc[0, "actual_trade_count"] == 0
    assert result.loc[0, "period_summary"] == "本周期无实际交易记录。"
    assert result.loc[0, "next_period_suggestion"] == "可继续观察系统计划表现，等待更多实盘样本。"


def test_period_review_counts_execution_status_and_flags():
    result = generate_period_review(
        actual_trades=_sample_actual_trades(),
        execution_review=_sample_execution_review(),
        trade_performance=_sample_performance(),
        daily_review=pd.DataFrame({"trade_date": ["2025-01-10", "2025-01-13"], "execution_score": [90, 70]}),
        start_date="2025-01-10",
        end_date="2025-01-31",
    )
    row = result.iloc[0]

    assert row["trading_days"] == 3
    assert row["actual_trade_count"] == 4
    assert row["buy_count"] == 3
    assert row["sell_count"] == 1
    assert row["follow_plan_count"] == 1
    assert row["off_plan_count"] == 1
    assert row["deviation_count"] == 2
    assert row["chase_count"] == 1
    assert row["over_position_count"] == 1
    assert row["bought_watch_only_count"] == 1
    assert row["avg_execution_score"] == pytest.approx(80)


def test_period_review_calculates_returns_and_extreme_codes():
    result = generate_period_review(
        actual_trades=_sample_actual_trades(),
        execution_review=_sample_execution_review(),
        trade_performance=_sample_performance(),
    )
    row = result.iloc[0]

    assert row["valid_performance_count"] == 3
    assert row["avg_return_1d"] == pytest.approx((0.01 - 0.02 + 0.03) / 3)
    assert row["avg_return_3d"] == pytest.approx((0.03 - 0.04 + 0.06) / 3)
    assert row["avg_return_5d"] == pytest.approx((0.05 - 0.06 + 0.08) / 3)
    assert row["plan_trade_avg_return_3d"] == pytest.approx(0.03)
    assert row["off_plan_avg_return_3d"] == pytest.approx(-0.04)
    assert row["chase_avg_return_3d"] == pytest.approx(0.06)
    assert row["over_position_avg_return_3d"] == pytest.approx(0.06)
    assert row["best_trade_code"] == "300001"
    assert row["worst_trade_code"] == "000001"


def test_period_review_filters_by_date_range():
    result = generate_period_review(
        actual_trades=_sample_actual_trades(),
        execution_review=_sample_execution_review(),
        trade_performance=_sample_performance(),
        start_date="2025-01-13",
        end_date="2025-01-14",
    )

    assert result.loc[0, "actual_trade_count"] == 2
    assert result.loc[0, "best_trade_code"] == "300001"
