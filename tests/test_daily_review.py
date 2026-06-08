import pandas as pd

from src.trading.daily_review import DAILY_REVIEW_COLUMNS, generate_daily_review


def test_generate_daily_review_without_actual_trades_returns_no_trade_summary():
    result = generate_daily_review(
        actual_trades=pd.DataFrame(),
        execution_review=pd.DataFrame(),
        trade_plan=pd.DataFrame({"trade_date": ["2025-01-10"], "code": ["600000"]}),
        trade_date="2025-01-10",
    )

    assert result.columns.tolist() == DAILY_REVIEW_COLUMNS
    assert result.loc[0, "actual_trade_count"] == 0
    assert result.loc[0, "review_summary"] == "当日无实际交易记录。"
    assert result.loc[0, "next_action_suggestion"] == "无需执行偏差复盘，可继续观察系统计划表现。"


def test_generate_daily_review_counts_follow_plan_trade():
    actual = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10"],
            "side": ["buy", "sell"],
        }
    )
    execution = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10"],
            "plan_match_status": ["matched", "matched"],
            "execution_status": ["follow_plan", "recorded_sell"],
            "execution_flags": ["price_in_range", "sell_recorded"],
        }
    )

    result = generate_daily_review(actual, execution, pd.DataFrame({"code": ["600000"]}), "2025-01-10")

    assert result.loc[0, "actual_trade_count"] == 2
    assert result.loc[0, "buy_count"] == 1
    assert result.loc[0, "sell_count"] == 1
    assert result.loc[0, "planned_trade_count"] == 1
    assert result.loc[0, "matched_plan_count"] == 2
    assert result.loc[0, "follow_plan_count"] == 1
    assert result.loc[0, "execution_score"] == 100


def test_generate_daily_review_counts_issues_and_score_floor():
    actual = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"] * 4,
            "side": ["buy", "buy", "buy", "buy"],
        }
    )
    execution = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"] * 4,
            "plan_match_status": ["no_plan", "matched", "matched", "matched"],
            "execution_status": ["off_plan", "deviation", "deviation", "deviation"],
            "execution_flags": [
                "plan_not_found",
                "chase_above_entry,over_position",
                "bought_watch_only",
                "chase_above_entry,over_position,bought_watch_only",
            ],
        }
    )

    result = generate_daily_review(actual, execution, pd.DataFrame(), "2025-01-10")

    assert result.loc[0, "off_plan_count"] == 1
    assert result.loc[0, "deviation_count"] == 3
    assert result.loc[0, "chase_count"] == 2
    assert result.loc[0, "over_position_count"] == 2
    assert result.loc[0, "bought_watch_only_count"] == 2
    assert result.loc[0, "execution_score"] == 0
    assert "存在计划外交易" in result.loc[0, "main_issues"]
    assert "存在仓位超过计划上限" in result.loc[0, "main_issues"]


def test_generate_daily_review_includes_trade_performance_fields_and_summary_notes():
    actual = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10"],
            "side": ["buy", "buy"],
        }
    )
    execution = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10"],
            "plan_match_status": ["matched", "no_plan"],
            "execution_status": ["follow_plan", "off_plan"],
            "execution_flags": ["", "chase_above_entry"],
        }
    )
    performance = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10"],
            "is_valid": [True, True],
            "execution_status": ["follow_plan", "off_plan"],
            "execution_flags": ["", "chase_above_entry"],
            "return_1d": [0.01, -0.02],
            "return_3d": [0.03, -0.04],
            "return_5d": [0.05, -0.01],
            "max_drawdown_3d": [-0.01, -0.08],
        }
    )

    result = generate_daily_review(
        actual,
        execution,
        pd.DataFrame(),
        "2025-01-10",
        actual_trade_performance=performance,
    )

    assert result.loc[0, "valid_performance_count"] == 2
    assert result.loc[0, "avg_return_3d"] == -0.005000000000000001
    assert result.loc[0, "plan_trade_avg_return_3d"] == 0.03
    assert result.loc[0, "off_plan_avg_return_3d"] == -0.04
    assert result.loc[0, "chase_trade_count"] == 1
    assert "计划外交易表现偏弱" in result.loc[0, "review_summary"]
    assert "追高交易风险较高" in result.loc[0, "review_summary"]
