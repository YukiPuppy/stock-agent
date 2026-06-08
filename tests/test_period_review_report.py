import pandas as pd

from src.reports.period_review_report import generate_period_review_report


def test_generate_period_review_report_creates_markdown_without_forbidden_phrases():
    period_review = pd.DataFrame(
        {
            "start_date": ["2025-01-01"],
            "end_date": ["2025-01-31"],
            "actual_trade_count": [2],
            "buy_count": [2],
            "sell_count": [0],
            "follow_plan_count": [1],
            "off_plan_count": [1],
            "deviation_count": [1],
            "chase_count": [1],
            "over_position_count": [0],
            "avg_execution_score": [85.0],
            "valid_performance_count": [2],
            "avg_return_1d": [0.0],
            "avg_return_3d": [0.01],
            "avg_return_5d": [0.02],
            "plan_trade_avg_return_3d": [0.03],
            "off_plan_avg_return_3d": [-0.01],
            "chase_avg_return_3d": [-0.01],
            "over_position_avg_return_3d": [None],
            "best_trade_code": ["600000"],
            "worst_trade_code": ["000001"],
            "main_issues": ["计划外交易 1 笔；追高偏差 1 笔"],
            "next_period_suggestion": ["建议减少计划外交易。"],
        }
    )
    trade_performance = pd.DataFrame(
        {
            "code": ["600000", "000001"],
            "execution_status": ["follow_plan", "off_plan"],
            "execution_flags": ["price_in_range", "chase_above_entry"],
            "return_3d": [0.03, -0.01],
            "is_valid": [True, True],
        }
    )

    report = generate_period_review_report(period_review, trade_performance=trade_performance)

    assert "# A股周期执行复盘报告" in report
    assert "## 五、执行与结果关系" in report
    assert "样本数量较少，暂不宜据此调整策略参数" in report
    for phrase in ["保证" + "盈利", "稳" + "赚", "满" + "仓"]:
        assert phrase not in report
