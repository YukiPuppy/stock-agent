import pandas as pd

from src.reports.daily_review_report import generate_daily_review_report


def test_generate_daily_review_report_renders_markdown_without_forbidden_phrases():
    daily_review = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "actual_trade_count": [1],
            "buy_count": [1],
            "sell_count": [0],
            "matched_plan_count": [1],
            "off_plan_count": [0],
            "deviation_count": [0],
            "chase_count": [0],
            "over_position_count": [0],
            "execution_score": [100],
            "main_issues": ["未发现明显执行偏差"],
            "next_action_suggestion": ["执行良好，建议继续保持，后续结合收益结果评估策略有效性。"],
        }
    )
    execution_review = pd.DataFrame(
        {
            "code": ["600000"],
            "name": ["浦发银行"],
            "side": ["buy"],
            "actual_price": [10.5],
            "plan_match_status": ["matched"],
            "execution_status": ["follow_plan"],
            "execution_flags": ["price_in_range"],
            "execution_comment": [""],
        }
    )
    actual_trades = pd.DataFrame(
        {
            "trade_time": ["09:45:00"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "side": ["buy"],
            "price": [10.5],
            "volume": [100],
            "amount": [1050.0],
            "position_ratio": [0.1],
            "reason": ["按计划"],
            "note": [""],
        }
    )

    report = generate_daily_review_report(daily_review, execution_review, actual_trades)

    assert "# A股盘后执行复盘报告" in report
    assert "## 二、当日执行总览" in report
    assert "| code | name | side | actual_price |" in report
    assert "600000" in report
    for phrase in ("保证" + "盈利", "稳" + "赚", "满" + "仓"):
        assert phrase not in report
