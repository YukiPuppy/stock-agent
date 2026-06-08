import pandas as pd

from src.reports.position_review_report import generate_position_review_report


def test_generate_position_review_report_creates_markdown_without_forbidden_phrases():
    positions = pd.DataFrame(
        {
            "as_of_date": ["2025-01-10"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "holding_volume": [100],
            "available_volume": [0],
            "frozen_volume": [100],
            "cost_price": [10.0],
            "latest_price": [9.0],
            "market_value": [900.0],
            "floating_pnl": [-100.0],
            "floating_pnl_pct": [-0.1],
            "t_plus_1_status": ["not_sellable_today"],
            "position_status": ["loss_warning"],
        }
    )
    review = pd.DataFrame(
        {
            "as_of_date": ["2025-01-10"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "position_risk_level": ["high"],
            "position_flags": ["below_stop_loss,t_plus_1_locked"],
            "position_comment": ["当前价格低于或接近计划止损价"],
            "next_action_hint": ["受 T+1 限制，需次日优先处理风险"],
        }
    )

    report = generate_position_review_report(positions, review, as_of_date="2025-01-10")

    assert report.startswith("# A股持仓与T+1风险检查报告")
    assert "## 三、持仓明细" in report
    assert "below_stop_loss" in report
    for phrase in ["保证" + "盈利", "稳" + "赚", "满" + "仓"]:
        assert phrase not in report
