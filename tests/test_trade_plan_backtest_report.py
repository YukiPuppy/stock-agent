import pandas as pd

from src.reports.trade_plan_backtest_report import generate_trade_plan_backtest_report


def test_generate_trade_plan_backtest_report_builds_markdown_without_forbidden_phrases():
    results = pd.DataFrame(
        {
            "plan_date": ["2026-01-01"],
            "code": ["600000"],
            "name": ["测试股"],
            "action": ["回踩低吸"],
            "entry_price": [10.0],
            "exit_price": [11.0],
            "exit_reason": ["take_profit_1"],
            "return_pct": [0.1],
            "max_drawdown": [-0.02],
            "max_favorable": [0.12],
            "invalid_reason": [""],
            "is_triggered": [True],
            "is_valid": [True],
        }
    )
    performance = pd.DataFrame(
        {
            "strategy_names": ["trend"],
            "strategy_versions": ["v1"],
            "action": ["回踩低吸"],
            "plan_count": [1],
            "trigger_rate": [1.0],
            "win_rate": [1.0],
            "avg_return": [0.1],
            "avg_max_drawdown": [-0.02],
            "stop_loss_rate": [0.0],
            "take_profit_rate": [1.0],
            "time_exit_rate": [0.0],
        }
    )

    report = generate_trade_plan_backtest_report(results, performance, report_date="2026-01-02")

    assert report.startswith("# 交易计划规则回测报告")
    assert "## 三、策略版本表现" in report
    for phrase in ("保证盈利", "稳赚", "满仓"):
        assert phrase not in report
