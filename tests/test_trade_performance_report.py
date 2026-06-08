import pandas as pd

from src.reports.trade_performance_report import generate_trade_performance_report


def test_generate_trade_performance_report_outputs_markdown_and_avoids_forbidden_phrases():
    performance = pd.DataFrame(
        {
            "trade_date": ["2025-01-10", "2025-01-10"],
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "entry_price": [10.0, 20.0],
            "execution_status": ["follow_plan", "off_plan"],
            "execution_flags": ["price_in_range", "chase_above_entry"],
            "return_1d": [0.02, -0.01],
            "return_3d": [0.04, -0.03],
            "return_5d": [0.06, -0.02],
            "max_drawdown_3d": [-0.01, -0.06],
            "max_favorable_3d": [0.06, 0.01],
            "performance_comment": ["短期表现较稳", "买入后回撤较大"],
            "is_valid": [True, True],
        }
    )

    report = generate_trade_performance_report(performance, trade_date="2025-01-10")

    assert report.startswith("# A股实盘交易表现复盘报告")
    assert "## 二、交易表现总览" in report
    assert "| code | name | entry_price |" in report
    assert "计划外交易表现" in report
    for phrase in ("保证" + "盈利", "稳" + "赚", "满" + "仓"):
        assert phrase not in report
