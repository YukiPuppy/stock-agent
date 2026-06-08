import pandas as pd

from src.reports.parameter_search_report import generate_parameter_search_report


def test_generate_parameter_search_report_outputs_markdown_and_risk_warning():
    report = generate_parameter_search_report(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "valid_count": [30],
                "win_rate_3d": [0.6],
                "avg_return_3d": [0.02],
                "median_return_3d": [0.01],
                "avg_max_drawdown_3d": [-0.03],
                "evaluation_score": [24.5],
                "evaluation_status": ["qualified"],
                "recommendation": ["enable_observation"],
            }
        ),
        report_date="2026-06-01",
    )

    assert "# 策略参数搜索报告" in report
    assert "参数搜索可能过拟合" in report
    assert "trend_pullback" in report


def test_parameter_search_report_does_not_include_forbidden_phrases():
    report = generate_parameter_search_report(pd.DataFrame())

    assert "保证盈利" not in report
    assert "稳赚" not in report
    assert "满仓" not in report
