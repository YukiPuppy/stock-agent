import pandas as pd

from src.reports.strategy_admission_report import generate_strategy_admission_report


def test_generate_strategy_admission_report_builds_markdown_without_forbidden_phrases():
    admission = pd.DataFrame(
        {
            "strategy_name": ["trend", "breakout"],
            "strategy_version": ["v1", "v2"],
            "source": ["manual_version", "parameter_search"],
            "valid_count": [40, 20],
            "evaluation_recommendation": ["enable_observation", "pause"],
            "oos_status": ["passed_oos", "failed_oos"],
            "oos_risk": ["low", "high"],
            "trade_plan_trigger_rate": [0.4, 0.1],
            "trade_plan_win_rate": [0.55, 0.3],
            "trade_plan_avg_return": [0.02, -0.01],
            "admission_score": [100.0, -100.0],
            "admission_status": ["qualified_for_observation", "oos_failed"],
            "admission_recommendation": ["enable_observation_candidate", "do_not_enable"],
            "admission_reason": ["满足观察候选条件。", "样本外验证未通过或不稳定。"],
        }
    )

    report = generate_strategy_admission_report(admission, report_date="2026-01-02")

    assert report.startswith("# 策略准入与观察候选报告")
    assert "## 三、策略准入总表" in report
    assert "40.00%" in report
    assert "100.0000" in report
    for phrase in ["保证盈利", "稳赚", "满仓"]:
        assert phrase not in report


def test_generate_strategy_admission_report_empty_candidate_message():
    report = generate_strategy_admission_report(pd.DataFrame(), report_date="2026-01-02")

    assert "当前没有满足观察候选条件的策略版本。" in report
