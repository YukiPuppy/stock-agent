import pandas as pd

from src.reports.strategy_evaluation_report import generate_strategy_evaluation_report


def _evaluation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_name": [
                "trend_pullback",
                "breakout_volume",
                "support_rebound",
                "weak_strategy",
            ],
            "strategy_version": ["v1", "v2", "v1", "v3"],
            "valid_count": [40, 8, 30, 25],
            "win_rate_3d": [0.625, 0.5, 0.45, 0.2],
            "avg_return_3d": [0.0312, 0.01, -0.004, -0.025],
            "median_return_3d": [0.02, 0.008, -0.002, -0.02],
            "avg_max_drawdown_3d": [-0.018, -0.035, -0.04, -0.08],
            "evaluation_score": [0.812345, 0.45, 0.25, 0.1],
            "evaluation_status": ["qualified", "insufficient_samples", "weak", "weak"],
            "risk_level": ["low", "medium", "medium", "high"],
            "recommendation": [
                "enable_observation",
                "continue_backtest",
                "reduce_or_pause",
                "pause",
            ],
            "evaluation_reason": [
                "胜率和收益达到观察条件。",
                "样本数量不足，需要继续验证。",
                "收益偏弱，建议降低权重。",
                "回撤和收益表现较弱。",
            ],
        }
    )


def _performance_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "win_rate_1d": [0.55],
            "win_rate_5d": [0.65],
            "avg_return_1d": [0.011],
            "avg_return_5d": [0.052],
            "avg_max_drawdown_1d": [-0.01],
            "avg_max_drawdown_5d": [-0.025],
        }
    )


def test_generate_strategy_evaluation_report_contains_core_sections_and_values():
    report = generate_strategy_evaluation_report(
        _evaluation_df(),
        _performance_df(),
        report_date="2026-01-02",
    )

    assert "# 策略版本评价报告" in report
    assert "trend_pullback" in report
    assert "v1" in report
    assert "enable_observation" in report
    assert "## 八、风险提示" in report
    assert "回测结果可能存在过拟合风险" in report


def test_generate_strategy_evaluation_report_empty_evaluation_message():
    report = generate_strategy_evaluation_report(pd.DataFrame(), report_date="2026-01-02")

    assert "当前没有可用的策略版本评价结果，请先运行 backtest_strategy_versions 和 evaluate_strategy_versions。" in report


def test_generate_strategy_evaluation_report_formats_percentages():
    report = generate_strategy_evaluation_report(_evaluation_df().iloc[[0]], report_date="2026-01-02")

    assert "62.50%" in report
    assert "3.12%" in report
    assert "-1.80%" in report
    assert "0.8123" in report


def test_enable_observation_strategy_appears_in_section():
    report = generate_strategy_evaluation_report(_evaluation_df(), report_date="2026-01-02")

    section = report.split("## 四、建议启用观察的策略", maxsplit=1)[1].split("## 五、需要继续回测的策略", maxsplit=1)[0]
    assert "trend_pullback:v1" in section
    assert "推荐理由" in section


def test_continue_backtest_strategy_appears_in_section():
    report = generate_strategy_evaluation_report(_evaluation_df(), report_date="2026-01-02")

    section = report.split("## 五、需要继续回测的策略", maxsplit=1)[1].split("## 六、建议降权或暂停的策略", maxsplit=1)[0]
    assert "breakout_volume:v2" in section


def test_pause_and_reduce_or_pause_strategies_appear_in_section():
    report = generate_strategy_evaluation_report(_evaluation_df(), report_date="2026-01-02")

    section = report.split("## 六、建议降权或暂停的策略", maxsplit=1)[1].split("## 七、策略版本详细评价", maxsplit=1)[0]
    assert "support_rebound:v1" in section
    assert "weak_strategy:v3" in section


def test_strategy_evaluation_report_does_not_include_forbidden_phrases():
    report = generate_strategy_evaluation_report(_evaluation_df(), report_date="2026-01-02")

    for phrase in ("保证" + "盈利", "稳" + "赚", "满" + "仓"):
        assert phrase not in report
