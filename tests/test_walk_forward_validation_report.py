import pandas as pd

from src.reports.walk_forward_validation_report import generate_walk_forward_validation_report


def test_generate_walk_forward_validation_report_creates_markdown_without_forbidden_phrases():
    validation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "breakout_volume"],
            "strategy_version": ["search_001", "search_002"],
            "train_valid_count": [30, 30],
            "train_win_rate_3d": [0.60, 0.70],
            "train_avg_return_3d": [0.02, 0.03],
            "validation_valid_count": [12, 12],
            "validation_win_rate_3d": [0.58, 0.40],
            "validation_avg_return_3d": [0.01, -0.02],
            "return_decay": [-0.01, -0.05],
            "win_rate_decay": [-0.02, -0.30],
            "stability_score": [20.12346, 5.0],
            "overfit_risk": ["low", "high"],
            "validation_status": ["passed_oos", "failed_oos"],
            "validation_reason": ["样本外表现基本稳定。", "训练区间为正但样本外转负。"],
        }
    )

    report = generate_walk_forward_validation_report(
        validation,
        train_start_date="2026-01-01",
        train_end_date="2026-01-30",
        validation_start_date="2026-02-01",
        validation_end_date="2026-02-28",
        report_date="2026-06-01",
    )

    assert report.startswith("# 策略样本外验证报告")
    assert "## 四、样本外验证总表" in report
    assert "passed_oos 数量：1" in report
    assert "58.00%" in report
    assert "20.1235" in report
    for phrase in ["保证盈利", "稳赚", "满仓"]:
        assert phrase not in report
