import pandas as pd

from src.reports.system_health_report import generate_system_health_report


def test_generate_system_health_report_renders_markdown():
    summary = {
        "overall_status": "partial",
        "blocking_issues": ["daily_bars 为空"],
        "warnings": ["actual_trades 为空"],
        "next_suggestions": ["daily_bars 为空：建议先运行 update_daily_bars"],
        "table_health": pd.DataFrame(
            [{"table_name": "daily_bars", "row_count": 0, "status": "empty", "message": "table is empty"}]
        ),
        "config_files": pd.DataFrame(
            [{"file_name": "strategy_versions.json", "path": "configs/strategy_versions.json", "exists": True}]
        ),
        "report_files": pd.DataFrame(
            [{"pattern": "daily_report_*.md", "file_count": 1, "latest_file": "reports/daily_report.md"}]
        ),
    }

    report = generate_system_health_report(summary, report_date="2026-01-02")

    assert "# A股多智能体选股系统健康检查报告" in report
    assert "## 一、总体状态" in report
    assert "partial" in report
    assert "| table_name | row_count | status | message |" in report
    assert "不构成投资建议" in report


def test_generate_system_health_report_does_not_include_forbidden_phrases():
    phrase_1 = "保证" + "盈利"
    phrase_2 = "稳" + "赚"
    phrase_3 = "满" + "仓"
    report = generate_system_health_report(
        {
            "overall_status": "partial",
            "blocking_issues": [phrase_1],
            "warnings": [phrase_2],
            "next_suggestions": [phrase_3],
        },
        report_date="2026-01-02",
    )

    assert phrase_1 not in report
    assert phrase_2 not in report
    assert phrase_3 not in report
