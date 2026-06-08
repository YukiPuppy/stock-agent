import pandas as pd

from src.reports.data_quality_report import generate_data_quality_report


def test_generate_data_quality_report_renders_markdown():
    report = generate_data_quality_report(
        pd.DataFrame(
            [{"check_name": "empty_data", "status": "error", "issue_count": 1, "message": "empty"}]
        ),
        compare_summary=pd.DataFrame([{"field": "close", "issue_count": 1, "max_relative_diff": 0.02}]),
        report_date="2026-01-02",
    )

    assert "# 数据质量与数据源对齐检查报告" in report
    assert "正式行情源为 Tushare Pro，AKShare 仅作为可选诊断源" in report
    assert "daily_bars 使用 Tushare Pro 标准单位：volume 为手，amount 为千元" in report
    assert "volume 为手" in report
    assert "amount 为千元" in report
    assert "daily_factors.amount_ma5 和 min_amount_ma5 单位为千元" in report
    assert "AKShare 数据在 Provider 层将 amount 转换为千元，volume 保持与 Tushare vol 对齐" not in report
    assert "AKShare volume 除以 100" not in report
    assert "不构成投资建议" in report
    assert "保证盈利" not in report
    assert "稳赚" not in report
    assert "满仓" not in report


def test_generate_data_quality_report_renders_empty_provider_compare_message():
    report = generate_data_quality_report(
        pd.DataFrame([{"check_name": "empty_data", "status": "ok", "issue_count": 0, "message": "ok"}]),
        compare_result=pd.DataFrame(
            columns=[
                "trade_date",
                "code",
                "field",
                "left_value",
                "right_value",
                "relative_diff",
                "status",
                "message",
            ]
        ),
        compare_summary=pd.DataFrame(columns=["field", "issue_count", "max_relative_diff", "avg_relative_diff", "status"]),
        report_date="2026-01-02",
    )

    assert "当前未执行或暂无数据源对齐异常。" in report
    assert "actual_trades.amount 仍为元" in report
    assert "positions 金额字段仍为元" in report
    assert "volume relative_diff" not in report
    assert "amount relative_diff" not in report
