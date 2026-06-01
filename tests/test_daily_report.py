import pandas as pd

from src.reports.daily_report import generate_daily_report


def _trade_plan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "rank": [1],
            "close": [10.5],
            "strategy_type": ["trend_pullback"],
            "action": ["回踩低吸"],
            "entry_low": [10.24],
            "entry_high": [10.45],
            "position_low": [0.1],
            "position_high": [0.2],
            "stop_loss": [9.97],
            "take_profit_1": [10.92],
            "take_profit_2": [11.34],
            "invalid_condition": ["跌破买入区间下沿且无法收回，计划失效。"],
            "t_plus_1_risk": ["A股 T+1 风险需控制隔夜风险。"],
            "plan_reason": ["趋势较强，量价配合。"],
        }
    )


def _candidate_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "rank": [1],
            "code": ["600000"],
            "name": ["浦发银行"],
            "close": [10.5],
            "pct_chg_5d": [0.05],
            "volume_ratio_5": [1.3],
            "close_position_20": [0.8],
            "score": [88.0],
            "reason": ["趋势评分靠前"],
        }
    )


def test_generate_daily_report_contains_trade_plan_details():
    report = generate_daily_report(_trade_plan(), _candidate_pool())

    assert "600000" in report
    assert "浦发银行" in report
    assert "买入区间：10.24 ~ 10.45" in report
    assert "止损价：9.97" in report
    assert "止盈价：10.92 ~ 11.34" in report
    assert "T+1 风险" in report


def test_generate_daily_report_empty_trade_plan_message():
    report = generate_daily_report(pd.DataFrame(), _candidate_pool())

    assert "当前没有生成可执行交易计划。" in report
    assert "报告日期：2026-01-02" in report


def test_generate_daily_report_empty_candidate_pool_message():
    report = generate_daily_report(_trade_plan(), pd.DataFrame())

    assert "当前候选股池为空。" in report


def test_generate_daily_report_does_not_include_forbidden_phrases():
    report = generate_daily_report(_trade_plan(), _candidate_pool())

    for phrase in ("保证盈利", "稳赚", "满仓"):
        assert phrase not in report
