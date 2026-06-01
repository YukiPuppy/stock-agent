import pandas as pd
import pytest

from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS, generate_trade_plan


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02", "2026-01-02", "2026-01-02", "2026-01-02"],
            "code": ["600000", "000001", "000002", "000003", "000004"],
            "name": ["趋势股", "突破股", "支撑股", "观察股", "超出股"],
            "close": [10.123, 20.0, 30.0, 40.0, 50.0],
            "pct_chg_1d": [0.01, 0.02, -0.04, 0.01, 0.01],
            "pct_chg_3d": [0.02, 0.03, -0.02, 0.01, 0.01],
            "pct_chg_5d": [0.04, 0.06, -0.01, 0.01, 0.01],
            "pct_chg_10d": [0.05, 0.07, 0.02, 0.01, 0.01],
            "volume_ratio_5": [1.0, 1.2, 0.9, 1.0, 1.0],
            "close_position_20": [0.6, 0.7, 0.4, 0.5, 0.5],
            "above_ma5": [True, False, False, False, False],
            "above_ma10": [True, False, False, False, False],
            "above_ma20": [True, True, True, False, False],
            "amount_ma5": [200000000.0] * 5,
            "score": [90.0, 80.0, 70.0, 60.0, 50.0],
            "rank": [1, 2, 3, 4, 6],
            "reason": ["趋势较强", "突破形态", "支撑附近", "条件不足", "超出范围"],
        }
    )


def test_generate_trade_plan_trend_pullback():
    result = generate_trade_plan(_candidate_rows(), max_items=5)
    row = result[result["code"] == "600000"].iloc[0]

    assert row["strategy_type"] == "trend_pullback"
    assert row["action"] == "回踩低吸"
    assert row["entry_low"] == 9.87
    assert row["entry_high"] == 10.07
    assert row["stop_loss"] == 9.62
    assert row["take_profit_1"] == 10.53
    assert row["take_profit_2"] == 10.93
    assert row["position_low"] == pytest.approx(0.10)
    assert row["position_high"] == pytest.approx(0.20)


def test_generate_trade_plan_breakout_watch():
    result = generate_trade_plan(_candidate_rows(), max_items=5)
    row = result[result["code"] == "000001"].iloc[0]

    assert row["strategy_type"] == "breakout_watch"
    assert row["action"] == "突破观察"
    assert row["entry_low"] == 20.10
    assert row["entry_high"] == 20.50
    assert row["stop_loss"] == 19.40
    assert row["take_profit_1"] == 21.00
    assert row["take_profit_2"] == 22.00
    assert row["position_low"] == pytest.approx(0.05)
    assert row["position_high"] == pytest.approx(0.15)


def test_generate_trade_plan_support_watch():
    result = generate_trade_plan(_candidate_rows(), max_items=5)
    row = result[result["code"] == "000002"].iloc[0]

    assert row["strategy_type"] == "support_watch"
    assert row["action"] == "支撑观察"
    assert row["entry_low"] == 28.80
    assert row["entry_high"] == 29.55
    assert row["stop_loss"] == 28.20
    assert row["take_profit_1"] == 31.05
    assert row["take_profit_2"] == 32.10
    assert row["position_low"] == pytest.approx(0.05)
    assert row["position_high"] == pytest.approx(0.15)


def test_generate_trade_plan_watch_only():
    result = generate_trade_plan(_candidate_rows(), max_items=5)
    row = result[result["code"] == "000003"].iloc[0]

    assert row["strategy_type"] == "watch_only"
    assert row["action"] == "仅观察"
    assert pd.isna(row["entry_low"])
    assert pd.isna(row["entry_high"])
    assert pd.isna(row["stop_loss"])
    assert pd.isna(row["take_profit_1"])
    assert pd.isna(row["take_profit_2"])
    assert row["position_low"] == 0
    assert row["position_high"] == 0


def test_generate_trade_plan_max_items_and_standard_empty_columns():
    result = generate_trade_plan(_candidate_rows(), max_items=3)

    assert result["code"].tolist() == ["600000", "000001", "000002"]

    empty = generate_trade_plan(pd.DataFrame())
    assert empty.empty
    assert empty.columns.tolist() == TRADE_PLAN_COLUMNS


def test_generate_trade_plan_contains_risk_and_reason_text():
    result = generate_trade_plan(_candidate_rows(), max_items=1)
    row = result.iloc[0]

    assert "A股T+1机制" in row["t_plus_1_risk"]
    assert "趋势较强" in row["plan_reason"]
    assert "score=90.00" in row["plan_reason"]
    assert "5日涨跌幅=4.00%" in row["plan_reason"]
