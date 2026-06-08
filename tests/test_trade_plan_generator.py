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
            "strategy_names": ["trend_pullback", "breakout_volume", "support_rebound", None, None],
            "strategy_versions": ["v1", "v2", "v3_loose", None, None],
            "active_signal_count": [1, 1, 1, None, None],
            "avg_strategy_weight": [1.2, 0.9, 1.0, None, None],
            "recommendations": ["enable_observation", "reduce_or_pause", "enable_observation", None, None],
            "risk_flags": ["low_risk", "high_drawdown", "", None, None],
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


def test_generate_trade_plan_keeps_strategy_evaluation_fields_and_reason():
    result = generate_trade_plan(_candidate_rows(), max_items=1)
    row = result.iloc[0]

    assert row["strategy_names"] == "trend_pullback"
    assert row["strategy_versions"] == "v1"
    assert row["recommendations"] == "enable_observation"
    assert row["active_signal_count"] == 1
    assert row["avg_strategy_weight"] == pytest.approx(1.2)
    assert "策略来源：trend_pullback:v1" in row["plan_reason"]
    assert "策略建议：enable_observation" in row["plan_reason"]
    assert "平均策略权重：1.20" in row["plan_reason"]


def test_generate_trade_plan_allows_missing_strategy_evaluation_fields():
    candidates = _candidate_rows().drop(
        columns=[
            "strategy_names",
            "strategy_versions",
            "active_signal_count",
            "avg_strategy_weight",
            "recommendations",
            "risk_flags",
        ]
    )

    result = generate_trade_plan(candidates, max_items=1)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "strategy_versions"])
    assert "策略来源" not in result.loc[0, "plan_reason"]


def test_generate_trade_plan_writes_limit_risk_reasons():
    candidates = _candidate_rows()
    candidates.loc[candidates["code"] == "600000", "risk_flags"] = "limit_up_close"
    candidates.loc[candidates["code"] == "000001", "risk_flags"] = "limit_down_close"

    result = generate_trade_plan(candidates, max_items=2).set_index("code")

    assert "涨停收盘，次日可能存在买入不可执行风险" in result.loc["600000", "plan_reason"]
    assert result.loc["000001", "action"] == "仅观察"
    assert "跌停收盘，短期流动性和风险较高" in result.loc["000001", "plan_reason"]


def test_generate_trade_plan_downgrades_suspended_to_watch_only():
    candidates = _candidate_rows()
    candidates.loc[candidates["code"] == "600000", "risk_flags"] = "suspended"

    result = generate_trade_plan(candidates, max_items=1)

    assert result.loc[0, "strategy_type"] == "watch_only"
    assert result.loc[0, "action"] == "仅观察"
    assert "停牌或停牌风险，暂不生成买入计划" in result.loc[0, "plan_reason"]


def test_generate_trade_plan_writes_daily_basic_missing_risk_reason():
    candidates = _candidate_rows()
    candidates.loc[candidates["code"] == "600000", "risk_flags"] = "missing_daily_basic,missing_market_value"

    result = generate_trade_plan(candidates, max_items=1)

    assert "部分 daily_basic 扩展指标缺失，需降低置信度" in result.loc[0, "plan_reason"]


def test_generate_trade_plan_writes_market_high_risk_reason():
    candidates = _candidate_rows()
    candidates.loc[candidates["code"] == "600000", "risk_flags"] = "market_high_risk"

    result = generate_trade_plan(candidates, max_items=1)

    assert "当前市场环境偏弱，计划置信度需降低" in result.loc[0, "plan_reason"]


def test_generate_trade_plan_writes_moneyflow_risk_reasons():
    candidates = _candidate_rows()
    candidates.loc[candidates["code"] == "600000", "risk_flags"] = "strong_main_outflow"
    candidates.loc[candidates["code"] == "000001", "risk_flags"] = "main_outflow"
    candidates.loc[candidates["code"] == "000002", "risk_flags"] = "strong_main_inflow"

    result = generate_trade_plan(candidates, max_items=3).set_index("code")

    assert "主力资金明显流出，计划置信度降低" in result.loc["600000", "plan_reason"]
    assert "资金流偏弱，需观察承接" in result.loc["000001", "plan_reason"]
    assert "资金流相对积极，但仍需结合价格和计划区间执行" in result.loc["000002", "plan_reason"]


def test_generate_trade_plan_writes_industry_risk_reasons():
    candidates = _candidate_rows()
    candidates.loc[candidates["code"] == "600000", "risk_flags"] = "weak_industry"
    candidates.loc[candidates["code"] == "000001", "risk_flags"] = "strong_industry"
    candidates.loc[candidates["code"] == "000002", "risk_flags"] = "missing_industry_strength"

    result = generate_trade_plan(candidates, max_items=3).set_index("code")

    assert "所属行业相对弱势，计划置信度降低" in result.loc["600000", "plan_reason"]
    assert "所属行业相对强势，存在板块共振加分，但仍需按计划执行" in result.loc["000001", "plan_reason"]
    assert "行业强度数据缺失，需降低判断置信度" in result.loc["000002", "plan_reason"]
