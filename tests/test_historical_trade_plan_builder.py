import pandas as pd

from src.backtest.historical_trade_plan_builder import build_historical_trade_plans
from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS


def test_build_historical_trade_plans_generates_multiple_dates():
    signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02"],
            "code": ["600000", "600000"],
            "strategy_name": ["trend_pullback", "trend_pullback"],
            "strategy_version": ["v1", "v1"],
            "signal_strength": [20.0, 30.0],
            "entry_reason": ["趋势", "趋势"],
            "risk_flags": ["", ""],
        }
    )
    factors = pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02"],
            "code": ["600000", "600000"],
            "close": [10.0, 11.0],
            "pct_chg_1d": [0.01, 0.01],
            "pct_chg_3d": [0.02, 0.02],
            "pct_chg_5d": [0.03, 0.03],
            "pct_chg_10d": [0.04, 0.04],
            "volume_ratio_5": [1.2, 1.3],
            "close_position_20": [0.7, 0.75],
            "above_ma5": [True, True],
            "above_ma10": [True, True],
            "above_ma20": [True, True],
            "amount_ma5": [1000.0, 1000.0],
        }
    )
    stock_basic = pd.DataFrame({"code": ["600000"], "name": ["测试股"], "market": ["SH"], "board": ["main"]})

    result = build_historical_trade_plans(signals, factors, stock_basic=stock_basic, top_n=1, max_plan_items=1)

    assert result["trade_date"].tolist() == ["2026-01-01", "2026-01-02"]
    assert result["code"].tolist() == ["600000", "600000"]
    assert result.columns.tolist() == TRADE_PLAN_COLUMNS


def test_build_historical_trade_plans_empty_signals_returns_standard_columns():
    result = build_historical_trade_plans(pd.DataFrame(), pd.DataFrame())

    assert result.empty
    assert result.columns.tolist() == TRADE_PLAN_COLUMNS
