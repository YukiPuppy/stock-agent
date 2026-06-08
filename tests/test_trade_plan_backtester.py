import pandas as pd

from src.backtest.trade_plan_backtester import (
    BACKTEST_RESULT_COLUMNS,
    backtest_trade_plans,
    evaluate_trade_plan_backtest,
)


def _plan(**overrides):
    row = {
        "trade_date": "2026-01-01",
        "code": "600000",
        "name": "测试股",
        "action": "回踩低吸",
        "entry_low": 10.0,
        "entry_high": 10.5,
        "stop_loss": 9.5,
        "take_profit_1": 11.0,
        "take_profit_2": 12.0,
        "position_low": 0.1,
        "position_high": 0.2,
        "strategy_names": "trend_pullback",
        "strategy_versions": "v1",
        "recommendations": "observe",
        "avg_strategy_weight": 1.0,
        "risk_flags": "",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _bars(rows):
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "code": "600000",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000,
                "amount": 10000,
            }
            for trade_date, open_, high, low, close in rows
        ]
    )


def test_empty_trade_plans_returns_standard_columns():
    result = backtest_trade_plans(pd.DataFrame(), pd.DataFrame())

    assert result.empty
    assert result.columns.tolist() == BACKTEST_RESULT_COLUMNS


def test_watch_only_or_no_entry_range_is_not_triggered():
    result = backtest_trade_plans(_plan(action="仅观察", entry_low=None), _bars([("2026-01-02", 10, 11, 9, 10)]))

    assert result.loc[0, "is_triggered"] == False
    assert result.loc[0, "is_valid"] == False
    assert result.loc[0, "invalid_reason"] == "watch_only_or_no_entry_range"


def test_entry_range_trigger_uses_open_price():
    result = backtest_trade_plans(
        _plan(),
        _bars([("2026-01-02", 10.2, 10.8, 10.0, 10.6), ("2026-01-03", 10.7, 10.9, 10.1, 10.8)]),
    )

    assert result.loc[0, "is_triggered"] == True
    assert result.loc[0, "entry_price"] == 10.2


def test_gap_above_entry_range_does_not_chase():
    result = backtest_trade_plans(_plan(), _bars([("2026-01-02", 11.0, 12.0, 10.6, 11.5)]))

    assert result.loc[0, "is_triggered"] == False
    assert result.loc[0, "invalid_reason"] == "gap_above_entry_range"


def test_not_reach_entry_range_is_not_triggered():
    result = backtest_trade_plans(_plan(), _bars([("2026-01-02", 9.5, 9.9, 9.2, 9.8)]))

    assert result.loc[0, "is_triggered"] == False
    assert result.loc[0, "invalid_reason"] == "not_reach_entry_range"


def test_stop_loss_triggers_and_same_day_priority_over_take_profit():
    result = backtest_trade_plans(_plan(), _bars([("2026-01-02", 10.2, 12.5, 9.4, 10.5)]))

    assert result.loc[0, "exit_reason"] == "stop_loss"
    assert result.loc[0, "exit_price"] == 9.5


def test_take_profit_1_and_take_profit_2_trigger():
    tp1 = backtest_trade_plans(_plan(take_profit_2=12.5), _bars([("2026-01-02", 10.2, 11.2, 10.0, 10.8)]))
    tp2 = backtest_trade_plans(_plan(), _bars([("2026-01-02", 10.2, 12.2, 10.0, 11.8)]))

    assert tp1.loc[0, "exit_reason"] == "take_profit_1"
    assert tp2.loc[0, "exit_reason"] == "take_profit_2"


def test_time_exit_and_risk_metrics_are_calculated():
    result = backtest_trade_plans(
        _plan(stop_loss=8.0, take_profit_1=20.0, take_profit_2=21.0),
        _bars([("2026-01-02", 10.0, 10.7, 9.8, 10.4), ("2026-01-03", 10.4, 10.9, 9.7, 10.5)]),
        max_holding_days=2,
    )

    assert result.loc[0, "exit_reason"] == "time_exit"
    assert result.loc[0, "holding_days"] == 2
    assert round(result.loc[0, "return_pct"], 4) == 0.05
    assert round(result.loc[0, "max_drawdown"], 4) == -0.03
    assert round(result.loc[0, "max_favorable"], 4) == 0.09


def test_evaluate_trade_plan_backtest_summarizes_rates():
    results = pd.DataFrame(
        {
            "strategy_names": ["trend", "trend", "trend"],
            "strategy_versions": ["v1", "v1", "v1"],
            "action": ["回踩低吸", "回踩低吸", "回踩低吸"],
            "is_triggered": [True, True, False],
            "is_valid": [True, True, False],
            "return_pct": [0.1, -0.05, None],
            "max_drawdown": [-0.02, -0.06, None],
            "max_favorable": [0.12, 0.03, None],
            "exit_reason": ["take_profit_1", "stop_loss", ""],
        }
    )

    summary = evaluate_trade_plan_backtest(results)

    assert summary.loc[0, "plan_count"] == 3
    assert summary.loc[0, "trigger_rate"] == 2 / 3
    assert summary.loc[0, "win_rate"] == 0.5
    assert summary.loc[0, "avg_return"] == 0.025
