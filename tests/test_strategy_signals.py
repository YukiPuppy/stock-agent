import pandas as pd

from src.strategy.base_strategy import SIGNAL_COLUMNS
from src.strategy.breakout_volume_strategy import BreakoutVolumeStrategy
from src.strategy.support_rebound_strategy import SupportReboundStrategy
from src.strategy.trend_pullback_strategy import TrendPullbackStrategy


def _factors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-02"],
            "code": ["OLD", "TREND", "BREAK", "SUPPORT"],
            "pct_chg_1d": [0.01, 0.055, 0.08, -0.04],
            "pct_chg_3d": [0.01, 0.02, 0.03, -0.07],
            "pct_chg_5d": [0.01, 0.06, 0.08, -0.01],
            "volume_ratio_5": [1.0, 1.2, 2.0, 3.0],
            "close_position_20": [0.5, 0.92, 0.96, 0.5],
            "above_ma5": [True, True, True, False],
            "above_ma10": [True, True, False, False],
            "above_ma20": [True, True, True, True],
            "amount_ma5": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_three_strategies_generate_signals_when_conditions_match():
    factors = _factors()

    trend = TrendPullbackStrategy().generate_signals(factors, trade_date="2026-01-02")
    breakout = BreakoutVolumeStrategy().generate_signals(factors, trade_date="2026-01-02")
    support = SupportReboundStrategy().generate_signals(factors, trade_date="2026-01-02")

    assert trend["code"].tolist() == ["TREND"]
    assert trend.loc[0, "strategy_version"] == "v1"
    assert trend.loc[0, "risk_flags"] == "near_20d_high,short_term_chase_risk"
    assert breakout["code"].tolist() == ["BREAK"]
    assert breakout.loc[0, "strategy_version"] == "v1"
    assert breakout.loc[0, "risk_flags"] == "near_limit_chase_risk,extended_position"
    assert support["code"].tolist() == ["SUPPORT"]
    assert support.loc[0, "strategy_version"] == "v1"
    assert support.loc[0, "risk_flags"] == "short_term_weakness,panic_volume_possible"


def test_strategy_returns_empty_standard_columns_when_no_signal():
    result = TrendPullbackStrategy().generate_signals(pd.DataFrame({"trade_date": ["2026-01-02"]}))

    assert result.empty
    assert result.columns.tolist() == SIGNAL_COLUMNS


def test_strategy_uses_latest_trade_date_when_missing():
    result = TrendPullbackStrategy().generate_signals(_factors())

    assert set(result["trade_date"]) == {"2026-01-02"}


def test_disabled_strategy_returns_empty_standard_dataframe():
    result = TrendPullbackStrategy({"enabled": False}).generate_signals(
        _factors(),
        trade_date="2026-01-02",
    )

    assert result.empty
    assert result.columns.tolist() == SIGNAL_COLUMNS


def test_strategy_config_parameter_changes_filter_result():
    result = TrendPullbackStrategy({"min_pct_chg_5d": 0.07}).generate_signals(
        _factors(),
        trade_date="2026-01-02",
    )

    assert result.empty
