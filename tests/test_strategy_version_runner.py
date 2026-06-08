import pandas as pd

from src.backtest.strategy_version_runner import (
    generate_historical_signals_for_version,
    generate_historical_signals_for_versions,
)


def _daily_factors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260101", "20260102", "20260101", "20260102"],
            "code": ["600000", "600000", "000001", "000001"],
            "close": [10.0, 10.5, 20.0, 19.5],
            "pct_chg_1d": [0.02, 0.03, -0.03, -0.04],
            "pct_chg_3d": [0.03, 0.04, -0.04, -0.05],
            "pct_chg_5d": [0.04, 0.05, 0.01, 0.01],
            "pct_chg_10d": [0.05, 0.06, 0.01, 0.01],
            "ma5": [9.8, 10.0, 20.2, 19.8],
            "ma10": [9.7, 9.9, 20.3, 19.9],
            "ma20": [9.5, 9.8, 19.0, 19.0],
            "volume_ma5": [1000.0] * 4,
            "amount_ma5": [10000.0] * 4,
            "volume_ratio_5": [1.2, 1.3, 1.1, 1.2],
            "high_20": [11.0, 11.0, 22.0, 22.0],
            "low_20": [8.0, 8.0, 18.0, 18.0],
            "close_position_20": [0.67, 0.83, 0.50, 0.38],
            "above_ma5": [True, True, False, False],
            "above_ma10": [True, True, False, False],
            "above_ma20": [True, True, True, True],
        }
    )


def test_generate_historical_signals_for_version_handles_multiple_trade_dates():
    result = generate_historical_signals_for_version(
        daily_factors=_daily_factors(),
        strategy_name="trend_pullback",
        strategy_version="v2",
        params={
            "min_pct_chg_5d": 0.03,
            "max_pct_chg_1d": 0.06,
            "min_close_position_20": 0.55,
            "min_volume_ratio_5": 1.0,
            "require_above_ma5": True,
            "require_above_ma10": True,
        },
    )

    assert result["trade_date"].tolist() == ["20260101", "20260102"]
    assert result["strategy_version"].tolist() == ["v2", "v2"]


def test_generate_historical_signals_for_versions_combines_enabled_versions():
    versions = [
        {
            "strategy_name": "trend_pullback",
            "strategy_version": "v1",
            "enabled": True,
            "params": {
                "min_pct_chg_5d": 0.03,
                "max_pct_chg_1d": 0.06,
                "min_close_position_20": 0.55,
                "min_volume_ratio_5": 1.0,
                "require_above_ma5": True,
                "require_above_ma10": True,
            },
        },
        {
            "strategy_name": "trend_pullback",
            "strategy_version": "off",
            "enabled": False,
            "params": {},
        },
        {
            "strategy_name": "support_rebound",
            "strategy_version": "v1",
            "enabled": True,
            "params": {
                "min_pct_chg_1d": -0.095,
                "max_pct_chg_1d": -0.02,
                "min_close_position_20": 0.35,
                "max_close_position_20": 0.75,
                "require_above_ma20": True,
                "min_amount_ma5": 0,
            },
        },
    ]

    result = generate_historical_signals_for_versions(_daily_factors(), versions)

    assert set(result["strategy_name"]) == {"trend_pullback", "support_rebound"}
    assert "off" not in set(result["strategy_version"])
