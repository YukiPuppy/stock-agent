from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.research.walk_forward_validation import validate_strategy_versions_out_of_sample


def _dates(start: str, count: int) -> list[str]:
    start_date = date.fromisoformat(start)
    return [(start_date + timedelta(days=index)).isoformat() for index in range(count)]


def _market_data(validation_down: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _dates("2026-01-01", 80)
    prices = []
    price = 10.0
    for index, _ in enumerate(dates):
        if validation_down and index >= 40:
            price -= 0.2
        else:
            price += 0.2
        prices.append(round(price, 2))

    daily_factors = pd.DataFrame(
        {
            "trade_date": dates,
            "code": ["600000"] * len(dates),
            "close": prices,
            "pct_chg_1d": [0.01] * len(dates),
            "pct_chg_3d": [0.03] * len(dates),
            "pct_chg_5d": [0.06] * len(dates),
            "pct_chg_10d": [0.10] * len(dates),
            "ma5": [9.0] * len(dates),
            "ma10": [8.8] * len(dates),
            "ma20": [8.5] * len(dates),
            "volume_ma5": [1000.0] * len(dates),
            "amount_ma5": [10000.0] * len(dates),
            "volume_ratio_5": [1.5] * len(dates),
            "high_20": [12.0] * len(dates),
            "low_20": [8.0] * len(dates),
            "close_position_20": [0.7] * len(dates),
            "above_ma5": [True] * len(dates),
            "above_ma10": [True] * len(dates),
            "above_ma20": [True] * len(dates),
        }
    )
    daily_bars = pd.DataFrame(
        {
            "trade_date": dates,
            "code": ["600000"] * len(dates),
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1000.0] * len(dates),
            "amount": [10000.0] * len(dates),
        }
    )
    return daily_factors, daily_bars


def _versions() -> list[dict]:
    return [
        {
            "strategy_name": "trend_pullback",
            "strategy_version": "search_001",
            "enabled": True,
            "params": {
                "require_above_ma5": True,
                "require_above_ma10": True,
                "min_pct_chg_5d": 0.02,
                "max_pct_chg_1d": 0.04,
                "min_close_position_20": 0.55,
                "min_volume_ratio_5": 1.0,
            },
        }
    ]


def test_validate_strategy_versions_out_of_sample_generates_train_and_validation_performance():
    daily_factors, daily_bars = _market_data()

    result = validate_strategy_versions_out_of_sample(
        daily_factors,
        daily_bars,
        _versions(),
        "2026-01-01",
        "2026-01-30",
        "2026-02-10",
        "2026-03-05",
        min_valid_count_train=1,
        min_valid_count_validation=1,
    )

    assert len(result) == 1
    assert result.loc[0, "train_valid_count"] > 0
    assert result.loc[0, "validation_valid_count"] > 0
    assert result.loc[0, "validation_status"] == "passed_oos"


def test_validate_strategy_versions_out_of_sample_marks_insufficient_train_samples():
    daily_factors, daily_bars = _market_data()

    result = validate_strategy_versions_out_of_sample(
        daily_factors,
        daily_bars,
        _versions(),
        "2026-01-01",
        "2026-01-10",
        "2026-02-10",
        "2026-03-05",
        min_valid_count_train=30,
        min_valid_count_validation=1,
    )

    assert result.loc[0, "validation_status"] == "insufficient_train_samples"
    assert "训练区间有效样本不足" in result.loc[0, "validation_reason"]


def test_validate_strategy_versions_out_of_sample_marks_insufficient_validation_samples():
    daily_factors, daily_bars = _market_data()

    result = validate_strategy_versions_out_of_sample(
        daily_factors,
        daily_bars,
        _versions(),
        "2026-01-01",
        "2026-01-30",
        "2026-02-10",
        "2026-02-14",
        min_valid_count_train=1,
        min_valid_count_validation=10,
    )

    assert result.loc[0, "validation_status"] == "insufficient_validation_samples"
    assert "验证区间有效样本不足" in result.loc[0, "validation_reason"]


def test_validate_strategy_versions_out_of_sample_marks_failed_when_oos_turns_negative():
    daily_factors, daily_bars = _market_data(validation_down=True)

    result = validate_strategy_versions_out_of_sample(
        daily_factors,
        daily_bars,
        _versions(),
        "2026-01-01",
        "2026-01-30",
        "2026-02-10",
        "2026-03-05",
        min_valid_count_train=1,
        min_valid_count_validation=1,
    )

    assert result.loc[0, "train_avg_return_3d"] > 0
    assert result.loc[0, "validation_avg_return_3d"] < 0
    assert result.loc[0, "validation_status"] == "failed_oos"


def test_validate_strategy_versions_out_of_sample_stability_score_sorts_descending():
    daily_factors, daily_bars = _market_data()
    versions = _versions() + [{**_versions()[0], "strategy_version": "search_002"}]

    result = validate_strategy_versions_out_of_sample(
        daily_factors,
        daily_bars,
        versions,
        "2026-01-01",
        "2026-01-30",
        "2026-02-10",
        "2026-03-05",
        min_valid_count_train=1,
        min_valid_count_validation=1,
    )

    assert "stability_score" in result.columns
    assert result["stability_score"].tolist() == sorted(result["stability_score"], reverse=True)
