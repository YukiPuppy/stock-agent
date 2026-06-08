from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.validate_strategy_oos import run_oos_validation


def _dates(start: str, count: int) -> list[str]:
    start_date = date.fromisoformat(start)
    return [(start_date + timedelta(days=index)).isoformat() for index in range(count)]


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _dates("2026-01-01", 80)
    prices = [10 + index * 0.2 for index in range(len(dates))]
    factors = pd.DataFrame(
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
    bars = pd.DataFrame(
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
    return factors, bars


def test_run_oos_validation_reads_duckdb_and_saves_result(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    config_path = tmp_path / "parameter_search_space.json"
    config_path.write_text(
        json.dumps(
            {
                "trend_pullback": {
                    "enabled": True,
                    "max_combinations": 1,
                    "base_params": {"require_above_ma5": True, "require_above_ma10": True},
                    "search_space": {
                        "min_pct_chg_5d": [0.02],
                        "max_pct_chg_1d": [0.04],
                        "min_close_position_20": [0.55],
                        "min_volume_ratio_5": [1.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    store = StockAgentStore(str(db_path))
    factors, bars = _data()
    store.save_daily_factors(factors)
    store.save_daily_bars(bars)

    result = run_oos_validation(
        train_start_date="2026-01-01",
        train_end_date="2026-01-30",
        validation_start_date="2026-02-10",
        validation_end_date="2026-03-05",
        config_path=str(config_path),
        db_path=str(db_path),
        min_valid_count_train=1,
        min_valid_count_validation=1,
    )

    saved = store.load_walk_forward_validation()
    assert len(result) == 1
    assert len(saved) == 1
    assert saved.loc[0, "strategy_version"] == "search_001"
