import json

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.backtest_strategy_versions import run_strategy_version_backtest


def test_run_strategy_version_backtest_reads_duckdb_and_saves_performance(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    config_path = tmp_path / "strategy_versions.json"
    config_path.write_text(
        json.dumps(
            {
                "trend_pullback": [
                    {
                        "version": "v1",
                        "enabled": True,
                        "params": {
                            "min_pct_chg_5d": 0.03,
                            "max_pct_chg_1d": 0.06,
                            "min_close_position_20": 0.55,
                            "min_volume_ratio_5": 1.0,
                            "require_above_ma5": True,
                            "require_above_ma10": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    store = StockAgentStore(str(db_path))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["20260101"],
                "code": ["600000"],
                "close": [10.0],
                "pct_chg_1d": [0.02],
                "pct_chg_3d": [0.03],
                "pct_chg_5d": [0.04],
                "pct_chg_10d": [0.05],
                "ma5": [9.8],
                "ma10": [9.7],
                "ma20": [9.5],
                "volume_ma5": [1000.0],
                "amount_ma5": [10000.0],
                "volume_ratio_5": [1.2],
                "high_20": [11.0],
                "low_20": [8.0],
                "close_position_20": [0.67],
                "above_ma5": [True],
                "above_ma10": [True],
                "above_ma20": [True],
            }
        )
    )
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": [
                    "20260101",
                    "20260102",
                    "20260105",
                    "20260106",
                    "20260107",
                    "20260108",
                    "20260109",
                ],
                "code": ["600000"] * 7,
                "open": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "high": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                "low": [8.5, 9.0, 9.5, 8.0, 10.0, 7.0, 13.0],
                "close": [9.2, 10.5, 11.0, 12.0, 13.0, 15.0, 16.0],
                "volume": [1000.0] * 7,
                "amount": [10000.0] * 7,
            }
        )
    )

    backtest_results, performance = run_strategy_version_backtest(
        config_path=str(config_path),
        db_path=str(db_path),
    )
    saved_performance = store.load_strategy_version_performance()

    assert len(backtest_results) == 1
    assert backtest_results.loc[0, "strategy_version"] == "v1"
    assert len(performance) == 1
    assert saved_performance.loc[0, "strategy_name"] == "trend_pullback"
    assert saved_performance.loc[0, "strategy_version"] == "v1"


def test_run_strategy_version_backtest_passes_date_range_to_store(monkeypatch):
    from src.pipeline import backtest_strategy_versions as pipeline

    calls = []

    class FakeStore:
        def __init__(self, db_path):
            self.db_path = db_path

        def load_daily_factors(self, **kwargs):
            calls.append(("load_daily_factors", kwargs))
            return pd.DataFrame(columns=["trade_date"])

        def load_daily_bars(self, **kwargs):
            calls.append(("load_daily_bars", kwargs))
            return pd.DataFrame()

        def save_backtest_results(self, df):
            calls.append(("save_backtest_results", len(df)))

        def save_strategy_version_performance(self, df):
            calls.append(("save_strategy_version_performance", len(df)))

    monkeypatch.setattr(pipeline, "StockAgentStore", FakeStore)
    monkeypatch.setattr(pipeline, "load_strategy_versions", lambda config_path=None: {"trend_pullback": []})

    pipeline.run_strategy_version_backtest(
        start_date="2026-01-01",
        end_date="2026-01-31",
        db_path="test.duckdb",
    )

    assert ("load_daily_factors", {"start_date": "2026-01-01", "end_date": "2026-01-31"}) in calls
    assert ("load_daily_bars", {"start_date": "2026-01-01", "end_date": "2026-01-31"}) in calls
