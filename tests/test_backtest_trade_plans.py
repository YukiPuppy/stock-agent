import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.backtest_trade_plans import run_trade_plan_backtest


def test_run_trade_plan_backtest_reads_local_duckdb_and_saves_results(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_stock_basic(pd.DataFrame({"code": ["600000"], "name": ["测试股"], "market": ["SH"], "board": ["main"]}))
    store.save_strategy_signals(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-01"],
                "code": ["600000"],
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "signal_strength": [20.0],
                "entry_reason": ["趋势"],
                "risk_flags": [""],
            }
        )
    )
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-01"],
                "code": ["600000"],
                "close": [10.0],
                "pct_chg_1d": [0.01],
                "pct_chg_3d": [0.02],
                "pct_chg_5d": [0.03],
                "pct_chg_10d": [0.04],
                "ma5": [9.8],
                "ma10": [9.7],
                "ma20": [9.6],
                "volume_ma5": [1000.0],
                "amount_ma5": [1000.0],
                "volume_ratio_5": [1.2],
                "high_20": [10.2],
                "low_20": [8.0],
                "close_position_20": [0.7],
                "above_ma5": [True],
                "above_ma10": [True],
                "above_ma20": [True],
            }
        )
    )
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-03"],
                "code": ["600000", "600000"],
                "open": [9.9, 10.2],
                "high": [10.6, 10.8],
                "low": [9.7, 10.0],
                "close": [10.4, 10.5],
                "volume": [1000, 1000],
                "amount": [10000, 10000],
            }
        )
    )

    plans, results, performance = run_trade_plan_backtest(db_path=str(db_path), top_n=1, max_plan_items=1)

    assert len(plans) == 1
    assert len(results) == 1
    assert len(performance) == 1
    assert len(store.load_historical_trade_plans()) == 1
    assert len(store.load_trade_plan_backtest_results()) == 1
    assert len(store.load_trade_plan_backtest_performance()) == 1
