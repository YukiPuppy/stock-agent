import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline.backtest_strategy_signals import run_signal_backtest


def test_run_signal_backtest_reads_duckdb_backtests_and_saves_results(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_strategy_signals(
        pd.DataFrame(
            {
                "trade_date": ["20260101", "20260101"],
                "code": ["600000", "000001"],
                "strategy_name": ["trend_pullback", "support_rebound"],
                "signal_strength": [30.0, 20.0],
                "entry_reason": ["trend", "support"],
                "risk_flags": ["", ""],
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
                ]
                * 2,
                "code": ["600000"] * 7 + ["000001"] * 7,
                "open": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0] + [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                "high": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5] + [19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5],
                "low": [8.5, 9.0, 9.5, 8.0, 10.0, 7.0, 13.0] + [18.5, 19.0, 19.5, 18.0, 17.0, 16.0, 15.0],
                "close": [9.2, 10.5, 11.0, 12.0, 13.0, 15.0, 16.0] + [19.2, 19.5, 19.0, 18.0, 17.0, 16.0, 15.0],
                "volume": [1000.0] * 14,
                "amount": [10000.0] * 14,
            }
        )
    )

    backtest_results, strategy_performance = run_signal_backtest(str(db_path))
    saved_results = store.load_backtest_results()
    saved_performance = store.load_strategy_performance()

    assert len(backtest_results) == 2
    assert len(strategy_performance) == 2
    assert len(saved_results) == 2
    assert len(saved_performance) == 2
    assert saved_results["is_valid"].tolist() == [True, True]
    assert set(saved_performance["strategy_name"]) == {"trend_pullback", "support_rebound"}
