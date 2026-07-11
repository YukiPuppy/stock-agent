from __future__ import annotations

import json
from concurrent.futures import Future
from datetime import date, timedelta

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import parallel_strategy_research as parallel
from src.pipeline.backtest_trade_plans import run_trade_plan_backtest
from src.pipeline.search_strategy_params import run_parameter_search
from src.pipeline.validate_strategy_oos import run_oos_validation
from src.research.parameter_search import generate_search_versions


def _dates(start: str, count: int) -> list[str]:
    start_date = date.fromisoformat(start)
    return [(start_date + timedelta(days=index)).isoformat() for index in range(count)]


def _write_search_config(path) -> None:
    path.write_text(
        json.dumps(
            {
                "trend_pullback": {
                    "enabled": True,
                    "max_combinations": 2,
                    "base_params": {"require_above_ma5": True, "require_above_ma10": True},
                    "search_space": {
                        "min_pct_chg_5d": [0.02, 0.03],
                        "max_pct_chg_1d": [0.04],
                        "min_close_position_20": [0.55],
                        "min_volume_ratio_5": [1.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _populate_market_data(db_path) -> None:
    dates = _dates("2026-01-01", 80)
    prices = [10 + index * 0.05 for index in range(len(dates))]
    store = StockAgentStore(str(db_path))
    store.save_stock_basic(pd.DataFrame({"code": ["600000"], "name": ["浦发银行"], "market": ["SH"], "board": ["main"], "industry": ["银行"], "list_status": ["L"]}))
    store.save_daily_factors(
        pd.DataFrame(
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
    )
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": dates,
                "code": ["600000"] * len(dates),
                "open": prices,
                "high": [price + 0.3 for price in prices],
                "low": [price - 0.3 for price in prices],
                "close": [price + 0.1 for price in prices],
                "volume": [1000.0] * len(dates),
                "amount": [10000.0] * len(dates),
            }
        )
    )


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    comparable = df.drop(columns=["run_id"], errors="ignore").copy()
    if comparable.empty:
        return comparable.reset_index(drop=True)
    return comparable.sort_values(list(comparable.columns)).reset_index(drop=True)


def test_run_parameter_search_parallel_matches_serial_and_preserves_run_id(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    config_path = tmp_path / "parameter_search_space.json"
    _write_search_config(config_path)
    _populate_market_data(db_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    versions = generate_search_versions(config)

    serial = run_parameter_search(
        start_date="2026-01-01",
        end_date="2026-03-15",
        config_path=str(config_path),
        db_path=str(db_path),
        min_valid_count=1,
        run_id="serial-run",
    )
    output = parallel.run_parameter_search_parallel(
        db_path=str(db_path),
        versions=versions,
        start_date="2026-01-01",
        end_date="2026-03-15",
        run_id="parallel-run",
        workers=2,
        min_valid_count=1,
    )

    assert output.summary["workers"] == 2
    for serial_df, parallel_df in zip(serial, output.result, strict=True):
        assert set(parallel_df["run_id"]) == {"parallel-run"}
        pd.testing.assert_frame_equal(_sort_frame(serial_df), _sort_frame(parallel_df), check_dtype=False)
    assert set(StockAgentStore(str(db_path)).load_parameter_search_results(run_id="parallel-run")["run_id"]) == {
        "parallel-run"
    }


def test_run_oos_validation_parallel_matches_serial_and_preserves_run_id(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    config_path = tmp_path / "parameter_search_space.json"
    _write_search_config(config_path)
    _populate_market_data(db_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    versions = generate_search_versions(config)

    serial = run_oos_validation(
        train_start_date="2026-01-01",
        train_end_date="2026-01-30",
        validation_start_date="2026-02-01",
        validation_end_date="2026-03-10",
        config_path=str(config_path),
        db_path=str(db_path),
        min_valid_count_train=1,
        min_valid_count_validation=1,
        run_id="serial-run",
    )
    output = parallel.run_oos_validation_parallel(
        db_path=str(db_path),
        versions=versions,
        train_start_date="2026-01-01",
        train_end_date="2026-01-30",
        validation_start_date="2026-02-01",
        validation_end_date="2026-03-10",
        run_id="parallel-run",
        workers=2,
        min_valid_count_train=1,
        min_valid_count_validation=1,
    )

    assert output.summary["workers"] == 2
    assert set(output.result["run_id"]) == {"parallel-run"}
    pd.testing.assert_frame_equal(_sort_frame(serial), _sort_frame(output.result), check_dtype=False)
    assert set(StockAgentStore(str(db_path)).load_walk_forward_validation(run_id="parallel-run")["run_id"]) == {
        "parallel-run"
    }


def test_run_trade_plan_backtest_parallel_matches_serial_and_preserves_chain(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    _populate_market_data(db_path)
    signals = pd.DataFrame(
        {
            "trade_date": ["20260105", "20260106", "20260107"],
            "code": ["600000", "600000", "600000"],
            "strategy_name": ["trend_pullback"] * 3,
            "strategy_version": ["v1"] * 3,
            "signal_strength": [12.0, 13.0, 14.0],
            "entry_reason": ["test"] * 3,
            "risk_flags": [""] * 3,
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "recommendation": ["research_candidate"],
            "risk_level": ["low"],
        }
    )

    serial = run_trade_plan_backtest(
        db_path=str(db_path),
        start_date="2026-01-01",
        end_date="2026-03-15",
        strategy_signals=signals,
        strategy_evaluation=evaluation,
        run_id="serial-run",
        return_diagnostics=True,
    )
    output = parallel.run_trade_plan_backtest_parallel(
        db_path=str(db_path),
        start_date="2026-01-01",
        end_date="2026-03-15",
        strategy_signals=signals,
        strategy_evaluation=evaluation,
        run_id="parallel-run",
        workers=2,
    )
    historical_trade_plans, backtest_results, performance, diagnostics = output.result

    assert output.summary["workers"] == 2
    assert diagnostics["historical_signals"] == 3
    assert len(historical_trade_plans) == diagnostics["historical_trade_plans"]
    for frame in [historical_trade_plans, backtest_results, performance]:
        assert set(frame["run_id"]) == {"parallel-run"}
    for serial_df, parallel_df in zip(serial[:3], output.result[:3], strict=True):
        pd.testing.assert_frame_equal(_sort_frame(serial_df), _sort_frame(parallel_df), check_dtype=False)
    saved = StockAgentStore(str(db_path)).load_trade_plan_backtest_results(run_id="parallel-run")
    assert set(saved["run_id"]) == {"parallel-run"}


def test_worker_connects_to_duckdb_read_only(monkeypatch):
    calls = []

    def fake_connect(db_path, read_only=False):
        calls.append({"db_path": db_path, "read_only": read_only})
        raise RuntimeError("stop before read")

    monkeypatch.setattr(parallel.duckdb, "connect", fake_connect)

    result = parallel._parameter_search_worker(
        {
            "worker_index": 0,
            "db_path": "test.duckdb",
            "versions": [{"strategy_name": "trend_pullback", "strategy_version": "search_001"}],
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
            "run_id": "run-id",
            "min_valid_count": 1,
            "min_win_rate_3d": 0.5,
            "min_avg_return_3d": 0.0,
            "max_avg_drawdown_3d": -0.08,
        }
    )

    assert calls == [{"db_path": "test.duckdb", "read_only": True}]
    assert result["status"] == "failed"
    assert "stop before read" in result["error"]


def test_parallel_worker_error_is_not_silent():
    errors = [
        {
            "worker_index": 1,
            "pid": 123,
            "error": "boom",
            "traceback": "trace",
        }
    ]
    exc = parallel.ParallelWorkerError("run_parameter_search", errors)

    assert "run_parameter_search parallel worker failed" in str(exc)
    assert exc.worker_errors == errors


def test_effective_worker_count_is_bounded_by_cpu_and_tasks(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 4)

    assert parallel.effective_worker_count(8, 2) == 2
    assert parallel.effective_worker_count(8, 20) == 4
    assert parallel.effective_worker_count(2, 20) == 2


def test_future_level_worker_error_is_captured(monkeypatch):
    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, worker_fn, task):
            future = Future()
            future.set_exception(RuntimeError("process crashed"))
            return future

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)

    tasks = [{"worker_index": 0, "versions": [{"strategy_name": "trend_pullback", "strategy_version": "v1"}]}]
    try:
        parallel._execute_parallel("run_parameter_search", tasks, lambda task: task, 1)
    except parallel.ParallelWorkerError as exc:
        assert exc.worker_errors[0]["worker_index"] == 0
        assert "process crashed" in exc.worker_errors[0]["error"]
    else:
        raise AssertionError("future exception was not propagated")
