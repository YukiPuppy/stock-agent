from __future__ import annotations

import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_after_market_review as workflow


def _seed_actual_trades(db_path: str) -> None:
    store = StockAgentStore(db_path)
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-09", "2025-01-10", "2025-01-10"],
                "trade_time": ["10:00:00", "10:00:00", "10:30:00"],
                "code": ["600000", "600000", "000001"],
                "side": ["buy", "buy", "sell"],
                "price": [9.8, 10.0, 11.0],
                "volume": [100, 200, 100],
            }
        )
    )


def test_run_after_market_review_calls_steps_and_exports_reports(monkeypatch, tmp_path):
    db_path = str(tmp_path / "stock_agent.duckdb")
    output_dir = str(tmp_path / "reports")
    _seed_actual_trades(db_path)
    calls: list[tuple[str, dict]] = []

    def record(name: str, result):
        def inner(**kwargs):
            calls.append((name, kwargs))
            return result

        return inner

    monkeypatch.setattr(workflow, "run_execution_review", record("execution", pd.DataFrame({"a": [1, 2]})))
    monkeypatch.setattr(workflow, "build_trade_performance", record("performance", pd.DataFrame({"a": [1]})))
    monkeypatch.setattr(workflow, "build_daily_review", record("daily", pd.DataFrame({"a": [1]})))
    monkeypatch.setattr(
        workflow,
        "build_positions",
        record("positions", (pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": [1, 2]}))),
    )
    monkeypatch.setattr(workflow, "export_daily_review_report", record("daily_report", "daily.md"))
    monkeypatch.setattr(workflow, "export_trade_performance_report", record("performance_report", "performance.md"))
    monkeypatch.setattr(workflow, "export_position_review_report", record("position_report", "position.md"))
    monkeypatch.setattr(
        workflow,
        "run_daily_review_agent_pipeline",
        record("llm_daily_review", "llm_daily.md"),
    )

    summary = workflow.run_after_market_review(db_path=db_path, output_dir=output_dir)

    assert [name for name, _ in calls] == [
        "execution",
        "performance",
        "daily",
        "positions",
        "daily_report",
        "performance_report",
        "position_report",
    ]
    assert summary["trade_date"] == "2025-01-10"
    assert summary["db_path"] == db_path
    assert summary["actual_trades_rows"] == 2
    assert summary["execution_review_rows"] == 2
    assert summary["actual_trade_performance_rows"] == 1
    assert summary["daily_review_rows"] == 1
    assert summary["positions_rows"] == 3
    assert summary["position_review_rows"] == 2
    assert summary["daily_review_report_path"] == "daily.md"
    assert summary["trade_performance_report_path"] == "performance.md"
    assert summary["position_review_report_path"] == "position.md"
    assert summary["llm_daily_review_report_path"] is None
    assert calls[0][1] == {"trade_date": "2025-01-10", "db_path": db_path}
    assert calls[3][1] == {"as_of_date": "2025-01-10", "db_path": db_path}
    assert calls[4][1] == {"trade_date": "2025-01-10", "db_path": db_path, "output_dir": output_dir}
    assert calls[6][1] == {"as_of_date": "2025-01-10", "db_path": db_path, "output_dir": output_dir}


def test_run_after_market_review_no_report_skips_exports(monkeypatch, tmp_path):
    db_path = str(tmp_path / "stock_agent.duckdb")
    _seed_actual_trades(db_path)
    exported = []

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_trade_performance", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_daily_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_positions", lambda **kwargs: (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]})))
    monkeypatch.setattr(workflow, "export_daily_review_report", lambda **kwargs: exported.append(kwargs))
    monkeypatch.setattr(workflow, "export_trade_performance_report", lambda **kwargs: exported.append(kwargs))
    monkeypatch.setattr(workflow, "export_position_review_report", lambda **kwargs: exported.append(kwargs))
    monkeypatch.setattr(
        workflow,
        "run_daily_review_agent_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run llm")),
    )

    summary = workflow.run_after_market_review(db_path=db_path, export_reports=False)

    assert exported == []
    assert summary["daily_review_report_path"] is None
    assert summary["trade_performance_report_path"] is None
    assert summary["position_review_report_path"] is None
    assert summary["llm_daily_review_report_path"] is None


def test_run_after_market_review_passes_explicit_parameters(monkeypatch, tmp_path):
    db_path = str(tmp_path / "stock_agent.duckdb")
    output_dir = str(tmp_path / "custom_reports")
    _seed_actual_trades(db_path)
    calls = []

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: calls.append(kwargs) or pd.DataFrame())
    monkeypatch.setattr(workflow, "build_trade_performance", lambda **kwargs: calls.append(kwargs) or pd.DataFrame())
    monkeypatch.setattr(workflow, "build_daily_review", lambda **kwargs: calls.append(kwargs) or pd.DataFrame())
    monkeypatch.setattr(workflow, "build_positions", lambda **kwargs: calls.append(kwargs) or (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(workflow, "export_daily_review_report", lambda **kwargs: calls.append(kwargs) or "daily.md")
    monkeypatch.setattr(workflow, "export_trade_performance_report", lambda **kwargs: calls.append(kwargs) or "performance.md")
    monkeypatch.setattr(workflow, "export_position_review_report", lambda **kwargs: calls.append(kwargs) or "position.md")

    summary = workflow.run_after_market_review(
        trade_date="2025-01-09",
        db_path=db_path,
        output_dir=output_dir,
    )

    assert summary["trade_date"] == "2025-01-09"
    assert summary["actual_trades_rows"] == 1
    assert calls == [
        {"trade_date": "2025-01-09", "db_path": db_path},
        {"trade_date": "2025-01-09", "db_path": db_path},
        {"trade_date": "2025-01-09", "db_path": db_path},
        {"as_of_date": "2025-01-09", "db_path": db_path},
        {"trade_date": "2025-01-09", "db_path": db_path, "output_dir": output_dir},
        {"trade_date": "2025-01-09", "db_path": db_path, "output_dir": output_dir},
        {"as_of_date": "2025-01-09", "db_path": db_path, "output_dir": output_dir},
    ]


def test_run_after_market_review_empty_actual_trades_does_not_crash(monkeypatch, tmp_path):
    db_path = str(tmp_path / "stock_agent.duckdb")
    StockAgentStore(db_path).init_tables()
    called = []

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: called.append(kwargs))

    summary = workflow.run_after_market_review(db_path=db_path)

    assert called == []
    assert summary["trade_date"] is None
    assert summary["actual_trades_rows"] == 0
    assert summary["execution_review_rows"] == 0
    assert summary["positions_rows"] == 0


def test_run_after_market_review_counts_empty_step_results(monkeypatch, tmp_path):
    db_path = str(tmp_path / "stock_agent.duckdb")
    _seed_actual_trades(db_path)

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "build_trade_performance", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "build_daily_review", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "build_positions", lambda **kwargs: (pd.DataFrame(), pd.DataFrame()))

    summary = workflow.run_after_market_review(db_path=db_path, export_reports=False)

    assert summary["execution_review_rows"] == 0
    assert summary["actual_trade_performance_rows"] == 0
    assert summary["daily_review_rows"] == 0
    assert summary["positions_rows"] == 0
    assert summary["position_review_rows"] == 0


def test_run_after_market_review_runs_llm_only_when_requested(monkeypatch, tmp_path):
    db_path = str(tmp_path / "stock_agent.duckdb")
    output_dir = str(tmp_path / "reports")
    _seed_actual_trades(db_path)
    llm_calls = []

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_trade_performance", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_daily_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_positions", lambda **kwargs: (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]})))
    monkeypatch.setattr(workflow, "export_daily_review_report", lambda **kwargs: "daily.md")
    monkeypatch.setattr(workflow, "export_trade_performance_report", lambda **kwargs: "performance.md")
    monkeypatch.setattr(workflow, "export_position_review_report", lambda **kwargs: "position.md")
    monkeypatch.setattr(
        workflow,
        "run_daily_review_agent_pipeline",
        lambda **kwargs: llm_calls.append(kwargs) or "llm_daily.md",
    )

    default_summary = workflow.run_after_market_review(db_path=db_path, output_dir=output_dir)
    requested_summary = workflow.run_after_market_review(
        db_path=db_path,
        output_dir=output_dir,
        run_llm_daily_review=True,
    )

    assert default_summary["llm_daily_review_report_path"] is None
    assert requested_summary["llm_daily_review_report_path"] == "llm_daily.md"
    assert llm_calls == [
        {
            "trade_date": "2025-01-10",
            "db_path": db_path,
            "output_dir": output_dir,
            "report_date": "2025-01-10",
        }
    ]


def test_main_no_report_prints_summary_and_skips_exports(monkeypatch, tmp_path, capsys):
    db_path = str(tmp_path / "stock_agent.duckdb")
    _seed_actual_trades(db_path)

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_trade_performance", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_daily_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_positions", lambda **kwargs: (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]})))
    monkeypatch.setattr(
        workflow,
        "export_daily_review_report",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not export")),
    )

    workflow.main(["--trade-date", "2025-01-10", "--db-path", db_path, "--no-report"])

    output = capsys.readouterr().out
    assert "After-market review finished." in output
    assert "trade_date: 2025-01-10" in output
    assert "daily_review_report_path: None" in output


def test_main_run_llm_daily_review_prints_report_path(monkeypatch, tmp_path, capsys):
    db_path = str(tmp_path / "stock_agent.duckdb")
    _seed_actual_trades(db_path)

    monkeypatch.setattr(workflow, "run_execution_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_trade_performance", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_daily_review", lambda **kwargs: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(workflow, "build_positions", lambda **kwargs: (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]})))
    monkeypatch.setattr(workflow, "export_daily_review_report", lambda **kwargs: "daily.md")
    monkeypatch.setattr(workflow, "export_trade_performance_report", lambda **kwargs: "performance.md")
    monkeypatch.setattr(workflow, "export_position_review_report", lambda **kwargs: "position.md")
    monkeypatch.setattr(workflow, "run_daily_review_agent_pipeline", lambda **kwargs: "llm_daily.md")

    workflow.main(["--trade-date", "2025-01-10", "--db-path", db_path, "--run-llm-daily-review"])

    output = capsys.readouterr().out
    assert "llm_daily_review_report_path: llm_daily.md" in output
