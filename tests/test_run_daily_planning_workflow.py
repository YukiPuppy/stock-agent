import pandas as pd
import pytest

from src.pipeline import run_daily_planning_workflow as workflow


def _df(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"value": list(range(rows))})


def _patch_steps(monkeypatch, calls: list | None = None, rows: dict | None = None) -> None:
    rows = rows or {}

    def record(name: str, kwargs: dict):
        if calls is not None:
            calls.append((name, kwargs))

    monkeypatch.setattr(
        workflow,
        "update_daily_bars",
        lambda **kwargs: record("update_daily_bars", kwargs) or _df(rows.get("daily_bars", 2)),
    )
    monkeypatch.setattr(
        workflow,
        "build_daily_factors",
        lambda **kwargs: record("build_daily_factors", kwargs) or _df(rows.get("daily_factors", 3)),
    )
    monkeypatch.setattr(
        workflow,
        "build_strategy_signals",
        lambda **kwargs: record("build_strategy_signals", kwargs) or _df(rows.get("strategy_signals", 4)),
    )
    monkeypatch.setattr(
        workflow,
        "build_candidate_pool",
        lambda **kwargs: record("build_candidate_pool", kwargs) or _df(rows.get("candidate_pool", 5)),
    )
    monkeypatch.setattr(
        workflow,
        "build_trade_plan",
        lambda **kwargs: record("build_trade_plan", kwargs) or _df(rows.get("trade_plan", 6)),
    )
    monkeypatch.setattr(
        workflow,
        "export_daily_report",
        lambda **kwargs: record("export_daily_report", kwargs) or "reports/daily_report_2026-01-02.md",
    )


def test_default_does_not_call_update_daily_bars(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_daily_planning_workflow(db_path="test.duckdb")

    assert [call[0] for call in calls] == [
        "build_daily_factors",
        "build_strategy_signals",
        "build_candidate_pool",
        "build_trade_plan",
        "export_daily_report",
    ]
    assert summary["update_data"] is False


def test_update_data_calls_update_daily_bars(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    workflow.run_daily_planning_workflow(
        start_date="20260101",
        end_date="20260131",
        update_data=True,
        db_path="test.duckdb",
    )

    assert calls[0] == (
        "update_daily_bars",
        {
            "start_date": "20260101",
            "end_date": "20260131",
            "db_path": "test.duckdb",
            "provider": "tushare",
            "limit": None,
            "sleep_seconds": 1.0,
        },
    )


def test_update_data_requires_dates(monkeypatch):
    _patch_steps(monkeypatch)

    with pytest.raises(ValueError, match="start_date and end_date are required"):
        workflow.run_daily_planning_workflow(update_data=True)


def test_default_use_active_candidates_true(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_daily_planning_workflow(db_path="test.duckdb")

    signal_call = next(call for call in calls if call[0] == "build_strategy_signals")
    assert signal_call[1]["use_active_candidates"] is True
    assert signal_call[1]["active_config_path"] == "configs/active_strategies_candidate.json"
    assert summary["use_active_candidates"] is True


def test_parse_args_no_active_candidates_maps_false():
    args = workflow._parse_args(["--no-active-candidates"])

    assert args.no_active_candidates is True


def test_main_passes_no_active_candidates(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(workflow, "clear_proxy_env_for_process", lambda: calls.append(("clear_proxy", {})))
    monkeypatch.setattr(
        workflow,
        "run_daily_planning_workflow",
        lambda **kwargs: calls.append(("workflow", kwargs))
        or {
            "update_data": kwargs["update_data"],
            "use_active_candidates": kwargs["use_active_candidates"],
            "daily_factors_rows": 1,
            "strategy_signals_rows": 2,
            "candidate_pool_rows": 3,
            "trade_plan_rows": 4,
            "daily_report_path": None,
        },
    )

    workflow.main(["--no-active-candidates", "--no-report"])

    output = capsys.readouterr().out
    assert calls[1][1]["use_active_candidates"] is False
    assert calls[1][1]["export_report"] is False
    assert "Daily planning workflow finished." in output
    assert "use_active_candidates: False" in output


def test_export_report_false_skips_report(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_daily_planning_workflow(export_report=False, db_path="test.duckdb")

    assert "export_daily_report" not in [call[0] for call in calls]
    assert summary["daily_report_path"] is None


def test_summary_rows_are_counted(monkeypatch):
    _patch_steps(
        monkeypatch,
        rows={
            "daily_factors": 7,
            "strategy_signals": 8,
            "candidate_pool": 9,
            "trade_plan": 10,
        },
    )

    summary = workflow.run_daily_planning_workflow(db_path="test.duckdb")

    assert summary["daily_factors_rows"] == 7
    assert summary["strategy_signals_rows"] == 8
    assert summary["candidate_pool_rows"] == 9
    assert summary["trade_plan_rows"] == 10


def test_empty_dataframe_counts_zero_and_workflow_continues(monkeypatch):
    calls = []
    _patch_steps(
        monkeypatch,
        calls,
        rows={
            "strategy_signals": 0,
            "candidate_pool": 0,
            "trade_plan": 0,
        },
    )

    summary = workflow.run_daily_planning_workflow(db_path="test.duckdb")

    assert [call[0] for call in calls] == [
        "build_daily_factors",
        "build_strategy_signals",
        "build_candidate_pool",
        "build_trade_plan",
        "export_daily_report",
    ]
    assert summary["strategy_signals_rows"] == 0
    assert summary["candidate_pool_rows"] == 0
    assert summary["trade_plan_rows"] == 0


def test_parameters_are_passed_to_steps(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_daily_planning_workflow(
        start_date="20260101",
        end_date="20260131",
        db_path="custom.duckdb",
        provider="akshare",
        limit=30,
        sleep_seconds=0.25,
        update_data=True,
        active_config_path="custom_active.json",
        top_n=12,
        max_plan_items=4,
        min_amount_ma5=123.0,
        output_dir="custom_reports",
    )

    assert calls[0][1] == {
        "start_date": "20260101",
        "end_date": "20260131",
        "db_path": "custom.duckdb",
        "provider": "akshare",
        "limit": 30,
        "sleep_seconds": 0.25,
    }
    assert calls[1] == ("build_daily_factors", {"db_path": "custom.duckdb"})
    assert calls[2] == (
        "build_strategy_signals",
        {
            "db_path": "custom.duckdb",
            "use_active_candidates": True,
            "active_config_path": "custom_active.json",
        },
    )
    assert calls[3] == (
        "build_candidate_pool",
        {"db_path": "custom.duckdb", "top_n": 12, "min_amount_ma5": 123.0},
    )
    assert calls[4] == ("build_trade_plan", {"db_path": "custom.duckdb", "max_items": 4})
    assert calls[5] == ("export_daily_report", {"db_path": "custom.duckdb", "output_dir": "custom_reports"})
    assert summary["start_date"] == "20260101"
    assert summary["end_date"] == "20260131"
    assert summary["provider"] == "akshare"
    assert summary["db_path"] == "custom.duckdb"


def test_default_provider_uses_settings(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow.run_daily_planning_workflow(
        start_date="20260101",
        end_date="20260131",
        update_data=True,
        db_path="test.duckdb",
    )

    assert calls[0][1]["provider"] == "tushare"
    assert summary["provider"] == "tushare"


def test_no_build_factors_counts_zero(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_daily_planning_workflow(build_factors=False, db_path="test.duckdb")

    assert "build_daily_factors" not in [call[0] for call in calls]
    assert summary["daily_factors_rows"] == 0
