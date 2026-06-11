import pandas as pd

from src.pipeline import run_strategy_research_workflow as workflow


def _df(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"value": range(rows)})


def _patch_steps(monkeypatch, calls: list[str]) -> None:
    def fake_version_backtest(**kwargs):
        calls.append("version_backtest")
        calls.append(("version_backtest_kwargs", kwargs))
        return _df(2), _df(3)

    def fake_version_evaluation(**kwargs):
        calls.append("version_evaluation")
        calls.append(("version_evaluation_kwargs", kwargs))
        return _df(4)

    def fake_parameter_search(**kwargs):
        calls.append("parameter_search")
        calls.append(("parameter_search_kwargs", kwargs))
        return _df(5), _df(6), _df(7)

    def fake_oos_validation(**kwargs):
        calls.append("oos_validation")
        calls.append(("oos_validation_kwargs", kwargs))
        return _df(8)

    def fake_trade_plan_backtest(**kwargs):
        calls.append("trade_plan_backtest")
        calls.append(("trade_plan_backtest_kwargs", kwargs))
        return _df(9), _df(10), _df(11)

    def fake_strategy_admission(**kwargs):
        calls.append("strategy_admission")
        calls.append(("strategy_admission_kwargs", kwargs))
        return _df(12)

    def fake_export_strategy_evaluation_report(**kwargs):
        calls.append("export_strategy_evaluation_report")
        return "reports/strategy_evaluation.md"

    def fake_export_parameter_search_report(**kwargs):
        calls.append("export_parameter_search_report")
        return "reports/parameter_search.md"

    def fake_export_walk_forward_validation_report(**kwargs):
        calls.append("export_walk_forward_validation_report")
        return "reports/walk_forward_validation.md"

    def fake_export_trade_plan_backtest_report(**kwargs):
        calls.append("export_trade_plan_backtest_report")
        return "reports/trade_plan_backtest.md"

    def fake_export_strategy_admission_report(**kwargs):
        calls.append("export_strategy_admission_report")
        return "reports/strategy_admission.md"

    monkeypatch.setattr(workflow, "run_strategy_version_backtest", fake_version_backtest)
    monkeypatch.setattr(workflow, "run_strategy_version_evaluation", fake_version_evaluation)
    monkeypatch.setattr(workflow, "run_parameter_search", fake_parameter_search)
    monkeypatch.setattr(workflow, "run_oos_validation", fake_oos_validation)
    monkeypatch.setattr(workflow, "run_trade_plan_backtest", fake_trade_plan_backtest)
    monkeypatch.setattr(workflow, "run_strategy_admission", fake_strategy_admission)
    monkeypatch.setattr(
        workflow,
        "export_strategy_evaluation_report",
        fake_export_strategy_evaluation_report,
    )
    monkeypatch.setattr(
        workflow,
        "export_parameter_search_report",
        fake_export_parameter_search_report,
    )
    monkeypatch.setattr(
        workflow,
        "export_walk_forward_validation_report",
        fake_export_walk_forward_validation_report,
    )
    monkeypatch.setattr(
        workflow,
        "export_trade_plan_backtest_report",
        fake_export_trade_plan_backtest_report,
    )
    monkeypatch.setattr(
        workflow,
        "export_strategy_admission_report",
        fake_export_strategy_admission_report,
    )


def _call_names(calls: list[object]) -> list[str]:
    return [item for item in calls if isinstance(item, str)]


def _kwargs_for(calls: list[object], name: str) -> dict:
    key = f"{name}_kwargs"
    for item in calls:
        if isinstance(item, tuple) and item[0] == key:
            return item[1]
    raise AssertionError(f"missing kwargs for {name}")


def test_workflow_calls_core_steps_in_order_without_oos_when_dates_missing(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(export_reports=False)

    assert _call_names(calls) == [
        "version_backtest",
        "version_evaluation",
        "parameter_search",
        "trade_plan_backtest",
        "strategy_admission",
    ]
    assert summary["skipped_oos"] is True
    assert summary["walk_forward_validation_rows"] == 0


def test_workflow_runs_oos_validation_when_all_dates_are_present(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
        export_reports=False,
    )

    assert _call_names(calls) == [
        "version_backtest",
        "version_evaluation",
        "parameter_search",
        "oos_validation",
        "trade_plan_backtest",
        "strategy_admission",
    ]
    assert summary["skipped_oos"] is False
    assert summary["walk_forward_validation_rows"] == 8


def test_workflow_exports_reports_when_enabled(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
    )

    assert "export_strategy_evaluation_report" in calls
    assert "export_parameter_search_report" in calls
    assert "export_walk_forward_validation_report" in calls
    assert "export_trade_plan_backtest_report" in calls
    assert "export_strategy_admission_report" in calls
    assert summary["strategy_evaluation_report_path"] == "reports/strategy_evaluation.md"
    assert summary["parameter_search_report_path"] == "reports/parameter_search.md"
    assert summary["walk_forward_validation_report_path"] == "reports/walk_forward_validation.md"
    assert summary["trade_plan_backtest_report_path"] == "reports/trade_plan_backtest.md"
    assert summary["strategy_admission_report_path"] == "reports/strategy_admission.md"


def test_workflow_no_report_disables_all_report_exports(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
        export_reports=False,
    )

    assert not any(name.startswith("export_") for name in _call_names(calls))
    assert summary["strategy_evaluation_report_path"] is None
    assert summary["parameter_search_report_path"] is None
    assert summary["walk_forward_validation_report_path"] is None
    assert summary["trade_plan_backtest_report_path"] is None
    assert summary["strategy_admission_report_path"] is None


def test_workflow_no_candidate_config_disables_candidate_export(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(
        export_reports=False,
        export_candidate_config=False,
        candidate_config_path="configs/custom_candidate.json",
    )

    admission_kwargs = _kwargs_for(calls, "strategy_admission")
    assert admission_kwargs["export_candidate_config"] is False
    assert admission_kwargs["candidate_config_path"] == "configs/custom_candidate.json"
    assert summary["active_candidate_config_path"] is None


def test_workflow_summary_counts_rows(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
        export_reports=False,
    )

    assert summary["strategy_version_backtest_results_rows"] == 2
    assert summary["strategy_version_performance_rows"] == 3
    assert summary["strategy_version_evaluation_rows"] == 4
    assert summary["parameter_search_backtest_rows"] == 5
    assert summary["parameter_search_performance_rows"] == 6
    assert summary["parameter_search_results_rows"] == 7
    assert summary["walk_forward_validation_rows"] == 8
    assert summary["trade_plan_backtest_results_rows"] == 10
    assert summary["trade_plan_backtest_performance_rows"] == 11
    assert summary["strategy_admission_rows"] == 12


def test_workflow_passes_paths_and_config_parameters(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    summary = workflow.run_strategy_research_workflow(
        db_path="custom.duckdb",
        output_dir="custom_reports",
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
        parameter_search_start_date="2024-08-01",
        parameter_search_end_date="2024-08-31",
        strategy_versions_config_path="configs/versions.json",
        parameter_search_config_path="configs/search.json",
        candidate_config_path="configs/candidate.json",
    )

    assert summary["db_path"] == "custom.duckdb"
    assert summary["output_dir"] == "custom_reports"
    assert summary["active_candidate_config_path"] == "configs/candidate.json"
    assert _kwargs_for(calls, "version_backtest") == {
        "start_date": "2024-09-01",
        "end_date": "2024-12-01",
        "config_path": "configs/versions.json",
        "db_path": "custom.duckdb",
        "limit_strategies": None,
    }
    assert _kwargs_for(calls, "parameter_search") == {
        "start_date": "2024-08-01",
        "end_date": "2024-08-31",
        "config_path": "configs/search.json",
        "db_path": "custom.duckdb",
        "limit_strategies": None,
        "limit_param_combinations": None,
    }
    assert _kwargs_for(calls, "oos_validation") == {
        "train_start_date": "2024-09-01",
        "train_end_date": "2024-12-01",
        "validation_start_date": "2024-12-02",
        "validation_end_date": "2025-01-10",
        "config_path": "configs/search.json",
        "db_path": "custom.duckdb",
        "limit_strategies": None,
        "limit_param_combinations": None,
    }
    assert _kwargs_for(calls, "trade_plan_backtest") == {
        "db_path": "custom.duckdb",
        "start_date": "2024-09-01",
        "end_date": "2024-12-01",
    }
    assert _kwargs_for(calls, "strategy_admission") == {
        "db_path": "custom.duckdb",
        "export_candidate_config": True,
        "candidate_config_path": "configs/candidate.json",
    }


def test_workflow_passes_date_ranges_and_smoke_limits_to_data_loading_steps(monkeypatch):
    calls = []
    _patch_steps(monkeypatch, calls)

    workflow.run_strategy_research_workflow(
        db_path="custom.duckdb",
        train_start_date="2026-01-01",
        train_end_date="2026-01-31",
        validation_start_date="2026-02-01",
        validation_end_date="2026-02-15",
        parameter_search_start_date="2026-01-05",
        parameter_search_end_date="2026-01-20",
        limit_strategies=1,
        limit_param_combinations=3,
        export_reports=False,
    )

    assert _kwargs_for(calls, "version_backtest") == {
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "config_path": None,
        "db_path": "custom.duckdb",
        "limit_strategies": 1,
    }
    assert _kwargs_for(calls, "parameter_search") == {
        "start_date": "2026-01-05",
        "end_date": "2026-01-20",
        "config_path": None,
        "db_path": "custom.duckdb",
        "limit_strategies": 1,
        "limit_param_combinations": 3,
    }
    assert _kwargs_for(calls, "oos_validation")["limit_param_combinations"] == 3
    assert _kwargs_for(calls, "trade_plan_backtest") == {
        "db_path": "custom.duckdb",
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
    }


def test_workflow_empty_dataframes_count_as_zero(monkeypatch):
    def empty_pair(**kwargs):
        return pd.DataFrame(), pd.DataFrame()

    def empty_triplet(**kwargs):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(workflow, "run_strategy_version_backtest", empty_pair)
    monkeypatch.setattr(workflow, "run_strategy_version_evaluation", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "run_parameter_search", empty_triplet)
    monkeypatch.setattr(workflow, "run_oos_validation", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "run_trade_plan_backtest", empty_triplet)
    monkeypatch.setattr(workflow, "run_strategy_admission", lambda **kwargs: pd.DataFrame())

    summary = workflow.run_strategy_research_workflow(
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
        export_reports=False,
    )

    row_keys = [key for key in summary if key.endswith("_rows")]
    assert all(summary[key] == 0 for key in row_keys)


def test_cli_no_report_and_no_candidate_config(monkeypatch, capsys):
    calls = []
    _patch_steps(monkeypatch, calls)

    workflow.main(["--no-report", "--no-candidate-config"])

    captured = capsys.readouterr()
    assert "Strategy research workflow finished." in captured.out
    assert "active_candidate_config_path: None" in captured.out
    assert not any(name.startswith("export_") for name in _call_names(calls))
    assert _kwargs_for(calls, "strategy_admission")["export_candidate_config"] is False
