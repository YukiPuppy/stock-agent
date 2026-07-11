from datetime import date
from pathlib import Path

from src.pipeline import run_strategy_ops_workflow as workflow


def _patch_successful_steps(monkeypatch, calls):
    def fake_factor_build(**kwargs):
        calls.append(("factor_build", kwargs))
        return {"factor_build_report_path": "reports/factor_build_workflow_2026-06-05.md"}

    def fake_strategy_research(**kwargs):
        calls.append(("strategy_research", kwargs))
        return {
            "strategy_evaluation_report_path": "reports/strategy_evaluation_2026-06-05.md",
            "parameter_search_report_path": "reports/parameter_search_2026-06-05.md",
            "walk_forward_validation_report_path": "reports/walk_forward_validation_2026-06-05.md",
            "trade_plan_backtest_report_path": "reports/trade_plan_backtest_2026-06-05.md",
            "strategy_admission_report_path": "reports/strategy_admission_2026-06-05.md",
        }

    def fake_backtest_agent(**kwargs):
        calls.append(("backtest_agent", kwargs))
        return "reports/llm_backtest_analysis_2026-06-05.md"

    def fake_strategy_agent(**kwargs):
        calls.append(("strategy_agent", kwargs))
        return {
            "strategy_research_report_path": "reports/llm_strategy_research_2026-06-05.md",
            "strategy_research_suggestions_path": "reports/strategy_research_suggestions_2026-06-05.json",
        }

    def fake_parameter_agent(**kwargs):
        calls.append(("parameter_agent", kwargs))
        return {
            "parameter_iteration_report_path": "reports/llm_parameter_iteration_2026-06-05.md",
            "parameter_search_space_candidate_path": "reports/parameter_search_space_candidate_2026-06-05.json",
        }

    def fake_health_check(**kwargs):
        calls.append(("health_check", kwargs))
        return {"overall_status": "partial"}

    def fake_health_report(summary, output_dir="reports", report_date=None):
        calls.append(("health_report", {"summary": summary, "output_dir": output_dir, "report_date": report_date}))
        return str(Path(output_dir) / "system_health_2026-06-05.md")

    monkeypatch.setattr(workflow, "run_factor_build_workflow", fake_factor_build)
    monkeypatch.setattr(workflow, "run_strategy_research_workflow", fake_strategy_research)
    monkeypatch.setattr(workflow, "run_backtest_analysis_agent_pipeline", fake_backtest_agent)
    monkeypatch.setattr(workflow, "run_strategy_research_agent_pipeline", fake_strategy_agent)
    monkeypatch.setattr(workflow, "run_parameter_iteration_agent_pipeline", fake_parameter_agent)
    monkeypatch.setattr(workflow, "run_system_health_check", fake_health_check)
    monkeypatch.setattr(workflow, "export_system_health_report", fake_health_report)


def test_default_calls_all_strategy_ops_steps(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_strategy_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    assert [name for name, _ in calls] == [
        "factor_build",
        "strategy_research",
        "backtest_agent",
        "strategy_agent",
        "parameter_agent",
        "health_check",
        "health_report",
    ]
    assert summary["success_count"] == 6
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["backtest_analysis_agent_path"] == "reports/llm_backtest_analysis_2026-06-05.md"
    assert summary["strategy_research_suggestions_path"] == "reports/strategy_research_suggestions_2026-06-05.json"
    assert summary["parameter_search_space_candidate_path"] == "reports/parameter_search_space_candidate_2026-06-05.json"
    assert summary["workers"] == 1
    assert summary["parallel_enabled"] is False


def test_cli_workers_default_and_override():
    assert workflow._parse_args([]).workers == 1
    assert workflow._parse_args(["--workers", "2"]).workers == 2


def test_dry_run_plan_displays_workers_without_running_workflow(monkeypatch, capsys):
    monkeypatch.setattr(
        workflow,
        "build_strategy_research_dry_run_plan",
        lambda **kwargs: {
            "enabled_strategy_versions_count": 1,
            "parameter_search_combinations_count": 2,
            "estimated_admission_candidates_count": 3,
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_strategy_ops_workflow",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("workflow must not run")),
    )

    workflow.main(["--dry-run-plan", "--workers", "2"])

    output = capsys.readouterr().out
    assert "workers: 2" in output
    assert "parallel enabled: True" in output
    assert "does not execute parallel computation" in output


def test_strategy_research_disables_candidate_config_and_passes_dates(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    workflow.run_strategy_ops_workflow(
        train_start_date="2024-09-01",
        train_end_date="2024-12-01",
        validation_start_date="2024-12-02",
        validation_end_date="2025-01-10",
        db_path="test.duckdb",
        output_dir=str(tmp_path),
    )

    strategy_kwargs = [kwargs for name, kwargs in calls if name == "strategy_research"][0]
    assert strategy_kwargs["train_start_date"] == "2024-09-01"
    assert strategy_kwargs["train_end_date"] == "2024-12-01"
    assert strategy_kwargs["validation_start_date"] == "2024-12-02"
    assert strategy_kwargs["validation_end_date"] == "2025-01-10"
    assert strategy_kwargs["parameter_search_start_date"] == "2024-09-01"
    assert strategy_kwargs["parameter_search_end_date"] == "2024-12-01"
    assert strategy_kwargs["export_candidate_config"] is False


def test_skip_flags_skip_requested_steps(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_strategy_ops_workflow(
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        build_factors=False,
        run_strategy_research=False,
        run_backtest_analysis_agent=False,
        run_strategy_research_agent=False,
        run_parameter_iteration_agent=False,
        run_health_check=False,
    )

    assert calls == []
    assert summary["success_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 6
    assert {step["status"] for step in summary["steps"]} == {"skipped"}


def test_step_failure_records_error_and_continues(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    def fail_strategy_research(**kwargs):
        calls.append(("strategy_research", kwargs))
        raise RuntimeError("strategy failed")

    monkeypatch.setattr(workflow, "run_strategy_research_workflow", fail_strategy_research)

    summary = workflow.run_strategy_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    call_names = [name for name, _ in calls]
    assert "backtest_agent" in call_names
    assert "strategy_agent" in call_names
    assert "parameter_agent" in call_names
    assert "health_check" in call_names
    assert summary["success_count"] == 5
    assert summary["failed_count"] == 1
    assert summary["skipped_count"] == 0
    assert summary["errors"][0]["step_name"] == "run_strategy_research_workflow"
    assert summary["errors"][0]["error"] == "strategy failed"
    assert "RuntimeError: strategy failed" in summary["errors"][0]["traceback"]


def test_generates_strategy_ops_markdown_report(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_strategy_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    report_path = Path(summary["strategy_ops_report_path"])
    assert report_path.name == f"strategy_ops_workflow_{summary['run_id']}.md"
    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "# 策略研究总流程运行报告" in content
    assert "本流程不自动下单" in content
    assert "不直接修改正式策略配置" in content
    assert "run_factor_build_workflow" in content
    assert "strategy_research_suggestions JSON 路径" in content
    assert "parameter_search_space_candidate JSON 路径" in content
    assert f"run_id: {summary['run_id']}" in content


def test_same_day_runs_do_not_overwrite_strategy_ops_report(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    first = workflow.run_strategy_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))
    second = workflow.run_strategy_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    first_path = Path(first["strategy_ops_report_path"])
    second_path = Path(second["strategy_ops_report_path"])
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()


def test_outputs_step_level_logs(tmp_path, monkeypatch, capsys):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    workflow.run_strategy_ops_workflow(
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        build_factors=False,
        run_strategy_research=False,
        run_backtest_analysis_agent=False,
        run_strategy_research_agent=False,
        run_parameter_iteration_agent=False,
        run_health_check=False,
    )

    output = capsys.readouterr().out
    assert "[start] run_factor_build_workflow" in output
    assert "[start] run_strategy_research_workflow" in output
    assert "[success]" not in output


def test_outputs_success_and_failed_logs(tmp_path, monkeypatch, capsys):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    def fail_strategy_research(**kwargs):
        raise RuntimeError("strategy failed")

    monkeypatch.setattr(workflow, "run_strategy_research_workflow", fail_strategy_research)

    workflow.run_strategy_ops_workflow(
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        run_backtest_analysis_agent=False,
        run_strategy_research_agent=False,
        run_parameter_iteration_agent=False,
        run_health_check=False,
    )

    output = capsys.readouterr().out
    assert "[start] run_factor_build_workflow" in output
    assert "[success] run_factor_build_workflow rows=0 elapsed=" in output
    assert "[failed] run_strategy_research_workflow error=strategy failed elapsed=" in output


def test_failure_still_generates_strategy_ops_report_with_traceback(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    def fail_strategy_research(**kwargs):
        raise RuntimeError("strategy failed")

    monkeypatch.setattr(workflow, "run_strategy_research_workflow", fail_strategy_research)

    summary = workflow.run_strategy_ops_workflow(
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        run_backtest_analysis_agent=False,
        run_strategy_research_agent=False,
        run_parameter_iteration_agent=False,
        run_health_check=False,
    )

    report_path = Path(summary["strategy_ops_report_path"])
    content = report_path.read_text(encoding="utf-8")
    assert report_path.is_file()
    assert "success_count: 1" in content
    assert "failed_count: 1" in content
    assert "skipped_count: 4" in content
    assert "run_strategy_research_workflow | failed" in content
    assert "RuntimeError: strategy failed" in content


def test_parallel_worker_failure_is_preserved_in_summary_and_report(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    def fail_strategy_research(**kwargs):
        exc = RuntimeError("parallel stage failed")
        worker_error = {
            "stage_name": "run_parameter_search",
            "worker_index": 1,
            "pid": 1234,
            "error": "worker boom",
        }
        exc.parallel_stage_summaries = [
            {
                "stage_name": "run_parameter_search",
                "status": "failed",
                "requested_workers": 2,
                "workers": 1,
                "elapsed_seconds": 0.25,
                "rows": 0,
                "worker_errors": [worker_error],
            }
        ]
        exc.profile_steps = [
            {
                "function_name": "run_parameter_search",
                "status": "failed",
                "elapsed_seconds": 0.25,
                "rows": 0,
            }
        ]
        raise exc

    monkeypatch.setattr(workflow, "run_strategy_research_workflow", fail_strategy_research)

    summary = workflow.run_strategy_ops_workflow(
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        build_factors=False,
        workers=2,
        skip_llm_agents=True,
        run_health_check=False,
    )

    assert summary["failed_count"] == 1
    assert summary["parallel_worker_errors"][0]["error"] == "worker boom"
    content = Path(summary["strategy_ops_report_path"]).read_text(encoding="utf-8")
    assert "run_parameter_search | failed | 2 | 1" in content
    assert "worker boom" in content


def test_smoke_mode_limits_strategy_research_and_skips_expensive_steps(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_strategy_ops_workflow(
        train_start_date="2026-01-01",
        train_end_date="2026-01-05",
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        mode="smoke",
        limit_strategies=1,
        limit_param_combinations=3,
        skip_llm_agents=True,
    )

    strategy_kwargs = [kwargs for name, kwargs in calls if name == "strategy_research"][0]
    assert strategy_kwargs["limit_strategies"] == 1
    assert strategy_kwargs["limit_param_combinations"] == 3
    assert summary["mode"] == "smoke"
    assert summary["success_count"] == 1
    assert summary["skipped_count"] == 5
    assert [name for name, _ in calls] == ["strategy_research"]
