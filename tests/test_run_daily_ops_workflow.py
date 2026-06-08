from datetime import date
from pathlib import Path

import pytest

from src.pipeline import run_daily_ops_workflow as workflow


def _patch_successful_steps(monkeypatch, calls):
    def fake_data_update(**kwargs):
        calls.append(("data_update", kwargs))
        return {"data_update_report_path": "reports/data_update_workflow_2026-06-04.md"}

    def fake_factor_build(**kwargs):
        calls.append(("factor_build", kwargs))
        return {"factor_build_report_path": "reports/factor_build_workflow_2026-06-04.md"}

    def fake_daily_plan(**kwargs):
        calls.append(("daily_plan", kwargs))
        return {"daily_report_path": "reports/daily_report_2026-06-04.md"}

    def fake_health_check(**kwargs):
        calls.append(("health_check", kwargs))
        return {"overall_status": "partial"}

    def fake_health_report(summary, output_dir="reports", report_date=None):
        calls.append(("health_report", {"summary": summary, "output_dir": output_dir, "report_date": report_date}))
        return str(Path(output_dir) / "system_health_2026-06-04.md")

    def fake_llm_agents(**kwargs):
        calls.append(("llm_agents", kwargs))
        return {"llm_agents_index_path": "reports/llm_agents_index_2026-06-04.md"}

    monkeypatch.setattr(workflow, "run_data_update_workflow", fake_data_update)
    monkeypatch.setattr(workflow, "run_factor_build_workflow", fake_factor_build)
    monkeypatch.setattr(workflow, "run_daily_planning_workflow", fake_daily_plan)
    monkeypatch.setattr(workflow, "run_system_health_check", fake_health_check)
    monkeypatch.setattr(workflow, "export_system_health_report", fake_health_report)
    monkeypatch.setattr(workflow, "run_llm_agents_workflow", fake_llm_agents)


def test_default_does_not_call_data_update_workflow(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_daily_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    call_names = [name for name, _ in calls]
    assert "data_update" not in call_names
    assert summary["steps"][0]["step_name"] == "run_data_update_workflow"
    assert summary["steps"][0]["status"] == "skipped"
    assert summary["skipped_count"] == 1


def test_update_data_true_calls_data_update_workflow(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_daily_ops_workflow(
        start_date="20250101",
        end_date="20250110",
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        update_data=True,
        data_update_mode="test",
    )

    data_update_calls = [kwargs for name, kwargs in calls if name == "data_update"]
    assert len(data_update_calls) == 1
    assert data_update_calls[0]["start_date"] == "20250101"
    assert data_update_calls[0]["end_date"] == "20250110"
    assert data_update_calls[0]["mode"] == "test"
    assert summary["data_update_report_path"] == "reports/data_update_workflow_2026-06-04.md"


def test_update_data_requires_start_and_end_date(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    with pytest.raises(ValueError, match="start_date and end_date are required when update_data=True"):
        workflow.run_daily_ops_workflow(update_data=True, output_dir=str(tmp_path))

    assert calls == []


def test_default_calls_factor_daily_plan_health_and_llm(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_daily_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    call_names = [name for name, _ in calls]
    assert "factor_build" in call_names
    assert "daily_plan" in call_names
    assert "health_check" in call_names
    assert "health_report" in call_names
    assert "llm_agents" in call_names
    assert summary["success_count"] == 4
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 1


def test_skip_flags_skip_requested_steps(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_daily_ops_workflow(
        db_path="test.duckdb",
        output_dir=str(tmp_path),
        build_factors=False,
        run_daily_plan=False,
        run_health_check=False,
        run_llm_agents=False,
    )

    assert calls == []
    assert summary["success_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 5
    assert {step["status"] for step in summary["steps"]} == {"skipped"}


def test_step_failure_records_error_and_continues(tmp_path, monkeypatch):
    calls = []
    secret = "secret-llm-key"
    monkeypatch.setattr(workflow.settings, "LLM_API_KEY", secret)

    def fail_factor_build(**kwargs):
        calls.append(("factor_build", kwargs))
        raise RuntimeError(f"factor failed with {secret}")

    _patch_successful_steps(monkeypatch, calls)
    monkeypatch.setattr(workflow, "run_factor_build_workflow", fail_factor_build)

    summary = workflow.run_daily_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    call_names = [name for name, _ in calls]
    assert "daily_plan" in call_names
    assert "health_check" in call_names
    assert "llm_agents" in call_names
    assert summary["success_count"] == 3
    assert summary["failed_count"] == 1
    assert summary["skipped_count"] == 1
    assert summary["errors"][0]["step_name"] == "run_factor_build_workflow"
    assert secret not in str(summary["errors"])
    assert "***" in str(summary["errors"])


def test_generates_daily_ops_markdown_report(tmp_path, monkeypatch):
    calls = []
    _patch_successful_steps(monkeypatch, calls)

    summary = workflow.run_daily_ops_workflow(db_path="test.duckdb", output_dir=str(tmp_path))

    report_path = Path(summary["daily_ops_report_path"])
    assert report_path.name == f"daily_ops_workflow_{date.today().isoformat()}.md"
    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "# 日度总流程运行报告" in content
    assert "本流程不自动下单，不构成投资建议。" in content
    assert "run_factor_build_workflow" in content
    assert str(report_path) in content
