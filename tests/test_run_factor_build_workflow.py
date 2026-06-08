from pathlib import Path

import pandas as pd

from src.pipeline import run_factor_build_workflow as workflow


def _patch_builders(monkeypatch, calls: list[str], fail_step: str | None = None) -> None:
    def runner(step_name: str):
        def _run(*, db_path=None):
            calls.append(step_name)
            if step_name == fail_step:
                raise RuntimeError(f"{step_name} failed")
            return pd.DataFrame({"value": [1, 2]})

        return _run

    monkeypatch.setattr(workflow, "build_and_save_moneyflow_factors", runner("build_moneyflow_factors"))
    monkeypatch.setattr(workflow, "build_market_regime", runner("build_market_regime"))
    monkeypatch.setattr(workflow, "build_and_save_industry_strength", runner("build_industry_strength"))
    monkeypatch.setattr(workflow, "build_daily_factors", runner("build_daily_factors"))
    monkeypatch.setattr(workflow, "run_build_factor_diagnostics", runner("build_factor_diagnostics"))


def test_run_factor_build_workflow_calls_default_steps_in_order(tmp_path, monkeypatch):
    calls: list[str] = []
    _patch_builders(monkeypatch, calls)

    summary = workflow.run_factor_build_workflow(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )

    assert calls == workflow.WORKFLOW_STEPS
    assert [step["step_name"] for step in summary["steps"]] == workflow.WORKFLOW_STEPS
    assert summary["success_count"] == 5
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 0


def test_run_factor_build_workflow_skip_flags_skip_steps(tmp_path, monkeypatch):
    calls: list[str] = []
    _patch_builders(monkeypatch, calls)

    summary = workflow.run_factor_build_workflow(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
        build_moneyflow_factors_enabled=False,
        build_industry_strength_enabled=False,
        build_factor_diagnostics_enabled=False,
    )

    assert calls == ["build_market_regime", "build_daily_factors"]
    statuses = {step["step_name"]: step["status"] for step in summary["steps"]}
    assert statuses["build_moneyflow_factors"] == "skipped"
    assert statuses["build_industry_strength"] == "skipped"
    assert statuses["build_factor_diagnostics"] == "skipped"
    assert summary["success_count"] == 2
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 3


def test_run_factor_build_workflow_records_failure_and_continues(tmp_path, monkeypatch):
    calls: list[str] = []
    _patch_builders(monkeypatch, calls, fail_step="build_daily_factors")

    summary = workflow.run_factor_build_workflow(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )

    assert calls == workflow.WORKFLOW_STEPS
    statuses = {step["step_name"]: step["status"] for step in summary["steps"]}
    assert statuses["build_daily_factors"] == "failed"
    assert statuses["build_factor_diagnostics"] == "success"
    assert summary["success_count"] == 4
    assert summary["failed_count"] == 1
    assert summary["skipped_count"] == 0
    assert summary["errors"] == [{"step_name": "build_daily_factors", "error": "build_daily_factors failed"}]
    assert "daily_factors 可能不是最新" in summary["warnings"][0]


def test_run_factor_build_workflow_generates_markdown_report(tmp_path, monkeypatch):
    calls: list[str] = []
    _patch_builders(monkeypatch, calls)

    summary = workflow.run_factor_build_workflow(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )

    report_path = Path(summary["factor_build_report_path"])
    assert report_path.exists()
    assert report_path.name.startswith("factor_build_workflow_")
    content = report_path.read_text(encoding="utf-8")
    assert "# 因子构建工作流报告" in content
    assert "不拉取数据" in content
    assert "不自动交易" in content
    assert "build_moneyflow_factors" in content
