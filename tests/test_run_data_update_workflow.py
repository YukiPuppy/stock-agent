from pathlib import Path

import pandas as pd

import src.pipeline.run_data_update_workflow as workflow


def _patch_all_steps(monkeypatch, calls: list[tuple[str, dict]] | None = None) -> None:
    def fake_step(name):
        def runner(*args, **kwargs):
            if calls is not None:
                calls.append((name, kwargs))
            return pd.DataFrame({"value": [1, 2]})

        return runner

    for step_name in workflow.WORKFLOW_STEPS:
        monkeypatch.setattr(workflow, step_name, fake_step(step_name))


def test_run_data_update_workflow_test_mode_defaults_limits(tmp_path, monkeypatch):
    calls = []
    _patch_all_steps(monkeypatch, calls)
    monkeypatch.setattr(workflow.settings, "DB_PATH", str(tmp_path / "stock_agent.duckdb"))
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow.run_data_update_workflow(
        start_date="20250101",
        end_date="20250110",
        mode="test",
        output_dir=str(tmp_path / "reports"),
        export_report=False,
    )

    daily_bars_call = dict(calls)["update_daily_bars"]
    daily_basic_call = dict(calls)["update_daily_basic"]
    assert summary["limit_stocks"] == 50
    assert summary["limit_days"] == 10
    assert daily_bars_call["limit"] == 50
    assert daily_basic_call["limit_days"] == 10


def test_run_data_update_workflow_full_mode_does_not_default_limits(tmp_path, monkeypatch):
    calls = []
    _patch_all_steps(monkeypatch, calls)
    monkeypatch.setattr(workflow.settings, "DB_PATH", str(tmp_path / "stock_agent.duckdb"))
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow.run_data_update_workflow(
        start_date="20250101",
        end_date="20250110",
        mode="full",
        output_dir=str(tmp_path / "reports"),
        export_report=False,
    )

    daily_bars_call = dict(calls)["update_daily_bars"]
    daily_basic_call = dict(calls)["update_daily_basic"]
    assert summary["limit_stocks"] is None
    assert summary["limit_days"] is None
    assert daily_bars_call["limit"] is None
    assert daily_basic_call["limit_days"] is None


def test_run_data_update_workflow_skip_step_counts_as_skipped(tmp_path, monkeypatch):
    _patch_all_steps(monkeypatch)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow.run_data_update_workflow(
        start_date="20250101",
        end_date="20250110",
        update_moneyflow_enabled=False,
        output_dir=str(tmp_path / "reports"),
        export_report=False,
    )

    steps = {step["step_name"]: step for step in summary["steps"]}
    assert steps["update_moneyflow"]["status"] == "skipped"
    assert summary["success_count"] == 10
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 1


def test_run_data_update_workflow_failure_records_error_and_continues(tmp_path, monkeypatch):
    calls = []
    _patch_all_steps(monkeypatch, calls)

    def fail_stock_basic(*args, **kwargs):
        calls.append(("update_stock_basic", kwargs))
        raise RuntimeError("stock basic failed")

    monkeypatch.setattr(workflow, "update_stock_basic", fail_stock_basic)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow.run_data_update_workflow(
        start_date="20250101",
        end_date="20250110",
        output_dir=str(tmp_path / "reports"),
        export_report=False,
    )

    called_names = [name for name, _ in calls]
    assert "update_stock_basic" in called_names
    assert "update_trade_calendar" in called_names
    assert "update_daily_bars" in called_names
    assert summary["failed_count"] == 1
    assert summary["success_count"] == 10
    assert summary["skipped_count"] == 0
    assert summary["critical_failure"] is True
    assert summary["errors"] == [{"step_name": "update_stock_basic", "error": "stock basic failed"}]


def test_run_data_update_workflow_sanitizes_tokens_in_summary_and_output(tmp_path, monkeypatch, capsys):
    _patch_all_steps(monkeypatch)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr(workflow.settings, "TUSHARE_TOKEN", "secret-tushare-token")
    monkeypatch.setattr(workflow.settings, "LLM_API_KEY", "secret-llm-key")

    def fail_moneyflow(*args, **kwargs):
        raise RuntimeError("bad token secret-tushare-token and key secret-llm-key")

    monkeypatch.setattr(workflow, "update_moneyflow", fail_moneyflow)
    summary = workflow.run_data_update_workflow(
        start_date="20250101",
        end_date="20250110",
        output_dir=str(tmp_path / "reports"),
        export_report=False,
    )
    workflow._print_summary(summary)
    output = capsys.readouterr().out

    assert "secret-tushare-token" not in str(summary)
    assert "secret-llm-key" not in str(summary)
    assert "secret-tushare-token" not in output
    assert "secret-llm-key" not in output
    assert "***" in output


def test_run_data_update_workflow_generates_markdown_report(tmp_path, monkeypatch):
    _patch_all_steps(monkeypatch)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow.run_data_update_workflow(
        start_date="20250101",
        end_date="20250110",
        output_dir=str(tmp_path / "reports"),
    )

    report_path = Path(summary["data_update_report_path"])
    assert report_path.exists()
    assert report_path.name.startswith("data_update_workflow_")
    content = report_path.read_text(encoding="utf-8")
    assert "# 数据更新工作流报告" in content
    assert "update_daily_bars" in content
