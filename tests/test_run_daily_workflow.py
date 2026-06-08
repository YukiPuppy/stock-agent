import pandas as pd
import pytest

from src.pipeline import run_daily_workflow as workflow


def _df(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"value": list(range(rows))})


def test_run_daily_workflow_calls_all_steps_in_order(monkeypatch):
    calls = []

    monkeypatch.setattr(
        workflow,
        "update_stock_basic",
        lambda db_path=None, provider="akshare": calls.append(("stock_basic", db_path, provider)) or _df(1),
    )
    monkeypatch.setattr(
        workflow,
        "update_daily_bars",
        lambda **kwargs: calls.append(("daily_bars", kwargs)) or _df(2),
    )
    monkeypatch.setattr(
        workflow,
        "build_daily_factors",
        lambda db_path=None: calls.append(("daily_factors", db_path)) or _df(3),
    )
    monkeypatch.setattr(
        workflow,
        "build_candidate_pool",
        lambda **kwargs: calls.append(("candidate_pool", kwargs)) or _df(4),
    )
    monkeypatch.setattr(
        workflow,
        "build_strategy_signals",
        lambda **kwargs: calls.append(("strategy_signals", kwargs)) or _df(6),
    )
    monkeypatch.setattr(
        workflow,
        "build_trade_plan",
        lambda **kwargs: calls.append(("trade_plan", kwargs)) or _df(5),
    )
    monkeypatch.setattr(
        workflow,
        "export_daily_report",
        lambda **kwargs: calls.append(("report", kwargs)) or "reports/daily_report_2026-01-02.md",
    )

    summary = workflow.run_daily_workflow(
        start_date="20241101",
        end_date="20250110",
        db_path="test.duckdb",
    )

    assert [call[0] for call in calls] == [
        "stock_basic",
        "daily_bars",
        "daily_factors",
        "strategy_signals",
        "candidate_pool",
        "trade_plan",
        "report",
    ]
    assert summary["stock_basic_rows"] == 1
    assert summary["daily_bars_rows"] == 2
    assert summary["daily_factors_rows"] == 3
    assert summary["strategy_signals_rows"] == 6
    assert summary["candidate_pool_rows"] == 4
    assert summary["trade_plan_rows"] == 5
    assert summary["report_path"] == "reports/daily_report_2026-01-02.md"


def test_run_daily_workflow_skip_stock_basic(monkeypatch):
    calls = []

    monkeypatch.setattr(
        workflow,
        "update_stock_basic",
        lambda **kwargs: calls.append("stock_basic") or _df(1),
    )
    monkeypatch.setattr(workflow, "update_daily_bars", lambda **kwargs: calls.append("daily_bars") or _df(2))
    monkeypatch.setattr(workflow, "build_daily_factors", lambda db_path=None: calls.append("daily_factors") or _df(3))
    monkeypatch.setattr(workflow, "build_strategy_signals", lambda **kwargs: calls.append("strategy_signals") or _df(6))
    monkeypatch.setattr(workflow, "build_candidate_pool", lambda **kwargs: calls.append("candidate_pool") or _df(4))
    monkeypatch.setattr(workflow, "build_trade_plan", lambda **kwargs: calls.append("trade_plan") or _df(5))
    monkeypatch.setattr(workflow, "export_daily_report", lambda **kwargs: calls.append("report") or "report.md")

    summary = workflow.run_daily_workflow(
        start_date="20241101",
        end_date="20250110",
        update_stock_basic_first=False,
    )

    assert calls == ["daily_bars", "daily_factors", "strategy_signals", "candidate_pool", "trade_plan", "report"]
    assert summary["stock_basic_rows"] == 0


def test_run_daily_workflow_passes_parameters(monkeypatch):
    seen = {}

    def fake_update_stock_basic(db_path=None, provider="akshare"):
        seen["stock_basic"] = {"db_path": db_path, "provider": provider}
        return _df(1)

    def fake_update_daily_bars(**kwargs):
        seen["daily_bars"] = kwargs
        return _df(1)

    def fake_build_candidate_pool(**kwargs):
        seen["candidate_pool"] = kwargs
        return _df(1)

    def fake_build_strategy_signals(**kwargs):
        seen["strategy_signals"] = kwargs
        return _df(1)

    def fake_build_trade_plan(**kwargs):
        seen["trade_plan"] = kwargs
        return _df(1)

    monkeypatch.setattr(workflow, "update_stock_basic", fake_update_stock_basic)
    monkeypatch.setattr(workflow, "update_daily_bars", fake_update_daily_bars)
    monkeypatch.setattr(workflow, "build_daily_factors", lambda db_path=None: seen.setdefault("daily_factors", db_path) or _df(1))
    monkeypatch.setattr(workflow, "build_strategy_signals", fake_build_strategy_signals)
    monkeypatch.setattr(workflow, "build_candidate_pool", fake_build_candidate_pool)
    monkeypatch.setattr(workflow, "build_trade_plan", fake_build_trade_plan)
    monkeypatch.setattr(workflow, "export_daily_report", lambda **kwargs: seen.setdefault("report", kwargs) or "custom_reports/report.md")

    summary = workflow.run_daily_workflow(
        start_date="20241101",
        end_date="20250110",
        provider="akshare",
        limit=30,
        sleep_seconds=0.25,
        top_n=12,
        max_plan_items=4,
        min_amount_ma5=123.0,
        db_path="custom.duckdb",
        report_path="custom_reports",
        use_active_candidates=True,
        active_config_path="custom_active.json",
    )

    assert seen["stock_basic"] == {"db_path": "custom.duckdb", "provider": "akshare"}
    assert seen["daily_bars"] == {
        "start_date": "20241101",
        "end_date": "20250110",
        "db_path": "custom.duckdb",
        "limit": 30,
        "sleep_seconds": 0.25,
        "provider": "akshare",
    }
    assert seen["daily_factors"] == "custom.duckdb"
    assert seen["strategy_signals"] == {
        "db_path": "custom.duckdb",
        "use_active_candidates": True,
        "active_config_path": "custom_active.json",
    }
    assert seen["candidate_pool"] == {
        "top_n": 12,
        "min_amount_ma5": 123.0,
        "db_path": "custom.duckdb",
    }
    assert seen["trade_plan"] == {"max_items": 4, "db_path": "custom.duckdb"}
    assert seen["report"] == {"db_path": "custom.duckdb", "output_dir": "custom_reports"}
    assert summary["provider"] == "akshare"
    assert summary["start_date"] == "20241101"
    assert summary["end_date"] == "20250110"
    assert summary["db_path"] == "custom.duckdb"
    assert summary["use_active_candidates"] is True
    assert summary["active_config_path"] == "custom_active.json"


def test_run_daily_workflow_counts_empty_dataframe_as_zero(monkeypatch):
    monkeypatch.setattr(workflow, "update_stock_basic", lambda **kwargs: _df(1))
    monkeypatch.setattr(workflow, "update_daily_bars", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "build_daily_factors", lambda **kwargs: _df(2))
    monkeypatch.setattr(workflow, "build_strategy_signals", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "build_candidate_pool", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(workflow, "build_trade_plan", lambda **kwargs: _df(3))
    monkeypatch.setattr(workflow, "export_daily_report", lambda **kwargs: "report.md")

    summary = workflow.run_daily_workflow("20241101", "20250110")

    assert summary["daily_bars_rows"] == 0
    assert summary["strategy_signals_rows"] == 0
    assert summary["candidate_pool_rows"] == 0


def test_run_daily_workflow_does_not_swallow_not_implemented(monkeypatch):
    def fake_update_stock_basic(**kwargs):
        raise NotImplementedError("TushareProvider will be added after AKShare MVP is verified.")

    monkeypatch.setattr(workflow, "update_stock_basic", fake_update_stock_basic)

    with pytest.raises(NotImplementedError, match="TushareProvider will be added"):
        workflow.run_daily_workflow("20241101", "20250110", provider="tushare")


def test_parse_args_maps_cli_options():
    args = workflow._parse_args(
        [
            "--start-date",
            "20241101",
            "--end-date",
            "20250110",
            "--provider",
            "akshare",
            "--limit",
            "30",
            "--sleep-seconds",
            "0.5",
            "--top-n",
            "10",
            "--max-plan-items",
            "3",
            "--min-amount-ma5",
            "100.5",
            "--db-path",
            "test.duckdb",
            "--skip-stock-basic",
            "--no-report",
            "--use-active-candidates",
            "--active-config-path",
            "custom_active.json",
        ]
    )

    assert args.start_date == "20241101"
    assert args.end_date == "20250110"
    assert args.provider == "akshare"
    assert args.limit == 30
    assert args.sleep_seconds == 0.5
    assert args.top_n == 10
    assert args.max_plan_items == 3
    assert args.min_amount_ma5 == 100.5
    assert args.db_path == "test.duckdb"
    assert args.skip_stock_basic is True
    assert args.no_report is True
    assert args.use_active_candidates is True
    assert args.active_config_path == "custom_active.json"


def test_run_daily_workflow_exports_report_when_enabled(monkeypatch):
    calls = []

    monkeypatch.setattr(workflow, "update_stock_basic", lambda **kwargs: _df(1))
    monkeypatch.setattr(workflow, "update_daily_bars", lambda **kwargs: _df(2))
    monkeypatch.setattr(workflow, "build_daily_factors", lambda **kwargs: _df(3))
    monkeypatch.setattr(workflow, "build_strategy_signals", lambda **kwargs: _df(6))
    monkeypatch.setattr(workflow, "build_candidate_pool", lambda **kwargs: _df(4))
    monkeypatch.setattr(workflow, "build_trade_plan", lambda **kwargs: _df(5))
    monkeypatch.setattr(
        workflow,
        "export_daily_report",
        lambda **kwargs: calls.append(kwargs) or "reports/daily_report_2026-01-02.md",
    )

    summary = workflow.run_daily_workflow(
        "20241101",
        "20250110",
        db_path="test.duckdb",
        export_report=True,
    )

    assert calls == [{"db_path": "test.duckdb", "output_dir": "reports"}]
    assert summary["report_path"] == "reports/daily_report_2026-01-02.md"


def test_run_daily_workflow_no_report_skips_export(monkeypatch):
    calls = []

    monkeypatch.setattr(workflow, "update_stock_basic", lambda **kwargs: _df(1))
    monkeypatch.setattr(workflow, "update_daily_bars", lambda **kwargs: _df(2))
    monkeypatch.setattr(workflow, "build_daily_factors", lambda **kwargs: _df(3))
    monkeypatch.setattr(workflow, "build_strategy_signals", lambda **kwargs: _df(6))
    monkeypatch.setattr(workflow, "build_candidate_pool", lambda **kwargs: _df(4))
    monkeypatch.setattr(workflow, "build_trade_plan", lambda **kwargs: _df(5))
    monkeypatch.setattr(workflow, "export_daily_report", lambda **kwargs: calls.append(kwargs) or "report.md")

    summary = workflow.run_daily_workflow(
        "20241101",
        "20250110",
        export_report=False,
    )

    assert calls == []
    assert summary["report_path"] is None


def test_main_clears_proxy_and_prints_summary(monkeypatch, capsys):
    calls = []

    def fake_clear_proxy_env_for_process():
        calls.append("clear_proxy")

    def fake_run_daily_workflow(**kwargs):
        calls.append(("workflow", kwargs))
        return {
            "provider": kwargs["provider"],
            "db_path": kwargs["db_path"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "stock_basic_rows": 0,
            "daily_bars_rows": 2,
            "daily_factors_rows": 3,
            "strategy_signals_rows": 6,
            "candidate_pool_rows": 4,
            "trade_plan_rows": 5,
            "use_active_candidates": kwargs["use_active_candidates"],
            "active_config_path": kwargs["active_config_path"],
        }

    monkeypatch.setattr(workflow, "clear_proxy_env_for_process", fake_clear_proxy_env_for_process)
    monkeypatch.setattr(workflow, "run_daily_workflow", fake_run_daily_workflow)
    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    workflow.main(
        [
            "--start-date",
            "20241101",
            "--end-date",
            "20250110",
            "--db-path",
            "test.duckdb",
            "--skip-stock-basic",
        ]
    )

    output = capsys.readouterr().out

    assert calls[0] == "clear_proxy"
    assert calls[1][1]["update_stock_basic_first"] is False
    assert calls[1][1]["export_report"] is True
    assert calls[1][1]["use_active_candidates"] is False
    assert "Daily workflow finished." in output
    assert "provider: tushare" in output
    assert "daily_bars_rows: 2" in output


def test_run_daily_workflow_default_provider_uses_settings(monkeypatch):
    seen = {}

    monkeypatch.setattr(workflow.settings, "DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr(workflow, "update_stock_basic", lambda **kwargs: seen.setdefault("stock_basic", kwargs) or _df(1))
    monkeypatch.setattr(workflow, "update_daily_bars", lambda **kwargs: seen.setdefault("daily_bars", kwargs) or _df(2))
    monkeypatch.setattr(workflow, "build_daily_factors", lambda **kwargs: _df(3))
    monkeypatch.setattr(workflow, "build_strategy_signals", lambda **kwargs: _df(4))
    monkeypatch.setattr(workflow, "build_candidate_pool", lambda **kwargs: _df(5))
    monkeypatch.setattr(workflow, "build_trade_plan", lambda **kwargs: _df(6))
    monkeypatch.setattr(workflow, "export_daily_report", lambda **kwargs: None)

    summary = workflow.run_daily_workflow("20241101", "20250110", db_path="test.duckdb")

    assert seen["stock_basic"]["provider"] == "tushare"
    assert seen["daily_bars"]["provider"] == "tushare"
    assert summary["provider"] == "tushare"
