import pandas as pd
import pytest

from src.graph import daily_workflow_graph as workflow_graph
from src.pipeline import run_daily_workflow_graph as workflow_cli


def _df(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"value": list(range(rows))})


def _patch_steps(monkeypatch, calls=None, rows=None):
    calls = calls if calls is not None else []
    rows = rows or {
        "stock_basic": 1,
        "daily_bars": 2,
        "daily_factors": 3,
        "strategy_signals": 6,
        "candidate_pool": 4,
        "trade_plan": 5,
    }

    monkeypatch.setattr(
        workflow_graph,
        "update_stock_basic",
        lambda **kwargs: calls.append(("stock_basic", kwargs)) or _df(rows["stock_basic"]),
    )
    monkeypatch.setattr(
        workflow_graph,
        "update_daily_bars",
        lambda **kwargs: calls.append(("daily_bars", kwargs)) or _df(rows["daily_bars"]),
    )
    monkeypatch.setattr(
        workflow_graph,
        "build_daily_factors",
        lambda **kwargs: calls.append(("daily_factors", kwargs)) or _df(rows["daily_factors"]),
    )
    monkeypatch.setattr(
        workflow_graph,
        "build_strategy_signals",
        lambda **kwargs: calls.append(("strategy_signals", kwargs)) or _df(rows["strategy_signals"]),
    )
    monkeypatch.setattr(
        workflow_graph,
        "build_candidate_pool",
        lambda **kwargs: calls.append(("candidate_pool", kwargs)) or _df(rows["candidate_pool"]),
    )
    monkeypatch.setattr(
        workflow_graph,
        "build_trade_plan",
        lambda **kwargs: calls.append(("trade_plan", kwargs)) or _df(rows["trade_plan"]),
    )
    monkeypatch.setattr(
        workflow_graph,
        "export_daily_report",
        lambda **kwargs: calls.append(("report", kwargs)) or "reports/daily_report_2025-01-10.md",
    )
    return calls


def test_graph_calls_all_steps_in_order(monkeypatch):
    calls = _patch_steps(monkeypatch)

    summary = workflow_graph.run_daily_workflow_graph(
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
    assert summary["report_path"] == "reports/daily_report_2025-01-10.md"
    assert summary["errors"] == []


def test_graph_skip_stock_basic(monkeypatch):
    calls = _patch_steps(monkeypatch)

    summary = workflow_graph.run_daily_workflow_graph(
        start_date="20241101",
        end_date="20250110",
        update_stock_basic_first=False,
    )

    assert "stock_basic" not in [call[0] for call in calls]
    assert [call[0] for call in calls] == [
        "daily_bars",
        "daily_factors",
        "strategy_signals",
        "candidate_pool",
        "trade_plan",
        "report",
    ]
    assert summary["stock_basic_rows"] == 0


def test_graph_no_report_skips_export(monkeypatch):
    calls = _patch_steps(monkeypatch)

    summary = workflow_graph.run_daily_workflow_graph(
        start_date="20241101",
        end_date="20250110",
        export_report=False,
    )

    assert "report" not in [call[0] for call in calls]
    assert summary["report_path"] is None


def test_graph_counts_rows(monkeypatch):
    _patch_steps(
        monkeypatch,
        rows={
            "stock_basic": 10,
            "daily_bars": 20,
            "daily_factors": 30,
            "strategy_signals": 6,
            "candidate_pool": 40,
            "trade_plan": 5,
        },
    )

    summary = workflow_graph.run_daily_workflow_graph("20241101", "20250110")

    assert summary["stock_basic_rows"] == 10
    assert summary["daily_bars_rows"] == 20
    assert summary["daily_factors_rows"] == 30
    assert summary["strategy_signals_rows"] == 6
    assert summary["candidate_pool_rows"] == 40
    assert summary["trade_plan_rows"] == 5


def test_graph_passes_provider_dates_and_db_path(monkeypatch):
    calls = _patch_steps(monkeypatch)

    summary = workflow_graph.run_daily_workflow_graph(
        start_date="20241101",
        end_date="20250110",
        provider="akshare",
        limit=10,
        sleep_seconds=0.5,
        top_n=8,
        max_plan_items=3,
        min_amount_ma5=123.0,
        db_path="custom.duckdb",
        use_active_candidates=True,
        active_config_path="custom_active.json",
    )

    assert summary["provider"] == "akshare"
    assert summary["start_date"] == "20241101"
    assert summary["end_date"] == "20250110"
    assert summary["db_path"] == "custom.duckdb"
    assert summary["use_active_candidates"] is True
    assert summary["active_config_path"] == "custom_active.json"
    assert calls[0] == ("stock_basic", {"db_path": "custom.duckdb", "provider": "akshare"})
    assert calls[1] == (
        "daily_bars",
        {
            "start_date": "20241101",
            "end_date": "20250110",
            "db_path": "custom.duckdb",
            "limit": 10,
            "sleep_seconds": 0.5,
            "provider": "akshare",
        },
    )
    assert calls[3] == (
        "strategy_signals",
        {
            "db_path": "custom.duckdb",
            "use_active_candidates": True,
            "active_config_path": "custom_active.json",
        },
    )
    assert calls[4] == (
        "candidate_pool",
        {"top_n": 8, "min_amount_ma5": 123.0, "db_path": "custom.duckdb"},
    )
    assert calls[5] == ("trade_plan", {"max_items": 3, "db_path": "custom.duckdb"})
    assert calls[6] == ("report", {"db_path": "custom.duckdb", "output_dir": "reports"})


def test_graph_default_provider_uses_settings(monkeypatch):
    calls = _patch_steps(monkeypatch)
    monkeypatch.setattr(workflow_graph.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    summary = workflow_graph.run_daily_workflow_graph(
        start_date="20241101",
        end_date="20250110",
        db_path="test.duckdb",
    )

    assert summary["provider"] == "tushare"
    assert calls[0][1]["provider"] == "tushare"
    assert calls[1][1]["provider"] == "tushare"


def test_graph_empty_dataframe_counts_zero(monkeypatch):
    _patch_steps(
        monkeypatch,
        rows={
            "stock_basic": 0,
            "daily_bars": 0,
            "daily_factors": 0,
            "strategy_signals": 0,
            "candidate_pool": 0,
            "trade_plan": 0,
        },
    )

    summary = workflow_graph.run_daily_workflow_graph("20241101", "20250110")

    assert summary["stock_basic_rows"] == 0
    assert summary["daily_bars_rows"] == 0
    assert summary["daily_factors_rows"] == 0
    assert summary["strategy_signals_rows"] == 0
    assert summary["candidate_pool_rows"] == 0
    assert summary["trade_plan_rows"] == 0


def test_graph_reraises_step_exception(monkeypatch):
    calls = []

    monkeypatch.setattr(workflow_graph, "update_stock_basic", lambda **kwargs: _df(1))

    def fail_daily_bars(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("boom")

    monkeypatch.setattr(workflow_graph, "update_daily_bars", fail_daily_bars)

    with pytest.raises(RuntimeError, match="boom"):
        workflow_graph.run_daily_workflow_graph("20241101", "20250110")

    assert calls


def test_cli_clears_proxy_and_prints_summary(monkeypatch, capsys):
    calls = []

    def fake_clear_proxy_env_for_process():
        calls.append("clear_proxy")

    def fake_run_daily_workflow_graph(**kwargs):
        calls.append(("workflow", kwargs))
        return {
            "provider": kwargs["provider"],
            "db_path": kwargs["db_path"],
            "stock_basic_rows": 1,
            "daily_bars_rows": 2,
            "daily_factors_rows": 3,
            "strategy_signals_rows": 6,
            "candidate_pool_rows": 4,
            "trade_plan_rows": 5,
            "use_active_candidates": kwargs["use_active_candidates"],
            "active_config_path": kwargs["active_config_path"],
            "report_path": None,
        }

    monkeypatch.setattr(workflow_cli, "clear_proxy_env_for_process", fake_clear_proxy_env_for_process)
    monkeypatch.setattr(workflow_cli, "run_daily_workflow_graph", fake_run_daily_workflow_graph)
    monkeypatch.setattr(workflow_cli.settings, "DEFAULT_DATA_PROVIDER", "tushare")

    workflow_cli.main(
        [
            "--start-date",
            "20241101",
            "--end-date",
            "20250110",
            "--db-path",
            "test.duckdb",
            "--limit",
            "10",
            "--skip-stock-basic",
            "--no-report",
            "--use-active-candidates",
            "--active-config-path",
            "custom_active.json",
        ]
    )

    output = capsys.readouterr().out

    assert calls[0] == "clear_proxy"
    assert calls[1][1]["limit"] == 10
    assert calls[1][1]["update_stock_basic_first"] is False
    assert calls[1][1]["export_report"] is False
    assert calls[1][1]["use_active_candidates"] is True
    assert calls[1][1]["active_config_path"] == "custom_active.json"
    assert "LangGraph daily workflow finished." in output
    assert "provider: tushare" in output
    assert "daily_bars_rows: 2" in output
    assert "report_path: None" in output
