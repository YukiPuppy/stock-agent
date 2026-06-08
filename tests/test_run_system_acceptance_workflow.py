from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.pipeline import run_system_acceptance_workflow as workflow


def _configure(monkeypatch, tushare_token: str = "test-token", llm_api_key: str = "") -> None:
    monkeypatch.setattr(settings, "DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr(settings, "TUSHARE_TOKEN", tushare_token)
    monkeypatch.setattr(settings, "TUSHARE_API_URL", "http://api.tushare.pro")
    monkeypatch.setattr(settings, "DATA_FETCH_DISABLE_PROXY", False)
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    monkeypatch.setattr(settings, "LLM_MODEL", "")
    monkeypatch.setattr(settings, "LLM_API_KEY", llm_api_key)
    monkeypatch.setattr(settings, "LLM_DISABLE_PROXY", False)


def test_acceptance_handles_empty_database_and_missing_tables(tmp_path, monkeypatch):
    _configure(monkeypatch)
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()

    summary = workflow.run_system_acceptance_workflow(db_path=str(db_path), output_dir=str(tmp_path / "reports"))

    assert summary["table_checks"]["daily_bars"]["status"] == "missing"
    assert summary["table_checks"]["daily_factors"]["status"] == "missing"
    assert summary["acceptance_status"] == "warning"
    assert Path(summary["system_acceptance_report_path"]).is_file()


def test_acceptance_fails_when_tushare_token_missing(tmp_path, monkeypatch):
    _configure(monkeypatch, tushare_token="")

    summary = workflow.run_system_acceptance_workflow(
        db_path=str(tmp_path / "missing.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )

    assert summary["acceptance_status"] == "failed"
    assert "TUSHARE_TOKEN 未配置。" in summary["errors"]


def test_acceptance_warns_for_empty_daily_bars_and_daily_factors(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()

    summary = workflow.run_system_acceptance_workflow(db_path=store.db_path, output_dir=str(tmp_path / "reports"))

    assert summary["table_checks"]["daily_bars"]["exists"] is True
    assert summary["table_checks"]["daily_bars"]["row_count"] == 0
    assert "daily_bars 为空。" in summary["warnings"]
    assert "daily_factors 为空。" in summary["warnings"]
    assert summary["acceptance_status"] == "warning"


def test_acceptance_reads_table_counts_and_date_range(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-03"],
                "code": ["000001", "000001"],
                "open": [10.0, 10.1],
                "high": [10.5, 10.6],
                "low": [9.8, 9.9],
                "close": [10.2, 10.3],
                "volume": [1000, 1100],
                "amount": [10200, 11300],
            }
        )
    )

    summary = workflow.run_system_acceptance_workflow(db_path=store.db_path, output_dir=str(tmp_path / "reports"))

    check = summary["table_checks"]["daily_bars"]
    assert check["row_count"] == 2
    assert check["date_range"] == "2026-01-02 - 2026-01-03"
    assert check["status"] == "ok"


def test_acceptance_identifies_key_report_files(tmp_path, monkeypatch):
    _configure(monkeypatch)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    latest = reports_dir / "llm_agents_index_2026-06-05.md"
    latest.write_text("# index", encoding="utf-8")
    (reports_dir / "llm_agents_index_2026-06-04.md").write_text("# older", encoding="utf-8")
    data_update = reports_dir / "data_update_workflow_2026-06-05.md"
    data_update.write_text("# data", encoding="utf-8")

    summary = workflow.run_system_acceptance_workflow(
        db_path=str(tmp_path / "missing.duckdb"),
        output_dir=str(reports_dir),
    )

    assert summary["report_checks"]["llm_agents_index_*.md"]["latest_file"] == str(latest)
    assert summary["report_checks"]["data_update_workflow_*.md"]["latest_file"] == str(data_update)


def test_acceptance_report_does_not_expose_secret_values(tmp_path, monkeypatch):
    token = "secret-tushare-token"
    api_key = "secret-llm-key"
    _configure(monkeypatch, tushare_token=token, llm_api_key=api_key)
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")

    summary = workflow.run_system_acceptance_workflow(
        db_path=str(tmp_path / "missing.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )
    report = Path(summary["system_acceptance_report_path"]).read_text(encoding="utf-8")

    assert token not in str(summary)
    assert api_key not in str(summary)
    assert token not in report
    assert api_key not in report
    assert "TUSHARE_TOKEN configured" in report
    assert "LLM_API_KEY configured" in report


def test_acceptance_generates_dated_report_with_scope_statement(tmp_path, monkeypatch):
    _configure(monkeypatch)

    summary = workflow.run_system_acceptance_workflow(
        db_path=str(tmp_path / "missing.duckdb"),
        output_dir=str(tmp_path / "reports"),
    )

    path = Path(summary["system_acceptance_report_path"])
    assert path.name == f"system_acceptance_{date.today().isoformat()}.md"
    content = path.read_text(encoding="utf-8")
    assert "不拉取数据" in content
    assert "不自动下单" in content
    assert "不会修改 active_strategies.json" in content
    assert "parameter_search_space.json" in content
