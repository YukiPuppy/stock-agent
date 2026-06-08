import duckdb
import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.ui.streamlit_app import (
    ACTIVE_STRATEGY_CANDIDATE_COLUMNS,
    LLM_REPORT_PATTERNS,
    RECENT_OVERVIEW_REPORT_PATTERNS,
    REPORT_PATTERNS,
    TABLE_NAMES,
    TRADE_PLAN_COLUMNS,
    _count_equal,
    _preferred_columns,
    _sum_value,
    build_system_health_key_table_status,
    get_table_summary,
    get_latest_trade_date,
    load_active_strategy_candidate_table,
    load_latest_parameter_search_space_candidate,
    load_latest_strategy_research_suggestions,
    list_report_files,
    mask_secret_config,
    read_markdown_file,
    safe_load_table,
)


def test_get_latest_trade_date_returns_none_for_empty_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()

    assert get_latest_trade_date(store) is None


def test_build_system_health_key_table_status_marks_table_data(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000],
                "amount": [10200],
            }
        )
    )

    result = build_system_health_key_table_status(store).set_index("table_name")

    assert bool(result.loc["daily_bars", "has_data"]) is True
    assert "data_quality_report" in result.index
    assert "factor_diagnostics" in result.index
    assert "provider_compare_result" in result.index
    assert bool(result.loc["daily_factors", "has_data"]) is False


def test_safe_load_table_supports_data_quality_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_data_quality_report(
        pd.DataFrame([{"check_name": "empty_data", "status": "ok", "issue_count": 0, "message": "ok"}])
    )

    result = safe_load_table(store, "data_quality_report")

    assert len(result) == 1
    assert "provider_compare_result" in TABLE_NAMES
    assert "factor_diagnostics" in TABLE_NAMES


def test_streamlit_helpers_support_factor_diagnostics_and_reports(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_factor_diagnostics(
        pd.DataFrame(
            {
                "factor_name": ["turnover_rate"],
                "total_count": [1],
                "non_null_count": [1],
                "missing_count": [0],
                "missing_rate": [0.0],
                "diagnostic_status": ["ok"],
                "diagnostic_message": ["ok"],
            }
        )
    )
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = reports_dir / "llm_factor_insight_2026-01-02.md"
    report.write_text("# factor", encoding="utf-8")

    loaded = safe_load_table(store, "factor_diagnostics")

    assert len(loaded) == 1
    assert list_report_files(str(reports_dir), pattern="llm_factor_insight_*.md") == [str(report)]


def test_safe_load_table_supports_tushare_extension_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_trade_calendar(pd.DataFrame({"trade_date": ["2025-01-02"], "exchange": ["SSE"], "is_open": [1]}))

    result = safe_load_table(store, "trade_calendar")

    assert "trade_calendar" in TABLE_NAMES
    assert "daily_basic" in TABLE_NAMES
    assert "stock_limits" in TABLE_NAMES
    assert "suspend_daily" in TABLE_NAMES
    assert "index_daily" in TABLE_NAMES
    assert "limit_list_daily" in TABLE_NAMES
    assert "market_regime" in TABLE_NAMES
    assert "sw_industry_classification" in TABLE_NAMES
    assert "sw_daily" in TABLE_NAMES
    assert "stock_industry_map" in TABLE_NAMES
    assert "industry_strength" in TABLE_NAMES
    assert len(result) == 1


def test_safe_load_table_supports_moneyflow_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_moneyflow(pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"], "buy_lg_amount": [100.0]}))
    store.save_moneyflow_factors(
        pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"], "moneyflow_score": [30.0]})
    )

    moneyflow = safe_load_table(store, "moneyflow")
    status = build_system_health_key_table_status(store).set_index("table_name")

    assert "moneyflow" in TABLE_NAMES
    assert "moneyflow_factors" in TABLE_NAMES
    assert len(moneyflow) == 1
    assert bool(status.loc["moneyflow", "has_data"]) is True


def test_safe_load_table_supports_market_environment_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_market_regime(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "market_regime": ["strong"],
                "risk_level": ["low"],
                "limit_up_count": [80],
                "limit_down_count": [1],
            }
        )
    )

    result = safe_load_table(store, "market_regime")
    status = build_system_health_key_table_status(store).set_index("table_name")

    assert len(result) == 1
    assert bool(status.loc["market_regime", "has_data"]) is True
    assert "index_daily" in status.index
    assert "limit_list_daily" in status.index


def test_safe_load_table_supports_industry_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_industry_strength(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "industry_code": ["801780.SI"],
                "industry_strength_score": [65],
            }
        )
    )

    result = safe_load_table(store, "industry_strength")
    status = build_system_health_key_table_status(store).set_index("table_name")

    assert len(result) == 1
    assert "industry_strength" in TABLE_NAMES
    assert "stock_industry_map" in status.index
    assert bool(status.loc["industry_strength", "has_data"]) is True


def test_get_latest_trade_date_uses_candidate_pool_and_trade_plan(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_candidate_pool(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "close": [10.5],
                "pct_chg_5d": [0.07],
                "volume_ratio_5": [1.5],
                "close_position_20": [0.75],
                "score": [44.5],
                "rank": [1],
                "reason": ["趋势较强"],
            }
        )
    )
    store.save_trade_plan(
        pd.DataFrame(
            {
                "trade_date": ["20260103"],
                "code": ["000001"],
                "name": ["平安银行"],
                "rank": [1],
                "close": [20.5],
                "strategy_type": ["watch_only"],
                "action": ["仅观察"],
                "position_low": [0.0],
                "position_high": [0.0],
                "invalid_condition": ["等待新的量价确认。"],
                "t_plus_1_risk": ["T+1 风险"],
                "plan_reason": ["条件不足"],
            }
        )
    )

    assert get_latest_trade_date(store) == "20260103"


def test_list_report_files_finds_daily_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "daily_report_2026-01-02.md"
    report_2 = reports_dir / "daily_report_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "other.md").write_text("# other", encoding="utf-8")

    assert list_report_files(str(reports_dir)) == [str(report_2), str(report_1)]


def test_read_markdown_file_reads_content(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# 标题\n\n正文", encoding="utf-8")

    assert read_markdown_file(report) == "# 标题\n\n正文"


def test_get_table_summary_returns_row_count_and_trade_date_range():
    df = pd.DataFrame({"trade_date": ["2026-01-03", "2026-01-01"], "code": ["1", "2"]})

    summary = get_table_summary(df)

    assert summary["row_count"] == 2
    assert summary["date_column"] == "trade_date"
    assert summary["date_range"] == "2026-01-01 - 2026-01-03"


def test_mask_secret_config_does_not_leak_secret_values():
    assert mask_secret_config("real-token-value") == "true"
    assert "real-token-value" not in mask_secret_config("real-token-value")
    assert mask_secret_config("") == "false"
    assert mask_secret_config(None) == "false"


def test_report_pattern_constants_include_dashboard_report_types():
    assert RECENT_OVERVIEW_REPORT_PATTERNS["日度总流程报告"] == "daily_ops_workflow_*.md"
    assert RECENT_OVERVIEW_REPORT_PATTERNS["策略研究总流程报告"] == "strategy_ops_workflow_*.md"
    assert RECENT_OVERVIEW_REPORT_PATTERNS["系统总体验收报告"] == "system_acceptance_*.md"
    assert LLM_REPORT_PATTERNS["策略研究 Agent 报告"] == "llm_strategy_research_*.md"
    assert LLM_REPORT_PATTERNS["参数迭代 Agent 报告"] == "llm_parameter_iteration_*.md"
    for pattern in [
        "daily_ops_workflow_*.md",
        "strategy_ops_workflow_*.md",
        "system_acceptance_*.md",
        "llm_strategy_research_*.md",
        "llm_parameter_iteration_*.md",
    ]:
        assert pattern in REPORT_PATTERNS.values()


def test_list_report_files_finds_data_update_workflow_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "data_update_workflow_2026-06-02.md"
    report_2 = reports_dir / "data_update_workflow_2026-06-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "daily_report_2026-06-03.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="data_update_workflow_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_factor_build_workflow_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "factor_build_workflow_2026-06-03.md"
    report_2 = reports_dir / "factor_build_workflow_2026-06-04.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "data_update_workflow_2026-06-04.md").write_text("# update", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="factor_build_workflow_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_daily_ops_workflow_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "daily_ops_workflow_2026-06-03.md"
    report_2 = reports_dir / "daily_ops_workflow_2026-06-04.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "factor_build_workflow_2026-06-04.md").write_text("# factor", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="daily_ops_workflow_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_strategy_ops_workflow_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "strategy_ops_workflow_2026-06-04.md"
    report_2 = reports_dir / "strategy_ops_workflow_2026-06-05.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "daily_ops_workflow_2026-06-05.md").write_text("# daily ops", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="strategy_ops_workflow_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_system_acceptance_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "system_acceptance_2026-06-04.md"
    report_2 = reports_dir / "system_acceptance_2026-06-05.md"
    report_1.write_text("# acceptance 1", encoding="utf-8")
    report_2.write_text("# acceptance 2", encoding="utf-8")
    (reports_dir / "system_health_2026-06-05.md").write_text("# health", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="system_acceptance_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_strategy_evaluation_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "strategy_evaluation_2026-01-02.md"
    report_2 = reports_dir / "strategy_evaluation_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="strategy_evaluation_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_parameter_search_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "parameter_search_2026-01-02.md"
    report_2 = reports_dir / "parameter_search_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "strategy_evaluation_2026-01-02.md").write_text("# evaluation", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="parameter_search_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_walk_forward_validation_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "walk_forward_validation_2026-01-02.md"
    report_2 = reports_dir / "walk_forward_validation_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "parameter_search_2026-01-02.md").write_text("# search", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="walk_forward_validation_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_strategy_admission_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "strategy_admission_2026-01-02.md"
    report_2 = reports_dir / "strategy_admission_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "walk_forward_validation_2026-01-02.md").write_text("# validation", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="strategy_admission_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_report_summary_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_report_summary_2026-01-02.md"
    report_2 = reports_dir / "llm_report_summary_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_report_summary_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_agents_index_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_agents_index_2026-01-02.md"
    report_2 = reports_dir / "llm_agents_index_2026-01-03.md"
    report_1.write_text("# index 1", encoding="utf-8")
    report_2.write_text("# index 2", encoding="utf-8")
    (reports_dir / "llm_daily_review_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_agents_index_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_market_regime_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_market_regime_2026-01-02.md"
    report_2 = reports_dir / "llm_market_regime_2026-01-03.md"
    report_1.write_text("# market 1", encoding="utf-8")
    report_2.write_text("# market 2", encoding="utf-8")
    (reports_dir / "llm_daily_review_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_market_regime_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_strategy_research_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_strategy_research_2026-01-02.md"
    report_2 = reports_dir / "llm_strategy_research_2026-01-03.md"
    report_1.write_text("# strategy 1", encoding="utf-8")
    report_2.write_text("# strategy 2", encoding="utf-8")
    (reports_dir / "llm_daily_review_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_strategy_research_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_parameter_iteration_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_parameter_iteration_2026-01-02.md"
    report_2 = reports_dir / "llm_parameter_iteration_2026-01-03.md"
    report_1.write_text("# parameter 1", encoding="utf-8")
    report_2.write_text("# parameter 2", encoding="utf-8")
    (reports_dir / "llm_daily_review_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_parameter_iteration_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_load_latest_strategy_research_suggestions(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    older = reports_dir / "strategy_research_suggestions_2026-01-02.json"
    latest = reports_dir / "strategy_research_suggestions_2026-01-03.json"
    older.write_text('{"requires_human_review": true}', encoding="utf-8")
    latest.write_text('{"requires_human_review": true, "strategy_hypotheses": ["候选"]}', encoding="utf-8")

    path, payload = load_latest_strategy_research_suggestions(str(reports_dir))

    assert path == str(latest)
    assert payload["strategy_hypotheses"] == ["候选"]


def test_load_latest_parameter_search_space_candidate(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    older = reports_dir / "parameter_search_space_candidate_2026-01-02.json"
    latest = reports_dir / "parameter_search_space_candidate_2026-01-03.json"
    older.write_text('{"requires_human_review": true}', encoding="utf-8")
    latest.write_text('{"requires_human_review": true, "parameter_search_space_candidates": ["候选"]}', encoding="utf-8")

    path, payload = load_latest_parameter_search_space_candidate(str(reports_dir))

    assert path == str(latest)
    assert payload["parameter_search_space_candidates"] == ["候选"]


def test_list_report_files_finds_llm_industry_insight_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_industry_insight_2026-01-02.md"
    report_2 = reports_dir / "llm_industry_insight_2026-01-03.md"
    report_1.write_text("# industry 1", encoding="utf-8")
    report_2.write_text("# industry 2", encoding="utf-8")
    (reports_dir / "llm_daily_review_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_industry_insight_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_all_llm_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    paths = [
        reports_dir / "llm_agents_index_2026-01-02.md",
        reports_dir / "llm_report_summary_2026-01-02.md",
        reports_dir / "llm_backtest_analysis_2026-01-02.md",
        reports_dir / "llm_market_regime_2026-01-02.md",
        reports_dir / "llm_industry_insight_2026-01-02.md",
        reports_dir / "llm_strategy_research_2026-01-02.md",
        reports_dir / "llm_parameter_iteration_2026-01-02.md",
        reports_dir / "llm_risk_review_2026-01-02.md",
        reports_dir / "llm_daily_review_2026-01-02.md",
    ]
    for path in paths:
        path.write_text("# report", encoding="utf-8")
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")

    expected = [
        str(path)
        for path in sorted(
            paths,
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
    ]
    assert list_report_files(str(reports_dir), pattern="llm_*.md") == expected


def test_list_report_files_finds_llm_backtest_analysis_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_backtest_analysis_2026-01-02.md"
    report_2 = reports_dir / "llm_backtest_analysis_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "llm_report_summary_2026-01-02.md").write_text("# summary", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_backtest_analysis_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_risk_review_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_risk_review_2026-01-02.md"
    report_2 = reports_dir / "llm_risk_review_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "llm_backtest_analysis_2026-01-02.md").write_text("# backtest", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_risk_review_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_list_report_files_finds_llm_daily_review_reports(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_1 = reports_dir / "llm_daily_review_2026-01-02.md"
    report_2 = reports_dir / "llm_daily_review_2026-01-03.md"
    report_1.write_text("# report 1", encoding="utf-8")
    report_2.write_text("# report 2", encoding="utf-8")
    (reports_dir / "daily_review_2026-01-02.md").write_text("# daily", encoding="utf-8")

    assert list_report_files(str(reports_dir), pattern="llm_daily_review_*.md") == [
        str(report_2),
        str(report_1),
    ]


def test_load_active_strategy_candidate_table_reads_config(tmp_path):
    config_path = tmp_path / "active_strategies_candidate.json"
    config_path.write_text(
        """
{
  "active_strategy_candidates": [
    {
      "strategy_name": "trend_pullback",
      "strategy_version": "v1",
      "admission_score": 90.0,
      "admission_status": "qualified_for_observation",
      "admission_recommendation": "enable_observation_candidate",
      "admission_reason": "满足观察候选条件。"
    }
  ]
}
""",
        encoding="utf-8",
    )

    exists, result = load_active_strategy_candidate_table(str(config_path))

    assert exists is True
    assert result.columns.tolist() == ACTIVE_STRATEGY_CANDIDATE_COLUMNS
    assert result.loc[0, "strategy_name"] == "trend_pullback"
    assert result.loc[0, "strategy_version"] == "v1"


def test_load_active_strategy_candidate_table_missing_file_returns_empty_table(tmp_path):
    exists, result = load_active_strategy_candidate_table(str(tmp_path / "missing.json"))

    assert exists is False
    assert result.empty
    assert result.columns.tolist() == ACTIVE_STRATEGY_CANDIDATE_COLUMNS


def test_safe_load_table_returns_empty_dataframe_for_missing_table(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    with duckdb.connect(str(db_path)):
        pass

    store = StockAgentStore(str(db_path))
    result = safe_load_table(store, "candidate_pool")

    assert result.empty


def test_safe_load_table_can_read_strategy_version_evaluation(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_strategy_version_evaluation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "evaluation_score": [80.0],
                "evaluation_status": ["ready"],
                "risk_level": ["low"],
                "recommendation": ["enable_observation"],
            }
        )
    )

    result = safe_load_table(store, "strategy_version_evaluation")

    assert result["strategy_name"].tolist() == ["trend_pullback"]
    assert result.loc[0, "recommendation"] == "enable_observation"


def test_safe_load_table_can_read_parameter_search_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_parameter_search_results(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "evaluation_score": [80.0],
                "evaluation_status": ["qualified"],
                "risk_level": ["low"],
                "recommendation": ["enable_observation"],
            }
        )
    )
    store.save_parameter_search_performance(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "sample_count": [1],
                "valid_count": [1],
            }
        )
    )
    store.save_parameter_search_backtest_results(
        pd.DataFrame(
            {
                "signal_date": ["20260101"],
                "code": ["600000"],
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "signal_strength": [1.0],
            }
        )
    )

    assert "parameter_search_results" in TABLE_NAMES
    assert "parameter_search_performance" in TABLE_NAMES
    assert "parameter_search_backtest_results" in TABLE_NAMES
    assert len(safe_load_table(store, "parameter_search_results")) == 1
    assert len(safe_load_table(store, "parameter_search_performance")) == 1
    assert len(safe_load_table(store, "parameter_search_backtest_results")) == 1


def test_safe_load_table_allows_walk_forward_validation(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_walk_forward_validation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["search_001"],
                "train_valid_count": [30],
                "validation_valid_count": [12],
                "stability_score": [20.0],
                "overfit_risk": ["low"],
                "validation_status": ["passed_oos"],
                "validation_reason": ["样本外表现基本稳定。"],
            }
        )
    )

    assert "walk_forward_validation" in TABLE_NAMES
    assert len(safe_load_table(store, "walk_forward_validation")) == 1


def test_safe_load_table_allows_strategy_admission(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_strategy_admission(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "source": ["manual_version"],
                "valid_count": [40],
                "evaluation_recommendation": ["enable_observation"],
                "evaluation_score": [45.0],
                "admission_score": [90.0],
                "admission_status": ["qualified_for_observation"],
                "admission_recommendation": ["enable_observation_candidate"],
                "admission_reason": ["满足观察候选条件。"],
            }
        )
    )

    assert "strategy_admission" in TABLE_NAMES
    assert len(safe_load_table(store, "strategy_admission")) == 1


def test_safe_load_table_allows_trade_plan_backtest_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_historical_trade_plans(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-01"],
                "code": ["600000"],
                "strategy_names": ["trend"],
                "strategy_versions": ["v1"],
                "action": ["回踩低吸"],
            }
        )
    )
    store.save_trade_plan_backtest_results(
        pd.DataFrame(
            {
                "plan_date": ["2026-01-01"],
                "code": ["600000"],
                "strategy_names": ["trend"],
                "strategy_versions": ["v1"],
                "action": ["回踩低吸"],
                "is_triggered": [True],
                "is_valid": [True],
            }
        )
    )
    store.save_trade_plan_backtest_performance(
        pd.DataFrame(
            {
                "strategy_names": ["trend"],
                "strategy_versions": ["v1"],
                "action": ["回踩低吸"],
                "plan_count": [1],
            }
        )
    )

    assert "historical_trade_plans" in TABLE_NAMES
    assert "trade_plan_backtest_results" in TABLE_NAMES
    assert "trade_plan_backtest_performance" in TABLE_NAMES
    assert len(safe_load_table(store, "historical_trade_plans")) == 1
    assert len(safe_load_table(store, "trade_plan_backtest_results")) == 1
    assert len(safe_load_table(store, "trade_plan_backtest_performance")) == 1
    assert get_latest_trade_date(store) == "2026-01-01"


def test_safe_load_table_allows_actual_trades_and_execution_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "code": ["600000"],
                "side": ["buy"],
                "price": [10.0],
                "volume": [100],
            }
        )
    )
    store.save_execution_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": [""],
                "code": ["600000"],
                "side": ["buy"],
                "execution_status": ["follow_plan"],
            }
        )
    )

    assert "actual_trades" in TABLE_NAMES
    assert "execution_review" in TABLE_NAMES
    assert len(safe_load_table(store, "actual_trades")) == 1
    assert len(safe_load_table(store, "execution_review")) == 1


def test_safe_load_table_allows_daily_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "actual_trade_count": [1],
                "buy_count": [1],
                "sell_count": [0],
                "planned_trade_count": [1],
                "matched_plan_count": [1],
                "off_plan_count": [0],
                "follow_plan_count": [1],
                "deviation_count": [0],
                "chase_count": [0],
                "over_position_count": [0],
                "bought_watch_only_count": [0],
                "execution_score": [100],
                "main_issues": ["未发现明显执行偏差"],
                "review_summary": ["执行良好"],
                "next_action_suggestion": ["继续保持"],
            }
        )
    )

    assert "daily_review" in TABLE_NAMES
    assert len(safe_load_table(store, "daily_review")) == 1


def test_safe_load_table_allows_period_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_period_review(
        pd.DataFrame(
            {
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "actual_trade_count": [1],
                "off_plan_count": [0],
                "deviation_count": [0],
                "avg_execution_score": [100.0],
                "avg_return_3d": [0.02],
                "plan_trade_avg_return_3d": [0.02],
                "off_plan_avg_return_3d": [None],
                "period_summary": ["本周期执行较好。"],
                "next_period_suggestion": ["执行良好，建议继续保持，并等待更多样本评估策略有效性。"],
            }
        )
    )

    assert "period_review" in TABLE_NAMES
    assert len(safe_load_table(store, "period_review")) == 1


def test_safe_load_table_allows_actual_trade_performance(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_actual_trade_performance(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["10:00:00"],
                "code": ["600000"],
                "side": ["buy"],
                "entry_price": [10.0],
                "entry_volume": [100],
                "is_valid": [True],
            }
        )
    )

    assert "actual_trade_performance" in TABLE_NAMES
    assert len(safe_load_table(store, "actual_trade_performance")) == 1


def test_safe_load_table_allows_positions_and_position_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_positions(
        pd.DataFrame(
            {
                "as_of_date": ["2025-01-10"],
                "code": ["600000"],
                "holding_volume": [100],
                "available_volume": [0],
                "frozen_volume": [100],
                "cost_amount": [1000.0],
                "cost_price": [10.0],
                "t_plus_1_status": ["not_sellable_today"],
                "position_status": ["unknown"],
            }
        )
    )
    store.save_position_review(
        pd.DataFrame(
            {
                "as_of_date": ["2025-01-10"],
                "code": ["600000"],
                "holding_volume": [100],
                "available_volume": [0],
                "frozen_volume": [100],
                "cost_amount": [1000.0],
                "cost_price": [10.0],
                "t_plus_1_status": ["not_sellable_today"],
                "position_status": ["unknown"],
                "position_risk_level": ["high"],
                "position_flags": ["t_plus_1_locked"],
            }
        )
    )

    assert "positions" in TABLE_NAMES
    assert "position_review" in TABLE_NAMES
    assert len(safe_load_table(store, "positions")) == 1
    assert len(safe_load_table(store, "position_review")) == 1
    assert get_latest_trade_date(store) == "2025-01-10"


def test_position_metric_helpers():
    df = pd.DataFrame(
        {
            "market_value": [100.0, 200.0, None],
            "position_risk_level": ["high", "low", "high"],
        }
    )

    assert _sum_value(df, "market_value") == 300
    assert _count_equal(df, "position_risk_level", "high") == 2


def test_trade_plan_preferred_columns_include_strategy_evaluation_fields():
    df = pd.DataFrame(
        {
            "code": ["600000"],
            "strategy_versions": ["v1"],
            "recommendations": ["enable_observation"],
            "avg_strategy_weight": [1.2],
            "extra": ["x"],
        }
    )

    result = _preferred_columns(df, TRADE_PLAN_COLUMNS)

    assert result.columns[:4].tolist() == [
        "code",
        "strategy_versions",
        "recommendations",
        "avg_strategy_weight",
    ]
