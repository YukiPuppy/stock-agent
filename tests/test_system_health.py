import pandas as pd

from src.database.duckdb_store import StockAgentStore
from src.diagnostics.system_health import check_table_health, run_system_health_check


def test_check_table_health_identifies_ok_empty_and_error(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))

    result = check_table_health(
        store,
        {
            "ok_table": lambda: pd.DataFrame({"value": [1]}),
            "empty_table": lambda: pd.DataFrame(),
            "error_table": lambda: (_ for _ in ()).throw(RuntimeError("broken loader")),
        },
    )

    rows = result.set_index("table_name")
    assert rows.loc["ok_table", "status"] == "ok"
    assert rows.loc["ok_table", "row_count"] == 1
    assert rows.loc["empty_table", "status"] == "empty"
    assert rows.loc["error_table", "status"] == "error"
    assert "broken loader" in rows.loc["error_table", "message"]


def test_run_system_health_check_checks_core_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    table_names = set(summary["table_health"]["table_name"])
    assert {
        "daily_bars",
        "trade_calendar",
        "daily_basic",
        "stock_limits",
        "suspend_daily",
        "index_daily",
        "limit_list_daily",
        "moneyflow",
        "moneyflow_factors",
        "market_regime",
        "sw_industry_classification",
        "sw_daily",
        "stock_industry_map",
        "industry_strength",
        "data_quality_report",
        "provider_compare_result",
        "daily_factors",
        "trade_plan",
        "period_review",
    }.issubset(table_names)


def test_run_system_health_check_detects_data_update_workflow_report(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = reports_dir / "data_update_workflow_2026-06-03.md"
    report.write_text("# 数据更新工作流报告", encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    status = summary["data_update_report_status"].iloc[0]
    assert report_files.loc["data_update_workflow_*.md", "latest_file"] == str(report)
    assert summary["latest_data_update_report_path"] == str(report)
    assert bool(status["exists"]) is True


def test_run_system_health_check_detects_daily_ops_workflow_report(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = reports_dir / "daily_ops_workflow_2026-06-04.md"
    report.write_text("# 日度总流程运行报告", encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    status = summary["daily_ops_report_status"].iloc[0]
    assert report_files.loc["daily_ops_workflow_*.md", "latest_file"] == str(report)
    assert summary["latest_daily_ops_report_path"] == str(report)
    assert bool(status["exists"]) is True
    assert all("daily_ops_workflow" not in issue for issue in summary["blocking_issues"])


def test_run_system_health_check_detects_strategy_ops_workflow_report(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = reports_dir / "strategy_ops_workflow_2026-06-05.md"
    report.write_text("# 策略研究总流程运行报告", encoding="utf-8")
    suggestions = reports_dir / "strategy_research_suggestions_2026-06-05.json"
    candidate = reports_dir / "parameter_search_space_candidate_2026-06-05.json"
    suggestions.write_text('{"requires_human_review": true}', encoding="utf-8")
    candidate.write_text('{"requires_human_review": true}', encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    status = summary["strategy_ops_report_status"].iloc[0]
    assert report_files.loc["strategy_ops_workflow_*.md", "latest_file"] == str(report)
    assert report_files.loc["strategy_research_suggestions_*.json", "latest_file"] == str(suggestions)
    assert report_files.loc["parameter_search_space_candidate_*.json", "latest_file"] == str(candidate)
    assert summary["latest_strategy_ops_report_path"] == str(report)
    assert bool(status["exists"]) is True
    assert all("strategy_ops_workflow" not in issue for issue in summary["blocking_issues"])


def test_run_system_health_check_detects_factor_build_workflow_report(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = reports_dir / "factor_build_workflow_2026-06-04.md"
    report.write_text("# 因子构建工作流报告", encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    status = summary["factor_build_report_status"].iloc[0]
    factor_tables = summary["factor_build_table_state"].set_index("table_name")
    assert report_files.loc["factor_build_workflow_*.md", "latest_file"] == str(report)
    assert summary["latest_factor_build_report_path"] == str(report)
    assert bool(status["exists"]) is True
    assert "moneyflow_factors" in factor_tables.index
    assert "market_regime" in factor_tables.index
    assert "industry_strength" in factor_tables.index
    assert "daily_factors" in factor_tables.index
    assert "factor_diagnostics" in factor_tables.index
    assert all("factor_build_workflow" not in issue for issue in summary["blocking_issues"])


def test_run_system_health_check_detects_system_acceptance_report(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = reports_dir / "system_acceptance_2026-06-05.md"
    report.write_text("# 系统总体验收报告\n\nacceptance_status: warning\n", encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    status = summary["system_acceptance_report_status"].iloc[0]
    assert report_files.loc["system_acceptance_*.md", "latest_file"] == str(report)
    assert summary["latest_system_acceptance_report_path"] == str(report)
    assert bool(status["exists"]) is True
    assert status["acceptance_status"] == "warning"
    assert all("system_acceptance" not in issue for issue in summary["blocking_issues"])


def test_run_system_health_check_warns_for_empty_extension_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert "trade_calendar 为空" in summary["warnings"]
    assert "daily_basic 为空" in summary["warnings"]
    assert "stock_limits 为空" in summary["warnings"]
    assert "suspend_daily 为空" in summary["warnings"]
    assert "index_daily 为空" in summary["warnings"]
    assert "limit_list_daily 为空" in summary["warnings"]
    assert "moneyflow 为空" in summary["warnings"]
    assert "moneyflow_factors 为空" in summary["warnings"]
    assert "market_regime 为空" in summary["warnings"]
    assert "sw_industry_classification 为空" in summary["warnings"]
    assert "sw_daily 为空" in summary["warnings"]
    assert "stock_industry_map 为空" in summary["warnings"]
    assert "industry_strength 为空" in summary["warnings"]
    assert all("trade_calendar" not in issue for issue in summary["blocking_issues"])
    assert all("market_regime" not in issue for issue in summary["blocking_issues"])
    enriched = summary["enriched_factors"].iloc[0]
    assert enriched["has_enriched_fields"] == True
    assert enriched["daily_basic_missing_rate"] == 0.0


def test_run_system_health_check_reports_moneyflow_state(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_moneyflow(pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"]}))
    store.save_moneyflow_factors(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "moneyflow_score": [30.0],
                "main_net_amount": [100.0],
                "main_net_amount_ratio": [0.2],
            }
        )
    )
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "moneyflow_score": [30.0],
                "main_net_amount": [100.0],
                "main_net_amount_ratio": [0.2],
            }
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    state = summary["moneyflow_state"].iloc[0]
    assert summary["latest_moneyflow_date"] == "2025-01-02"
    assert state["moneyflow_rows"] == 1
    assert state["moneyflow_factors_rows"] == 1
    assert bool(state["merged_to_daily_factors"]) is True


def test_run_system_health_check_reports_industry_state(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_sw_industry_classification(pd.DataFrame({"industry_code": ["801780.SI"], "industry_name": ["银行"], "level": ["L1"], "src": ["SW2021"]}))
    store.save_sw_daily(pd.DataFrame({"trade_date": ["2025-01-02"], "industry_code": ["801780.SI"], "close": [1000.0]}))
    store.save_stock_industry_map(pd.DataFrame({"code": ["000001"], "industry_code": ["801780.SI"], "industry_name": ["银行"]}))
    store.save_industry_strength(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-02"],
                "industry_code": ["801780.SI", "801010.SI"],
                "industry_strength_score": [65.0, 10.0],
                "industry_strength_level": ["strong", "weak"],
                "industry_return_5d": [0.03, -0.04],
            }
        )
    )
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "industry_strength_score": [65.0],
                "industry_strength_level": ["strong"],
                "industry_return_5d": [0.03],
            }
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    state = summary["industry_state"].iloc[0]
    assert summary["latest_industry_strength_date"] == "20250102"
    assert summary["strong_industry_count"] == 1
    assert summary["weak_industry_count"] == 1
    assert bool(state["merged_to_daily_factors"]) is True


def test_run_system_health_check_reports_enriched_factors_missing_rate_warning(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["20260102"] * 5,
                "code": ["600000", "600001", "600002", "600003", "600004"],
                "turnover_rate": [1.0] * 5,
                "volume_ratio_daily_basic": [pd.NA, pd.NA, 1.2, 1.3, 1.4],
                "total_mv": [1000.0] * 5,
                "circ_mv": [800.0] * 5,
                "up_limit": [11.0] * 5,
                "down_limit": [9.0] * 5,
                "is_suspended": [False] * 5,
                "is_limit_up_close": [False] * 5,
                "is_limit_down_close": [False] * 5,
            }
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    enriched = summary["enriched_factors"].iloc[0]
    assert enriched["has_enriched_fields"] == True
    assert enriched["daily_basic_missing_rate"] == 0.4
    assert "daily_factors daily_basic 扩展字段缺失率较高：40.0%" in summary["warnings"]


def test_run_system_health_check_blocks_only_when_enriched_missing_rate_extreme(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["20260102"] * 10,
                "code": [f"60{i:04d}" for i in range(10)],
                "turnover_rate": [1.0] * 10,
                "volume_ratio_daily_basic": [pd.NA] * 9 + [1.2],
                "total_mv": [1000.0] * 10,
                "circ_mv": [800.0] * 10,
                "up_limit": [11.0] * 10,
                "down_limit": [9.0] * 10,
                "is_suspended": [False] * 10,
                "is_limit_up_close": [False] * 10,
                "is_limit_down_close": [False] * 10,
            }
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert "daily_factors daily_basic 扩展字段缺失率过高：90.0%" in summary["blocking_issues"]


def test_run_system_health_check_reports_latest_market_regime(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_market_regime(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "market_regime": ["weak"],
                "risk_level": ["high"],
                "limit_up_count": [12],
                "limit_down_count": [24],
            }
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert summary["latest_market_regime"] == "weak"
    assert summary["latest_market_risk_level"] == "high"
    assert summary["latest_limit_up_count"] == 12
    assert summary["latest_limit_down_count"] == 24


def test_run_system_health_check_not_ready_when_base_data_empty(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.init_tables()

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert summary["overall_status"] in {"not_ready", "partial"}
    assert "daily_bars 为空" in summary["blocking_issues"]
    assert "daily_factors 为空" in summary["blocking_issues"]


def test_run_system_health_check_ready_for_daily_planning(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    _seed_daily_planning_tables(store)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily_report_2026-01-02.md").write_text("# report", encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    assert summary["overall_status"] == "ready_for_daily_planning"


def test_run_system_health_check_ready_for_research(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    _seed_daily_planning_tables(store)
    store.save_parameter_search_results(_evaluation_df())
    store.save_trade_plan_backtest_performance(
        pd.DataFrame(
            {
                "strategy_names": ["trend_pullback"],
                "strategy_versions": ["v1"],
                "action": ["watch"],
                "plan_count": [1],
                "triggered_count": [1],
                "valid_count": [1],
            }
        )
    )
    store.save_strategy_admission(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "source": ["parameter_search"],
                "valid_count": [30],
                "admission_score": [80.0],
                "admission_status": ["qualified_for_observation"],
                "admission_recommendation": ["enable_observation_candidate"],
            }
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert summary["overall_status"] == "ready_for_research"


def test_run_system_health_check_config_and_report_files(tmp_path):
    configs_dir = tmp_path / "configs"
    reports_dir = tmp_path / "reports"
    configs_dir.mkdir()
    reports_dir.mkdir()
    (configs_dir / "strategy_versions.json").write_text("{}", encoding="utf-8")
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(reports_dir),
        configs_dir=str(configs_dir),
    )

    configs = summary["config_files"].set_index("file_name")
    reports = summary["report_files"].set_index("pattern")
    assert configs.loc["strategy_versions.json", "status"] == "ok"
    assert configs.loc["active_strategies_candidate.json", "status"] == "missing"
    assert reports.loc["daily_report_*.md", "file_count"] == 1
    assert reports.loc["period_review_*.md", "status"] == "missing"
    assert reports.loc["llm_agents_index_*.md", "status"] == "missing"
    assert reports.loc["llm_report_summary_*.md", "status"] == "missing"
    assert reports.loc["llm_backtest_analysis_*.md", "status"] == "missing"
    assert reports.loc["llm_market_regime_*.md", "status"] == "missing"
    assert reports.loc["llm_strategy_research_*.md", "status"] == "missing"
    assert reports.loc["strategy_research_suggestions_*.json", "status"] == "missing"
    assert reports.loc["llm_risk_review_*.md", "status"] == "missing"
    assert all("llm_agents_index" not in issue for issue in summary["blocking_issues"])
    assert all("llm_report_summary" not in issue for issue in summary["blocking_issues"])
    assert all("llm_backtest_analysis" not in issue for issue in summary["blocking_issues"])
    assert all("llm_market_regime" not in issue for issue in summary["blocking_issues"])
    assert all("llm_strategy_research" not in issue for issue in summary["blocking_issues"])
    assert all("strategy_research_suggestions" not in issue for issue in summary["blocking_issues"])
    assert all("llm_risk_review" not in issue for issue in summary["blocking_issues"])


def test_run_system_health_check_warns_when_tushare_token_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.diagnostics.system_health.settings.DEFAULT_DATA_PROVIDER", "akshare")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_TOKEN", "")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_API_URL", "http://api.tushare.pro")
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        False,
    )
    monkeypatch.setattr("src.diagnostics.system_health.settings.DATA_FETCH_DISABLE_PROXY", False)

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    data_source = summary["data_source_config"].iloc[0]
    assert data_source["default_data_provider"] == "akshare"
    assert data_source["official_data_provider"] == "tushare"
    assert data_source["tushare_token_configured"] == False
    assert data_source["tushare_api_url"] == "http://api.tushare.pro"
    assert data_source["tushare_api_url_is_official"] == True
    assert data_source["data_fetch_disable_proxy"] == False
    assert "AKShare is currently not recommended as the primary provider due to unit consistency concerns." in summary["warnings"]
    assert "TUSHARE_TOKEN 未配置；仅在使用 provider=tushare 时需要" in summary["warnings"]
    assert "DEFAULT_DATA_PROVIDER=tushare 但 TUSHARE_TOKEN 未配置" not in summary["blocking_issues"]


def test_run_system_health_check_blocks_when_default_tushare_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr("src.diagnostics.system_health.settings.DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_TOKEN", "")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_API_URL", "http://api.tushare.pro")
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        False,
    )

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert "DEFAULT_DATA_PROVIDER=tushare 但 TUSHARE_TOKEN 未配置" in summary["blocking_issues"]


def test_run_system_health_check_reports_official_provider_and_units(tmp_path, monkeypatch):
    monkeypatch.setattr("src.diagnostics.system_health.settings.DEFAULT_DATA_PROVIDER", "tushare")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_TOKEN", "configured-token")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_API_URL", "http://api.tushare.pro")
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        False,
    )
    monkeypatch.setattr("src.diagnostics.system_health.settings.DATA_FETCH_DISABLE_PROXY", False)

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    data_source = summary["data_source_config"].iloc[0]
    assert data_source["official_data_provider"] == "tushare"
    assert data_source["daily_bars_volume_unit"] == "手"
    assert data_source["daily_bars_amount_unit"] == "千元"
    assert data_source["actual_trades_amount_unit"] == "元"
    assert data_source["positions_amount_unit"] == "元"


def test_run_system_health_check_reports_llm_config_without_key_plaintext(tmp_path, monkeypatch):
    secret = "llm-secret-key"
    monkeypatch.setattr("src.diagnostics.system_health.settings.ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_PROVIDER", "deepseek")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_BASE_URL", "")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_API_KEY", secret)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_DISABLE_PROXY", False)

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    llm_config = summary["llm_config"].iloc[0]
    assert llm_config["enable_llm_report_agent"] == True
    assert llm_config["llm_provider"] == "deepseek"
    assert llm_config["llm_model"] == "deepseek-v4-flash"
    assert llm_config["llm_base_url"] == "https://api.deepseek.com"
    assert llm_config["llm_disable_proxy"] == False
    assert llm_config["llm_api_key_configured"] == True
    assert llm_config["message"] == "ReportAgent 已配置 DeepSeek 后端"
    assert "ReportAgent 已配置 DeepSeek 后端" in summary["warnings"]
    assert all(secret not in str(value) for value in summary.values())
    assert "llm_api_key_configured" in summary["llm_config"].columns


def test_run_system_health_check_warns_when_deepseek_key_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.diagnostics.system_health.settings.ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_PROVIDER", "deepseek")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_API_KEY", "")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_DISABLE_PROXY", False)

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert "ENABLE_LLM_REPORT_AGENT=true 且 LLM_PROVIDER=deepseek，但 LLM_API_KEY 未配置" in summary["warnings"]


def test_run_system_health_check_reports_llm_disable_proxy_status(tmp_path, monkeypatch):
    monkeypatch.setattr("src.diagnostics.system_health.settings.ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_PROVIDER", "deepseek")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_API_KEY", "configured-key")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_DISABLE_PROXY", True)

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    llm_config = summary["llm_config"].iloc[0]
    warning = "LLM API 调用时将临时绕过 HTTP/HTTPS/ALL proxy 环境变量。"
    assert llm_config["llm_disable_proxy"] == True
    assert warning in summary["warnings"]
    assert warning not in summary["blocking_issues"]


def test_run_system_health_check_checks_llm_daily_review_report_as_warning(tmp_path, monkeypatch):
    secret = "llm-secret-key"
    monkeypatch.setattr("src.diagnostics.system_health.settings.ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_PROVIDER", "deepseek")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_API_KEY", secret)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr("src.diagnostics.system_health.settings.LLM_DISABLE_PROXY", True)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    llm_config = summary["llm_config"].iloc[0]
    warning = "llm_daily_review_*.md 暂无报告；仅影响 LLM 每日执行复盘看板展示"
    parameter_report_warning = "llm_parameter_iteration_*.md 暂无报告；仅影响 LLM 参数迭代建议看板展示"
    parameter_json_warning = "parameter_search_space_candidate_*.json 暂无候选参数研究建议；仅影响 ParameterIterationAgent JSON 展示"
    assert "llm_daily_review_*.md" in report_files.index
    assert "llm_parameter_iteration_*.md" in report_files.index
    assert "parameter_search_space_candidate_*.json" in report_files.index
    assert report_files.loc["llm_daily_review_*.md", "status"] == "missing"
    assert report_files.loc["llm_parameter_iteration_*.md", "status"] == "missing"
    assert report_files.loc["parameter_search_space_candidate_*.json", "status"] == "missing"
    assert warning in summary["warnings"]
    assert parameter_report_warning in summary["warnings"]
    assert parameter_json_warning in summary["warnings"]
    assert warning not in summary["blocking_issues"]
    assert parameter_report_warning not in summary["blocking_issues"]
    assert parameter_json_warning not in summary["blocking_issues"]
    assert llm_config["LLM_PROVIDER"] == "deepseek"
    assert llm_config["ENABLE_LLM_REPORT_AGENT"] == True
    assert llm_config["LLM_DISABLE_PROXY"] == True
    assert llm_config["LLM_MODEL"] == "deepseek-v4-flash"
    assert llm_config["LLM_API_KEY_configured"] == True
    assert secret not in str(summary)


def test_run_system_health_check_checks_llm_market_regime_report_as_warning(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")
    (reports_dir / "llm_agents_index_2026-01-02.md").write_text("# LLM Index\n\n- ReportAgent", encoding="utf-8")

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    missing_report_warning = "llm_market_regime_*.md 暂无报告；仅影响 LLM 市场环境解释看板展示"
    missing_index_warning = "llm_agents_index_*.md 未包含 MarketRegimeAgent；仅影响 LLM Agent 报告索引展示"
    assert report_files.loc["llm_market_regime_*.md", "status"] == "missing"
    assert missing_report_warning in summary["warnings"]
    assert missing_index_warning in summary["warnings"]
    assert missing_report_warning not in summary["blocking_issues"]
    assert missing_index_warning not in summary["blocking_issues"]


def test_run_system_health_check_checks_llm_industry_insight_report_as_warning(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")
    (reports_dir / "llm_agents_index_2026-01-02.md").write_text("# LLM Index\n\n- MarketRegimeAgent", encoding="utf-8")

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    missing_report_warning = "llm_industry_insight_*.md 暂无报告；仅影响 LLM 行业洞察看板展示"
    missing_index_warning = "llm_agents_index_*.md 未包含 IndustryInsightAgent；仅影响 LLM Agent 报告索引展示"
    assert report_files.loc["llm_industry_insight_*.md", "status"] == "missing"
    assert missing_report_warning in summary["warnings"]
    assert missing_index_warning in summary["warnings"]
    assert missing_report_warning not in summary["blocking_issues"]
    assert missing_index_warning not in summary["blocking_issues"]


def test_run_system_health_check_marks_llm_reports_present(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "llm_agents_index_2026-01-02.md").write_text(
        "# llm index\n\nMarketRegimeAgent\nIndustryInsightAgent\nFactorInsightAgent\nStrategyResearchAgent\nParameterIterationAgent",
        encoding="utf-8",
    )
    (reports_dir / "llm_report_summary_2026-01-02.md").write_text("# llm summary", encoding="utf-8")
    (reports_dir / "llm_backtest_analysis_2026-01-02.md").write_text("# llm backtest", encoding="utf-8")
    (reports_dir / "llm_market_regime_2026-01-02.md").write_text("# llm market", encoding="utf-8")
    (reports_dir / "llm_industry_insight_2026-01-02.md").write_text("# llm industry", encoding="utf-8")
    (reports_dir / "llm_factor_insight_2026-01-02.md").write_text("# llm factor", encoding="utf-8")
    (reports_dir / "llm_strategy_research_2026-01-02.md").write_text("# llm strategy", encoding="utf-8")
    (reports_dir / "strategy_research_suggestions_2026-01-02.json").write_text("{}", encoding="utf-8")
    (reports_dir / "llm_parameter_iteration_2026-01-02.md").write_text("# llm parameter", encoding="utf-8")
    (reports_dir / "parameter_search_space_candidate_2026-01-02.json").write_text("{}", encoding="utf-8")
    (reports_dir / "llm_risk_review_2026-01-02.md").write_text("# llm risk", encoding="utf-8")
    (reports_dir / "llm_daily_review_2026-01-02.md").write_text("# llm daily", encoding="utf-8")

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    report_files = summary["report_files"].set_index("pattern")
    assert report_files.loc["llm_agents_index_*.md", "status"] == "ok"
    assert report_files.loc["llm_report_summary_*.md", "status"] == "ok"
    assert report_files.loc["llm_backtest_analysis_*.md", "status"] == "ok"
    assert report_files.loc["llm_market_regime_*.md", "status"] == "ok"
    assert report_files.loc["llm_industry_insight_*.md", "status"] == "ok"
    assert report_files.loc["llm_factor_insight_*.md", "status"] == "ok"
    assert report_files.loc["llm_strategy_research_*.md", "status"] == "ok"
    assert report_files.loc["strategy_research_suggestions_*.json", "status"] == "ok"
    assert report_files.loc["llm_parameter_iteration_*.md", "status"] == "ok"
    assert report_files.loc["parameter_search_space_candidate_*.json", "status"] == "ok"
    assert report_files.loc["llm_risk_review_*.md", "status"] == "ok"
    assert report_files.loc["llm_daily_review_*.md", "status"] == "ok"
    assert summary["llm_agents_index_content"].iloc[0]["contains_market_regime_agent"] == True
    assert summary["llm_agents_index_content"].iloc[0]["contains_industry_insight_agent"] == True
    assert summary["llm_agents_index_content"].iloc[0]["contains_factor_insight_agent"] == True
    assert summary["llm_agents_index_content"].iloc[0]["contains_strategy_research_agent"] == True
    assert summary["llm_agents_index_content"].iloc[0]["contains_parameter_iteration_agent"] == True


def test_run_system_health_check_reports_factor_diagnostics_state(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_factor_diagnostics(
        pd.DataFrame(
            [
                {
                    "factor_name": "moneyflow_score",
                    "total_count": 10,
                    "non_null_count": 1,
                    "missing_count": 9,
                    "missing_rate": 0.9,
                    "diagnostic_status": "high_missing",
                    "diagnostic_message": "high",
                },
                {
                    "factor_name": "turnover_rate",
                    "total_count": 10,
                    "non_null_count": 5,
                    "missing_count": 5,
                    "missing_rate": 0.5,
                    "diagnostic_status": "medium_missing",
                    "diagnostic_message": "medium",
                },
            ]
        )
    )
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily_report_2026-01-02.md").write_text("# daily", encoding="utf-8")

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(reports_dir),
        configs_dir=str(tmp_path / "configs"),
    )

    assert summary["factor_high_missing_count"] == 1
    assert summary["factor_medium_missing_count"] == 1
    assert "factor_diagnostics high_missing 因子数量：1" in summary["warnings"]
    assert "factor_diagnostics high_missing 因子数量：1" not in summary["blocking_issues"]


def test_run_system_health_check_warns_for_allowed_non_official_tushare_url(tmp_path, monkeypatch):
    dummy_token = "configured-" + "token"
    monkeypatch.setattr("src.diagnostics.system_health.settings.DEFAULT_DATA_PROVIDER", "akshare")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_TOKEN", dummy_token)
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_API_URL",
        "https://example.test/tushare",
    )
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        True,
    )

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    data_source = summary["data_source_config"].iloc[0]
    assert data_source["tushare_api_url"] == "https://example.test/tushare"
    assert data_source["tushare_api_url_is_official"] == False
    assert data_source["tushare_allow_non_official_api_url"] == True
    assert (
        "Using non-official Tushare API URL; token and data integrity risk should be reviewed."
        in summary["warnings"]
    )
    assert all(dummy_token not in str(value) for value in summary.values())


def test_run_system_health_check_blocks_for_disallowed_non_official_tushare_url(tmp_path, monkeypatch):
    dummy_token = "configured-" + "token"
    monkeypatch.setattr("src.diagnostics.system_health.settings.DEFAULT_DATA_PROVIDER", "akshare")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_TOKEN", dummy_token)
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_API_URL",
        "https://example.test/tushare",
    )
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        False,
    )

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert (
        "Non-official Tushare API URL is not allowed unless "
        "TUSHARE_ALLOW_NON_OFFICIAL_API_URL=true"
    ) not in summary["blocking_issues"]


def test_run_system_health_check_warns_when_data_fetch_disable_proxy_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("src.diagnostics.system_health.settings.DEFAULT_DATA_PROVIDER", "akshare")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_TOKEN", "")
    monkeypatch.setattr("src.diagnostics.system_health.settings.TUSHARE_API_URL", "http://api.tushare.pro")
    monkeypatch.setattr(
        "src.diagnostics.system_health.settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
        False,
    )
    monkeypatch.setattr("src.diagnostics.system_health.settings.DATA_FETCH_DISABLE_PROXY", True)

    summary = run_system_health_check(
        db_path=str(tmp_path / "stock_agent.duckdb"),
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    data_source = summary["data_source_config"].iloc[0]
    assert data_source["data_fetch_disable_proxy"] == True
    assert "数据拉取时将临时绕过 HTTP/HTTPS/ALL proxy 环境变量。" in summary["warnings"]
    assert "数据拉取时将临时绕过 HTTP/HTTPS/ALL proxy 环境变量。" not in summary["blocking_issues"]


def test_run_system_health_check_blocks_when_data_quality_has_error(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_data_quality_report(
        pd.DataFrame(
            [
                {"check_name": "empty_data", "status": "error", "issue_count": 1, "message": "empty"},
                {"check_name": "duplicated_rows", "status": "warning", "issue_count": 2, "message": "dup"},
            ]
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert "data_quality_report 存在 1 项 error" in summary["blocking_issues"]
    assert "data_quality_report 存在 1 项 warning" in summary["warnings"]


def test_run_system_health_check_does_not_block_for_provider_compare_warning(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_provider_compare_result(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-01-02",
                    "code": "600000",
                    "field": "amount",
                    "left_value": 1.0,
                    "right_value": 2.0,
                    "relative_diff": 0.5,
                    "status": "warning",
                    "message": "diagnostic only",
                }
            ]
        )
    )

    summary = run_system_health_check(
        db_path=store.db_path,
        reports_dir=str(tmp_path / "reports"),
        configs_dir=str(tmp_path / "configs"),
    )

    assert all("provider_compare" not in issue for issue in summary["blocking_issues"])


def _seed_daily_planning_tables(store: StockAgentStore) -> None:
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
    store.save_daily_factors(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "close": [10.2],
                "pct_chg_1d": [0.01],
            }
        )
    )
    store.save_strategy_signals(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "signal_strength": [1.0],
            }
        )
    )
    store.save_candidate_pool(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "close": [10.2],
                "score": [80.0],
                "rank": [1],
            }
        )
    )
    store.save_trade_plan(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "rank": [1],
                "close": [10.2],
                "strategy_type": ["watch_only"],
                "action": ["观察"],
                "position_low": [0.0],
                "position_high": [0.0],
            }
        )
    )


def _evaluation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "sample_count": [30],
            "valid_count": [30],
            "evaluation_score": [80.0],
            "evaluation_status": ["qualified"],
            "risk_level": ["medium"],
            "recommendation": ["enable_observation"],
        }
    )
