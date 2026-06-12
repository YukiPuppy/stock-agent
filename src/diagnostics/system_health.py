"""System health checks for local stock-agent state."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from src.agents.llm_client import resolve_llm_model
from src.config import settings
from src.config import units
from src.database.duckdb_store import StockAgentStore
from src.data_providers.tushare_provider import is_official_tushare_api_url


CORE_TABLES = [
    "stock_basic",
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
    "factor_diagnostics",
    "strategy_signals",
    "candidate_pool",
    "trade_plan",
    "strategy_version_evaluation",
    "parameter_search_results",
    "walk_forward_validation",
    "trade_plan_backtest_performance",
    "strategy_admission",
    "actual_trades",
    "execution_review",
    "daily_review",
    "actual_trade_performance",
    "positions",
    "position_review",
    "period_review",
]

CONFIG_FILES = [
    "active_strategies_candidate.json",
    "strategy_versions.json",
    "parameter_search_space.json",
]

REPORT_PATTERNS = [
    "daily_ops_workflow_*.md",
    "strategy_ops_workflow_*.md",
    "system_acceptance_*.md",
    "data_update_workflow_*.md",
    "factor_build_workflow_*.md",
    "daily_report_*.md",
    "strategy_evaluation_*.md",
    "parameter_search_*.md",
    "walk_forward_validation_*.md",
    "trade_plan_backtest_*.md",
    "strategy_admission_*.md",
    "llm_agents_index_*.md",
    "llm_report_summary_*.md",
    "llm_backtest_analysis_*.md",
    "llm_market_regime_*.md",
    "llm_industry_insight_*.md",
    "llm_factor_insight_*.md",
    "llm_strategy_research_*.md",
    "strategy_research_suggestions_*.json",
    "llm_parameter_iteration_*.md",
    "parameter_search_space_candidate_*.json",
    "llm_risk_review_*.md",
    "llm_daily_review_*.md",
    "daily_review_*.md",
    "trade_performance_*.md",
    "position_review_*.md",
    "period_review_*.md",
    "data_quality_*.md",
]


def check_table_health(
    store: StockAgentStore,
    table_loaders: dict[str, Callable[[], pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for table_name, loader in table_loaders.items():
        try:
            df = loader()
            row_count = len(df)
            if row_count > 0:
                status = "ok"
                message = "table has data"
            else:
                status = "empty"
                message = "table is empty"
        except Exception as exc:
            row_count = 0
            status = "error"
            message = str(exc)

        rows.append(
            {
                "table_name": table_name,
                "row_count": row_count,
                "status": status,
                "message": message,
            }
        )

    return pd.DataFrame(rows, columns=["table_name", "row_count", "status", "message"])


def run_system_health_check(
    db_path: str | None = None,
    reports_dir: str = "reports",
    configs_dir: str = "configs",
) -> dict:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    store = StockAgentStore(resolved_db_path)
    table_health = _check_core_tables(store)
    data_quality_status = _check_data_quality_status(store)
    config_files = _check_config_files(configs_dir)
    report_files = _check_report_files(reports_dir)
    daily_ops_report_status = _check_daily_ops_report_status(reports_dir)
    strategy_ops_report_status = _check_strategy_ops_report_status(reports_dir)
    data_update_report_status = _check_data_update_report_status(reports_dir)
    factor_build_report_status = _check_factor_build_report_status(reports_dir)
    system_acceptance_report_status = _check_system_acceptance_report_status(reports_dir)
    data_source_config = _check_data_source_config()
    llm_config = _check_llm_config()
    llm_index_content = _check_llm_agents_index_content(reports_dir)
    enriched_factors = _check_enriched_factors(store)
    factor_diagnostics_state = _check_factor_diagnostics_state(store)
    moneyflow_state = _check_moneyflow_state(store)
    industry_state = _check_industry_state(store)
    latest_market = _latest_market_regime(store)

    table_status = {
        str(row["table_name"]): (int(row["row_count"]), str(row["status"]))
        for _, row in table_health.iterrows()
    }
    factor_build_table_state = _check_factor_build_table_state(table_status)
    config_exists = {
        str(row["file_name"]): bool(row["exists"])
        for _, row in config_files.iterrows()
    }
    total_report_count = int(report_files["file_count"].sum()) if not report_files.empty else 0

    blocking_issues = _blocking_issues(
        table_status,
        config_exists,
        reports_dir,
        total_report_count,
        data_source_config,
        data_quality_status,
        enriched_factors,
        factor_diagnostics_state,
        moneyflow_state,
        industry_state,
    )
    warnings = _warnings(
        table_status,
        data_source_config,
        data_quality_status,
        llm_config,
        report_files,
        daily_ops_report_status,
        strategy_ops_report_status,
        data_update_report_status,
        factor_build_report_status,
        system_acceptance_report_status,
        enriched_factors,
        factor_diagnostics_state,
        moneyflow_state,
        industry_state,
        llm_index_content,
    )
    next_suggestions = _next_suggestions(table_status, total_report_count)
    overall_status = _overall_status(table_status)

    return {
        "db_path": resolved_db_path,
        "reports_dir": reports_dir,
        "configs_dir": configs_dir,
        "table_health": table_health,
        "config_files": config_files,
        "report_files": report_files,
        "daily_ops_report_status": daily_ops_report_status,
        "latest_daily_ops_report_path": _latest_daily_ops_report_path(reports_dir),
        "strategy_ops_report_status": strategy_ops_report_status,
        "latest_strategy_ops_report_path": _latest_strategy_ops_report_path(reports_dir),
        "data_update_report_status": data_update_report_status,
        "latest_data_update_report_path": _latest_data_update_report_path(reports_dir),
        "factor_build_report_status": factor_build_report_status,
        "latest_factor_build_report_path": _latest_factor_build_report_path(reports_dir),
        "system_acceptance_report_status": system_acceptance_report_status,
        "latest_system_acceptance_report_path": _latest_system_acceptance_report_path(reports_dir),
        "factor_build_table_state": factor_build_table_state,
        "data_source_config": data_source_config,
        "llm_config": llm_config,
        "llm_agents_index_content": llm_index_content,
        "data_quality_status": data_quality_status,
        "enriched_factors": enriched_factors,
        "factor_diagnostics_state": factor_diagnostics_state,
        "moneyflow_state": moneyflow_state,
        "industry_state": industry_state,
        "latest_moneyflow_date": _latest_moneyflow_date(store),
        "latest_industry_strength_date": _latest_industry_strength_date(store),
        "strong_industry_count": _industry_strength_level_count(store, "strong"),
        "weak_industry_count": _industry_strength_level_count(store, "weak"),
        "factor_high_missing_count": _factor_diagnostics_status_count(factor_diagnostics_state, "high_missing"),
        "factor_medium_missing_count": _factor_diagnostics_status_count(factor_diagnostics_state, "medium_missing"),
        "latest_market_regime": latest_market.get("market_regime", ""),
        "latest_market_risk_level": latest_market.get("risk_level", ""),
        "latest_limit_up_count": latest_market.get("limit_up_count", 0),
        "latest_limit_down_count": latest_market.get("limit_down_count", 0),
        "overall_status": overall_status,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "next_suggestions": next_suggestions,
    }


def _check_core_tables(store: StockAgentStore) -> pd.DataFrame:
    rows = []
    loaders: dict[str, Callable[[], pd.DataFrame]] = {}
    for table_name in CORE_TABLES:
        loader = getattr(store, f"load_{table_name}", None)
        if loader is None:
            rows.append(
                {
                    "table_name": table_name,
                    "row_count": 0,
                    "status": "missing_loader",
                    "message": "load method is missing",
                }
            )
        else:
            loaders[table_name] = loader

    checked = check_table_health(store, loaders)
    if rows:
        checked = pd.concat(
            [checked, pd.DataFrame(rows, columns=checked.columns)],
            ignore_index=True,
        )
    return checked.loc[:, ["table_name", "row_count", "status", "message"]]


def _check_config_files(configs_dir: str) -> pd.DataFrame:
    base = Path(configs_dir)
    rows = []
    for file_name in CONFIG_FILES:
        path = base / file_name
        exists = path.is_file()
        rows.append(
            {
                "file_name": file_name,
                "path": str(path),
                "exists": exists,
                "status": "ok" if exists else "missing",
                "message": "file exists" if exists else "file is missing",
            }
        )
    return pd.DataFrame(rows, columns=["file_name", "path", "exists", "status", "message"])


def _check_report_files(reports_dir: str) -> pd.DataFrame:
    base = Path(reports_dir)
    rows = []
    for pattern in REPORT_PATTERNS:
        paths = sorted(path for path in base.glob(pattern) if path.is_file()) if base.exists() else []
        rows.append(
            {
                "pattern": pattern,
                "file_count": len(paths),
                "latest_file": str(paths[-1]) if paths else "",
                "status": "ok" if paths else "missing",
                "message": "report files exist" if paths else "no report files",
            }
        )
    return pd.DataFrame(rows, columns=["pattern", "file_count", "latest_file", "status", "message"])


def _latest_data_update_report_path(reports_dir: str) -> str:
    base = Path(reports_dir)
    paths = sorted(path for path in base.glob("data_update_workflow_*.md") if path.is_file()) if base.exists() else []
    return str(paths[-1]) if paths else ""


def _latest_daily_ops_report_path(reports_dir: str) -> str:
    base = Path(reports_dir)
    paths = sorted(path for path in base.glob("daily_ops_workflow_*.md") if path.is_file()) if base.exists() else []
    return str(paths[-1]) if paths else ""


def _latest_strategy_ops_report_path(reports_dir: str) -> str:
    base = Path(reports_dir)
    paths = sorted(path for path in base.glob("strategy_ops_workflow_*.md") if path.is_file()) if base.exists() else []
    return str(paths[-1]) if paths else ""


def _latest_factor_build_report_path(reports_dir: str) -> str:
    base = Path(reports_dir)
    paths = sorted(path for path in base.glob("factor_build_workflow_*.md") if path.is_file()) if base.exists() else []
    return str(paths[-1]) if paths else ""


def _latest_system_acceptance_report_path(reports_dir: str) -> str:
    base = Path(reports_dir)
    paths = sorted(path for path in base.glob("system_acceptance_*.md") if path.is_file()) if base.exists() else []
    return str(paths[-1]) if paths else ""


def _check_daily_ops_report_status(reports_dir: str) -> pd.DataFrame:
    latest_file = _latest_daily_ops_report_path(reports_dir)
    exists = bool(latest_file)
    return pd.DataFrame(
        [
            {
                "exists": exists,
                "latest_file": latest_file,
                "status": "ok" if exists else "missing",
                "message": "daily ops workflow report exists" if exists else "no daily ops workflow report",
            }
        ],
        columns=["exists", "latest_file", "status", "message"],
    )


def _check_strategy_ops_report_status(reports_dir: str) -> pd.DataFrame:
    latest_file = _latest_strategy_ops_report_path(reports_dir)
    exists = bool(latest_file)
    return pd.DataFrame(
        [
            {
                "exists": exists,
                "latest_file": latest_file,
                "status": "ok" if exists else "missing",
                "message": "strategy ops workflow report exists" if exists else "no strategy ops workflow report",
            }
        ],
        columns=["exists", "latest_file", "status", "message"],
    )


def _check_data_update_report_status(reports_dir: str) -> pd.DataFrame:
    latest_file = _latest_data_update_report_path(reports_dir)
    exists = bool(latest_file)
    return pd.DataFrame(
        [
            {
                "exists": exists,
                "latest_file": latest_file,
                "status": "ok" if exists else "missing",
                "message": "data update workflow report exists" if exists else "no data update workflow report",
            }
        ],
        columns=["exists", "latest_file", "status", "message"],
    )


def _check_factor_build_report_status(reports_dir: str) -> pd.DataFrame:
    latest_file = _latest_factor_build_report_path(reports_dir)
    exists = bool(latest_file)
    return pd.DataFrame(
        [
            {
                "exists": exists,
                "latest_file": latest_file,
                "status": "ok" if exists else "missing",
                "message": "factor build workflow report exists" if exists else "no factor build workflow report",
            }
        ],
        columns=["exists", "latest_file", "status", "message"],
    )


def _check_system_acceptance_report_status(reports_dir: str) -> pd.DataFrame:
    latest_file = _latest_system_acceptance_report_path(reports_dir)
    exists = bool(latest_file)
    acceptance_status = _read_acceptance_status(latest_file) if exists else ""
    return pd.DataFrame(
        [
            {
                "exists": exists,
                "latest_file": latest_file,
                "acceptance_status": acceptance_status,
                "status": "ok" if exists else "missing",
                "message": "system acceptance report exists" if exists else "no system acceptance report",
            }
        ],
        columns=["exists", "latest_file", "acceptance_status", "status", "message"],
    )


def _read_acceptance_status(path: str) -> str:
    if not path:
        return ""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""
    for status in ["pass", "warning", "failed"]:
        if f"acceptance_status: {status}" in content:
            return status
    return ""


def _check_factor_build_table_state(table_status: dict[str, tuple[int, str]]) -> pd.DataFrame:
    rows = []
    for table_name in [
        "moneyflow_factors",
        "market_regime",
        "industry_strength",
        "daily_factors",
        "factor_diagnostics",
    ]:
        row_count, status = table_status.get(table_name, (0, "missing_loader"))
        rows.append(
            {
                "table_name": table_name,
                "row_count": row_count,
                "status": status,
                "has_data": status == "ok" and row_count > 0,
            }
        )
    return pd.DataFrame(rows, columns=["table_name", "row_count", "status", "has_data"])


def _check_data_source_config() -> pd.DataFrame:
    default_provider = str(getattr(settings, "DEFAULT_DATA_PROVIDER", "tushare") or "tushare").strip().lower()
    tushare_token_configured = bool(str(getattr(settings, "TUSHARE_TOKEN", "") or "").strip())
    tushare_api_url = str(getattr(settings, "TUSHARE_API_URL", "http://api.tushare.pro") or "").strip()
    if not tushare_api_url:
        tushare_api_url = "http://api.tushare.pro"
    tushare_api_url_is_official = is_official_tushare_api_url(tushare_api_url)
    allow_non_official = bool(getattr(settings, "TUSHARE_ALLOW_NON_OFFICIAL_API_URL", False))
    data_fetch_disable_proxy = bool(getattr(settings, "DATA_FETCH_DISABLE_PROXY", False))

    return pd.DataFrame(
        [
            {
                "default_data_provider": default_provider,
                "official_data_provider": units.OFFICIAL_DATA_PROVIDER,
                "tushare_token_configured": tushare_token_configured,
                "tushare_api_url": tushare_api_url,
                "tushare_api_url_is_official": tushare_api_url_is_official,
                "tushare_allow_non_official_api_url": allow_non_official,
                "data_fetch_disable_proxy": data_fetch_disable_proxy,
                "daily_bars_volume_unit": units.DAILY_BARS_VOLUME_UNIT,
                "daily_bars_amount_unit": units.DAILY_BARS_AMOUNT_UNIT,
                "actual_trades_amount_unit": units.ACTUAL_TRADES_AMOUNT_UNIT,
                "positions_amount_unit": units.POSITIONS_AMOUNT_UNIT,
            }
        ],
        columns=[
            "default_data_provider",
            "official_data_provider",
            "tushare_token_configured",
            "tushare_api_url",
            "tushare_api_url_is_official",
            "tushare_allow_non_official_api_url",
            "data_fetch_disable_proxy",
            "daily_bars_volume_unit",
            "daily_bars_amount_unit",
            "actual_trades_amount_unit",
            "positions_amount_unit",
        ],
    )


def _check_llm_config() -> pd.DataFrame:
    enabled = bool(getattr(settings, "ENABLE_LLM_REPORT_AGENT", False))
    provider = str(getattr(settings, "LLM_PROVIDER", "none") or "").strip().lower()
    default_model = str(getattr(settings, "DEFAULT_LLM_MODEL", "") or "").strip()
    legacy_model = str(getattr(settings, "LLM_MODEL", "") or "").strip()
    model = resolve_llm_model("ReportAgent")
    base_url = str(getattr(settings, "LLM_BASE_URL", "") or "").strip()
    api_key_configured = bool(str(getattr(settings, "LLM_API_KEY", "") or "").strip())
    timeout_seconds = int(getattr(settings, "LLM_TIMEOUT_SECONDS", 60) or 60)
    llm_disable_proxy = bool(getattr(settings, "LLM_DISABLE_PROXY", False))
    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"
    status = "configured" if enabled and provider == "deepseek" and api_key_configured else "disabled"
    message = "ReportAgent 已配置 DeepSeek 后端" if provider == "deepseek" else "LLM ReportAgent is disabled or not configured"

    return pd.DataFrame(
        [
            {
                "ENABLE_LLM_REPORT_AGENT": enabled,
                "LLM_PROVIDER": provider,
                "DEFAULT_LLM_MODEL": default_model,
                "LLM_MODEL": model,
                "legacy_LLM_MODEL": legacy_model,
                "LLM_DISABLE_PROXY": llm_disable_proxy,
                "LLM_API_KEY_configured": api_key_configured,
                "enable_llm_report_agent": enabled,
                "llm_provider": provider,
                "llm_model": model,
                "llm_base_url": base_url,
                "llm_timeout_seconds": timeout_seconds,
                "llm_disable_proxy": llm_disable_proxy,
                "llm_api_key_configured": api_key_configured,
                "status": status,
                "message": message,
            }
        ],
        columns=[
            "ENABLE_LLM_REPORT_AGENT",
            "LLM_PROVIDER",
            "DEFAULT_LLM_MODEL",
            "LLM_MODEL",
            "legacy_LLM_MODEL",
            "LLM_DISABLE_PROXY",
            "LLM_API_KEY_configured",
            "enable_llm_report_agent",
            "llm_provider",
            "llm_model",
            "llm_base_url",
            "llm_timeout_seconds",
            "llm_disable_proxy",
            "llm_api_key_configured",
            "status",
            "message",
        ],
    )


def _check_llm_agents_index_content(reports_dir: str) -> pd.DataFrame:
    base = Path(reports_dir)
    paths = sorted(path for path in base.glob("llm_agents_index_*.md") if path.is_file()) if base.exists() else []
    latest_file = str(paths[-1]) if paths else ""
    contains_market_regime = False
    contains_industry_insight = False
    contains_factor_insight = False
    contains_strategy_research = False
    contains_parameter_iteration = False
    if paths:
        try:
            content = paths[-1].read_text(encoding="utf-8")
            contains_market_regime = "MarketRegimeAgent" in content
            contains_industry_insight = "IndustryInsightAgent" in content
            contains_factor_insight = "FactorInsightAgent" in content
            contains_strategy_research = "StrategyResearchAgent" in content
            contains_parameter_iteration = "ParameterIterationAgent" in content
        except Exception:
            contains_market_regime = False
            contains_industry_insight = False
            contains_factor_insight = False
            contains_strategy_research = False
            contains_parameter_iteration = False
    status = (
        "ok"
        if (
            contains_market_regime
            and contains_industry_insight
            and contains_factor_insight
            and contains_strategy_research
            and contains_parameter_iteration
        )
        else "missing"
    )
    return pd.DataFrame(
        [
            {
                "latest_file": latest_file,
                "contains_market_regime_agent": contains_market_regime,
                "contains_industry_insight_agent": contains_industry_insight,
                "contains_factor_insight_agent": contains_factor_insight,
                "contains_strategy_research_agent": contains_strategy_research,
                "contains_parameter_iteration_agent": contains_parameter_iteration,
                "status": status,
                "message": (
                    "llm_agents_index contains all required LLM agents"
                    if status == "ok"
                    else "llm_agents_index does not contain all required LLM agents"
                ),
            }
        ],
        columns=[
            "latest_file",
            "contains_market_regime_agent",
            "contains_industry_insight_agent",
            "contains_factor_insight_agent",
            "contains_strategy_research_agent",
            "contains_parameter_iteration_agent",
            "status",
            "message",
        ],
    )


def _check_data_quality_status(store: StockAgentStore) -> pd.DataFrame:
    try:
        report = store.load_data_quality_report()
    except Exception as exc:
        return pd.DataFrame(
            [{"status": "missing", "error_count": 0, "warning_count": 0, "message": str(exc)}],
            columns=["status", "error_count", "warning_count", "message"],
        )
    if report.empty:
        return pd.DataFrame(
            [{"status": "empty", "error_count": 0, "warning_count": 0, "message": "data_quality_report is empty"}],
            columns=["status", "error_count", "warning_count", "message"],
        )
    error_count = int((report["status"] == "error").sum()) if "status" in report.columns else 0
    warning_count = int((report["status"] == "warning").sum()) if "status" in report.columns else 0
    status = "error" if error_count else "warning" if warning_count else "ok"
    return pd.DataFrame(
        [
            {
                "status": status,
                "error_count": error_count,
                "warning_count": warning_count,
                "message": "data quality report checked",
            }
        ],
        columns=["status", "error_count", "warning_count", "message"],
    )


def _has_data(table_status: dict[str, tuple[int, str]], table_name: str) -> bool:
    row_count, status = table_status.get(table_name, (0, "missing_loader"))
    return status == "ok" and row_count > 0


def _overall_status(table_status: dict[str, tuple[int, str]]) -> str:
    if not _has_data(table_status, "daily_bars") or not _has_data(table_status, "daily_factors"):
        return "not_ready"

    research_ready = all(
        _has_data(table_status, table_name)
        for table_name in [
            "strategy_admission",
            "parameter_search_results",
            "trade_plan_backtest_performance",
        ]
    )
    if research_ready:
        return "ready_for_research"

    daily_ready = all(
        _has_data(table_status, table_name)
        for table_name in [
            "daily_bars",
            "daily_factors",
            "strategy_signals",
            "candidate_pool",
            "trade_plan",
        ]
    )
    if daily_ready:
        return "ready_for_daily_planning"

    return "partial"


def _blocking_issues(
    table_status: dict[str, tuple[int, str]],
    config_exists: dict[str, bool],
    reports_dir: str,
    total_report_count: int,
    data_source_config: pd.DataFrame | None = None,
    data_quality_status: pd.DataFrame | None = None,
    enriched_factors: pd.DataFrame | None = None,
    factor_diagnostics_state: pd.DataFrame | None = None,
    moneyflow_state: pd.DataFrame | None = None,
    industry_state: pd.DataFrame | None = None,
) -> list[str]:
    issues = []
    if not _has_data(table_status, "daily_bars"):
        issues.append("daily_bars 为空")
    if not _has_data(table_status, "daily_factors"):
        issues.append("daily_factors 为空")
    if not _has_data(table_status, "trade_plan"):
        issues.append("trade_plan 为空")
    if not config_exists.get("active_strategies_candidate.json", False):
        issues.append("active_strategies_candidate.json 不存在，若使用观察候选策略需先生成配置")
    if not Path(reports_dir).exists() or total_report_count == 0:
        issues.append("reports 目录不存在或没有任何报告")
    if _default_provider(data_source_config) == "tushare" and not _tushare_token_configured(data_source_config):
        issues.append("DEFAULT_DATA_PROVIDER=tushare 但 TUSHARE_TOKEN 未配置")
    if (
        _default_provider(data_source_config) == "tushare"
        and _tushare_api_url_non_official(data_source_config)
        and not _tushare_allow_non_official(data_source_config)
    ):
        issues.append(
            "Non-official Tushare API URL is not allowed unless "
            "TUSHARE_ALLOW_NON_OFFICIAL_API_URL=true"
        )
    data_quality_error_count = _data_quality_error_count(data_quality_status)
    if data_quality_error_count > 0:
        issues.append(f"data_quality_report 存在 {data_quality_error_count} 项 error")
    enriched_missing_rate = _enriched_missing_rate(enriched_factors)
    if enriched_missing_rate > 0.80:
        issues.append(f"daily_factors daily_basic 扩展字段缺失率过高：{enriched_missing_rate:.1%}")
    return issues


def _warnings(
    table_status: dict[str, tuple[int, str]],
    data_source_config: pd.DataFrame | None = None,
    data_quality_status: pd.DataFrame | None = None,
    llm_config: pd.DataFrame | None = None,
    report_files: pd.DataFrame | None = None,
    daily_ops_report_status: pd.DataFrame | None = None,
    strategy_ops_report_status: pd.DataFrame | None = None,
    data_update_report_status: pd.DataFrame | None = None,
    factor_build_report_status: pd.DataFrame | None = None,
    system_acceptance_report_status: pd.DataFrame | None = None,
    enriched_factors: pd.DataFrame | None = None,
    factor_diagnostics_state: pd.DataFrame | None = None,
    moneyflow_state: pd.DataFrame | None = None,
    industry_state: pd.DataFrame | None = None,
    llm_index_content: pd.DataFrame | None = None,
) -> list[str]:
    rows = []
    for table_name in [
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
        "actual_trades",
        "execution_review",
        "period_review",
        "walk_forward_validation",
        "parameter_search_results",
        "factor_diagnostics",
    ]:
        if not _has_data(table_status, table_name):
            rows.append(f"{table_name} 为空")
    default_provider = _default_provider(data_source_config)
    if default_provider == "akshare":
        rows.append("AKShare is currently not recommended as the primary provider due to unit consistency concerns.")
    if default_provider != "tushare" and not _tushare_token_configured(data_source_config):
        rows.append("TUSHARE_TOKEN 未配置；仅在使用 provider=tushare 时需要")
    if _tushare_api_url_non_official(data_source_config) and _tushare_allow_non_official(data_source_config):
        rows.append("Using non-official Tushare API URL; token and data integrity risk should be reviewed.")
    if _data_fetch_disable_proxy(data_source_config):
        rows.append("数据拉取时将临时绕过 HTTP/HTTPS/ALL proxy 环境变量。")
    if not _enriched_has_required_columns(enriched_factors):
        rows.append("daily_factors 缺少部分 daily_basic 扩展字段")
    enriched_missing_rate = _enriched_missing_rate(enriched_factors)
    if 0.30 < enriched_missing_rate <= 0.80:
        rows.append(f"daily_factors daily_basic 扩展字段缺失率较高：{enriched_missing_rate:.1%}")
    factor_high_missing_count = _factor_diagnostics_status_count(factor_diagnostics_state, "high_missing")
    factor_medium_missing_count = _factor_diagnostics_status_count(factor_diagnostics_state, "medium_missing")
    if _factor_diagnostics_empty(factor_diagnostics_state):
        rows.append("factor_diagnostics 为空；仅影响因子诊断和 FactorInsightAgent 报告")
    if factor_high_missing_count > 0:
        rows.append(f"factor_diagnostics high_missing 因子数量：{factor_high_missing_count}")
    if factor_medium_missing_count > 0:
        rows.append(f"factor_diagnostics medium_missing 因子数量：{factor_medium_missing_count}")
    moneyflow_missing_rate = _moneyflow_missing_rate(moneyflow_state)
    if moneyflow_missing_rate > 0.50:
        rows.append(f"daily_factors moneyflow_factors 缺失率较高：{moneyflow_missing_rate:.1%}")
    if not _moneyflow_merged(moneyflow_state):
        rows.append("daily_factors 尚未合并 moneyflow_factors 字段")
    if not _industry_merged(industry_state):
        rows.append("daily_factors 尚未合并 industry_strength 字段")
    industry_missing_rate = _industry_missing_rate(industry_state)
    if industry_missing_rate > 0.50:
        rows.append(f"daily_factors industry_strength 缺失率较高：{industry_missing_rate:.1%}")
    data_quality_warning_count = _data_quality_warning_count(data_quality_status)
    if data_quality_warning_count > 0:
        rows.append(f"data_quality_report 存在 {data_quality_warning_count} 项 warning")
    if _llm_enabled(llm_config) and _llm_provider(llm_config) == "deepseek" and not _llm_api_key_configured(llm_config):
        rows.append("ENABLE_LLM_REPORT_AGENT=true 且 LLM_PROVIDER=deepseek，但 LLM_API_KEY 未配置")
    if _llm_disable_proxy(llm_config):
        rows.append("LLM API 调用时将临时绕过 HTTP/HTTPS/ALL proxy 环境变量。")
    if _llm_provider(llm_config) == "deepseek":
        rows.append("ReportAgent 已配置 DeepSeek 后端")
    if not _workflow_report_exists(daily_ops_report_status):
        rows.append("daily_ops_workflow_*.md 暂无报告；仅影响最近一次日度总流程展示")
    if not _workflow_report_exists(strategy_ops_report_status):
        rows.append("strategy_ops_workflow_*.md 暂无报告；仅影响最近一次策略研究总流程展示")
    if not _workflow_report_exists(data_update_report_status):
        rows.append("data_update_workflow_*.md 暂无报告；仅影响最近一次数据更新工作流展示")
    if not _workflow_report_exists(factor_build_report_status):
        rows.append("factor_build_workflow_*.md 暂无报告；仅影响最近一次因子构建工作流展示")
    if not _workflow_report_exists(system_acceptance_report_status):
        rows.append("system_acceptance_*.md 暂无报告；仅影响最近一次系统验收展示")
    llm_report_messages = {
        "llm_agents_index_*.md": "llm_agents_index_*.md 暂无报告；仅影响 LLM Agent 报告索引展示",
        "llm_report_summary_*.md": "llm_report_summary_*.md 暂无报告；仅影响 ReportAgent 看板展示",
        "llm_backtest_analysis_*.md": "llm_backtest_analysis_*.md 暂无报告；仅影响 LLM 回测分析看板展示",
        "llm_market_regime_*.md": "llm_market_regime_*.md 暂无报告；仅影响 LLM 市场环境解释看板展示",
        "llm_industry_insight_*.md": "llm_industry_insight_*.md 暂无报告；仅影响 LLM 行业洞察看板展示",
        "llm_factor_insight_*.md": "llm_factor_insight_*.md 暂无报告；仅影响 LLM 因子诊断看板展示",
        "llm_strategy_research_*.md": "llm_strategy_research_*.md 暂无报告；仅影响 LLM 策略研究建议看板展示",
        "strategy_research_suggestions_*.json": "strategy_research_suggestions_*.json 暂无候选研究建议；仅影响 StrategyResearchAgent JSON 展示",
        "llm_parameter_iteration_*.md": "llm_parameter_iteration_*.md 暂无报告；仅影响 LLM 参数迭代建议看板展示",
        "parameter_search_space_candidate_*.json": "parameter_search_space_candidate_*.json 暂无候选参数研究建议；仅影响 ParameterIterationAgent JSON 展示",
        "llm_risk_review_*.md": "llm_risk_review_*.md 暂无报告；仅影响 LLM 风险审查看板展示",
        "llm_daily_review_*.md": "llm_daily_review_*.md 暂无报告；仅影响 LLM 每日执行复盘看板展示",
    }
    for pattern, message in llm_report_messages.items():
        if not _report_pattern_exists(report_files, pattern):
            rows.append(message)
    if not _llm_index_contains_market_regime(llm_index_content):
        rows.append("llm_agents_index_*.md 未包含 MarketRegimeAgent；仅影响 LLM Agent 报告索引展示")
    if not _llm_index_contains_industry_insight(llm_index_content):
        rows.append("llm_agents_index_*.md 未包含 IndustryInsightAgent；仅影响 LLM Agent 报告索引展示")
    if not _llm_index_contains_factor_insight(llm_index_content):
        rows.append("llm_agents_index_*.md 未包含 FactorInsightAgent；仅影响 LLM Agent 报告索引展示")
    if not _llm_index_contains_strategy_research(llm_index_content):
        rows.append("llm_agents_index_*.md 未包含 StrategyResearchAgent；仅影响 LLM Agent 报告索引展示")
    if not _llm_index_contains_parameter_iteration(llm_index_content):
        rows.append("llm_agents_index_*.md 未包含 ParameterIterationAgent；仅影响 LLM Agent 报告索引展示")
    return rows


def _workflow_report_exists(report_status: pd.DataFrame | None) -> bool:
    if report_status is None or report_status.empty:
        return False
    return bool(report_status.iloc[0].get("exists", False))


def _latest_moneyflow_date(store: StockAgentStore) -> str:
    try:
        df = store.load_moneyflow()
    except Exception:
        return ""
    if df.empty or "trade_date" not in df.columns:
        return ""
    dates = df["trade_date"].dropna().astype(str)
    return "" if dates.empty else str(dates.max())


def _latest_industry_strength_date(store: StockAgentStore) -> str:
    try:
        df = store.load_industry_strength()
    except Exception:
        return ""
    if df.empty or "trade_date" not in df.columns:
        return ""
    dates = df["trade_date"].dropna().astype(str)
    return "" if dates.empty else str(dates.max())


def _industry_strength_level_count(store: StockAgentStore, level: str) -> int:
    try:
        df = store.load_industry_strength(trade_date=_latest_industry_strength_date(store))
    except Exception:
        return 0
    if df.empty or "industry_strength_level" not in df.columns:
        return 0
    return int((df["industry_strength_level"].fillna("").astype(str) == level).sum())


def _check_moneyflow_state(store: StockAgentStore) -> pd.DataFrame:
    required_columns = ["moneyflow_score", "main_net_amount", "main_net_amount_ratio"]
    try:
        moneyflow = store.load_moneyflow()
        moneyflow_factors = store.load_moneyflow_factors()
        daily_factors = store.load_daily_factors()
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "status": "warning",
                    "moneyflow_rows": 0,
                    "moneyflow_factors_rows": 0,
                    "latest_moneyflow_date": "",
                    "merged_to_daily_factors": False,
                    "moneyflow_factors_missing_rate": None,
                    "message": str(exc),
                }
            ]
        )
    merged = all(column in daily_factors.columns for column in required_columns)
    present = [column for column in required_columns if column in daily_factors.columns]
    missing_rate = 0.0
    if not daily_factors.empty and present:
        missing_rate = float(daily_factors[present].isna().any(axis=1).sum() / len(daily_factors))
    status = "warning" if moneyflow.empty or moneyflow_factors.empty or not merged or missing_rate > 0.50 else "ok"
    return pd.DataFrame(
        [
            {
                "status": status,
                "moneyflow_rows": len(moneyflow),
                "moneyflow_factors_rows": len(moneyflow_factors),
                "latest_moneyflow_date": _latest_moneyflow_date(store),
                "merged_to_daily_factors": merged,
                "moneyflow_factors_missing_rate": missing_rate,
                "message": (
                    f"moneyflow rows={len(moneyflow)}, moneyflow_factors rows={len(moneyflow_factors)}, "
                    f"merged_to_daily_factors={merged}, missing_rate={missing_rate:.1%}"
                ),
            }
        ]
    )


def _check_industry_state(store: StockAgentStore) -> pd.DataFrame:
    required_columns = ["industry_strength_score", "industry_strength_level"]
    try:
        classification = store.load_sw_industry_classification()
        sw_daily = store.load_sw_daily()
        stock_map = store.load_stock_industry_map()
        industry_strength = store.load_industry_strength()
        daily_factors = store.load_daily_factors()
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "status": "warning",
                    "sw_industry_classification_rows": 0,
                    "sw_daily_rows": 0,
                    "stock_industry_map_rows": 0,
                    "industry_strength_rows": 0,
                    "latest_industry_strength_date": "",
                    "strong_industry_count": 0,
                    "weak_industry_count": 0,
                    "merged_to_daily_factors": False,
                    "industry_strength_missing_rate": None,
                    "message": str(exc),
                }
            ]
        )
    merged = all(column in daily_factors.columns for column in required_columns)
    present = [column for column in required_columns if column in daily_factors.columns]
    missing_rate = 0.0
    if not daily_factors.empty and present:
        missing_rate = float(daily_factors[present].isna().any(axis=1).sum() / len(daily_factors))
    latest_date = _latest_industry_strength_date(store)
    latest = industry_strength[industry_strength["trade_date"].astype(str) == latest_date] if latest_date and not industry_strength.empty else pd.DataFrame()
    strong_count = int((latest.get("industry_strength_level", pd.Series(dtype=str)).fillna("").astype(str) == "strong").sum())
    weak_count = int((latest.get("industry_strength_level", pd.Series(dtype=str)).fillna("").astype(str) == "weak").sum())
    status = (
        "warning"
        if classification.empty or sw_daily.empty or stock_map.empty or industry_strength.empty or not merged or missing_rate > 0.50
        else "ok"
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "sw_industry_classification_rows": len(classification),
                "sw_daily_rows": len(sw_daily),
                "stock_industry_map_rows": len(stock_map),
                "industry_strength_rows": len(industry_strength),
                "latest_industry_strength_date": latest_date,
                "strong_industry_count": strong_count,
                "weak_industry_count": weak_count,
                "merged_to_daily_factors": merged,
                "industry_strength_missing_rate": missing_rate,
                "message": (
                    f"industry_strength rows={len(industry_strength)}, merged_to_daily_factors={merged}, "
                    f"missing_rate={missing_rate:.1%}, strong={strong_count}, weak={weak_count}"
                ),
            }
        ]
    )


def _latest_market_regime(store: StockAgentStore) -> dict[str, object]:
    try:
        df = store.load_market_regime()
    except Exception:
        return {}
    if df.empty:
        return {}
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date")
    return df.iloc[-1].to_dict()


def _check_enriched_factors(store: StockAgentStore) -> pd.DataFrame:
    required_columns = [
        "turnover_rate",
        "volume_ratio_daily_basic",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
        "is_suspended",
        "is_limit_up_close",
        "is_limit_down_close",
    ]
    missing_rate_columns = ["volume_ratio_daily_basic", "total_mv", "circ_mv"]
    try:
        factors = store.load_daily_factors()
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "status": "error",
                    "row_count": 0,
                    "has_enriched_fields": False,
                    "missing_columns": ",".join(required_columns),
                    "daily_basic_missing_rate": None,
                    "message": str(exc),
                }
            ]
        )
    existing_columns = set(factors.columns)
    missing_columns = [column for column in required_columns if column not in existing_columns]
    present_missing_rate_columns = [column for column in missing_rate_columns if column in existing_columns]
    missing_rate = 0.0
    if not factors.empty and present_missing_rate_columns:
        missing_count = int(factors[present_missing_rate_columns].isna().any(axis=1).sum())
        missing_rate = missing_count / len(factors) if len(factors) else 0.0
    has_enriched_fields = not missing_columns
    status = "warning" if missing_columns or missing_rate > 0.30 else "ok"
    if missing_rate > 0.80:
        status = "blocking"
    message = (
        f"daily_factors enriched fields present={has_enriched_fields}; "
        f"daily_basic missing rate={missing_rate:.1%}"
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "row_count": len(factors),
                "has_enriched_fields": has_enriched_fields,
                "missing_columns": ",".join(missing_columns),
                "daily_basic_missing_rate": missing_rate,
                "message": message,
            }
        ],
        columns=[
            "status",
            "row_count",
            "has_enriched_fields",
            "missing_columns",
            "daily_basic_missing_rate",
            "message",
        ],
    )


def _check_factor_diagnostics_state(store: StockAgentStore) -> pd.DataFrame:
    try:
        diagnostics = store.load_factor_diagnostics()
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "status": "warning",
                    "row_count": 0,
                    "high_missing_count": 0,
                    "medium_missing_count": 0,
                    "message": str(exc),
                }
            ]
        )
    if diagnostics.empty or "diagnostic_status" not in diagnostics.columns:
        return pd.DataFrame(
            [
                {
                    "status": "warning",
                    "row_count": len(diagnostics),
                    "high_missing_count": 0,
                    "medium_missing_count": 0,
                    "message": "factor_diagnostics is empty",
                }
            ]
        )
    statuses = diagnostics["diagnostic_status"].fillna("").astype(str)
    high_missing_count = int((statuses == "high_missing").sum())
    medium_missing_count = int((statuses == "medium_missing").sum())
    status = "warning" if high_missing_count > 0 else "ok"
    return pd.DataFrame(
        [
            {
                "status": status,
                "row_count": len(diagnostics),
                "high_missing_count": high_missing_count,
                "medium_missing_count": medium_missing_count,
                "message": (
                    f"factor_diagnostics rows={len(diagnostics)}, "
                    f"high_missing={high_missing_count}, medium_missing={medium_missing_count}"
                ),
            }
        ]
    )


def _enriched_missing_rate(enriched_factors: pd.DataFrame | None) -> float:
    if (
        enriched_factors is None
        or enriched_factors.empty
        or "daily_basic_missing_rate" not in enriched_factors.columns
        or pd.isna(enriched_factors["daily_basic_missing_rate"].iloc[0])
    ):
        return 0.0
    return float(enriched_factors["daily_basic_missing_rate"].iloc[0])


def _enriched_has_required_columns(enriched_factors: pd.DataFrame | None) -> bool:
    if enriched_factors is None or enriched_factors.empty or "has_enriched_fields" not in enriched_factors.columns:
        return False
    return bool(enriched_factors["has_enriched_fields"].iloc[0])


def _moneyflow_missing_rate(moneyflow_state: pd.DataFrame | None) -> float:
    if (
        moneyflow_state is None
        or moneyflow_state.empty
        or "moneyflow_factors_missing_rate" not in moneyflow_state.columns
        or pd.isna(moneyflow_state["moneyflow_factors_missing_rate"].iloc[0])
    ):
        return 0.0
    return float(moneyflow_state["moneyflow_factors_missing_rate"].iloc[0])


def _moneyflow_merged(moneyflow_state: pd.DataFrame | None) -> bool:
    if moneyflow_state is None or moneyflow_state.empty or "merged_to_daily_factors" not in moneyflow_state.columns:
        return True
    return bool(moneyflow_state["merged_to_daily_factors"].iloc[0])


def _industry_missing_rate(industry_state: pd.DataFrame | None) -> float:
    if (
        industry_state is None
        or industry_state.empty
        or "industry_strength_missing_rate" not in industry_state.columns
        or pd.isna(industry_state["industry_strength_missing_rate"].iloc[0])
    ):
        return 0.0
    return float(industry_state["industry_strength_missing_rate"].iloc[0])


def _industry_merged(industry_state: pd.DataFrame | None) -> bool:
    if industry_state is None or industry_state.empty or "merged_to_daily_factors" not in industry_state.columns:
        return True
    return bool(industry_state["merged_to_daily_factors"].iloc[0])


def _default_provider(data_source_config: pd.DataFrame | None) -> str:
    if data_source_config is None or data_source_config.empty:
        return "tushare"
    if "default_data_provider" not in data_source_config.columns:
        return "tushare"
    return str(data_source_config["default_data_provider"].iloc[0]).strip().lower()


def _tushare_token_configured(data_source_config: pd.DataFrame | None) -> bool:
    if data_source_config is None or data_source_config.empty:
        return False
    if "tushare_token_configured" not in data_source_config.columns:
        return False
    return bool(data_source_config["tushare_token_configured"].iloc[0])


def _tushare_api_url_non_official(data_source_config: pd.DataFrame | None) -> bool:
    if data_source_config is None or data_source_config.empty:
        return False
    if "tushare_api_url_is_official" not in data_source_config.columns:
        return False
    return not bool(data_source_config["tushare_api_url_is_official"].iloc[0])


def _tushare_allow_non_official(data_source_config: pd.DataFrame | None) -> bool:
    if data_source_config is None or data_source_config.empty:
        return False
    if "tushare_allow_non_official_api_url" not in data_source_config.columns:
        return False
    return bool(data_source_config["tushare_allow_non_official_api_url"].iloc[0])


def _data_fetch_disable_proxy(data_source_config: pd.DataFrame | None) -> bool:
    if data_source_config is None or data_source_config.empty:
        return False
    if "data_fetch_disable_proxy" not in data_source_config.columns:
        return False
    return bool(data_source_config["data_fetch_disable_proxy"].iloc[0])


def _llm_enabled(llm_config: pd.DataFrame | None) -> bool:
    if llm_config is None or llm_config.empty or "enable_llm_report_agent" not in llm_config.columns:
        return False
    return bool(llm_config["enable_llm_report_agent"].iloc[0])


def _llm_provider(llm_config: pd.DataFrame | None) -> str:
    if llm_config is None or llm_config.empty or "llm_provider" not in llm_config.columns:
        return "none"
    return str(llm_config["llm_provider"].iloc[0]).strip().lower()


def _llm_api_key_configured(llm_config: pd.DataFrame | None) -> bool:
    if llm_config is None or llm_config.empty or "llm_api_key_configured" not in llm_config.columns:
        return False
    return bool(llm_config["llm_api_key_configured"].iloc[0])


def _llm_disable_proxy(llm_config: pd.DataFrame | None) -> bool:
    if llm_config is None or llm_config.empty or "llm_disable_proxy" not in llm_config.columns:
        return False
    return bool(llm_config["llm_disable_proxy"].iloc[0])


def _report_pattern_exists(report_files: pd.DataFrame | None, pattern: str) -> bool:
    if report_files is None or report_files.empty:
        return False
    if "pattern" not in report_files.columns or "file_count" not in report_files.columns:
        return False
    matched = report_files[report_files["pattern"].astype(str) == pattern]
    if matched.empty:
        return False
    return int(matched["file_count"].iloc[0]) > 0


def _llm_index_contains_market_regime(llm_index_content: pd.DataFrame | None) -> bool:
    if (
        llm_index_content is None
        or llm_index_content.empty
        or "contains_market_regime_agent" not in llm_index_content.columns
    ):
        return False
    return bool(llm_index_content["contains_market_regime_agent"].iloc[0])


def _llm_index_contains_industry_insight(llm_index_content: pd.DataFrame | None) -> bool:
    if (
        llm_index_content is None
        or llm_index_content.empty
        or "contains_industry_insight_agent" not in llm_index_content.columns
    ):
        return False
    return bool(llm_index_content["contains_industry_insight_agent"].iloc[0])


def _llm_index_contains_factor_insight(llm_index_content: pd.DataFrame | None) -> bool:
    if (
        llm_index_content is None
        or llm_index_content.empty
        or "contains_factor_insight_agent" not in llm_index_content.columns
    ):
        return False
    return bool(llm_index_content["contains_factor_insight_agent"].iloc[0])


def _llm_index_contains_strategy_research(llm_index_content: pd.DataFrame | None) -> bool:
    if (
        llm_index_content is None
        or llm_index_content.empty
        or "contains_strategy_research_agent" not in llm_index_content.columns
    ):
        return False
    return bool(llm_index_content["contains_strategy_research_agent"].iloc[0])


def _llm_index_contains_parameter_iteration(llm_index_content: pd.DataFrame | None) -> bool:
    if (
        llm_index_content is None
        or llm_index_content.empty
        or "contains_parameter_iteration_agent" not in llm_index_content.columns
    ):
        return False
    return bool(llm_index_content["contains_parameter_iteration_agent"].iloc[0])


def _factor_diagnostics_empty(factor_diagnostics_state: pd.DataFrame | None) -> bool:
    if factor_diagnostics_state is None or factor_diagnostics_state.empty or "row_count" not in factor_diagnostics_state.columns:
        return True
    return int(factor_diagnostics_state["row_count"].iloc[0]) == 0


def _factor_diagnostics_status_count(factor_diagnostics_state: pd.DataFrame | None, status: str) -> int:
    if factor_diagnostics_state is None or factor_diagnostics_state.empty:
        return 0
    column = "high_missing_count" if status == "high_missing" else "medium_missing_count"
    if column not in factor_diagnostics_state.columns:
        return 0
    return int(factor_diagnostics_state[column].iloc[0])


def _data_quality_error_count(data_quality_status: pd.DataFrame | None) -> int:
    if data_quality_status is None or data_quality_status.empty or "error_count" not in data_quality_status.columns:
        return 0
    return int(data_quality_status["error_count"].iloc[0])


def _data_quality_warning_count(data_quality_status: pd.DataFrame | None) -> int:
    if data_quality_status is None or data_quality_status.empty or "warning_count" not in data_quality_status.columns:
        return 0
    return int(data_quality_status["warning_count"].iloc[0])


def _next_suggestions(table_status: dict[str, tuple[int, str]], total_report_count: int) -> list[str]:
    suggestions = []
    if not _has_data(table_status, "daily_bars"):
        suggestions.append("daily_bars 为空：建议先运行 update_daily_bars")
    if not _has_data(table_status, "daily_factors"):
        suggestions.append("daily_factors 为空：建议运行 build_daily_factors")
    if not _has_data(table_status, "strategy_signals"):
        suggestions.append("strategy_signals 为空：建议运行 run_daily_planning_workflow 或 build_strategy_signals")
    if not _has_data(table_status, "strategy_admission"):
        suggestions.append("strategy_admission 为空：建议运行 run_strategy_research_workflow")
    if not _has_data(table_status, "actual_trades"):
        suggestions.append("actual_trades 为空：建议导入 actual_trades CSV")
    if total_report_count == 0:
        suggestions.append("reports 为空：建议导出对应 Markdown 报告")
    return suggestions
