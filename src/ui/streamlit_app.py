from pathlib import Path
import json
import sys
from typing import Callable

import duckdb
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.database.duckdb_store import StockAgentStore
from src.strategy.active_strategy_config import load_active_strategy_candidates
from src.ui.labels import (
    get_field_label,
    get_table_label,
    translate_dataframe_columns,
    translate_risk_flags,
    translate_value,
)


TABLE_NAMES = [
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
    "strategy_version_evaluation",
    "parameter_search_results",
    "parameter_search_performance",
    "parameter_search_backtest_results",
    "walk_forward_validation",
    "strategy_admission",
    "trade_plan",
    "historical_trade_plans",
    "trade_plan_backtest_results",
    "trade_plan_backtest_performance",
    "actual_trades",
    "execution_review",
    "actual_trade_performance",
    "daily_review",
    "period_review",
    "positions",
    "position_review",
]
OVERVIEW_TABLES = [
    "stock_basic",
    "daily_bars",
    "daily_factors",
    "candidate_pool",
    "trade_plan",
    "strategy_admission",
    "market_regime",
    "moneyflow_factors",
    "industry_strength",
    "factor_diagnostics",
]
DATA_TABLE_GROUPS = {
    "基础数据": [
        "stock_basic",
        "trade_calendar",
        "daily_bars",
        "daily_basic",
        "stock_limits",
        "suspend_daily",
    ],
    "市场环境": [
        "index_daily",
        "limit_list_daily",
        "market_regime",
    ],
    "资金流": [
        "moneyflow",
        "moneyflow_factors",
    ],
    "行业": [
        "sw_industry_classification",
        "stock_industry_map",
        "sw_daily",
        "industry_strength",
    ],
    "因子": [
        "daily_factors",
        "factor_diagnostics",
    ],
}
STRATEGY_RESEARCH_TABLES = [
    "strategy_version_evaluation",
    "parameter_search_results",
    "walk_forward_validation",
    "trade_plan_backtest_performance",
    "strategy_admission",
]
RECENT_OVERVIEW_REPORT_PATTERNS = {
    "系统总体验收报告": "system_acceptance_*.md",
    "日度总流程报告": "daily_ops_workflow_*.md",
    "策略研究总流程报告": "strategy_ops_workflow_*.md",
    "LLM Agent 报告索引": "llm_agents_index_*.md",
}
LLM_REPORT_PATTERNS = {
    "LLM Agent 报告索引": "llm_agents_index_*.md",
    "LLM 综合总结": "llm_report_summary_*.md",
    "回测分析 Agent 报告": "llm_backtest_analysis_*.md",
    "风险审查 Agent 报告": "llm_risk_review_*.md",
    "交易复盘 Agent 报告": "llm_daily_review_*.md",
    "市场环境 Agent 报告": "llm_market_regime_*.md",
    "行业洞察 Agent 报告": "llm_industry_insight_*.md",
    "因子洞察 Agent 报告": "llm_factor_insight_*.md",
    "策略研究 Agent 报告": "llm_strategy_research_*.md",
    "参数迭代 Agent 报告": "llm_parameter_iteration_*.md",
}
SYSTEM_ACCEPTANCE_REPORT_PATTERNS = {
    "系统总体验收报告": "system_acceptance_*.md",
    "系统健康检查报告": "system_health_*.md",
}
REPORT_PATTERNS = {
    **RECENT_OVERVIEW_REPORT_PATTERNS,
    **LLM_REPORT_PATTERNS,
    **SYSTEM_ACCEPTANCE_REPORT_PATTERNS,
    "数据更新工作流报告": "data_update_workflow_*.md",
    "因子构建工作流报告": "factor_build_workflow_*.md",
    "策略研究建议 JSON": "strategy_research_suggestions_*.json",
    "参数搜索空间候选 JSON": "parameter_search_space_candidate_*.json",
}
DATE_TABLES = [
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
    "sw_daily",
    "industry_strength",
    "provider_compare_result",
    "daily_factors",
    "strategy_signals",
    "candidate_pool",
    "trade_plan",
    "historical_trade_plans",
    "actual_trades",
    "execution_review",
    "actual_trade_performance",
    "daily_review",
    "positions",
    "position_review",
]
CANDIDATE_COLUMNS = [
    "rank",
    "code",
    "name",
    "strategy_names",
    "strategy_versions",
    "signal_count",
    "active_signal_count",
    "total_weighted_signal_strength",
    "avg_strategy_weight",
    "recommendations",
    "risk_flags",
    "turnover_rate",
    "circ_mv",
    "is_suspended",
    "is_limit_up_close",
    "is_limit_down_close",
    "moneyflow_score",
    "main_net_amount",
    "main_net_amount_ratio",
    "moneyflow_risk_flags",
    "industry_name",
    "industry_strength_score",
    "industry_strength_level",
    "industry_risk_flags",
    "score",
    "close",
    "pct_chg_5d",
    "reason",
]
TRADE_PLAN_COLUMNS = [
    "trade_date",
    "rank",
    "code",
    "name",
    "action",
    "entry_low",
    "entry_high",
    "strategy_names",
    "strategy_versions",
    "recommendations",
    "avg_strategy_weight",
    "risk_flags",
    "turnover_rate",
    "circ_mv",
    "is_suspended",
    "is_limit_up_close",
    "is_limit_down_close",
    "moneyflow_score",
    "main_net_amount",
    "main_net_amount_ratio",
    "moneyflow_risk_flags",
    "industry_name",
    "industry_strength_score",
    "industry_strength_level",
    "industry_risk_flags",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "position_low",
    "position_high",
    "plan_reason",
]
TRADE_PLAN_REQUESTED_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "action",
    "buy_price_low",
    "buy_price_high",
    "entry_low",
    "entry_high",
    "stop_loss",
    "take_profit",
    "take_profit_1",
    "take_profit_2",
    "position_ratio",
    "position_low",
    "position_high",
    "plan_reason",
    "risk_flags",
]
CANDIDATE_REQUESTED_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "strategy_name",
    "strategy_names",
    "strategy_versions",
    "score",
    "risk_flags",
    "moneyflow_score",
    "industry_strength_score",
    "industry_strength_level",
    "market_regime",
    "action",
]
TRADE_PLAN_DETAIL_COLUMNS = [
    "action",
    "strategy_names",
    "strategy_versions",
    "recommendations",
    "avg_strategy_weight",
    "risk_flags",
    "entry_low",
    "entry_high",
    "position_low",
    "position_high",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "invalid_condition",
    "t_plus_1_risk",
    "plan_reason",
]
TRADE_PLAN_DETAIL_LABELS = {
    "strategy_names": "策略来源",
    "strategy_versions": "策略版本",
    "recommendations": "策略评价建议",
    "avg_strategy_weight": "平均策略权重",
    "risk_flags": "风险标记",
    "industry_name": "所属行业",
    "industry_strength_score": "行业强度分",
    "industry_strength_level": "行业强度",
    "industry_risk_flags": "行业风险标记",
    "plan_reason": "计划理由",
}
POSITION_COLUMNS = [
    "code",
    "name",
    "holding_volume",
    "available_volume",
    "frozen_volume",
    "cost_price",
    "latest_price",
    "market_value",
    "floating_pnl",
    "floating_pnl_pct",
    "t_plus_1_status",
    "position_status",
]
POSITION_REVIEW_COLUMNS = [
    "code",
    "name",
    "position_risk_level",
    "position_flags",
    "position_comment",
    "next_action_hint",
    "planned_stop_loss",
    "planned_take_profit_1",
    "planned_take_profit_2",
]
ACTIVE_STRATEGY_CANDIDATE_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "admission_score",
    "admission_status",
    "admission_recommendation",
    "admission_reason",
]
SYSTEM_HEALTH_KEY_TABLES = [
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
    "strategy_admission",
    "actual_trades",
    "daily_review",
    "period_review",
]


def _show_error(message: str) -> None:
    try:
        st.error(message)
    except Exception:
        pass


def _db_path(store: StockAgentStore) -> Path:
    return Path(store.db_path).expanduser()


def _connect_read_only(store: StockAgentStore) -> duckdb.DuckDBPyConnection | None:
    path = _db_path(store)
    if not path.exists():
        return None
    return duckdb.connect(str(path), read_only=True)


def safe_load_table(store: StockAgentStore, table_name: str) -> pd.DataFrame:
    """Load a local DuckDB table without creating or mutating database state."""
    if table_name not in TABLE_NAMES:
        _show_error(f"不支持的数据表: {table_name}")
        return pd.DataFrame()

    try:
        con = _connect_read_only(store)
        if con is None:
            return pd.DataFrame()
        with con:
            return con.execute(f"SELECT * FROM {table_name}").fetchdf()
    except Exception as exc:
        _show_error(f"读取数据表 {table_name} 失败: {exc}")
        return pd.DataFrame()


def get_available_trade_dates(store: StockAgentStore) -> list[str]:
    try:
        trade_dates: set[str] = set()
        for table_name in DATE_TABLES:
            df = safe_load_table(store, table_name)
            if not df.empty and "trade_date" in df.columns:
                trade_dates.update(str(value) for value in df["trade_date"].dropna())
            if not df.empty and "plan_date" in df.columns:
                trade_dates.update(str(value) for value in df["plan_date"].dropna())
            if not df.empty and "as_of_date" in df.columns:
                trade_dates.update(str(value) for value in df["as_of_date"].dropna())
        return sorted(trade_dates, reverse=True)
    except Exception as exc:
        _show_error(f"读取交易日期失败: {exc}")
        return []


def get_latest_trade_date(store: StockAgentStore) -> str | None:
    try:
        dates = get_available_trade_dates(store)
        return dates[0] if dates else None
    except Exception as exc:
        _show_error(f"读取最新交易日期失败: {exc}")
        return None


def list_report_files(output_dir: str = "reports", pattern: str = "daily_report_*.md") -> list[str]:
    """Return matching report paths ordered by file update time, newest first."""
    try:
        reports_dir = Path(output_dir)
        if not reports_dir.exists():
            return []
        paths = [path for path in reports_dir.glob(pattern) if path.is_file()]
        return [
            str(path)
            for path in sorted(
                paths,
                key=lambda path: (path.stat().st_mtime, path.name),
                reverse=True,
            )
        ]
    except Exception as exc:
        _show_error(f"读取报告失败: {exc}")
        return []


def read_markdown_file(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        _show_error(f"读取 Markdown 文件失败: {exc}")
        return ""


def mask_secret_config(value: object) -> str:
    text = "" if value is None else str(value)
    return "true" if bool(text.strip()) else "false"


def build_config_overview() -> pd.DataFrame:
    rows = [
        {"key": "DEFAULT_DATA_PROVIDER", "value": str(getattr(settings, "DEFAULT_DATA_PROVIDER", ""))},
        {"key": "TUSHARE_API_URL", "value": str(getattr(settings, "TUSHARE_API_URL", ""))},
        {"key": "DATA_FETCH_DISABLE_PROXY", "value": str(getattr(settings, "DATA_FETCH_DISABLE_PROXY", False))},
        {"key": "LLM_PROVIDER", "value": str(getattr(settings, "LLM_PROVIDER", ""))},
        {"key": "LLM_MODEL", "value": str(getattr(settings, "LLM_MODEL", ""))},
        {"key": "LLM_DISABLE_PROXY", "value": str(getattr(settings, "LLM_DISABLE_PROXY", False))},
        {"key": "TUSHARE_TOKEN configured", "value": mask_secret_config(getattr(settings, "TUSHARE_TOKEN", ""))},
        {"key": "LLM_API_KEY configured", "value": mask_secret_config(getattr(settings, "LLM_API_KEY", ""))},
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def get_table_summary(df: pd.DataFrame) -> dict:
    row_count = int(len(df)) if df is not None else 0
    date_column = None
    for candidate in ["trade_date", "plan_date", "as_of_date", "start_date", "end_date"]:
        if df is not None and candidate in df.columns:
            date_column = candidate
            break
    date_range = ""
    if row_count > 0 and date_column is not None:
        values = df[date_column].dropna().astype(str)
        if not values.empty:
            date_range = f"{values.min()} - {values.max()}"
    return {
        "row_count": row_count,
        "date_column": date_column or "",
        "date_range": date_range,
    }


def _display_technical_fields(df: pd.DataFrame) -> None:
    if df.empty:
        return
    with st.expander("查看技术字段名"):
        st.write("、".join(str(column) for column in df.columns))


def display_dataframe_preview(
    df: pd.DataFrame,
    max_rows: int = 20,
    *,
    show_technical_fields: bool = True,
    hide_index: bool = False,
) -> None:
    if df.empty:
        st.info("表为空或不存在。")
        return
    st.dataframe(translate_dataframe_columns(df.head(max_rows)), use_container_width=True, hide_index=hide_index)
    if show_technical_fields:
        _display_technical_fields(df)


def build_table_status(store: StockAgentStore, table_names: list[str]) -> pd.DataFrame:
    rows = []
    for table_name in table_names:
        df = safe_load_table(store, table_name)
        summary = get_table_summary(df)
        rows.append(
            {
                "table_name": get_table_label(table_name),
                "row_count": summary["row_count"],
                "date_range": summary["date_range"],
                "date_column": get_field_label(summary["date_column"]) if summary["date_column"] else "",
            }
        )
    return pd.DataFrame(rows, columns=["table_name", "row_count", "date_range", "date_column"])


def latest_report_path(output_dir: str = "reports", pattern: str = "daily_report_*.md") -> str | None:
    reports = list_report_files(output_dir, pattern=pattern)
    return reports[0] if reports else None


def load_json_file(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        _show_error(f"读取 JSON 文件失败: {exc}")
        return {}


def load_latest_strategy_research_suggestions(output_dir: str = "reports") -> tuple[str | None, dict]:
    reports = list_report_files(output_dir, pattern="strategy_research_suggestions_*.json")
    if not reports:
        return None, {}
    latest = reports[0]
    try:
        return latest, json.loads(Path(latest).read_text(encoding="utf-8"))
    except Exception as exc:
        _show_error(f"读取策略研究建议 JSON 失败: {exc}")
        return latest, {}


def load_latest_parameter_search_space_candidate(output_dir: str = "reports") -> tuple[str | None, dict]:
    reports = list_report_files(output_dir, pattern="parameter_search_space_candidate_*.json")
    if not reports:
        return None, {}
    latest = reports[0]
    try:
        return latest, json.loads(Path(latest).read_text(encoding="utf-8"))
    except Exception as exc:
        _show_error(f"读取参数搜索空间候选 JSON 失败: {exc}")
        return latest, {}


def load_active_strategy_candidate_table(
    config_path: str = "configs/active_strategies_candidate.json",
) -> tuple[bool, pd.DataFrame]:
    path = Path(config_path)
    config = load_active_strategy_candidates(config_path)
    candidates = config.get("active_strategy_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    df = pd.DataFrame([item for item in candidates if isinstance(item, dict)])
    for column in ACTIVE_STRATEGY_CANDIDATE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")
    return path.exists(), df.loc[:, ACTIVE_STRATEGY_CANDIDATE_COLUMNS]


def _filter_by_trade_date(df: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    if df.empty or trade_date in (None, "全部"):
        return df
    if "trade_date" in df.columns:
        return df[df["trade_date"].astype(str) == str(trade_date)].copy()
    if "plan_date" in df.columns:
        return df[df["plan_date"].astype(str) == str(trade_date)].copy()
    if "as_of_date" in df.columns:
        return df[df["as_of_date"].astype(str) == str(trade_date)].copy()
    return df


def _preferred_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    preferred = [column for column in columns if column in df.columns]
    remaining = [column for column in df.columns if column not in preferred]
    return df.loc[:, preferred + remaining] if preferred else df


def _row_count(store: StockAgentStore, table_name: str) -> int:
    return len(safe_load_table(store, table_name))


def _has_rows(store: StockAgentStore, table_name: str) -> str:
    return "有数据" if _row_count(store, table_name) > 0 else "无数据"


def build_system_health_key_table_status(store: StockAgentStore) -> pd.DataFrame:
    rows = []
    for table_name in SYSTEM_HEALTH_KEY_TABLES:
        row_count = _row_count(store, table_name)
        rows.append(
            {
                "table_name": table_name,
                "row_count": row_count,
                "has_data": row_count > 0,
            }
        )
    return pd.DataFrame(rows, columns=["table_name", "row_count", "has_data"])


def _latest_report_path(output_dir: str = "reports") -> str | None:
    reports = list_report_files(output_dir)
    return reports[0] if reports else None


def _render_report_viewer(
    title: str,
    patterns: dict[str, str],
    output_dir: str = "reports",
    key_prefix: str = "report",
) -> None:
    st.markdown(title)
    report_type = st.selectbox("报告类型", list(patterns), key=f"{key_prefix}_type")
    reports = list_report_files(output_dir, pattern=patterns[report_type])
    if not reports:
        st.info("reports 目录下暂无对应报告。")
        return

    selected = st.selectbox("报告文件", reports, index=0, key=f"{key_prefix}_file")
    content = read_markdown_file(selected)
    if content:
        st.markdown(content)


def _render_overview(store: StockAgentStore) -> None:
    st.subheader("系统总览")
    st.caption("本页面只读取本地配置、DuckDB 和 reports 文件，不触发数据拉取、LLM、策略研究或交易执行。")
    if not _db_path(store).exists():
        st.info("数据库文件不存在，请先运行本地数据流程生成 DuckDB。")

    st.markdown("#### 运行配置")
    display_dataframe_preview(build_config_overview(), max_rows=20, hide_index=True)

    st.markdown("#### 核心表行数")
    metrics = {f"{get_table_label(table_name)}行数": _row_count(store, table_name) for table_name in OVERVIEW_TABLES}
    cols = st.columns(3)
    for index, (label, value) in enumerate(metrics.items()):
        cols[index % 3].metric(label, value)

    st.metric("最新交易日期", get_latest_trade_date(store) or "暂无")

    st.markdown("#### 最近关键报告")
    report_rows = []
    for label, pattern in RECENT_OVERVIEW_REPORT_PATTERNS.items():
        report_rows.append({"report_type": label, "latest_path": latest_report_path(pattern=pattern) or "暂无"})
    display_dataframe_preview(pd.DataFrame(report_rows), max_rows=20, hide_index=True)

    st.markdown("#### 日度计划数据状态")
    status = pd.DataFrame(
        [
            {"table": get_table_label("strategy_signals"), "status": _has_rows(store, "strategy_signals")},
            {"table": get_table_label("candidate_pool"), "status": _has_rows(store, "candidate_pool")},
            {"table": get_table_label("trade_plan"), "status": _has_rows(store, "trade_plan")},
        ]
    )
    display_dataframe_preview(status, max_rows=20)


def _render_data_table_status(store: StockAgentStore) -> None:
    st.subheader("数据表状态")
    st.caption("本页面只读取本地 DuckDB。表不存在或为空时显示为空，不自动创建、更新或拉取数据。")
    for group_name, table_names in DATA_TABLE_GROUPS.items():
        st.markdown(f"#### {group_name}")
        display_dataframe_preview(build_table_status(store, table_names), max_rows=100, hide_index=True)
        for table_name in table_names:
            with st.expander(f"{get_table_label(table_name)}前 20 行预览"):
                df = safe_load_table(store, table_name)
                summary = get_table_summary(df)
                st.write(f"行数：{summary['row_count']}")
                if summary["date_range"]:
                    st.write(f"日期范围：{summary['date_range']}")
                _technical_fields_caption(df)
                display_dataframe_preview(df, max_rows=20, show_technical_fields=False)


def _render_daily_trade_plan_page(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("日度交易计划")
    st.warning("交易计划仅供人工复核参考；系统不会自动下单，也不会连接券商接口。")

    sections = [
        ("strategy_signals", ["trade_date", "code", "strategy_name", "strategy_version", "signal_strength", "entry_reason", "risk_flags"]),
        ("candidate_pool", CANDIDATE_REQUESTED_COLUMNS),
        ("trade_plan", TRADE_PLAN_REQUESTED_COLUMNS),
    ]
    for table_name, preferred_columns in sections:
        st.markdown(f"#### {get_table_label(table_name)}")
        df = _filter_by_trade_date(safe_load_table(store, table_name), selected_date)
        if df.empty:
            if table_name == "candidate_pool":
                st.info("候选池为空。可能原因：数据不足；策略条件过严；策略未准入；小样本日期太短。")
            elif table_name == "trade_plan":
                st.info("当前暂无交易计划，可能原因包括：数据不足、策略条件过严、策略未准入或样本日期太短。")
            else:
                st.info(f"{get_table_label(table_name)}为空或当前日期无数据。")
            continue
        display_dataframe_preview(_preferred_columns(df, preferred_columns), max_rows=20)


def _render_strategy_research_page(store: StockAgentStore, output_dir: str = "reports") -> None:
    st.subheader("策略研究")
    st.caption("本页只展示本地研究表和报告；不会运行策略研究，不修改 active_strategies.json 或 parameter_search_space.json。")

    for table_name in STRATEGY_RESEARCH_TABLES:
        st.markdown(f"#### {get_table_label(table_name)}")
        df = safe_load_table(store, table_name)
        summary = get_table_summary(df)
        st.write(f"行数：{summary['row_count']}")
        display_dataframe_preview(df, max_rows=20)

    st.markdown("#### 最新策略研究总流程报告")
    strategy_ops_path = latest_report_path(output_dir, "strategy_ops_workflow_*.md")
    if strategy_ops_path:
        st.write(f"报告路径：{strategy_ops_path}")
        content = read_markdown_file(strategy_ops_path)
        if content:
            st.markdown(content)
    else:
        st.info("暂无 reports/strategy_ops_workflow_*.md。")

    st.markdown("#### 候选研究建议 JSON")
    st.caption("这些只是候选研究建议，不能直接用于实盘，必须人工确认。")
    for label, pattern in [
        ("StrategyResearchAgent", "strategy_research_suggestions_*.json"),
        ("ParameterIterationAgent", "parameter_search_space_candidate_*.json"),
    ]:
        path = latest_report_path(output_dir, pattern)
        if not path:
            st.info(f"暂无 {pattern}。")
            continue
        payload = load_json_file(path)
        st.write(f"{label} JSON 路径：{path}")
        st.json(_json_summary(payload))


def _json_summary(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    summary: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = list(value.keys())[:20]
        elif key in {"requires_human_review", "do_not_auto_apply", "note"}:
            summary[key] = value
    return summary or {"keys": list(payload.keys())[:20]}


def _render_llm_reports_page(output_dir: str = "reports") -> None:
    st.subheader("LLM 报告")
    st.caption("LLM 只解释结构化结果，不直接做交易决策；本页面不会自动运行 LLM。")

    report_type = st.selectbox("报告类型", list(LLM_REPORT_PATTERNS), key="llm_report_type")
    reports = list_report_files(output_dir, pattern=LLM_REPORT_PATTERNS[report_type])
    if not reports:
        st.info("reports 目录下暂无对应 LLM 报告。")
        return

    selected = st.selectbox("报告文件（按更新时间倒序）", reports, index=0, key="llm_report_file")
    content = read_markdown_file(selected)
    if content:
        st.markdown(content)


def _render_system_acceptance_page(output_dir: str = "reports") -> None:
    st.subheader("系统验收")
    st.caption("本页只展示验收与健康检查报告，不自动执行检查。")

    has_any_report = False
    for title, pattern in SYSTEM_ACCEPTANCE_REPORT_PATTERNS.items():
        st.markdown(f"#### {title}")
        path = latest_report_path(output_dir, pattern)
        if not path:
            st.info(f"暂无 {pattern}。")
            continue
        has_any_report = True
        st.write(f"报告路径：{path}")
        content = read_markdown_file(path)
        if content:
            st.markdown(content)

    if not has_any_report:
        st.markdown("#### 建议运行")
        st.code("uv run python -m src.pipeline.run_system_acceptance_workflow", language="bash")
        st.code("uv run python -m src.pipeline.run_system_health_check --export-report", language="bash")


def _render_workflow_commands_page() -> None:
    st.subheader("工作流命令")
    st.caption("本页面只展示命令，不自动执行。全量数据建议晚上挂机；daily_basic / moneyflow 在家宽下可能超时，当前页面不处理网络机制。")

    commands = [
        (
            "系统验收",
            "uv run python -m src.pipeline.run_system_acceptance_workflow",
        ),
        (
            "小样本数据更新",
            "uv run python -m src.pipeline.run_data_update_workflow \\\n"
            "  --start-date 20250101 \\\n"
            "  --end-date 20250110 \\\n"
            "  --mode test",
        ),
        (
            "夜间全量挂机",
            "mkdir -p logs\n\n"
            "nohup uv run python -m src.pipeline.run_data_update_workflow \\\n"
            "  --start-date 20240901 \\\n"
            "  --end-date 20250110 \\\n"
            "  --mode full \\\n"
            "  --sleep-seconds 0.5 \\\n"
            "  > logs/data_update_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &",
        ),
        (
            "因子构建",
            "uv run python -m src.pipeline.run_factor_build_workflow",
        ),
        (
            "每日盘后总流程",
            "uv run python -m src.pipeline.run_daily_ops_workflow",
        ),
        (
            "策略研究总流程",
            "uv run python -m src.pipeline.run_strategy_ops_workflow",
        ),
        (
            "LLM 总控",
            "uv run python -m src.pipeline.run_llm_agents_workflow",
        ),
    ]
    for title, command in commands:
        st.markdown(f"#### {title}")
        st.code(command, language="bash")


def _latest_market_regime(store: StockAgentStore) -> pd.DataFrame:
    df = safe_load_table(store, "market_regime")
    if df.empty:
        return df
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date")
    return df.tail(1).reset_index(drop=True)


def _render_market_regime(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("市场环境")
    st.code("uv run python -m src.pipeline.update_index_daily --start-date YYYYMMDD --end-date YYYYMMDD", language="bash")
    st.code("uv run python -m src.pipeline.update_limit_list_daily --start-date YYYYMMDD --end-date YYYYMMDD", language="bash")
    st.code("uv run python -m src.pipeline.build_market_regime", language="bash")

    df = _filter_by_trade_date(safe_load_table(store, "market_regime"), selected_date)
    if df.empty:
        df = _latest_market_regime(store)
    if df.empty:
        st.info("当前暂无市场环境结果。")
        return

    row = df.sort_values("trade_date").iloc[-1] if "trade_date" in df.columns else df.iloc[-1]
    cols = st.columns(5)
    cols[0].metric(_metric_label("market_regime"), _format_value(row.get("market_regime")))
    cols[1].metric(_metric_label("risk_level"), _format_value(row.get("risk_level")))
    cols[2].metric(_metric_label("limit_up_count"), _format_value(row.get("limit_up_count")))
    cols[3].metric(_metric_label("limit_down_count"), _format_value(row.get("limit_down_count")))
    cols[4].metric(_metric_label("sentiment_score"), _format_value(row.get("sentiment_score")))
    st.write(f"{get_field_label('regime_reason')}：{_format_value(row.get('regime_reason'))}")
    display_dataframe_preview(df, max_rows=50)


def _render_industry_strength(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("行业强度")
    st.code("uv run python -m src.pipeline.update_sw_industry_classification --level L1", language="bash")
    st.code("uv run python -m src.pipeline.update_sw_daily --start-date YYYYMMDD --end-date YYYYMMDD", language="bash")
    st.code("uv run python -m src.pipeline.build_industry_strength", language="bash")
    df = _filter_by_trade_date(safe_load_table(store, "industry_strength"), selected_date)
    if df.empty:
        all_data = safe_load_table(store, "industry_strength")
        if not all_data.empty and "trade_date" in all_data.columns:
            df = all_data.sort_values("trade_date").groupby("industry_code", as_index=False).tail(1)
    if df.empty:
        st.info("当前暂无行业强度结果。")
        return
    preferred = [
        "trade_date",
        "industry_code",
        "industry_name",
        "industry_strength_score",
        "industry_strength_level",
        "pct_change",
        "industry_return_3d",
        "industry_return_5d",
        "industry_amount_ratio_5",
        "industry_risk_flags",
    ]
    display_dataframe_preview(_preferred_columns(df.sort_values("industry_strength_score", ascending=False), preferred), max_rows=50)


def _render_candidate_pool(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("候选股池")
    df = _filter_by_trade_date(safe_load_table(store, "candidate_pool"), selected_date)
    if df.empty:
        st.info("当前候选股池为空。")
        return
    st.caption("重点关注：综合评分、策略名称、风险标记、资金流评分、行业强度和市场风险。")
    display_dataframe_preview(_preferred_columns(df, CANDIDATE_COLUMNS), max_rows=100)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return translate_value(value)


def _metric_label(field_name: str) -> str:
    return get_field_label(field_name)


def _technical_fields_caption(df: pd.DataFrame) -> None:
    if not df.empty:
        st.caption(f"技术字段名：{', '.join(str(column) for column in df.columns)}")


def _mean_value(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _sum_value(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return None if values.empty else float(values.sum())


def _count_equal(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int((df[column].fillna("").astype(str) == value).sum())


def _valid_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_valid" not in df.columns:
        return pd.DataFrame()
    return df[df["is_valid"].fillna(False).astype(bool)].copy()


def _render_trade_plan(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("交易计划")
    df = _filter_by_trade_date(safe_load_table(store, "trade_plan"), selected_date)
    if df.empty:
        st.info("当前暂无交易计划，可能原因包括：数据不足、策略条件过严、策略未准入或样本日期太短。")
        return

    st.caption("本系统只生成条件化交易计划，不自动下单。重点查看计划动作、买入区间、止损价、止盈参考、仓位比例、计划理由和风险说明。")
    st.markdown("#### 交易计划总览")
    display_dataframe_preview(_preferred_columns(df, TRADE_PLAN_COLUMNS), max_rows=100)
    for _, row in df.iterrows():
        title = f"{_format_value(row.get('code'))} / {_format_value(row.get('name'))}"
        with st.expander(title):
            st.write(f"{get_field_label('action')}：{_format_value(row.get('action'))}")
            detail = {
                TRADE_PLAN_DETAIL_LABELS.get(column, get_field_label(column)): (
                    translate_risk_flags(row.get(column)) if "risk_flags" in column else _format_value(row.get(column))
                )
                for column in TRADE_PLAN_DETAIL_COLUMNS
                if column != "action"
            }
            st.table(pd.DataFrame(detail.items(), columns=["字段", "值"]))


def _render_actual_trades_and_execution_review(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("实盘记录与执行复盘")
    command_date = selected_date if selected_date not in (None, "全部") else "YYYY-MM-DD"
    st.markdown("#### 一键盘后复盘工作流")
    st.caption("该命令只基于本地数据库执行复盘与报告导出。")
    st.code(f"uv run python -m src.pipeline.run_after_market_review --trade-date {command_date}", language="bash")
    st.code(
        f"uv run python -m src.pipeline.run_after_market_review --trade-date {command_date} --run-llm-daily-review",
        language="bash",
    )

    daily_review = _filter_by_trade_date(safe_load_table(store, "daily_review"), selected_date)
    st.markdown("#### 盘后复盘总览")
    if daily_review.empty:
        st.info("当前暂无盘后复盘结果。")
    else:
        row = daily_review.iloc[0]
        cols = st.columns(4)
        cols[0].metric("执行评分", _format_value(row.get("execution_score")))
        cols[1].metric("计划外交易", _format_value(row.get("off_plan_count")))
        cols[2].metric("执行偏差", _format_value(row.get("deviation_count")))
        cols[3].metric("追高偏差", _format_value(row.get("chase_count")))
        st.write(f"主要问题：{_format_value(row.get('main_issues'))}")
        st.write(f"复盘总结：{_format_value(row.get('review_summary'))}")
        st.write(f"后续建议：{_format_value(row.get('next_action_suggestion'))}")

    actual_trades = _filter_by_trade_date(safe_load_table(store, "actual_trades"), selected_date)
    st.markdown("#### 实盘交易记录")
    if actual_trades.empty:
        st.info("当前暂无实盘交易记录。")
    else:
        display_dataframe_preview(actual_trades, max_rows=100)

    execution_review = _filter_by_trade_date(safe_load_table(store, "execution_review"), selected_date)
    st.markdown("#### 执行复盘结果")
    if execution_review.empty:
        st.info("当前暂无执行复盘结果。")
    else:
        display_dataframe_preview(execution_review, max_rows=100)

    trade_performance = _filter_by_trade_date(safe_load_table(store, "actual_trade_performance"), selected_date)
    st.markdown("#### 实盘交易表现")
    if trade_performance.empty:
        st.info("当前暂无实盘交易表现结果。")
    else:
        valid = _valid_performance(trade_performance)
        cols = st.columns(4)
        cols[0].metric("平均1日收益率", _format_value(_mean_value(valid, "return_1d")))
        cols[1].metric("平均3日收益率", _format_value(_mean_value(valid, "return_3d")))
        cols[2].metric("平均5日收益率", _format_value(_mean_value(valid, "return_5d")))
        cols[3].metric("平均3日最大回撤", _format_value(_mean_value(valid, "max_drawdown_3d")))
        display_dataframe_preview(trade_performance, max_rows=100)

    positions = _filter_by_trade_date(safe_load_table(store, "positions"), selected_date)
    position_review = _filter_by_trade_date(safe_load_table(store, "position_review"), selected_date)
    st.markdown("#### 当前持仓")
    if positions.empty:
        st.info("当前暂无持仓数据。")
    else:
        cols = st.columns(5)
        cols[0].metric("持仓股票数", len(positions))
        cols[1].metric("总市值", _format_value(_sum_value(positions, "market_value")))
        cols[2].metric("总浮动盈亏", _format_value(_sum_value(positions, "floating_pnl")))
        cols[3].metric("T+1 锁定数量", _count_equal(positions, "t_plus_1_status", "not_sellable_today"))
        cols[4].metric("高风险持仓数量", _count_equal(position_review, "position_risk_level", "high"))
        display_dataframe_preview(_preferred_columns(positions, POSITION_COLUMNS), max_rows=100)

    st.markdown("#### 持仓风险检查")
    if position_review.empty:
        st.info("当前暂无持仓风险检查结果。")
    else:
        display_dataframe_preview(_preferred_columns(position_review, POSITION_REVIEW_COLUMNS), max_rows=100)

    _render_report_viewer(
        "#### 盘后复盘报告",
        {
            "盘后执行复盘报告": "daily_review_*.md",
            "实盘交易表现报告": "trade_performance_*.md",
            "持仓风险检查报告": "position_review_*.md",
            "周期执行复盘报告": "period_review_*.md",
            "交易复盘 Agent 报告": "llm_daily_review_*.md",
        },
        key_prefix="after_market_report",
    )


def _render_period_review(store: StockAgentStore) -> None:
    st.subheader("周期复盘")
    st.code(
        "uv run python -m src.pipeline.build_period_review --start-date YYYY-MM-DD --end-date YYYY-MM-DD",
        language="bash",
    )
    period_review = safe_load_table(store, "period_review")
    if period_review.empty:
        st.info("当前暂无周期复盘结果。")
    else:
        row = period_review.iloc[-1]
        cols = st.columns(4)
        cols[0].metric(_metric_label("actual_trade_count"), _format_value(row.get("actual_trade_count")))
        cols[1].metric(_metric_label("off_plan_count"), _format_value(row.get("off_plan_count")))
        cols[2].metric(_metric_label("deviation_count"), _format_value(row.get("deviation_count")))
        cols[3].metric(_metric_label("avg_execution_score"), _format_value(row.get("avg_execution_score")))
        cols = st.columns(3)
        cols[0].metric(_metric_label("avg_return_3d"), _format_value(row.get("avg_return_3d")))
        cols[1].metric(_metric_label("plan_trade_avg_return_3d"), _format_value(row.get("plan_trade_avg_return_3d")))
        cols[2].metric(_metric_label("off_plan_avg_return_3d"), _format_value(row.get("off_plan_avg_return_3d")))
        display_dataframe_preview(period_review, max_rows=100)

    _render_report_viewer(
        "#### 周期复盘报告",
        {"周期执行复盘报告": "period_review_*.md"},
        key_prefix="period_review_report",
    )


def _render_daily_report(output_dir: str = "reports") -> None:
    st.subheader("日度报告")
    _render_report_viewer(
        "#### 报告内容",
        {
            "交易计划报告": "daily_report_*.md",
            "盘后执行复盘报告": "daily_review_*.md",
            "实盘交易表现报告": "trade_performance_*.md",
            "持仓风险检查报告": "position_review_*.md",
            "周期执行复盘报告": "period_review_*.md",
            "参数搜索报告": "parameter_search_*.md",
            "样本外验证报告": "walk_forward_validation_*.md",
            "交易计划规则回测报告": "trade_plan_backtest_*.md",
            "策略准入与观察候选报告": "strategy_admission_*.md",
            "数据更新工作流报告": "data_update_workflow_*.md",
            "日度总流程报告": "daily_ops_workflow_*.md",
            "策略研究总流程报告": "strategy_ops_workflow_*.md",
            "因子构建工作流报告": "factor_build_workflow_*.md",
            "系统总体验收报告": "system_acceptance_*.md",
            "LLM Agent 报告索引": "llm_agents_index_*.md",
            "回测分析 Agent 报告": "llm_backtest_analysis_*.md",
            "市场环境 Agent 报告": "llm_market_regime_*.md",
            "行业洞察 Agent 报告": "llm_industry_insight_*.md",
            "因子洞察 Agent 报告": "llm_factor_insight_*.md",
            "策略研究 Agent 报告": "llm_strategy_research_*.md",
            "参数迭代 Agent 报告": "llm_parameter_iteration_*.md",
            "数据质量与数据源对齐报告": "data_quality_*.md",
            "LLM 综合总结": "llm_report_summary_*.md",
            "风险审查 Agent 报告": "llm_risk_review_*.md",
            "交易复盘 Agent 报告": "llm_daily_review_*.md",
        },
        output_dir=output_dir,
        key_prefix="daily_report",
    )


def _render_strategy_evaluation_report(output_dir: str = "reports") -> None:
    st.subheader("策略评价报告")
    st.markdown("#### 当前观察候选策略配置")
    exists, candidates = load_active_strategy_candidate_table()
    st.write(f"active_strategies_candidate.json 是否存在：{'是' if exists else '否'}")
    if not exists:
        st.info("当前暂无观察候选策略配置，请先运行 build_strategy_admission --export-candidate-config。")
    else:
        st.metric("观察候选策略数量", len(candidates))
        display_dataframe_preview(candidates, max_rows=100)

    _render_report_viewer(
        "#### 报告内容",
        {
            "策略版本评价报告": "strategy_evaluation_*.md",
            "参数搜索报告": "parameter_search_*.md",
            "样本外验证报告": "walk_forward_validation_*.md",
            "交易计划规则回测报告": "trade_plan_backtest_*.md",
            "策略准入与观察候选报告": "strategy_admission_*.md",
            "回测分析 Agent 报告": "llm_backtest_analysis_*.md",
        },
        output_dir=output_dir,
        key_prefix="strategy_evaluation_report",
    )


def _render_strategy_research_workflow(store: StockAgentStore) -> None:
    st.subheader("策略研究工作流")
    st.caption("该命令只基于本地 DuckDB 现有数据运行，生成研究表和报告；策略研究总控流程不用于每日自动交易。")
    st.code(
        "uv run python -m src.pipeline.run_strategy_ops_workflow \\\n"
        "  --train-start-date YYYY-MM-DD \\\n"
        "  --train-end-date YYYY-MM-DD \\\n"
        "  --validation-start-date YYYY-MM-DD \\\n"
        "  --validation-end-date YYYY-MM-DD",
        language="bash",
    )
    st.code(
        "uv run python -m src.pipeline.run_strategy_research_workflow \\\n"
        "  --train-start-date YYYY-MM-DD \\\n"
        "  --train-end-date YYYY-MM-DD \\\n"
        "  --validation-start-date YYYY-MM-DD \\\n"
        "  --validation-end-date YYYY-MM-DD",
        language="bash",
    )
    st.code("uv run python -m src.pipeline.run_backtest_analysis_agent", language="bash")
    st.code("uv run python -m src.pipeline.run_strategy_research_agent", language="bash")
    st.code("uv run python -m src.pipeline.run_parameter_iteration_agent", language="bash")

    st.markdown("#### 研究结果表状态")
    table_names = [
        "strategy_version_evaluation",
        "parameter_search_results",
        "walk_forward_validation",
        "trade_plan_backtest_performance",
        "strategy_admission",
    ]
    rows_by_table = {table_name: _row_count(store, table_name) for table_name in table_names}
    status = pd.DataFrame(
        [
            {
                "table_name": table_name,
                "rows": rows_by_table[table_name],
                "has_data": rows_by_table[table_name] > 0,
            }
            for table_name in table_names
        ]
    )
    status["table_name"] = status["table_name"].map(get_table_label)
    display_dataframe_preview(status, max_rows=100)

    exists, candidates = load_active_strategy_candidate_table()
    st.markdown("#### 当前观察候选策略配置")
    st.write(f"active_strategies_candidate.json 是否存在：{'是' if exists else '否'}")
    st.metric("观察候选策略数量", len(candidates) if exists else 0)
    if exists and not candidates.empty:
        display_dataframe_preview(candidates, max_rows=100)

    st.markdown("#### StrategyResearchAgent 候选研究建议")
    st.caption("该 JSON 只是候选研究建议，不能直接用于实盘，也不会写入 active_strategies.json。")
    suggestions_path, suggestions = load_latest_strategy_research_suggestions()
    if not suggestions_path:
        st.info("当前暂无 strategy_research_suggestions_*.json。")
    else:
        st.write(f"JSON 路径：{suggestions_path}")
        st.json(
            {
                "requires_human_review": suggestions.get("requires_human_review", True),
                "strategy_hypotheses_count": len(suggestions.get("strategy_hypotheses", [])),
                "parameter_search_suggestions_count": len(suggestions.get("parameter_search_suggestions", [])),
                "risk_control_suggestions_count": len(suggestions.get("risk_control_suggestions", [])),
            }
        )

    _render_report_viewer(
        "#### 策略研究 Agent 报告",
        {
            "策略研究总流程运行报告": "strategy_ops_workflow_*.md",
            "策略研究 Agent 报告": "llm_strategy_research_*.md",
            "参数迭代 Agent 报告": "llm_parameter_iteration_*.md",
        },
        key_prefix="strategy_research_agent_report",
    )

    st.markdown("#### ParameterIterationAgent 候选参数搜索空间")
    st.caption("该 JSON 只是候选参数研究建议，不能直接用于实盘，也不会写入 parameter_search_space.json。")
    parameter_candidate_path, parameter_candidate = load_latest_parameter_search_space_candidate()
    if not parameter_candidate_path:
        st.info("当前暂无 parameter_search_space_candidate_*.json。")
    else:
        st.write(f"JSON 路径：{parameter_candidate_path}")
        st.json(
            {
                "requires_human_review": parameter_candidate.get("requires_human_review", True),
                "do_not_auto_apply": parameter_candidate.get("do_not_auto_apply", True),
                "parameter_search_space_candidates_count": len(
                    parameter_candidate.get("parameter_search_space_candidates", [])
                ),
                "risk_control_parameter_candidates_count": len(
                    parameter_candidate.get("risk_control_parameter_candidates", [])
                ),
                "research_questions_count": len(parameter_candidate.get("research_questions", [])),
            }
        )


def _render_table_check(store: StockAgentStore) -> None:
    st.subheader("数据表检查")
    tabs = st.tabs([get_table_label(table_name) for table_name in TABLE_NAMES])
    for tab, table_name in zip(tabs, TABLE_NAMES, strict=True):
        with tab:
            df = safe_load_table(store, table_name)
            if df.empty:
                st.info(f"{get_table_label(table_name)}为空或不存在。")
            else:
                display_dataframe_preview(df, max_rows=100)


def _render_system_health_check(store: StockAgentStore, output_dir: str = "reports") -> None:
    st.subheader("系统健康检查")
    st.markdown("#### 常用命令")
    st.code("uv run python -m src.pipeline.run_system_health_check --export-report", language="bash")
    st.code("uv run python -m src.pipeline.run_report_agent", language="bash")
    st.code("uv run python -m src.pipeline.run_risk_review_agent", language="bash")
    st.code("uv run python -m src.pipeline.run_daily_review_agent", language="bash")

    st.markdown("#### 关键表数据状态")
    key_status = build_system_health_key_table_status(store)
    key_status["table_name"] = key_status["table_name"].map(get_table_label)
    display_dataframe_preview(key_status, max_rows=100)

    st.markdown("#### 数据质量状态")
    quality_report = safe_load_table(store, "data_quality_report")
    if quality_report.empty:
        st.info("当前暂无数据质量报告。")
    else:
        display_dataframe_preview(quality_report, max_rows=100)

    _render_report_viewer(
        "#### 系统健康检查报告",
        {
            "日度总流程运行报告": "daily_ops_workflow_*.md",
            "策略研究总流程运行报告": "strategy_ops_workflow_*.md",
            "数据更新工作流报告": "data_update_workflow_*.md",
            "因子构建工作流报告": "factor_build_workflow_*.md",
            "系统健康检查报告": "system_health_*.md",
            "数据质量与数据源对齐报告": "data_quality_*.md",
            "LLM 综合总结": "llm_report_summary_*.md",
            "风险审查 Agent 报告": "llm_risk_review_*.md",
            "交易复盘 Agent 报告": "llm_daily_review_*.md",
        },
        output_dir=output_dir,
        key_prefix="system_health_report",
    )


def _trade_date_selectbox(
    dates: list[str],
    default_date: str | None,
    selectbox: Callable[..., str] = st.sidebar.selectbox,
) -> str | None:
    options = ["全部"] + dates
    default_index = options.index(default_date) if default_date in options else 0
    return selectbox("交易日期", options, index=default_index)


def main() -> None:
    st.set_page_config(page_title="stock-agent 看板", layout="wide")
    st.title("stock-agent 看板")

    db_path = st.sidebar.text_input("数据库路径", value=settings.DB_PATH)
    if st.sidebar.button("刷新"):
        st.rerun()

    store = StockAgentStore(db_path)
    dates = get_available_trade_dates(store)
    selected_date = _trade_date_selectbox(dates, get_latest_trade_date(store))
    page = st.sidebar.radio(
        "页面选择",
        [
            "系统总览",
            "数据表状态",
            "日度交易计划",
            "策略研究",
            "LLM 报告",
            "系统验收",
            "工作流命令",
        ],
    )

    if page == "系统总览":
        _render_overview(store)
    elif page == "数据表状态":
        _render_data_table_status(store)
    elif page == "日度交易计划":
        _render_daily_trade_plan_page(store, selected_date)
    elif page == "策略研究":
        _render_strategy_research_page(store)
    elif page == "LLM 报告":
        _render_llm_reports_page()
    elif page == "系统验收":
        _render_system_acceptance_page()
    elif page == "工作流命令":
        _render_workflow_commands_page()


if __name__ == "__main__":
    main()
