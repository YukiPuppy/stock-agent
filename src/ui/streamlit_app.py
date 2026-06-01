from pathlib import Path
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


TABLE_NAMES = [
    "stock_basic",
    "daily_bars",
    "daily_factors",
    "candidate_pool",
    "trade_plan",
]
DATE_TABLES = ["daily_bars", "daily_factors", "candidate_pool", "trade_plan"]
CANDIDATE_COLUMNS = [
    "rank",
    "code",
    "name",
    "close",
    "pct_chg_5d",
    "volume_ratio_5",
    "close_position_20",
    "score",
    "reason",
]
TRADE_PLAN_DETAIL_COLUMNS = [
    "action",
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


def list_report_files(output_dir: str = "reports") -> list[str]:
    try:
        reports_dir = Path(output_dir)
        if not reports_dir.exists():
            return []
        return sorted(
            str(path)
            for path in reports_dir.glob("daily_report_*.md")
            if path.is_file()
        )
    except Exception as exc:
        _show_error(f"读取日度报告失败: {exc}")
        return []


def _filter_by_trade_date(df: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    if df.empty or trade_date in (None, "全部") or "trade_date" not in df.columns:
        return df
    return df[df["trade_date"].astype(str) == str(trade_date)].copy()


def _preferred_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    preferred = [column for column in columns if column in df.columns]
    remaining = [column for column in df.columns if column not in preferred]
    return df.loc[:, preferred + remaining] if preferred else df


def _row_count(store: StockAgentStore, table_name: str) -> int:
    return len(safe_load_table(store, table_name))


def _latest_report_path(output_dir: str = "reports") -> str | None:
    reports = list_report_files(output_dir)
    return reports[-1] if reports else None


def _render_overview(store: StockAgentStore) -> None:
    st.subheader("总览")
    if not _db_path(store).exists():
        st.info("数据库文件不存在，请先运行本地数据流程生成 DuckDB。")

    metrics = {
        "stock_basic 行数": _row_count(store, "stock_basic"),
        "daily_bars 行数": _row_count(store, "daily_bars"),
        "daily_factors 行数": _row_count(store, "daily_factors"),
        "candidate_pool 行数": _row_count(store, "candidate_pool"),
        "trade_plan 行数": _row_count(store, "trade_plan"),
    }

    cols = st.columns(3)
    for index, (label, value) in enumerate(metrics.items()):
        cols[index % 3].metric(label, value)

    st.metric("最新交易日期", get_latest_trade_date(store) or "暂无")
    st.metric("最新报告文件路径", _latest_report_path() or "暂无")


def _render_candidate_pool(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("候选股池")
    df = _filter_by_trade_date(safe_load_table(store, "candidate_pool"), selected_date)
    if df.empty:
        st.info("当前候选股池为空。")
        return
    st.dataframe(_preferred_columns(df, CANDIDATE_COLUMNS), use_container_width=True)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_trade_plan(store: StockAgentStore, selected_date: str | None) -> None:
    st.subheader("交易计划")
    df = _filter_by_trade_date(safe_load_table(store, "trade_plan"), selected_date)
    if df.empty:
        st.info("当前没有生成交易计划。")
        return

    st.dataframe(df, use_container_width=True)
    for _, row in df.iterrows():
        title = f"{_format_value(row.get('code'))} / {_format_value(row.get('name'))}"
        with st.expander(title):
            st.write(f"action: {_format_value(row.get('action'))}")
            detail = {
                column: _format_value(row.get(column))
                for column in TRADE_PLAN_DETAIL_COLUMNS
                if column != "action"
            }
            st.table(pd.DataFrame(detail.items(), columns=["字段", "值"]))


def _render_daily_report(output_dir: str = "reports") -> None:
    st.subheader("日度报告")
    reports = list_report_files(output_dir)
    if not reports:
        st.info("reports 目录下暂无日度报告。")
        return

    selected = st.selectbox("报告文件", reports, index=len(reports) - 1)
    try:
        content = Path(selected).read_text(encoding="utf-8")
        st.markdown(content)
    except Exception as exc:
        st.error(f"读取报告文件失败: {exc}")


def _render_table_check(store: StockAgentStore) -> None:
    st.subheader("数据表检查")
    tabs = st.tabs(TABLE_NAMES)
    for tab, table_name in zip(tabs, TABLE_NAMES, strict=True):
        with tab:
            df = safe_load_table(store, table_name)
            if df.empty:
                st.info(f"{table_name} 为空或不存在。")
            else:
                st.dataframe(df.head(100), use_container_width=True)


def _trade_date_selectbox(
    dates: list[str],
    default_date: str | None,
    selectbox: Callable[..., str] = st.sidebar.selectbox,
) -> str | None:
    options = ["全部"] + dates
    default_index = options.index(default_date) if default_date in options else 0
    return selectbox("交易日期", options, index=default_index)


def main() -> None:
    st.set_page_config(page_title="A股多智能体选股系统 MVP", layout="wide")
    st.title("A股多智能体选股系统 MVP")

    db_path = st.sidebar.text_input("数据库路径", value=settings.DB_PATH)
    if st.sidebar.button("刷新"):
        st.rerun()

    store = StockAgentStore(db_path)
    dates = get_available_trade_dates(store)
    selected_date = _trade_date_selectbox(dates, get_latest_trade_date(store))
    page = st.sidebar.radio(
        "页面选择",
        ["总览", "候选股池", "交易计划", "日度报告", "数据表检查"],
    )

    if page == "总览":
        _render_overview(store)
    elif page == "候选股池":
        _render_candidate_pool(store, selected_date)
    elif page == "交易计划":
        _render_trade_plan(store, selected_date)
    elif page == "日度报告":
        _render_daily_report()
    elif page == "数据表检查":
        _render_table_check(store)


if __name__ == "__main__":
    main()
