"""Small RSS helpers shared by the memory-bounded research pipeline."""

from __future__ import annotations

import gc
import os

import pandas as pd

from src.database.duckdb_store import DAILY_FACTOR_COLUMNS


_BASE_FACTOR_COLUMNS = {
    "trade_date", "code", "close", "pct_chg_1d", "pct_chg_3d", "pct_chg_5d",
    "pct_chg_10d", "amount_ma5", "turnover_rate", "is_suspended",
    "is_limit_up_close", "is_limit_down_close",
}
_STRATEGY_FACTOR_COLUMNS = {
    "trend_pullback": {"above_ma5", "above_ma10", "close_position_20", "volume_ratio_5"},
    "breakout_volume": {"above_ma5", "close_position_20", "volume_ratio_5"},
    "support_rebound": {"above_ma20", "close_position_20"},
    "industry_rotation": {"close_position_20", "industry_amount_ratio_5", "industry_return_3d", "industry_return_5d", "industry_strength_level", "industry_strength_score", "moneyflow_score"},
    "moneyflow_accumulation": {"big_net_amount", "main_net_amount", "main_net_amount_ratio", "moneyflow_score", "net_mf_amount"},
    "low_vol_trend": {"above_ma5", "above_ma10", "above_ma20", "close_position_20", "volume_ratio_5"},
    "oversold_rebound": {"close_position_20", "industry_strength_score", "moneyflow_score"},
    "volume_dryup_breakout": {"above_ma5", "above_ma10", "close_position_20", "volume_ratio_5", "volume_ratio_daily_basic"},
    "relative_strength_pullback": {"above_ma10", "above_ma20", "close_position_20", "industry_return_5d", "industry_strength_score", "moneyflow_score"},
}


def rss_mb() -> float:
    """Return current resident memory without adding a psutil dependency."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return 0.0


def log_memory(stage: str, event: str = "checkpoint") -> float:
    value = rss_mb()
    print(f"[memory] stage={stage} event={event} RSS_MB={value:.1f}", flush=True)
    return value


def collect_memory(stage: str) -> float:
    gc.collect()
    return log_memory(stage, "released")


def load_factor_chunk(store, versions: list[dict], start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Load only factor columns consumed by the requested strategy chunk."""
    # Test doubles and external stores keep using their public compatibility path.
    if not hasattr(store, "_connect") or not hasattr(store, "_create_tables"):
        return store.load_daily_factors(start_date=start_date, end_date=end_date)
    requested = set(_BASE_FACTOR_COLUMNS)
    for version in versions:
        requested.update(_STRATEGY_FACTOR_COLUMNS.get(str(version.get("strategy_name", "")), set()))
    columns = [column for column in DAILY_FACTOR_COLUMNS if column in requested]
    conditions: list[str] = []
    params: list[object] = []
    if start_date is not None:
        conditions.append("trade_date >= ?")
        params.append(start_date.replace("-", ""))
    if end_date is not None:
        conditions.append("trade_date <= ?")
        params.append(end_date.replace("-", ""))
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with store._connect() as con:
        store._create_tables(con)
        return con.execute(
            f"SELECT {', '.join(columns)} FROM daily_factors {where_clause} ORDER BY trade_date, code",
            params,
        ).fetchdf()
