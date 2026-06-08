"""Build deterministic multi-strategy signal layer."""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest.strategy_version_runner import generate_historical_signals_for_versions
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.strategy.active_strategy_config import (
    filter_versions_by_active_candidates,
    get_active_strategy_version_set,
    load_active_strategy_candidates,
)
from src.strategy.base_strategy import empty_signals
from src.strategy.strategy_config import is_strategy_enabled, load_strategy_config
from src.strategy.strategy_runner import run_strategies
from src.strategy.strategy_versions import iter_strategy_versions, load_strategy_versions


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def build_strategy_signals(
    trade_date: str | None = None,
    db_path: str | None = None,
    config_path: str | None = None,
    use_active_candidates: bool = False,
    active_config_path: str = "configs/active_strategies_candidate.json",
    versions_config_path: str = "configs/strategy_versions.json",
) -> pd.DataFrame:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    if use_active_candidates:
        used_trade_date = _resolve_trade_date(daily_factors, trade_date)
        strategy_signals = _generate_active_candidate_signals(
            daily_factors=daily_factors,
            trade_date=used_trade_date,
            active_config_path=active_config_path,
            versions_config_path=versions_config_path,
        )
    else:
        strategy_signals = run_strategies(daily_factors, trade_date=trade_date, config_path=config_path)
    store.save_strategy_signals(strategy_signals)
    return strategy_signals


def _build_and_save_strategy_signals(
    trade_date: str | None = None,
    db_path: str | None = None,
    config_path: str | None = None,
    use_active_candidates: bool = False,
    active_config_path: str = "configs/active_strategies_candidate.json",
    versions_config_path: str = "configs/strategy_versions.json",
) -> tuple[pd.DataFrame, int, str, str | None, str | None, bool, str, int]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    used_trade_date = _resolve_trade_date(daily_factors, trade_date)

    active_candidate_count = 0
    if use_active_candidates:
        active_config = load_active_strategy_candidates(active_config_path)
        active_candidate_count = len(get_active_strategy_version_set(active_config))
        strategy_signals = _generate_active_candidate_signals(
            daily_factors=daily_factors,
            trade_date=used_trade_date,
            active_config_path=active_config_path,
            versions_config_path=versions_config_path,
            active_config=active_config,
        )
    else:
        strategy_signals = run_strategies(daily_factors, trade_date=trade_date, config_path=config_path)

    store.save_strategy_signals(strategy_signals)

    return (
        strategy_signals,
        len(daily_factors),
        resolved_db_path,
        used_trade_date,
        config_path,
        use_active_candidates,
        active_config_path,
        active_candidate_count,
    )


def _generate_active_candidate_signals(
    daily_factors: pd.DataFrame,
    trade_date: str | None,
    active_config_path: str,
    versions_config_path: str,
    active_config: dict | None = None,
) -> pd.DataFrame:
    active_config = active_config if active_config is not None else load_active_strategy_candidates(active_config_path)
    versions = iter_strategy_versions(load_strategy_versions(versions_config_path))
    filtered_versions = filter_versions_by_active_candidates(versions, active_config)
    if not filtered_versions:
        print("no active strategy candidates found, no signals generated")
        return empty_signals()
    return generate_historical_signals_for_versions(
        daily_factors=daily_factors,
        versions=filtered_versions,
        start_date=trade_date,
        end_date=trade_date,
    )


def _resolve_trade_date(daily_factors: pd.DataFrame, trade_date: str | None) -> str | None:
    if trade_date is not None or daily_factors.empty or "trade_date" not in daily_factors.columns:
        return trade_date
    trade_dates = daily_factors["trade_date"].dropna()
    if trade_dates.empty:
        return None
    return str(trade_dates.max())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic multi-strategy signals.")
    parser.add_argument("--trade-date", default=None, help="Optional trade date, format YYYY-MM-DD.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--use-active-candidates", action="store_true", default=False)
    parser.add_argument("--active-config-path", default="configs/active_strategies_candidate.json")
    parser.add_argument("--versions-config-path", default="configs/strategy_versions.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    (
        strategy_signals,
        daily_factors_count,
        resolved_db_path,
        used_trade_date,
        config_path,
        use_active_candidates,
        active_config_path,
        active_candidate_count,
    ) = _build_and_save_strategy_signals(
        trade_date=args.trade_date,
        db_path=args.db_path,
        config_path=args.config_path,
        use_active_candidates=args.use_active_candidates,
        active_config_path=args.active_config_path,
        versions_config_path=args.versions_config_path,
    )

    print(f"使用交易日期: {used_trade_date}")
    print(f"使用策略配置: {config_path or 'configs/strategies.yaml'}")
    if use_active_candidates:
        versions = pd.DataFrame(iter_strategy_versions(load_strategy_versions(args.versions_config_path)))
    else:
        config = load_strategy_config(config_path)
        versions = pd.DataFrame(
            [
                {"strategy_name": name, "strategy_version": values.get("version", "v1")}
                for name, values in config.items()
                if is_strategy_enabled(name, config)
            ]
        )
    print("策略版本:")
    print(versions.to_string(index=False))
    print(f"daily_factors 行数: {daily_factors_count}")
    print(f"use_active_candidates: {use_active_candidates}")
    print(f"active_config_path: {active_config_path}")
    print(f"active_candidate_count: {active_candidate_count}")
    print(f"strategy_signals 行数: {len(strategy_signals)}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 20 条信号:")
    print(strategy_signals.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
