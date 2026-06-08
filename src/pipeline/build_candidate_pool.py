"""Build deterministic A-share candidate pools."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.strategy.candidate_selector import select_candidates, select_candidates_from_signals


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def _build_and_save_candidate_pool(
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
    db_path: str | None = None,
) -> tuple[pd.DataFrame, int, int, int, str, str | None, str]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    daily_factors = store.load_daily_factors()
    stock_basic = store.load_stock_basic()
    strategy_signals = store.load_strategy_signals(trade_date=trade_date)
    strategy_evaluation = store.load_strategy_version_evaluation()
    market_regime = _load_latest_market_regime(store, trade_date)
    if not strategy_signals.empty:
        mode = "signals_weighted" if not strategy_evaluation.empty else "signals_unweighted"
        candidate_pool = select_candidates_from_signals(
            strategy_signals=strategy_signals,
            daily_factors=daily_factors,
            stock_basic=stock_basic,
            trade_date=trade_date,
            top_n=top_n,
            min_amount_ma5=min_amount_ma5,
            strategy_evaluation=strategy_evaluation if not strategy_evaluation.empty else None,
            market_regime=market_regime,
        )
    else:
        mode = "fallback_factors"
        candidate_pool = select_candidates(
            daily_factors=daily_factors,
            stock_basic=stock_basic,
            trade_date=trade_date,
            top_n=top_n,
            min_amount_ma5=min_amount_ma5,
            market_regime=market_regime,
        )
    store.save_candidate_pool(candidate_pool)

    used_trade_date = trade_date
    date_source = strategy_signals if mode in {"signals_weighted", "signals_unweighted"} else daily_factors
    if used_trade_date is None and not date_source.empty and "trade_date" in date_source.columns:
        trade_dates = date_source["trade_date"].dropna()
        if not trade_dates.empty:
            used_trade_date = str(trade_dates.max())

    return (
        candidate_pool,
        len(daily_factors),
        len(strategy_signals),
        len(strategy_evaluation),
        resolved_db_path,
        used_trade_date,
        mode,
    )


def _load_latest_market_regime(store: StockAgentStore, trade_date: str | None = None) -> pd.DataFrame:
    try:
        if trade_date is not None:
            selected = store.load_market_regime(trade_date=trade_date)
            if not selected.empty:
                return selected
        regime = store.load_market_regime()
    except Exception:
        return pd.DataFrame()
    if regime.empty or "trade_date" not in regime.columns:
        return pd.DataFrame()
    return regime.sort_values("trade_date").tail(1).reset_index(drop=True)


def build_candidate_pool(
    trade_date: str | None = None,
    top_n: int = 20,
    min_amount_ma5: float = 100000000.0,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Read factors and stock basics, build candidates, persist them, and return the result.

    ``min_amount_ma5`` uses the system daily_bars amount unit: thousand yuan.
    """
    candidate_pool, _, _, _, _, _, _ = _build_and_save_candidate_pool(
        trade_date=trade_date,
        top_n=top_n,
        min_amount_ma5=min_amount_ma5,
        db_path=db_path,
    )
    return candidate_pool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic A-share candidate pool.")
    parser.add_argument("--trade-date", default=None, help="Optional trade date, format YYYY-MM-DD.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--min-amount-ma5",
        type=float,
        default=100000000.0,
        help="Minimum amount_ma5 filter, in thousand yuan.",
    )
    parser.add_argument("--db-path", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    (
        candidate_pool,
        daily_factors_count,
        strategy_signals_count,
        strategy_evaluation_count,
        resolved_db_path,
        used_trade_date,
        mode,
    ) = _build_and_save_candidate_pool(
        trade_date=args.trade_date,
        top_n=args.top_n,
        min_amount_ma5=args.min_amount_ma5,
        db_path=args.db_path,
    )

    print(f"使用交易日期: {used_trade_date}")
    print(f"daily_factors 行数: {daily_factors_count}")
    print(f"strategy_signals 行数: {strategy_signals_count}")
    print(f"strategy_version_evaluation 行数: {strategy_evaluation_count}")
    print(f"candidate_pool 行数: {len(candidate_pool)}")
    print(f"候选池模式: {mode}")
    print(f"保存数据库路径: {resolved_db_path}")
    print("前 20 条候选股:")
    print(candidate_pool.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
