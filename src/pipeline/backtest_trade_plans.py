from __future__ import annotations

import argparse
import re
from collections.abc import Sequence

import pandas as pd

from src.backtest.historical_trade_plan_builder import build_historical_trade_plans
from src.backtest.trade_plan_backtester import backtest_trade_plans, evaluate_trade_plan_backtest
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_trade_plan_backtest(
    db_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    max_holding_days: int = 5,
    strategy_signals: pd.DataFrame | None = None,
    strategy_evaluation: pd.DataFrame | None = None,
    run_id: str | None = None,
    return_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    if strategy_signals is None:
        strategy_signals = store.load_strategy_signals(start_date=start_date, end_date=end_date)
    else:
        strategy_signals = _normalize_current_strategy_signals(strategy_signals)
    daily_factors = store.load_daily_factors(start_date=start_date, end_date=end_date)
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    stock_basic = store.load_stock_basic()
    try:
        market_regime = store.load_market_regime()
    except Exception:
        market_regime = pd.DataFrame()
    if strategy_evaluation is None:
        try:
            strategy_evaluation = store.load_strategy_version_evaluation(run_id=run_id)
        except Exception:
            strategy_evaluation = pd.DataFrame()

    historical_trade_plans, diagnostics = build_historical_trade_plans(
        strategy_signals=strategy_signals,
        daily_factors=daily_factors,
        stock_basic=stock_basic,
        strategy_evaluation=strategy_evaluation,
        top_n=top_n,
        max_plan_items=max_plan_items,
        min_amount_ma5=min_amount_ma5,
        market_regime=market_regime,
        return_diagnostics=True,
    )
    if run_id is not None:
        historical_trade_plans = historical_trade_plans.assign(run_id=run_id)
    store.save_historical_trade_plans(historical_trade_plans)

    backtest_results = backtest_trade_plans(
        trade_plans=historical_trade_plans,
        daily_bars=daily_bars,
        max_holding_days=max_holding_days,
    )
    performance = evaluate_trade_plan_backtest(backtest_results)
    if run_id is not None:
        backtest_results = backtest_results.assign(run_id=run_id)
        performance = performance.assign(run_id=run_id)
    store.save_trade_plan_backtest_results(backtest_results)
    store.save_trade_plan_backtest_performance(performance)
    diagnostics["triggered_trades"] = (
        int(backtest_results["is_triggered"].fillna(False).astype(bool).sum())
        if "is_triggered" in backtest_results.columns
        else 0
    )
    print(
        "[historical_trade_plan_chain] "
        f"historical_signals={diagnostics['historical_signals']} "
        f"historical_candidates={diagnostics['historical_candidates']} "
        f"historical_trade_plans={diagnostics['historical_trade_plans']} "
        f"triggered_trades={diagnostics['triggered_trades']}",
        flush=True,
    )

    if return_diagnostics:
        return historical_trade_plans, backtest_results, performance, diagnostics
    return historical_trade_plans, backtest_results, performance


def _normalize_current_strategy_signals(strategy_signals: pd.DataFrame) -> pd.DataFrame:
    signals = strategy_signals.copy()
    if "trade_date" in signals.columns:
        values = signals["trade_date"].fillna("").astype(str).str.strip()
        digits = values.str.replace(r"\D", "", regex=True)
        normalized = digits.where(digits.str.len() == 8, "")
        needs_parse = normalized.eq("") & values.ne("")
        if needs_parse.any():
            parsed = pd.to_datetime(values[needs_parse], errors="coerce")
            normalized.loc[needs_parse] = parsed.dt.strftime("%Y%m%d").fillna("")
        signals["trade_date"] = normalized
    if "code" in signals.columns:
        signals["code"] = signals["code"].fillna("").astype(str).map(_normalize_stock_code)
    return signals


def _normalize_stock_code(value: str) -> str:
    match = re.search(r"\d{6}", value)
    if match:
        return match.group(0)
    digits = re.sub(r"\D", "", value)
    return digits.zfill(6)[-6:] if digits else ""


def _run_and_report(
    db_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    max_holding_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
    strategy_signals = store.load_strategy_signals(start_date=start_date, end_date=end_date)
    daily_factors = store.load_daily_factors(start_date=start_date, end_date=end_date)
    daily_bars = store.load_daily_bars(start_date=start_date, end_date=end_date)
    historical_trade_plans, backtest_results, performance = run_trade_plan_backtest(
        db_path=resolved_db_path,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        max_plan_items=max_plan_items,
        min_amount_ma5=min_amount_ma5,
        max_holding_days=max_holding_days,
    )
    counts = {
        "strategy_signals": len(strategy_signals),
        "daily_factors": len(daily_factors),
        "daily_bars": len(daily_bars),
    }
    return historical_trade_plans, backtest_results, performance, counts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest generated trade plans with deterministic rules.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-plan-items", type=int, default=5)
    parser.add_argument(
        "--min-amount-ma5",
        type=float,
        default=0.0,
        help="Minimum amount_ma5 filter, in thousand yuan.",
    )
    parser.add_argument("--max-holding-days", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    historical_trade_plans, backtest_results, performance, counts = _run_and_report(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        max_plan_items=args.max_plan_items,
        min_amount_ma5=args.min_amount_ma5,
        max_holding_days=args.max_holding_days,
    )
    print(f"strategy_signals 行数: {counts['strategy_signals']}")
    print(f"daily_factors 行数: {counts['daily_factors']}")
    print(f"daily_bars 行数: {counts['daily_bars']}")
    print(f"historical_trade_plans 行数: {len(historical_trade_plans)}")
    print(f"trade_plan_backtest_results 行数: {len(backtest_results)}")
    print(f"trade_plan_backtest_performance 行数: {len(performance)}")
    print("前 20 条回测结果:")
    print(backtest_results.head(20).to_string(index=False))
    print("表现汇总:")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
