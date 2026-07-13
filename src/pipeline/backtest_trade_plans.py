from __future__ import annotations

import argparse
import gc
import re
from collections.abc import Sequence

import pandas as pd

from src.backtest.historical_trade_plan_builder import build_historical_trade_plans
from src.backtest.trade_plan_backtester import (
    DEFAULT_MAX_HOLDING_DAYS,
    backtest_trade_plans,
    evaluate_trade_plan_backtest,
    expand_trade_plans_for_holding_days,
)
from src.config.settings import DB_PATH
from src.database.duckdb_store import StockAgentStore
from src.pipeline.memory import collect_memory, log_memory


_DATE_CHUNK_SIZE = 20
_PLAN_CHUNK_SIZE = 1000


def _resolve_db_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


def run_trade_plan_backtest(
    db_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    holding_days_mode: str = "fixed",
    strategy_signals: pd.DataFrame | None = None,
    strategy_evaluation: pd.DataFrame | None = None,
    run_id: str | None = None,
    return_diagnostics: bool = False,
    materialize_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    resolved_db_path = _resolve_db_path(db_path)
    store = StockAgentStore(resolved_db_path)
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

    persisted_trade_dates: list[str] = []
    if run_id is not None:
        with store._connect() as con:
            store._create_tables(con)
            conditions = ["run_id = ?"]
            params: list[object] = [run_id]
            if start_date is not None:
                conditions.append("trade_date >= ?")
                params.append(start_date.replace("-", ""))
            if end_date is not None:
                conditions.append("trade_date <= ?")
                params.append(end_date.replace("-", ""))
            persisted_trade_dates = [
                str(row[0])
                for row in con.execute(
                    f"""
                    SELECT DISTINCT trade_date
                    FROM research_strategy_signals
                    WHERE {' AND '.join(conditions)}
                    ORDER BY trade_date
                    """,
                    params,
                ).fetchall()
            ]
    use_persisted_signals = bool(persisted_trade_dates)
    if not use_persisted_signals and strategy_signals is None:
        strategy_signals = store.load_strategy_signals(start_date=start_date, end_date=end_date)
    elif not use_persisted_signals:
        strategy_signals = _normalize_current_strategy_signals(strategy_signals)

    log_memory("trade_plan_backtest", "before_plan_chunks")
    if not materialize_results and run_id is None:
        raise ValueError("run_id is required when materialize_results=False")
    plan_frames: list[pd.DataFrame] = []
    persisted_plan_count = 0
    diagnostics = {"historical_signals": 0, "historical_candidates": 0, "historical_trade_plans": 0}
    trade_dates = persisted_trade_dates or (
        strategy_signals["trade_date"].dropna().astype(str).drop_duplicates().sort_values().tolist()
        if strategy_signals is not None and not strategy_signals.empty and "trade_date" in strategy_signals.columns
        else []
    )
    for offset in range(0, len(trade_dates), _DATE_CHUNK_SIZE):
        date_chunk = trade_dates[offset : offset + _DATE_CHUNK_SIZE]
        if use_persisted_signals:
            chunk_signals = store.load_research_strategy_signals(
                str(run_id), start_date=date_chunk[0], end_date=date_chunk[-1]
            )
        else:
            chunk_signals = strategy_signals[strategy_signals["trade_date"].astype(str).isin(date_chunk)].copy()
        daily_factors = store.load_daily_factors(start_date=date_chunk[0], end_date=date_chunk[-1])
        chunk_plans, chunk_diagnostics = build_historical_trade_plans(
            strategy_signals=chunk_signals,
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
            chunk_plans = chunk_plans.assign(run_id=run_id)
        store.save_historical_trade_plans(chunk_plans)
        persisted_plan_count += len(chunk_plans)
        if materialize_results:
            plan_frames.append(chunk_plans)
        for key in diagnostics:
            diagnostics[key] += int(chunk_diagnostics.get(key, 0))
        del chunk_signals, daily_factors, chunk_plans
        gc.collect()
        log_memory("trade_plan_backtest", f"plan_chunk_written_{offset // _DATE_CHUNK_SIZE + 1}")
    if materialize_results:
        historical_trade_plans = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
    else:
        historical_trade_plans = pd.DataFrame()
        historical_trade_plans.attrs["row_count"] = persisted_plan_count
    del plan_frames, strategy_signals, persisted_trade_dates
    collect_memory("trade_plan_backtest:plans")

    result_frames: list[pd.DataFrame] = []
    persisted_result_count = 0
    triggered_trade_count = 0
    plan_total = len(historical_trade_plans) if materialize_results else persisted_plan_count
    for offset in range(0, plan_total, _PLAN_CHUNK_SIZE):
        if materialize_results:
            plan_chunk = historical_trade_plans.iloc[offset : offset + _PLAN_CHUNK_SIZE].copy()
        else:
            plan_chunk = _load_historical_plan_chunk(store, str(run_id), _PLAN_CHUNK_SIZE, offset)
        plan_chunk = expand_trade_plans_for_holding_days(
            plan_chunk,
            mode=holding_days_mode,
            max_holding_days=max_holding_days,
        )
        chunk_max_holding_days = int(plan_chunk["max_holding_days"].max()) if not plan_chunk.empty else max_holding_days
        daily_bars = _load_daily_bars_for_plans(store, plan_chunk, end_date, chunk_max_holding_days)
        result_chunk = backtest_trade_plans(
            trade_plans=plan_chunk,
            daily_bars=daily_bars,
            max_holding_days=max_holding_days,
        )
        if run_id is not None:
            result_chunk = result_chunk.assign(run_id=run_id)
        store.save_trade_plan_backtest_results(result_chunk)
        persisted_result_count += len(result_chunk)
        triggered_trade_count += int(result_chunk["is_triggered"].fillna(False).astype(bool).sum())
        if materialize_results:
            result_frames.append(result_chunk)
        del plan_chunk, daily_bars, result_chunk
        gc.collect()
        log_memory("trade_plan_backtest", f"backtest_chunk_written_{offset // _PLAN_CHUNK_SIZE + 1}")
    if materialize_results:
        backtest_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        performance = evaluate_trade_plan_backtest(backtest_results)
    else:
        backtest_results = pd.DataFrame()
        backtest_results.attrs["row_count"] = persisted_result_count
        performance = store.aggregate_trade_plan_backtest_performance(str(run_id))
    del result_frames
    if run_id is not None:
        backtest_results = backtest_results.assign(run_id=run_id)
        if "run_id" not in performance.columns:
            performance = performance.assign(run_id=run_id)
    store.save_trade_plan_backtest_performance(performance)
    diagnostics["triggered_trades"] = triggered_trade_count
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


def _load_daily_bars_for_plans(
    store: StockAgentStore,
    plans: pd.DataFrame,
    end_date: str | None,
    max_holding_days: int,
) -> pd.DataFrame:
    """Read only bars for codes present in a plan chunk (never SELECT *)."""
    if plans.empty:
        return pd.DataFrame(columns=["trade_date", "code", "open", "high", "low", "close", "volume", "amount"])
    codes = plans[["code"]].dropna().astype(str).drop_duplicates()
    start = str(plans["plan_date" if "plan_date" in plans.columns else "trade_date"].min())
    # Holding-day exits require future rows; end_date is already the workflow's semantic boundary.
    with store._connect() as con:
        store._create_tables(con)
        con.register("plan_chunk_codes", codes)
        conditions = ["replace(b.trade_date, '-', '') >= ?"]
        params: list[object] = [start.replace("-", "")]
        if end_date is not None:
            conditions.append("replace(b.trade_date, '-', '') <= ?")
            params.append(end_date.replace("-", ""))
        result = con.execute(
            f"""
            SELECT b.trade_date, b.code, b.open, b.high, b.low, b.close, b.volume, b.amount
            FROM daily_bars AS b
            INNER JOIN plan_chunk_codes AS c ON b.code = c.code
            WHERE {' AND '.join(conditions)}
            ORDER BY b.trade_date, b.code
            """,
            params,
        ).fetchdf()
        con.unregister("plan_chunk_codes")
        return result


def _load_historical_plan_chunk(
    store: StockAgentStore,
    run_id: str,
    limit: int,
    offset: int,
) -> pd.DataFrame:
    with store._connect() as con:
        store._create_tables(con)
        return con.execute(
            """
            SELECT * EXCLUDE (run_id)
            FROM historical_trade_plans
            WHERE run_id = ?
            ORDER BY trade_date, rank, code, strategy_names, strategy_versions
            LIMIT ? OFFSET ?
            """,
            [run_id, limit, offset],
        ).fetchdf()


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
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    holding_days_mode: str = "fixed",
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
        holding_days_mode=holding_days_mode,
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
    parser.add_argument("--max-holding-days", type=int, default=DEFAULT_MAX_HOLDING_DAYS)
    parser.add_argument(
        "--holding-days-mode",
        choices=["fixed", "strategy_grid"],
        default="fixed",
        help="Use one fixed holding period or the configured per-strategy secondary-validation grid.",
    )
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
        holding_days_mode=args.holding_days_mode,
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
