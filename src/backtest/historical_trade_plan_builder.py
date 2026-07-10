from __future__ import annotations

import pandas as pd

from src.strategy.candidate_selector import select_candidates_from_signals
from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS, generate_trade_plan


def build_historical_trade_plans(
    strategy_signals: pd.DataFrame,
    daily_factors: pd.DataFrame,
    stock_basic: pd.DataFrame | None = None,
    strategy_evaluation: pd.DataFrame | None = None,
    top_n: int = 20,
    max_plan_items: int = 5,
    min_amount_ma5: float = 0.0,
    market_regime: pd.DataFrame | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Build deterministic historical trade plans for every signal date.

    ``min_amount_ma5`` uses amount_ma5 in thousand yuan.
    """
    diagnostics = {
        "historical_signals": int(len(strategy_signals)) if strategy_signals is not None else 0,
        "historical_candidates": 0,
        "historical_trade_plans": 0,
    }
    if strategy_signals is None or strategy_signals.empty or "trade_date" not in strategy_signals.columns:
        empty = pd.DataFrame(columns=TRADE_PLAN_COLUMNS)
        return (empty, diagnostics) if return_diagnostics else empty

    trade_dates = (
        strategy_signals["trade_date"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(key=lambda values: pd.to_datetime(values, errors="coerce"))
    )
    if trade_dates.empty:
        empty = pd.DataFrame(columns=TRADE_PLAN_COLUMNS)
        return (empty, diagnostics) if return_diagnostics else empty

    plans: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        candidate_pool = select_candidates_from_signals(
            strategy_signals=strategy_signals,
            daily_factors=daily_factors,
            stock_basic=stock_basic,
            trade_date=str(trade_date),
            top_n=top_n,
            min_amount_ma5=min_amount_ma5,
            strategy_evaluation=strategy_evaluation,
            market_regime=_market_regime_for_trade_date(market_regime, str(trade_date)),
        )
        diagnostics["historical_candidates"] += int(len(candidate_pool))
        trade_plan = generate_trade_plan(candidate_pool, max_items=max_plan_items).head(max_plan_items)
        if not trade_plan.empty:
            plans.append(trade_plan)

    if not plans:
        empty = pd.DataFrame(columns=TRADE_PLAN_COLUMNS)
        return (empty, diagnostics) if return_diagnostics else empty

    result = pd.concat(plans, ignore_index=True)
    for column in TRADE_PLAN_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result = result.loc[:, TRADE_PLAN_COLUMNS].reset_index(drop=True)
    diagnostics["historical_trade_plans"] = int(len(result))
    return (result, diagnostics) if return_diagnostics else result


def _market_regime_for_trade_date(market_regime: pd.DataFrame | None, trade_date: str) -> pd.DataFrame | None:
    if market_regime is None or market_regime.empty or "trade_date" not in market_regime.columns:
        return market_regime
    values = market_regime["trade_date"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    trade_date_key = str(trade_date).replace("-", "")
    selected = market_regime.loc[values == trade_date_key].copy()
    if not selected.empty:
        return selected
    before = market_regime.loc[values <= trade_date_key].copy()
    if before.empty:
        return pd.DataFrame()
    return before.assign(_trade_date_key=values.loc[before.index]).sort_values("_trade_date_key").tail(1).drop(
        columns=["_trade_date_key"]
    )
