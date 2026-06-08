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
) -> pd.DataFrame:
    """Build deterministic historical trade plans for every signal date.

    ``min_amount_ma5`` uses amount_ma5 in thousand yuan.
    """
    if strategy_signals.empty or "trade_date" not in strategy_signals.columns:
        return pd.DataFrame(columns=TRADE_PLAN_COLUMNS)

    trade_dates = (
        strategy_signals["trade_date"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(key=lambda values: pd.to_datetime(values, errors="coerce"))
    )
    if trade_dates.empty:
        return pd.DataFrame(columns=TRADE_PLAN_COLUMNS)

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
        )
        trade_plan = generate_trade_plan(candidate_pool, max_items=max_plan_items).head(max_plan_items)
        if not trade_plan.empty:
            plans.append(trade_plan)

    if not plans:
        return pd.DataFrame(columns=TRADE_PLAN_COLUMNS)

    result = pd.concat(plans, ignore_index=True)
    for column in TRADE_PLAN_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result.loc[:, TRADE_PLAN_COLUMNS].reset_index(drop=True)
