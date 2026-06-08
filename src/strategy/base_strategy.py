from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


SIGNAL_COLUMNS = [
    "trade_date",
    "code",
    "strategy_name",
    "strategy_version",
    "signal_strength",
    "entry_reason",
    "risk_flags",
]


class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def generate_signals(
        self,
        daily_factors: pd.DataFrame,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """Generate normalized strategy signals from daily factors."""


def empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)
