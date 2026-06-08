"""Base interfaces for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataProvider(ABC):
    """Abstract interface implemented by market data providers."""

    @abstractmethod
    def get_stock_basic(self) -> pd.DataFrame:
        """Return normalized stock-basic metadata."""

    @abstractmethod
    def get_daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized daily bars for one stock."""
