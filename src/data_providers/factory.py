"""Factory helpers for selecting market data providers."""

from __future__ import annotations

from src.data_providers.akshare_provider import AkshareProvider
from src.data_providers.base import BaseDataProvider


def get_data_provider(name: str | None = "akshare") -> BaseDataProvider:
    """Return a data provider by name."""
    normalized_name = "akshare" if name is None else name.strip().lower()

    if normalized_name == "akshare":
        return AkshareProvider()
    if normalized_name == "tushare":
        raise NotImplementedError("TushareProvider will be added after AKShare MVP is verified.")

    raise ValueError(f"Unsupported data provider: {name}")
