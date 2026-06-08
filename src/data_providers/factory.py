"""Factory helpers for selecting market data providers."""

from __future__ import annotations

from src.config import settings
from src.data_providers.akshare_provider import AkshareProvider
from src.data_providers.base import BaseDataProvider
from src.data_providers.tushare_provider import TushareProvider


def get_data_provider(name: str | None = None) -> BaseDataProvider:
    """Return a data provider by name."""
    configured_name = getattr(settings, "DEFAULT_DATA_PROVIDER", "tushare")
    normalized_name = str(configured_name if name is None else name).strip().lower()

    if normalized_name == "akshare":
        return AkshareProvider()
    if normalized_name == "tushare":
        return TushareProvider()

    raise ValueError(f"Unsupported data provider: {name}")
