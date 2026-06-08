"""Proxy environment helpers for data fetching."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


@contextmanager
def no_proxy_context(disable_proxy: bool = False) -> Iterator[None]:
    """Temporarily remove proxy variables when data fetching should bypass proxies."""
    if not disable_proxy:
        yield
        return

    saved = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    try:
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
