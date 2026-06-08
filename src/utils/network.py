"""Network environment helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from src.utils.proxy import PROXY_ENV_KEYS, no_proxy_context


@contextmanager
def without_proxy() -> Iterator[None]:
    """Temporarily remove proxy environment variables from the current process."""
    with no_proxy_context(True):
        yield


def clear_proxy_env_for_process() -> None:
    """Permanently remove proxy environment variables from the current process."""
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
