"""Network environment helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]


@contextmanager
def without_proxy() -> Iterator[None]:
    """Temporarily remove proxy environment variables from the current process."""
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


def clear_proxy_env_for_process() -> None:
    """Permanently remove proxy environment variables from the current process."""
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
