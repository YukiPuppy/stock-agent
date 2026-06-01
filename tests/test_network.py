import os

import pytest

from src.utils.network import PROXY_ENV_KEYS, clear_proxy_env_for_process, without_proxy


def test_without_proxy_temporarily_removes_proxy_variables(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")

    with without_proxy():
        assert "http_proxy" not in os.environ
        assert "https_proxy" not in os.environ


def test_without_proxy_restores_proxy_variables_after_exit(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7891")
    monkeypatch.delenv("all_proxy", raising=False)

    with without_proxy():
        os.environ["all_proxy"] = "http://temporary:7890"

    assert os.environ["http_proxy"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7891"
    assert "all_proxy" not in os.environ


def test_without_proxy_restores_proxy_variables_after_exception(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")

    with pytest.raises(RuntimeError, match="boom"):
        with without_proxy():
            raise RuntimeError("boom")

    assert os.environ["http_proxy"] == "http://127.0.0.1:7890"
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:7890"


def test_clear_proxy_env_for_process_removes_proxy_variables(monkeypatch):
    for key in PROXY_ENV_KEYS:
        monkeypatch.setenv(key, f"value-for-{key}")

    clear_proxy_env_for_process()

    for key in PROXY_ENV_KEYS:
        assert key not in os.environ
