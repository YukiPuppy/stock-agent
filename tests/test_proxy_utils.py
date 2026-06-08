import os

from src.utils.proxy import PROXY_ENV_KEYS, no_proxy_context


def test_no_proxy_context_false_does_not_modify_environment(monkeypatch):
    for key in PROXY_ENV_KEYS:
        monkeypatch.setenv(key, f"value-for-{key}")

    before = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}

    with no_proxy_context(False):
        assert {key: os.environ.get(key) for key in PROXY_ENV_KEYS} == before

    assert {key: os.environ.get(key) for key in PROXY_ENV_KEYS} == before


def test_no_proxy_context_true_temporarily_removes_proxy_variables(monkeypatch):
    for key in PROXY_ENV_KEYS:
        monkeypatch.setenv(key, f"value-for-{key}")

    with no_proxy_context(True):
        for key in PROXY_ENV_KEYS:
            assert key not in os.environ


def test_no_proxy_context_true_restores_proxy_variables_after_exit(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7891")
    monkeypatch.delenv("ALL_PROXY", raising=False)

    with no_proxy_context(True):
        os.environ["ALL_PROXY"] = "http://temporary:7890"

    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["https_proxy"] == "http://127.0.0.1:7891"
    assert "ALL_PROXY" not in os.environ
