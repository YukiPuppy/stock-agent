import io
import json
from contextlib import contextmanager
import urllib.error

import pytest

from src.agents.llm_client import DeepSeekClient, DisabledLLMClient, get_llm_client
from src.config import settings


def test_disabled_llm_client_returns_disabled_message():
    result = DisabledLLMClient().generate("hello")

    assert "LLM is disabled" in result
    assert "ENABLE_LLM_REPORT_AGENT" in result


def test_get_llm_client_returns_disabled_when_feature_flag_off(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", False)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")

    assert isinstance(get_llm_client(), DisabledLLMClient)


def test_get_llm_client_returns_disabled_for_none_provider(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    assert isinstance(get_llm_client(), DisabledLLMClient)


def test_get_llm_client_raises_for_unimplemented_provider(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: openai"):
        get_llm_client()


def test_get_llm_client_returns_deepseek_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LLM_REPORT_AGENT", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(settings, "LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 60)

    client = get_llm_client()

    assert isinstance(client, DeepSeekClient)
    assert client.base_url == "https://api.deepseek.com"


def test_deepseek_client_requires_api_key():
    with pytest.raises(ValueError, match="LLM_API_KEY is not configured"):
        DeepSeekClient(api_key="", model="deepseek-v4-flash")


def test_deepseek_client_posts_openai_compatible_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "# 总结\n\n需要人工复核。",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("src.agents.llm_client.urllib.request.urlopen", fake_urlopen)
    client = DeepSeekClient(
        api_key="secret-key",
        model="",
        base_url="https://api.deepseek.com/",
        timeout_seconds=12,
    )

    result = client.generate("请总结")

    request = captured["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert result == "# 总结\n\n需要人工复核。"
    assert captured["timeout"] == 12
    assert request.full_url == "https://api.deepseek.com/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer secret-key"
    assert request.get_header("Content-type") == "application/json"
    assert body["model"] == "deepseek-v4-flash"
    assert body["temperature"] == 0.2
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "system"
    assert "不得直接给出自动交易指令" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "请总结"}


def test_deepseek_client_generate_uses_configured_no_proxy_context(monkeypatch):
    calls = []

    @contextmanager
    def fake_no_proxy_context(disable_proxy=False):
        calls.append(disable_proxy)
        yield

    def fake_urlopen(request, timeout):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("src.agents.llm_client.no_proxy_context", fake_no_proxy_context)
    monkeypatch.setattr("src.agents.llm_client.settings.LLM_DISABLE_PROXY", True)
    monkeypatch.setattr("src.agents.llm_client.urllib.request.urlopen", fake_urlopen)
    client = DeepSeekClient(api_key="secret-key", model="deepseek-v4-flash")

    assert client.generate("prompt") == "ok"
    assert calls == [True]


def test_deepseek_client_http_error_does_not_leak_api_key(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"secret-key is invalid"}'),
        )

    monkeypatch.setattr("src.agents.llm_client.urllib.request.urlopen", fake_urlopen)
    client = DeepSeekClient(api_key="secret-key", model="deepseek-v4-flash")

    with pytest.raises(RuntimeError) as exc_info:
        client.generate("prompt")

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "secret-key" not in message
    assert "[REDACTED]" in message


def test_deepseek_client_raises_for_unexpected_response(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeResponse({"choices": []})

    monkeypatch.setattr("src.agents.llm_client.urllib.request.urlopen", fake_urlopen)
    client = DeepSeekClient(api_key="secret-key", model="deepseek-v4-flash")

    with pytest.raises(RuntimeError, match="unexpected structure"):
        client.generate("prompt")


class FakeResponse:
    def __init__(self, payload, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
