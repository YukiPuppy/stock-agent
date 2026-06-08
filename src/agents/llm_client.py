"""LLM client abstraction for low-risk report generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from src.config import settings
from src.utils.proxy import no_proxy_context


class BaseLLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DisabledLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        return "LLM is disabled. Please enable ENABLE_LLM_REPORT_AGENT and configure LLM_PROVIDER."


class DeepSeekClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: int = 60,
    ):
        if not str(api_key or "").strip():
            raise ValueError("LLM_API_KEY is not configured")
        self.api_key = str(api_key).strip()
        self.model = str(model or "").strip() or "deepseek-v4-flash"
        self.base_url = (str(base_url or "").strip() or "https://api.deepseek.com").rstrip("/")
        self.timeout_seconds = int(timeout_seconds)

    def generate(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是股票研究系统的报告总结助手，只能总结结构化结果，不得直接给出自动交易指令。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with no_proxy_context(settings.LLM_DISABLE_PROXY):
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    status = getattr(response, "status", response.getcode())
                    raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek API request failed with HTTP {exc.code}: {_redact_secret(error_body, self.api_key)}"
            ) from exc

        if status != 200:
            raise RuntimeError(f"DeepSeek API request failed with HTTP {status}: {_redact_secret(raw_body, self.api_key)}")

        try:
            payload = json.loads(raw_body)
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("DeepSeek API response has unexpected structure") from exc
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek API response message content is not a string")
        return content


def get_llm_client() -> BaseLLMClient:
    if not bool(getattr(settings, "ENABLE_LLM_REPORT_AGENT", False)):
        return DisabledLLMClient()

    provider = str(getattr(settings, "LLM_PROVIDER", "none") or "").strip().lower()
    if provider in {"none", "disabled", ""}:
        return DisabledLLMClient()

    if provider == "deepseek":
        return DeepSeekClient(
            api_key=str(getattr(settings, "LLM_API_KEY", "") or ""),
            model=str(getattr(settings, "LLM_MODEL", "") or ""),
            base_url=str(getattr(settings, "LLM_BASE_URL", "") or "https://api.deepseek.com"),
            timeout_seconds=int(getattr(settings, "LLM_TIMEOUT_SECONDS", 60) or 60),
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")
