"""LLM client abstraction for low-risk report generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from src.config import settings
from src.utils.proxy import no_proxy_context

DEFAULT_LLM_MODEL = "deepseek-v4-flash"
AGENT_MODEL_ENV_VARS = {
    "ReportAgent": "REPORT_AGENT_MODEL",
    "MarketRegimeAgent": "MARKET_REGIME_AGENT_MODEL",
    "IndustryInsightAgent": "INDUSTRY_INSIGHT_AGENT_MODEL",
    "FactorInsightAgent": "FACTOR_INSIGHT_AGENT_MODEL",
    "DailyReviewAgent": "DAILY_REVIEW_AGENT_MODEL",
    "RiskReviewAgent": "RISK_REVIEW_AGENT_MODEL",
    "BacktestAnalysisAgent": "BACKTEST_ANALYSIS_AGENT_MODEL",
    "StrategyResearchAgent": "STRATEGY_RESEARCH_AGENT_MODEL",
    "ParameterIterationAgent": "PARAMETER_ITERATION_AGENT_MODEL",
}
AGENT_DEFAULT_MODELS = {
    "ReportAgent": "deepseek-v4-flash",
    "MarketRegimeAgent": "deepseek-v4-flash",
    "IndustryInsightAgent": "deepseek-v4-flash",
    "FactorInsightAgent": "deepseek-v4-flash",
    "DailyReviewAgent": "deepseek-v4-flash",
    "RiskReviewAgent": "deepseek-v4-pro",
    "BacktestAnalysisAgent": "deepseek-v4-pro",
    "StrategyResearchAgent": "deepseek-v4-pro",
    "ParameterIterationAgent": "deepseek-v4-pro",
}


class BaseLLMClient:
    def generate(self, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError


class DisabledLLMClient(BaseLLMClient):
    def generate(self, prompt: str, *args, **kwargs) -> str:
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
        self.model = str(model or "").strip() or DEFAULT_LLM_MODEL
        self.base_url = (str(base_url or "").strip() or "https://api.deepseek.com").rstrip("/")
        self.timeout_seconds = int(timeout_seconds)

    def generate(self, prompt: str, *args, model: str | None = None, **kwargs) -> str:
        resolved_model = str(model or "").strip() or self.model
        body = {
            "model": resolved_model,
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


def get_llm_client(agent_name: str | None = None) -> BaseLLMClient:
    if not bool(getattr(settings, "ENABLE_LLM_REPORT_AGENT", False)):
        return DisabledLLMClient()

    provider = str(getattr(settings, "LLM_PROVIDER", "none") or "").strip().lower()
    if provider in {"none", "disabled", ""}:
        return DisabledLLMClient()

    if provider == "deepseek":
        return DeepSeekClient(
            api_key=str(getattr(settings, "LLM_API_KEY", "") or ""),
            model=resolve_llm_model(agent_name),
            base_url=str(getattr(settings, "LLM_BASE_URL", "") or "https://api.deepseek.com"),
            timeout_seconds=int(getattr(settings, "LLM_TIMEOUT_SECONDS", 60) or 60),
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def resolve_llm_model(agent_name: str | None = None) -> str:
    """Resolve model with Agent-specific config taking precedence over defaults."""
    normalized_agent_name = str(agent_name or "").strip()
    model_env = AGENT_MODEL_ENV_VARS.get(normalized_agent_name)
    if model_env:
        agent_model = str(getattr(settings, model_env, "") or "").strip()
        if agent_model:
            return agent_model

    default_model = str(getattr(settings, "DEFAULT_LLM_MODEL", "") or "").strip()
    if default_model:
        return default_model

    legacy_model = str(getattr(settings, "LLM_MODEL", "") or "").strip()
    if legacy_model:
        return legacy_model

    agent_default_model = AGENT_DEFAULT_MODELS.get(normalized_agent_name, "")
    if agent_default_model:
        return agent_default_model

    return DEFAULT_LLM_MODEL


def _redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")
