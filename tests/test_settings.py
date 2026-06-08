import importlib

import dotenv

from src.config import settings


TUSHARE_ENV_VARS = [
    "TUSHARE_TOKEN",
    "TUSHARE_API_URL",
    "TUSHARE_ALLOW_NON_OFFICIAL_API_URL",
    "DEFAULT_DATA_PROVIDER",
    "DATA_FETCH_DISABLE_PROXY",
    "TUSHARE_ADJ",
    "TUSHARE_RATE_LIMIT_SLEEP",
]
LLM_ENV_VARS = [
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_TIMEOUT_SECONDS",
    "ENABLE_LLM_REPORT_AGENT",
    "LLM_DISABLE_PROXY",
]


def reload_settings_without_dotenv(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    return importlib.reload(settings)


def test_settings_importable():
    assert settings is not None


def test_db_path_has_default_value(monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.DB_PATH == "data/stock_agent.duckdb"


def test_data_provider_settings_have_defaults(monkeypatch):
    for env_var in TUSHARE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.DEFAULT_DATA_PROVIDER == "tushare"
    assert reloaded_settings.DATA_FETCH_DISABLE_PROXY is False
    assert reloaded_settings.TUSHARE_TOKEN == ""
    assert reloaded_settings.TUSHARE_API_URL == "http://api.tushare.pro"
    assert reloaded_settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL is False
    assert reloaded_settings.TUSHARE_ADJ == "qfq"
    assert reloaded_settings.TUSHARE_RATE_LIMIT_SLEEP == 0.3


def test_llm_settings_have_safe_defaults(monkeypatch):
    for env_var in LLM_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.LLM_PROVIDER == "none"
    assert reloaded_settings.LLM_MODEL == ""
    assert reloaded_settings.LLM_API_KEY == ""
    assert reloaded_settings.LLM_BASE_URL == ""
    assert reloaded_settings.LLM_TIMEOUT_SECONDS == 60
    assert reloaded_settings.ENABLE_LLM_REPORT_AGENT is False
    assert reloaded_settings.LLM_DISABLE_PROXY is False


def test_enable_llm_report_agent_parses_true(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_REPORT_AGENT", "yes")

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.ENABLE_LLM_REPORT_AGENT is True


def test_llm_disable_proxy_parses_true(monkeypatch):
    monkeypatch.setenv("LLM_DISABLE_PROXY", "true")

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.LLM_DISABLE_PROXY is True


def test_tushare_token_can_be_read_from_environment(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.TUSHARE_TOKEN == "test-token"


def test_tushare_api_url_can_be_read_from_environment(monkeypatch):
    monkeypatch.setenv("TUSHARE_API_URL", "http://localhost:8000")

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.TUSHARE_API_URL == "http://localhost:8000"


def test_tushare_allow_non_official_api_url_parses_true(monkeypatch):
    monkeypatch.setenv("TUSHARE_ALLOW_NON_OFFICIAL_API_URL", "true")

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.TUSHARE_ALLOW_NON_OFFICIAL_API_URL is True


def test_data_fetch_disable_proxy_parses_true(monkeypatch):
    monkeypatch.setenv("DATA_FETCH_DISABLE_PROXY", "true")

    reloaded_settings = reload_settings_without_dotenv(monkeypatch)

    assert reloaded_settings.DATA_FETCH_DISABLE_PROXY is True


def test_env_bool_accepts_true_values(monkeypatch):
    for value in ["true", "1", "yes", "y"]:
        monkeypatch.setenv("BOOL_TEST", value)
        assert settings._env_bool("BOOL_TEST") is True


def test_env_or_default_uses_default_for_empty_value(monkeypatch):
    monkeypatch.setenv("EMPTY_TEST", "")

    assert settings._env_or_default("EMPTY_TEST", "fallback") == "fallback"
