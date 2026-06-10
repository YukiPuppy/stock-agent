import os

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _env_or_default(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = _env_or_default("LLM_PROVIDER", "none")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
REPORT_AGENT_MODEL = os.getenv("REPORT_AGENT_MODEL", "").strip()
MARKET_REGIME_AGENT_MODEL = os.getenv("MARKET_REGIME_AGENT_MODEL", "").strip()
INDUSTRY_INSIGHT_AGENT_MODEL = os.getenv("INDUSTRY_INSIGHT_AGENT_MODEL", "").strip()
FACTOR_INSIGHT_AGENT_MODEL = os.getenv("FACTOR_INSIGHT_AGENT_MODEL", "").strip()
DAILY_REVIEW_AGENT_MODEL = os.getenv("DAILY_REVIEW_AGENT_MODEL", "").strip()
RISK_REVIEW_AGENT_MODEL = os.getenv("RISK_REVIEW_AGENT_MODEL", "").strip()
BACKTEST_ANALYSIS_AGENT_MODEL = os.getenv("BACKTEST_ANALYSIS_AGENT_MODEL", "").strip()
STRATEGY_RESEARCH_AGENT_MODEL = os.getenv("STRATEGY_RESEARCH_AGENT_MODEL", "").strip()
PARAMETER_ITERATION_AGENT_MODEL = os.getenv("PARAMETER_ITERATION_AGENT_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
ENABLE_LLM_REPORT_AGENT = _env_bool("ENABLE_LLM_REPORT_AGENT", False)
LLM_DISABLE_PROXY = _env_bool("LLM_DISABLE_PROXY", False)
DB_PATH = os.getenv("DB_PATH", "data/stock_agent.duckdb")
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
TUSHARE_API_URL = _env_or_default("TUSHARE_API_URL", "http://api.tushare.pro")
TUSHARE_ALLOW_NON_OFFICIAL_API_URL = _env_bool("TUSHARE_ALLOW_NON_OFFICIAL_API_URL", False)
DEFAULT_DATA_PROVIDER = _env_or_default("DEFAULT_DATA_PROVIDER", "tushare")
DATA_FETCH_DISABLE_PROXY = _env_bool("DATA_FETCH_DISABLE_PROXY", False)
TUSHARE_ADJ = _env_or_default("TUSHARE_ADJ", "qfq")
TUSHARE_RATE_LIMIT_SLEEP = float(os.getenv("TUSHARE_RATE_LIMIT_SLEEP", "0.3"))
