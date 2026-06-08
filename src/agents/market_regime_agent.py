"""MarketRegimeAgent explains market conditions with an LLM."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


SENSITIVE_KEY_PARTS = (
    "token",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "credential",
    "account",
)
FORBIDDEN_OUTPUT_TERMS = {
    "保证盈利": "收益存在不确定性，需保持风险约束",
    "稳赚": "收益存在不确定性",
    "满仓": "需控制仓位并遵守风险约束",
    "自动下单": "仅供人工复核参考，不执行交易",
}
MAJOR_INDEX_CODES = ["000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH"]
LIMIT_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "pct_chg",
    "amp",
    "fc_ratio",
    "fl_ratio",
    "fd_amount",
    "first_time",
    "last_time",
    "open_times",
    "strth",
    "limit_type",
    "status",
]


def build_market_regime_agent_context(
    market_regime: pd.DataFrame | None = None,
    index_daily: pd.DataFrame | None = None,
    limit_list_daily: pd.DataFrame | None = None,
    candidate_pool: pd.DataFrame | None = None,
    trade_plan: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized context focused on market environment."""
    return {
        "agent_scope": "只做市场环境解释，不做交易决策；不直接选股、不调参、不自动启用策略、不自动下单。",
        "market_regime": _compact_dataframe(market_regime, sort_mode="latest_trade_date"),
        "index_daily": _compact_dataframe(index_daily, sort_mode="major_index_latest"),
        "limit_list_daily": _compact_dataframe(limit_list_daily, sort_mode="latest_limit_records"),
        "candidate_pool": _compact_dataframe(candidate_pool, sort_mode="market_high_risk_first"),
        "trade_plan": _compact_dataframe(trade_plan, sort_mode="market_risk_first"),
    }


def build_market_regime_agent_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 MarketRegimeAgent。"
        "你只做市场环境解释，不做交易决策；不直接选股、不调参、不自动启用策略、不自动下单。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、市场环境总体判断\n"
        "## 二、指数趋势分析\n"
        "## 三、涨跌停与短线情绪\n"
        "## 四、市场风险等级解释\n"
        "## 五、对候选池和交易计划的影响\n"
        "## 六、需要关注的市场信号\n"
        "## 七、下一步观察建议\n\n"
        "解释约束：\n"
        "- strong 不代表一定上涨。\n"
        "- weak 不代表一定下跌。\n"
        "- 市场环境只用于调整策略置信度和风险提示。\n\n"
        "硬性禁止：\n"
        "- 不得承诺收益，不得使用“保证盈利”“稳赚”“满仓”“自动下单”。\n"
        "- 不得建议绕过止损、仓位限制或风控。\n"
        "- 不得直接给出新的买入股票。\n"
        "- 不得直接修改策略参数。\n"
        "- 不得直接启用策略。\n"
        "- 不得输出 API key、token、账号、密码或任何凭证明文。\n"
        "- 所有建议都必须表述为人工复核参考，不构成交易指令。\n\n"
        "上下文：\n"
        f"{context_text}"
    )


def run_market_regime_agent(
    llm_client,
    market_regime=None,
    index_daily=None,
    limit_list_daily=None,
    candidate_pool=None,
    trade_plan=None,
) -> str:
    context = build_market_regime_agent_context(
        market_regime=market_regime,
        index_daily=index_daily,
        limit_list_daily=limit_list_daily,
        candidate_pool=candidate_pool,
        trade_plan=trade_plan,
    )
    prompt = build_market_regime_agent_prompt(context)
    markdown = str(llm_client.generate(prompt))
    return _neutralize_forbidden_terms(markdown)


def _compact_dataframe(df: pd.DataFrame | None, sort_mode: str = "") -> dict:
    if df is None:
        return {"is_empty": True, "row_count": 0, "columns": [], "rows": []}

    safe = df.copy()
    safe_columns = [column for column in safe.columns if not _is_sensitive_key(str(column))]
    safe = safe.loc[:, safe_columns]
    columns = [str(column) for column in safe.columns]
    if safe.empty:
        return {"is_empty": True, "row_count": int(len(df)), "columns": columns, "rows": []}

    sorted_df = _sort_dataframe(safe, sort_mode)
    compact = sorted_df.head(30).copy()
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "sort_by": sort_mode,
        "columns": [str(column) for column in compact.columns],
        "rows": _sanitize_value(compact.to_dict(orient="records")),
    }


def _sort_dataframe(df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if sort_mode == "latest_trade_date" and "trade_date" in df.columns:
        return df.sort_values("trade_date", ascending=False, kind="mergesort")

    if sort_mode == "major_index_latest":
        ranked = df.copy()
        if "index_code" in ranked.columns:
            index_rank = {code: idx for idx, code in enumerate(MAJOR_INDEX_CODES)}
            ranked["__index_rank"] = ranked["index_code"].astype(str).map(index_rank).fillna(len(index_rank))
        else:
            ranked["__index_rank"] = 0
        if "trade_date" in ranked.columns:
            return ranked.sort_values(
                ["trade_date", "__index_rank"], ascending=[False, True], kind="mergesort"
            ).drop(columns=["__index_rank"])
        return ranked.sort_values("__index_rank", kind="mergesort").drop(columns=["__index_rank"])

    if sort_mode == "latest_limit_records":
        narrowed = df.loc[:, [column for column in LIMIT_COLUMNS if column in df.columns]]
        if "trade_date" in narrowed.columns:
            return narrowed.sort_values("trade_date", ascending=False, kind="mergesort")
        return narrowed

    if sort_mode == "market_high_risk_first" and "risk_flags" in df.columns:
        return _sort_by_contains(df, ["risk_flags"], ["market_high_risk"])

    if sort_mode == "market_risk_first":
        return _sort_by_contains(df, ["risk_flags", "plan_reason"], ["market_high_risk", "市场风险"])

    return df


def _sort_by_contains(df: pd.DataFrame, columns: list[str], keywords: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    if not available:
        return df
    score = pd.Series(1, index=df.index)
    for column in available:
        text = df[column].fillna("").astype(str)
        for keyword in keywords:
            score = score.mask(text.str.contains(keyword, case=False, regex=False), 0)
    ranked = df.assign(__risk_rank=score)
    sort_columns = ["__risk_rank"]
    ascending = [True]
    if "trade_date" in ranked.columns:
        sort_columns.append("trade_date")
        ascending.append(False)
    return ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop(columns=["__risk_rank"])


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if not _is_sensitive_key(str(key))
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _compact_dataframe(value)
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _neutralize_forbidden_terms(markdown: str) -> str:
    sanitized = str(markdown)
    for term, replacement in FORBIDDEN_OUTPUT_TERMS.items():
        sanitized = sanitized.replace(term, replacement)
    return sanitized
