"""IndustryInsightAgent explains industry strength and sector resonance with an LLM."""

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
INDUSTRY_RISK_KEYWORDS = [
    "strong_industry",
    "weak_industry",
    "missing_industry_strength",
    "行业",
    "板块",
    "industry",
]


def build_industry_insight_context(
    industry_strength: pd.DataFrame | None = None,
    sw_daily: pd.DataFrame | None = None,
    stock_industry_map: pd.DataFrame | None = None,
    candidate_pool: pd.DataFrame | None = None,
    trade_plan: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized context focused on industry strength."""
    return {
        "agent_scope": "只做行业强弱解释，不做交易决策；不直接选股、不调参、不自动启用策略、不自动下单。",
        "industry_strength": _compact_dataframe(industry_strength, sort_mode="industry_strength_priority"),
        "sw_daily": _compact_dataframe(sw_daily, sort_mode="latest_trade_date"),
        "stock_industry_map": _compact_dataframe(stock_industry_map),
        "candidate_pool": _compact_dataframe(candidate_pool, sort_mode="candidate_industry_risk_first"),
        "trade_plan": _compact_dataframe(trade_plan, sort_mode="trade_plan_industry_risk_first"),
    }


def build_industry_insight_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 IndustryInsightAgent。"
        "你只做行业强弱解释，不做交易决策；不直接选股、不调参、不自动启用策略、不自动下单。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、行业强弱总体判断\n"
        "## 二、强势行业分析\n"
        "## 三、弱势行业风险\n"
        "## 四、候选池中的行业共振情况\n"
        "## 五、交易计划中的行业风险提示\n"
        "## 六、需要继续观察的行业信号\n"
        "## 七、下一步研究建议\n\n"
        "解释约束：\n"
        "- 行业强势不代表个股一定上涨。\n"
        "- 行业弱势不代表个股一定下跌。\n"
        "- 行业强度只用于调整策略置信度和风险提示。\n\n"
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


def run_industry_insight_agent(
    llm_client,
    industry_strength=None,
    sw_daily=None,
    stock_industry_map=None,
    candidate_pool=None,
    trade_plan=None,
) -> str:
    context = build_industry_insight_context(
        industry_strength=industry_strength,
        sw_daily=sw_daily,
        stock_industry_map=stock_industry_map,
        candidate_pool=candidate_pool,
        trade_plan=trade_plan,
    )
    prompt = build_industry_insight_prompt(context)
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
    if sort_mode == "industry_strength_priority":
        return _sort_industry_strength(df)
    if sort_mode == "latest_trade_date" and "trade_date" in df.columns:
        return df.sort_values("trade_date", ascending=False, kind="mergesort")
    if sort_mode == "candidate_industry_risk_first":
        return _sort_by_contains(df, ["risk_flags"], INDUSTRY_RISK_KEYWORDS)
    if sort_mode == "trade_plan_industry_risk_first":
        return _sort_by_contains(df, ["plan_reason", "risk_flags"], INDUSTRY_RISK_KEYWORDS)
    return df


def _sort_industry_strength(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.copy()
    if "trade_date" in latest.columns:
        dates = latest["trade_date"].dropna().astype(str)
        if not dates.empty:
            latest_date = str(dates.max())
            latest = latest[latest["trade_date"].astype(str) == latest_date].copy()

    priority = pd.Series(3, index=latest.index)
    if "industry_strength_level" in latest.columns:
        levels = latest["industry_strength_level"].fillna("").astype(str)
        priority = priority.mask(levels == "strong", 0)
        priority = priority.mask(levels == "weak", 1)

    ranked = latest.assign(__priority=priority)
    sort_columns = ["__priority"]
    ascending = [True]
    if "industry_strength_score" in ranked.columns:
        ranked["__score_rank"] = ranked["industry_strength_score"].rank(method="first", ascending=False)
        score_count = len(ranked)
        score_priority = pd.Series(2, index=ranked.index)
        score_priority = score_priority.mask(ranked["__score_rank"] <= 10, 0)
        score_priority = score_priority.mask(ranked["__score_rank"] > max(score_count - 10, 0), 1)
        ranked["__score_priority"] = score_priority
        sort_columns.extend(["__score_priority", "industry_strength_score"])
        ascending.extend([True, False])
    if "trade_date" in ranked.columns:
        sort_columns.append("trade_date")
        ascending.append(False)
    return ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop(
        columns=[column for column in ["__priority", "__score_rank", "__score_priority"] if column in ranked.columns]
    )


def _sort_by_contains(df: pd.DataFrame, columns: list[str], keywords: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    if not available:
        return df
    score = pd.Series(1, index=df.index)
    for column in available:
        text = df[column].fillna("").astype(str)
        for keyword in keywords:
            score = score.mask(text.str.contains(keyword, case=False, regex=False), 0)
    ranked = df.assign(__industry_rank=score)
    sort_columns = ["__industry_rank"]
    ascending = [True]
    if "trade_date" in ranked.columns:
        sort_columns.append("trade_date")
        ascending.append(False)
    if "rank" in ranked.columns:
        sort_columns.append("rank")
        ascending.append(True)
    return ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop(columns=["__industry_rank"])


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
    if isinstance(value, str) and _contains_sensitive_marker(value):
        return "[redacted]"
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _contains_sensitive_marker(value: str) -> bool:
    lowered = value.strip().lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _neutralize_forbidden_terms(markdown: str) -> str:
    sanitized = str(markdown)
    for term, replacement in FORBIDDEN_OUTPUT_TERMS.items():
        sanitized = sanitized.replace(term, replacement)
    return sanitized
