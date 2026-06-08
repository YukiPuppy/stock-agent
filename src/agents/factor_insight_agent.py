"""FactorInsightAgent explains factor diagnostics with an LLM."""

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
    "保证盈利": "收益存在不确定性，需经过验证",
    "稳赚": "收益存在不确定性",
    "满仓": "需控制仓位并遵守风险约束",
    "自动下单": "仅供人工复核参考，不执行交易",
}
FACTOR_CANDIDATE_KEYWORDS = [
    "missing_daily_basic",
    "missing_moneyflow",
    "missing_industry_strength",
    "strong_main_inflow",
    "strong_main_outflow",
    "strong_industry",
    "weak_industry",
]
TRADE_PLAN_FACTOR_KEYWORDS = ["资金流", "行业", "市场", "缺失"]


def build_factor_insight_context(
    factor_diagnostics: pd.DataFrame | None = None,
    daily_factors: pd.DataFrame | None = None,
    candidate_pool: pd.DataFrame | None = None,
    trade_plan: pd.DataFrame | None = None,
    strategy_admission: pd.DataFrame | None = None,
    trade_plan_backtest_performance: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized context focused on factor coverage and usage."""
    return {
        "agent_scope": "只做因子诊断和研究建议，不做交易决策；不直接选股、不调参、不自动启用策略、不自动下单。",
        "factor_diagnostics": _compact_dataframe(factor_diagnostics, sort_mode="factor_diagnostics_priority"),
        "daily_factors": _compact_dataframe(daily_factors),
        "candidate_pool": _compact_dataframe(candidate_pool, sort_mode="candidate_factor_risk_first"),
        "trade_plan": _compact_dataframe(trade_plan, sort_mode="trade_plan_factor_reason_first"),
        "strategy_admission": _compact_dataframe(strategy_admission),
        "trade_plan_backtest_performance": _compact_dataframe(trade_plan_backtest_performance),
    }


def build_factor_insight_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 FactorInsightAgent。"
        "你只做因子覆盖率、缺失率、分布、候选池和交易计划使用情况诊断，以及下一轮研究建议；"
        "不做交易决策，不直接选股，不直接调参，不启用策略，不执行交易。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、因子覆盖情况概览\n"
        "## 二、缺失率较高的因子\n"
        "## 三、候选池中的因子特征\n"
        "## 四、交易计划中的因子特征\n"
        "## 五、资金流、行业强度、市场环境因子的观察\n"
        "## 六、可能存在的问题\n"
        "## 七、下一轮因子研究建议\n\n"
        "必须说明：\n"
        "- 当前诊断不等于因子有效性证明。\n"
        "- 因子是否有效需要通过回测、样本外验证和交易计划级回测确认。\n"
        "- 小样本结果不能直接用于实盘判断。\n\n"
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


def run_factor_insight_agent(
    llm_client,
    factor_diagnostics=None,
    daily_factors=None,
    candidate_pool=None,
    trade_plan=None,
    strategy_admission=None,
    trade_plan_backtest_performance=None,
) -> str:
    context = build_factor_insight_context(
        factor_diagnostics=factor_diagnostics,
        daily_factors=daily_factors,
        candidate_pool=candidate_pool,
        trade_plan=trade_plan,
        strategy_admission=strategy_admission,
        trade_plan_backtest_performance=trade_plan_backtest_performance,
    )
    prompt = build_factor_insight_prompt(context)
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
    if sort_mode == "factor_diagnostics_priority":
        return _sort_factor_diagnostics(df)
    if sort_mode == "candidate_factor_risk_first":
        return _sort_by_contains(df, ["risk_flags"], FACTOR_CANDIDATE_KEYWORDS)
    if sort_mode == "trade_plan_factor_reason_first":
        return _sort_by_contains(df, ["plan_reason"], TRADE_PLAN_FACTOR_KEYWORDS)
    return df


def _sort_factor_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    status = ranked.get("diagnostic_status", pd.Series("", index=ranked.index)).fillna("").astype(str)
    missing_rate = pd.to_numeric(ranked.get("missing_rate", pd.Series(0, index=ranked.index)), errors="coerce").fillna(0)
    mean = pd.to_numeric(ranked.get("mean", pd.Series(0, index=ranked.index)), errors="coerce")
    candidate_mean = pd.to_numeric(ranked.get("candidate_mean", pd.Series(pd.NA, index=ranked.index)), errors="coerce")
    trade_plan_mean = pd.to_numeric(ranked.get("trade_plan_mean", pd.Series(pd.NA, index=ranked.index)), errors="coerce")
    scale = mean.abs().where(mean.abs() > 1e-9, 1.0)
    diff_score = ((candidate_mean - mean).abs().fillna(0) + (trade_plan_mean - mean).abs().fillna(0)) / scale
    ranked["__status_rank"] = status.eq("ok").astype(int)
    ranked["__missing_rate"] = missing_rate
    ranked["__diff_score"] = diff_score
    return ranked.sort_values(
        ["__status_rank", "__missing_rate", "__diff_score"],
        ascending=[True, False, False],
        kind="mergesort",
    ).drop(columns=["__status_rank", "__missing_rate", "__diff_score"])


def _sort_by_contains(df: pd.DataFrame, columns: list[str], keywords: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    if not available:
        return df
    score = pd.Series(1, index=df.index)
    for column in available:
        text = df[column].fillna("").astype(str)
        for keyword in keywords:
            score = score.mask(text.str.contains(keyword, case=False, regex=False), 0)
    ranked = df.assign(__factor_rank=score)
    sort_columns = ["__factor_rank"]
    ascending = [True]
    if "trade_date" in ranked.columns:
        sort_columns.append("trade_date")
        ascending.append(False)
    if "rank" in ranked.columns:
        sort_columns.append("rank")
        ascending.append(True)
    return ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop(columns=["__factor_rank"])


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
