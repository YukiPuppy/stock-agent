"""RiskReviewAgent reviews system-level risks with an LLM."""

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
    "保证盈利": "收益存在不确定性，需按风控流程复核",
    "稳赚": "收益存在不确定性",
    "满仓": "需控制仓位并遵守风险约束",
    "自动下单": "仅供人工复核参考，不执行交易",
}


def build_risk_review_context(
    system_health: dict | pd.DataFrame | None = None,
    data_quality_report: pd.DataFrame | None = None,
    strategy_admission: pd.DataFrame | None = None,
    trade_plan: pd.DataFrame | None = None,
    position_review: pd.DataFrame | None = None,
    execution_review: pd.DataFrame | None = None,
    daily_review: pd.DataFrame | None = None,
    period_review: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized context focused on higher-risk records."""
    return {
        "agent_scope": "只做系统风险审查，不直接选股、不直接调参、不自动启用策略、不自动下单。",
        "system_health": _compact_value(system_health),
        "data_quality_report": _compact_dataframe(
            data_quality_report,
            risk_column="status",
            risk_values=["error", "warning"],
        ),
        "strategy_admission": _compact_dataframe(
            strategy_admission,
            risk_column="admission_recommendation",
            risk_values=["do_not_enable", "continue_research"],
        ),
        "trade_plan": _compact_dataframe(trade_plan, nonempty_column="risk_flags"),
        "position_review": _compact_dataframe(
            position_review,
            risk_column="position_risk_level",
            risk_values=["high", "medium"],
        ),
        "execution_review": _compact_dataframe(
            execution_review,
            risk_column="execution_status",
            risk_values=["deviation", "off_plan"],
        ),
        "daily_review": _compact_dataframe(daily_review, ascending_numeric_column="execution_score"),
        "period_review": _compact_dataframe(
            period_review,
            descending_numeric_columns=["off_plan_count", "deviation_count"],
        ),
    }


def build_risk_review_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 RiskReviewAgent。"
        "你只做风险审查，不做交易决策；你不直接选股、不直接调参、不自动启用策略、不下单。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、总体风险结论\n"
        "## 二、数据质量风险\n"
        "## 三、策略准入风险\n"
        "## 四、交易计划风险\n"
        "## 五、持仓与 T+1 风险\n"
        "## 六、执行偏差风险\n"
        "## 七、需要立即关注的问题\n"
        "## 八、下一步风险控制建议\n\n"
        "硬性约束：\n"
        "- 不得承诺收益。\n"
        "- 不得使用“保证盈利”“稳赚”“满仓”“自动下单”。\n"
        "- 不得建议绕过止损、仓位限制或风控。\n"
        "- 不得直接修改策略参数。\n"
        "- 不得直接启用策略。\n"
        "- 不得输出 API key、token、账号、密码或任何凭证明文。\n"
        "- 所有建议都必须表述为人工复核参考，不构成交易指令。\n\n"
        "上下文：\n"
        f"{context_text}"
    )


def run_risk_review_agent(
    llm_client,
    system_health=None,
    data_quality_report=None,
    strategy_admission=None,
    trade_plan=None,
    position_review=None,
    execution_review=None,
    daily_review=None,
    period_review=None,
) -> str:
    context = build_risk_review_context(
        system_health=system_health,
        data_quality_report=data_quality_report,
        strategy_admission=strategy_admission,
        trade_plan=trade_plan,
        position_review=position_review,
        execution_review=execution_review,
        daily_review=daily_review,
        period_review=period_review,
    )
    prompt = build_risk_review_prompt(context)
    markdown = str(llm_client.generate(prompt))
    return _neutralize_forbidden_terms(markdown)


def _compact_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _compact_dataframe(value)
    if isinstance(value, dict):
        return _sanitize_value(value)
    if value is None:
        return {"is_empty": True, "rows": []}
    return _sanitize_value(value)


def _compact_dataframe(
    df: pd.DataFrame | None,
    risk_column: str | None = None,
    risk_values: list[str] | None = None,
    nonempty_column: str | None = None,
    ascending_numeric_column: str | None = None,
    descending_numeric_columns: list[str] | None = None,
) -> dict:
    if df is None:
        return {"is_empty": True, "row_count": 0, "columns": [], "rows": []}

    safe = df.copy()
    safe_columns = [column for column in safe.columns if not _is_sensitive_key(str(column))]
    safe = safe.loc[:, safe_columns]
    columns = [str(column) for column in safe.columns]
    if safe.empty:
        return {"is_empty": True, "row_count": int(len(df)), "columns": columns, "rows": []}

    sort_by = ""
    safe = _sort_for_risk(
        safe,
        risk_column=risk_column,
        risk_values=risk_values or [],
        nonempty_column=nonempty_column,
        ascending_numeric_column=ascending_numeric_column,
        descending_numeric_columns=descending_numeric_columns or [],
    )
    sort_by = _sort_description(risk_column, nonempty_column, ascending_numeric_column, descending_numeric_columns)

    compact = safe.head(30).copy()
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "sort_by": sort_by,
        "columns": [str(column) for column in compact.columns],
        "rows": _sanitize_value(compact.to_dict(orient="records")),
    }


def _sort_for_risk(
    df: pd.DataFrame,
    risk_column: str | None,
    risk_values: list[str],
    nonempty_column: str | None,
    ascending_numeric_column: str | None,
    descending_numeric_columns: list[str],
) -> pd.DataFrame:
    if risk_column and risk_column in df.columns:
        priority = {str(value).lower(): index for index, value in enumerate(risk_values)}
        ranked = df.assign(
            __risk_rank=df[risk_column].fillna("").astype(str).str.lower().map(priority).fillna(len(priority))
        )
        return ranked.sort_values("__risk_rank", ascending=True, kind="mergesort").drop(columns=["__risk_rank"])

    if nonempty_column and nonempty_column in df.columns:
        ranked = df.assign(
            __risk_rank=df[nonempty_column].fillna("").astype(str).str.strip().map(lambda value: 0 if value else 1)
        )
        return ranked.sort_values("__risk_rank", ascending=True, kind="mergesort").drop(columns=["__risk_rank"])

    if ascending_numeric_column and ascending_numeric_column in df.columns:
        ranked = df.assign(__risk_rank=pd.to_numeric(df[ascending_numeric_column], errors="coerce"))
        return ranked.sort_values("__risk_rank", ascending=True, na_position="last", kind="mergesort").drop(
            columns=["__risk_rank"]
        )

    available_desc = [column for column in descending_numeric_columns if column in df.columns]
    if available_desc:
        ranked = df.copy()
        for column in available_desc:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
        return ranked.sort_values(available_desc, ascending=[False] * len(available_desc), na_position="last")

    return df


def _sort_description(
    risk_column: str | None,
    nonempty_column: str | None,
    ascending_numeric_column: str | None,
    descending_numeric_columns: list[str] | None,
) -> str:
    if risk_column:
        return f"{risk_column}_risk_priority"
    if nonempty_column:
        return f"{nonempty_column}_nonempty_first"
    if ascending_numeric_column:
        return f"{ascending_numeric_column}_ascending"
    if descending_numeric_columns:
        return ",".join(descending_numeric_columns) + "_descending"
    return ""


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
