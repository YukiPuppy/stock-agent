"""DailyReviewAgent reviews actual trade execution discipline with an LLM."""

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
    "保证盈利": "收益存在不确定性，需按执行纪律复核",
    "稳赚": "收益存在不确定性",
    "满仓": "需控制仓位并遵守风险约束",
    "自动下单": "仅供人工复核参考，不执行交易",
}


def build_daily_review_agent_context(
    actual_trades: pd.DataFrame | None = None,
    execution_review: pd.DataFrame | None = None,
    daily_review: pd.DataFrame | None = None,
    period_review: pd.DataFrame | None = None,
    actual_trade_performance: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    position_review: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized context focused on execution review records."""
    return {
        "agent_scope": "只做交易复盘和执行纪律分析，不做交易决策；不选股、不调参、不启用策略、不下单。",
        "actual_trades": _compact_dataframe(actual_trades),
        "execution_review": _compact_dataframe(
            execution_review,
            risk_column="execution_status",
            risk_values=["deviation", "off_plan"],
        ),
        "daily_review": _compact_dataframe(daily_review, ascending_numeric_column="execution_score"),
        "period_review": _compact_dataframe(
            period_review,
            descending_numeric_columns=[
                "off_plan_count",
                "deviation_count",
                "chase_count",
                "over_position_count",
            ],
        ),
        "actual_trade_performance": _compact_actual_trade_performance(actual_trade_performance),
        "positions": _compact_dataframe(
            positions,
            risk_column="position_risk_level",
            risk_values=["high", "medium"],
        ),
        "position_review": _compact_dataframe(
            position_review,
            risk_column="position_risk_level",
            risk_values=["high", "medium"],
        ),
    }


def build_daily_review_agent_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 DailyReviewAgent。"
        "你只做交易复盘和执行纪律分析，不做交易决策。"
        "你只分析用户实际交易执行情况、执行偏差复盘、每日复盘、周期复盘、交易后表现和持仓风险。"
        "你不得直接选股、不得调参、不得自动启用策略、不得下单。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、当日执行总体评价\n"
        "## 二、计划内与计划外交易分析\n"
        "## 三、追高、超仓和偏离计划问题\n"
        "## 四、实际交易表现观察\n"
        "## 五、持仓与 T+1 风险\n"
        "## 六、周期执行纪律趋势\n"
        "## 七、主要问题归因\n"
        "## 八、下一交易日执行纪律建议\n\n"
        "归因要求：\n"
        "- 明确区分策略问题、执行问题、数据样本不足问题。\n"
        "- 样本不足时必须说明结论置信度有限，不得过度归因。\n"
        "- 对计划外交易、追高、超仓、偏离计划和持仓风险优先分析。\n\n"
        "硬性约束：\n"
        "- 不得承诺收益。\n"
        "- 不得使用“保证盈利”“稳赚”“满仓”“自动下单”。\n"
        "- 不得建议绕过止损、仓位限制或风控。\n"
        "- 不得直接给出新的买入股票。\n"
        "- 不得直接修改策略参数。\n"
        "- 不得直接启用策略。\n"
        "- 不得输出 API key、token、账号、密码或任何凭证明文。\n"
        "- 所有建议都必须表述为人工复核参考，不构成交易指令。\n\n"
        "上下文：\n"
        f"{context_text}"
    )


def run_daily_review_agent(
    llm_client,
    actual_trades=None,
    execution_review=None,
    daily_review=None,
    period_review=None,
    actual_trade_performance=None,
    positions=None,
    position_review=None,
) -> str:
    context = build_daily_review_agent_context(
        actual_trades=actual_trades,
        execution_review=execution_review,
        daily_review=daily_review,
        period_review=period_review,
        actual_trade_performance=actual_trade_performance,
        positions=positions,
        position_review=position_review,
    )
    prompt = build_daily_review_agent_prompt(context)
    markdown = str(llm_client.generate(prompt))
    return _neutralize_forbidden_terms(markdown)


def _compact_dataframe(
    df: pd.DataFrame | None,
    risk_column: str | None = None,
    risk_values: list[str] | None = None,
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

    safe = _sort_for_risk(
        safe,
        risk_column=risk_column,
        risk_values=risk_values or [],
        ascending_numeric_column=ascending_numeric_column,
        descending_numeric_columns=descending_numeric_columns or [],
    )
    compact = safe.head(30).copy()
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "sort_by": _sort_description(risk_column, ascending_numeric_column, descending_numeric_columns),
        "columns": [str(column) for column in compact.columns],
        "rows": _sanitize_value(compact.to_dict(orient="records")),
    }


def _compact_actual_trade_performance(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {"is_empty": True, "row_count": 0, "columns": [], "rows": []}

    safe = df.copy()
    safe_columns = [column for column in safe.columns if not _is_sensitive_key(str(column))]
    safe = safe.loc[:, safe_columns]
    columns = [str(column) for column in safe.columns]
    if safe.empty:
        return {"is_empty": True, "row_count": int(len(df)), "columns": columns, "rows": []}

    if "is_valid" in safe.columns and "return_3d" in safe.columns:
        valid = safe[safe["is_valid"].fillna(False).astype(bool)].copy()
        invalid = safe[~safe["is_valid"].fillna(False).astype(bool)].copy()
        if not valid.empty:
            valid["__return_3d"] = pd.to_numeric(valid["return_3d"], errors="coerce")
            weak = valid.sort_values("__return_3d", ascending=True, na_position="last", kind="mergesort")
            strong = valid.sort_values("__return_3d", ascending=False, na_position="last", kind="mergesort")
            safe = pd.concat(
                [
                    weak.head(20),
                    strong.head(10),
                    invalid.head(max(0, 30 - min(30, len(valid)))),
                ],
                ignore_index=True,
            ).drop_duplicates()
            if "__return_3d" in safe.columns:
                safe = safe.drop(columns=["__return_3d"])
        else:
            safe = invalid
    elif "return_3d" in safe.columns:
        safe = safe.assign(__return_3d=pd.to_numeric(safe["return_3d"], errors="coerce"))
        safe = safe.sort_values("__return_3d", ascending=True, na_position="last", kind="mergesort").drop(
            columns=["__return_3d"]
        )

    compact = safe.head(30).copy()
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "sort_by": "is_valid_true_return_3d_weak_first_with_strong_samples",
        "columns": [str(column) for column in compact.columns],
        "rows": _sanitize_value(compact.to_dict(orient="records")),
    }


def _sort_for_risk(
    df: pd.DataFrame,
    risk_column: str | None,
    risk_values: list[str],
    ascending_numeric_column: str | None,
    descending_numeric_columns: list[str],
) -> pd.DataFrame:
    if risk_column and risk_column in df.columns:
        priority = {str(value).lower(): index for index, value in enumerate(risk_values)}
        ranked = df.assign(
            __risk_rank=df[risk_column].fillna("").astype(str).str.lower().map(priority).fillna(len(priority))
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
    ascending_numeric_column: str | None,
    descending_numeric_columns: list[str] | None,
) -> str:
    if risk_column:
        return f"{risk_column}_risk_priority"
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
