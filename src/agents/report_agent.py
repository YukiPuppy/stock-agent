"""ReportAgent turns structured local outputs into a Chinese Markdown summary."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


SENSITIVE_KEY_PARTS = ("token", "api_key", "apikey", "password", "passwd", "secret", "credential")
FORBIDDEN_OUTPUT_TERMS = ("保证盈利", "稳赚", "满仓", "自动下单")


def build_report_agent_context(
    system_health: pd.DataFrame | dict | None = None,
    data_quality: pd.DataFrame | None = None,
    strategy_admission: pd.DataFrame | None = None,
    trade_plan: pd.DataFrame | None = None,
    candidate_pool: pd.DataFrame | None = None,
) -> dict:
    return {
        "system_health": _compact_value(system_health),
        "data_quality": _compact_dataframe(data_quality),
        "strategy_admission": _compact_dataframe(strategy_admission),
        "trade_plan": _compact_dataframe(trade_plan),
        "candidate_pool": _compact_dataframe(candidate_pool),
    }


def build_report_agent_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 ReportAgent，只负责读取结构化结果和已有报告，生成自然语言总结。"
        "你不参与选股、调参、策略启用、下单或任何自动交易决策。\n\n"
        "总结范围仅限：系统状态、数据质量、策略研究、交易计划风险。\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 系统状态概览\n"
        "## 数据质量摘要\n"
        "## 策略研究摘要\n"
        "## 日度计划摘要\n"
        "## 主要风险\n"
        "## 下一步建议\n\n"
        "硬性约束：\n"
        "- 不得承诺收益，不得使用“保证盈利”“稳赚”“满仓”。\n"
        "- 不得直接给出自动交易指令。\n"
        "- 不得建议绕过风控。\n"
        "- 不得修改 active_strategies.json。\n"
        "- 不得输出 API key、token 或任何凭证明文。\n"
        "- 只能做信息总结和风险提示，所有计划都应表述为人工复核参考。\n\n"
        "上下文：\n"
        f"{context_text}"
    )


def run_report_agent(
    llm_client,
    system_health=None,
    data_quality=None,
    strategy_admission=None,
    trade_plan=None,
    candidate_pool=None,
) -> str:
    context = build_report_agent_context(
        system_health=system_health,
        data_quality=data_quality,
        strategy_admission=strategy_admission,
        trade_plan=trade_plan,
        candidate_pool=candidate_pool,
    )
    prompt = build_report_agent_prompt(context)
    markdown = str(llm_client.generate(prompt))
    _raise_for_forbidden_terms(markdown)
    return markdown


def _compact_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _compact_dataframe(value)
    if isinstance(value, dict):
        return _sanitize_value(value)
    if value is None:
        return {"is_empty": True, "rows": []}
    return _sanitize_value(value)


def _compact_dataframe(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {"is_empty": True, "row_count": 0, "columns": [], "rows": []}
    columns = [str(column) for column in df.columns]
    if df.empty:
        return {"is_empty": True, "row_count": 0, "columns": columns, "rows": []}
    compact = df.head(20).copy()
    safe_columns = [column for column in compact.columns if not _is_sensitive_key(str(column))]
    compact = compact.loc[:, safe_columns]
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "columns": [str(column) for column in compact.columns],
        "rows": _sanitize_value(compact.to_dict(orient="records")),
    }


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


def _raise_for_forbidden_terms(markdown: str) -> None:
    found = [term for term in FORBIDDEN_OUTPUT_TERMS if term in markdown]
    if found:
        raise ValueError(f"ReportAgent output contains forbidden expression(s): {', '.join(found)}")
