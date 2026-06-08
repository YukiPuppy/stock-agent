"""BacktestAnalysisAgent summarizes computed backtest outputs with an LLM."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


SENSITIVE_KEY_PARTS = ("token", "api_key", "apikey", "password", "passwd", "secret", "credential", "account")
FORBIDDEN_OUTPUT_TERMS = {
    "保证盈利": "不应承诺收益，需关注回测与实盘偏差风险",
    "稳赚": "收益存在不确定性",
    "满仓": "需控制仓位并遵守风险约束",
    "自动下单": "仅供人工复核参考，不执行交易",
}


def build_backtest_analysis_context(
    strategy_evaluation: pd.DataFrame | None = None,
    parameter_search_results: pd.DataFrame | None = None,
    walk_forward_validation: pd.DataFrame | None = None,
    trade_plan_backtest_performance: pd.DataFrame | None = None,
    strategy_admission: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized context for LLM backtest analysis."""
    return {
        "agent_scope": "只分析程序计算出的回测结果，不计算收益、不选股、不自动调参、不启用策略、不下单。",
        "strategy_evaluation": _compact_dataframe(strategy_evaluation, sort_by="evaluation_score"),
        "parameter_search_results": _compact_dataframe(parameter_search_results, sort_by="evaluation_score"),
        "walk_forward_validation": _compact_dataframe(walk_forward_validation, sort_by="stability_score"),
        "trade_plan_backtest_performance": _compact_dataframe(
            trade_plan_backtest_performance,
            sort_by="avg_return",
        ),
        "strategy_admission": _compact_dataframe(strategy_admission, sort_by="admission_score"),
    }


def build_backtest_analysis_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 BacktestAnalysisAgent。"
        "你只做分析，不做交易决策；只分析程序已经计算出的回测、样本外验证、参数搜索和策略准入结果。"
        "你不得直接计算收益，不得选股，不得自动调参，不得自动启用策略，不得下单。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、总体结论\n"
        "## 二、策略版本表现分析\n"
        "## 三、参数搜索结果分析\n"
        "## 四、样本外验证与过拟合风险\n"
        "## 五、交易规则级回测观察\n"
        "## 六、策略准入建议解读\n"
        "## 七、主要风险\n"
        "## 八、下一轮研究建议\n\n"
        "硬性约束：\n"
        "- 不得承诺收益。\n"
        "- 不得使用“保证盈利”“稳赚”“满仓”“自动下单”。\n"
        "- 不得建议绕过风控。\n"
        "- 不得直接修改参数。\n"
        "- 不得直接启用策略。\n"
        "- 不得输出 API key、token、账号、密码或任何凭证明文。\n"
        "- 所有建议都必须表述为人工复核参考，不构成交易指令。\n\n"
        "上下文：\n"
        f"{context_text}"
    )


def run_backtest_analysis_agent(
    llm_client,
    strategy_evaluation=None,
    parameter_search_results=None,
    walk_forward_validation=None,
    trade_plan_backtest_performance=None,
    strategy_admission=None,
) -> str:
    context = build_backtest_analysis_context(
        strategy_evaluation=strategy_evaluation,
        parameter_search_results=parameter_search_results,
        walk_forward_validation=walk_forward_validation,
        trade_plan_backtest_performance=trade_plan_backtest_performance,
        strategy_admission=strategy_admission,
    )
    prompt = build_backtest_analysis_prompt(context)
    markdown = str(llm_client.generate(prompt))
    return _neutralize_forbidden_terms(markdown)


def _compact_dataframe(df: pd.DataFrame | None, sort_by: str) -> dict:
    if df is None:
        return {"is_empty": True, "row_count": 0, "columns": [], "rows": []}

    safe = df.copy()
    safe_columns = [column for column in safe.columns if not _is_sensitive_key(str(column))]
    safe = safe.loc[:, safe_columns]
    columns = [str(column) for column in safe.columns]
    if safe.empty:
        return {"is_empty": True, "row_count": int(len(df)), "columns": columns, "rows": []}

    if sort_by in safe.columns:
        safe = safe.sort_values(by=sort_by, ascending=False, na_position="last")
    compact = safe.head(30).copy()
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "sort_by": sort_by if sort_by in safe.columns else "",
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
        return _compact_dataframe(value, sort_by="")
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _neutralize_forbidden_terms(markdown: str) -> str:
    sanitized = str(markdown)
    for term, replacement in FORBIDDEN_OUTPUT_TERMS.items():
        sanitized = sanitized.replace(term, replacement)
    return sanitized
