"""StrategyResearchAgent proposes next-round strategy research candidates."""

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
ADMISSION_PRIORITY = ["do_not_enable", "continue_research", "observation_candidate"]
FACTOR_DIAGNOSTIC_PRIORITY = ["high_missing", "medium_missing", "missing_column"]
INDUSTRY_PRIORITY = ["strong", "weak"]
MONEYFLOW_PRIORITY = ["strong_main_outflow", "strong_main_inflow"]
TRADE_PLAN_PERFORMANCE_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "plan_date",
    "trade_date",
    "avg_return",
    "win_rate",
    "max_drawdown",
    "trigger_rate",
    "trade_count",
    "sample_count",
    "is_valid",
]


def build_strategy_research_context(
    strategy_evaluation: pd.DataFrame | None = None,
    parameter_search_results: pd.DataFrame | None = None,
    walk_forward_validation: pd.DataFrame | None = None,
    trade_plan_backtest_performance: pd.DataFrame | None = None,
    strategy_admission: pd.DataFrame | None = None,
    factor_diagnostics: pd.DataFrame | None = None,
    market_regime: pd.DataFrame | None = None,
    industry_strength: pd.DataFrame | None = None,
    moneyflow_factors: pd.DataFrame | None = None,
) -> dict:
    """Build a compact, sanitized research context for LLM strategy ideation."""
    return {
        "agent_scope": (
            "StrategyResearchAgent 只做策略研究建议，不做交易决策；"
            "不直接修改策略配置，不启用策略，不推荐买入股票，不自动下单。"
        ),
        "strategy_evaluation": _compact_dataframe(strategy_evaluation, sort_mode="evaluation_high_low"),
        "parameter_search_results": _compact_dataframe(parameter_search_results, sort_mode="evaluation_high"),
        "walk_forward_validation": _compact_dataframe(walk_forward_validation, sort_mode="stability_low_high"),
        "trade_plan_backtest_performance": _compact_dataframe(
            trade_plan_backtest_performance,
            sort_mode="trade_plan_backtest_key_fields",
            preferred_columns=TRADE_PLAN_PERFORMANCE_COLUMNS,
        ),
        "strategy_admission": _compact_dataframe(strategy_admission, sort_mode="strategy_admission_priority"),
        "factor_diagnostics": _compact_dataframe(factor_diagnostics, sort_mode="factor_diagnostics_priority"),
        "market_regime": _compact_dataframe(market_regime, sort_mode="latest_trade_date"),
        "industry_strength": _compact_dataframe(industry_strength, sort_mode="industry_strength_priority"),
        "moneyflow_factors": _compact_dataframe(moneyflow_factors, sort_mode="moneyflow_priority"),
    }


def build_strategy_research_prompt(context: dict) -> str:
    context_text = json.dumps(_sanitize_value(context), ensure_ascii=False, indent=2, default=str)
    return (
        "你是 stock-agent 项目的 StrategyResearchAgent。"
        "你只做策略研究建议，不做交易决策。"
        "你的输出只能提出 candidate 级别的下一轮策略研究假设和参数研究方向；"
        "不能直接修改策略配置，不能直接启用策略，不能推荐买入股票，不能执行交易。\n\n"
        "请基于以下上下文输出中文 Markdown，结构必须包括：\n"
        "## 一、当前策略研究状态判断\n"
        "## 二、现有策略主要问题\n"
        "## 三、可能的策略改进方向\n"
        "## 四、可新增的策略假设\n"
        "## 五、建议纳入下一轮参数搜索的变量\n"
        "## 六、需要优先验证的风控条件\n"
        "## 七、样本不足与过拟合风险\n"
        "## 八、下一轮研究计划\n\n"
        "必须强调：\n"
        "- 所有建议必须经过回测、样本外验证、交易计划级回测。\n"
        "- 小样本结果不能用于实盘判断。\n"
        "- Agent 建议不能直接进入正式策略配置，必须经过人工确认。\n\n"
        "硬性禁止：\n"
        "- 不得承诺收益，不得使用“保证盈利”“稳赚”“满仓”“自动下单”。\n"
        "- 不得直接给出新的买入股票。\n"
        "- 不得直接修改 active_strategies.json。\n"
        "- 不得直接修改 active_strategies_candidate.json。\n"
        "- 不得直接修改 strategy_versions.json。\n"
        "- 不得直接修改 parameter_search_space.json。\n"
        "- 不得建议绕过风控。\n"
        "- 不得输出 API key、token、账号、密码或任何凭证明文。\n\n"
        "上下文：\n"
        f"{context_text}"
    )


def run_strategy_research_agent(
    llm_client,
    strategy_evaluation=None,
    parameter_search_results=None,
    walk_forward_validation=None,
    trade_plan_backtest_performance=None,
    strategy_admission=None,
    factor_diagnostics=None,
    market_regime=None,
    industry_strength=None,
    moneyflow_factors=None,
) -> str:
    context = build_strategy_research_context(
        strategy_evaluation=strategy_evaluation,
        parameter_search_results=parameter_search_results,
        walk_forward_validation=walk_forward_validation,
        trade_plan_backtest_performance=trade_plan_backtest_performance,
        strategy_admission=strategy_admission,
        factor_diagnostics=factor_diagnostics,
        market_regime=market_regime,
        industry_strength=industry_strength,
        moneyflow_factors=moneyflow_factors,
    )
    prompt = build_strategy_research_prompt(context)
    markdown = str(llm_client.generate(prompt))
    return _neutralize_forbidden_terms(markdown)


def _compact_dataframe(
    df: pd.DataFrame | None,
    sort_mode: str = "",
    preferred_columns: list[str] | None = None,
) -> dict:
    if df is None:
        return {"is_empty": True, "row_count": 0, "columns": [], "rows": []}

    safe = df.copy()
    safe_columns = [column for column in safe.columns if not _is_sensitive_key(str(column))]
    safe = safe.loc[:, safe_columns]
    if preferred_columns:
        preferred = [column for column in preferred_columns if column in safe.columns]
        remaining = [column for column in safe.columns if column not in preferred]
        safe = safe.loc[:, preferred + remaining]
    columns = [str(column) for column in safe.columns]
    if safe.empty:
        return {"is_empty": True, "row_count": int(len(df)), "columns": columns, "rows": []}

    compact = _sort_dataframe(safe, sort_mode).head(30).copy()
    return {
        "is_empty": False,
        "row_count": int(len(df)),
        "included_rows": int(len(compact)),
        "sort_by": sort_mode,
        "columns": [str(column) for column in compact.columns],
        "rows": _sanitize_value(compact.to_dict(orient="records")),
    }


def _sort_dataframe(df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if sort_mode == "evaluation_high_low":
        return _head_tail_by_score(df, "evaluation_score")
    if sort_mode == "evaluation_high":
        return _sort_numeric(df, "evaluation_score", ascending=False)
    if sort_mode == "stability_low_high":
        return _head_tail_by_score(df, "stability_score", low_first=True)
    if sort_mode == "strategy_admission_priority":
        return _sort_by_status_priority(df, "admission_status", ADMISSION_PRIORITY)
    if sort_mode == "factor_diagnostics_priority":
        return _sort_by_status_priority(df, "diagnostic_status", FACTOR_DIAGNOSTIC_PRIORITY)
    if sort_mode == "latest_trade_date":
        return _sort_date_desc(df)
    if sort_mode == "industry_strength_priority":
        return _sort_by_status_priority(df, "industry_strength_level", INDUSTRY_PRIORITY)
    if sort_mode == "moneyflow_priority":
        return _sort_moneyflow_priority(df)
    if sort_mode == "trade_plan_backtest_key_fields":
        return _sort_numeric(df, "avg_return", ascending=False)
    return df


def _head_tail_by_score(df: pd.DataFrame, column: str, low_first: bool = False) -> pd.DataFrame:
    if column not in df.columns:
        return df
    scored = df.assign(__score=pd.to_numeric(df[column], errors="coerce"))
    low = scored.sort_values("__score", ascending=True, kind="mergesort").head(15)
    high = scored.sort_values("__score", ascending=False, kind="mergesort").head(15)
    combined = pd.concat([low, high] if low_first else [high, low])
    return combined.loc[~combined.index.duplicated(keep="first")].drop(columns=["__score"])


def _sort_numeric(df: pd.DataFrame, column: str, ascending: bool) -> pd.DataFrame:
    if column not in df.columns:
        return df
    return df.assign(__score=pd.to_numeric(df[column], errors="coerce")).sort_values(
        "__score",
        ascending=ascending,
        kind="mergesort",
    ).drop(columns=["__score"])


def _sort_by_status_priority(df: pd.DataFrame, column: str, priorities: list[str]) -> pd.DataFrame:
    if column not in df.columns:
        return df
    values = df[column].fillna("").astype(str)
    rank = pd.Series(len(priorities), index=df.index)
    for index, value in enumerate(priorities):
        rank = rank.mask(values == value, index)
    ranked = df.assign(__priority_rank=rank)
    sort_columns = ["__priority_rank"]
    ascending = [True]
    if "trade_date" in ranked.columns:
        sort_columns.append("trade_date")
        ascending.append(False)
    return ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop(columns=["__priority_rank"])


def _sort_date_desc(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["trade_date", "as_of_date", "plan_date"]:
        if column in df.columns:
            return df.sort_values(column, ascending=False, kind="mergesort")
    return df


def _sort_moneyflow_priority(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["moneyflow_signal", "moneyflow_risk_flags", "risk_flags"] if column in df.columns]
    if not columns:
        return _sort_date_desc(df)
    score = pd.Series(len(MONEYFLOW_PRIORITY), index=df.index)
    for column in columns:
        text = df[column].fillna("").astype(str)
        for index, keyword in enumerate(MONEYFLOW_PRIORITY):
            score = score.mask(text.str.contains(keyword, case=False, regex=False), index)
    ranked = df.assign(__moneyflow_rank=score)
    sort_columns = ["__moneyflow_rank"]
    ascending = [True]
    if "trade_date" in ranked.columns:
        sort_columns.append("trade_date")
        ascending.append(False)
    return ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop(columns=["__moneyflow_rank"])


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items() if not _is_sensitive_key(str(key))}
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
