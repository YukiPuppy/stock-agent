"""Strategy admission and observation-candidate configuration."""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


STRATEGY_ADMISSION_COLUMNS = [
    "strategy_name",
    "strategy_version",
    "source",
    "valid_count",
    "evaluation_recommendation",
    "evaluation_score",
    "oos_status",
    "oos_risk",
    "oos_stability_score",
    "trade_plan_valid_count",
    "trade_plan_trigger_rate",
    "trade_plan_win_rate",
    "trade_plan_avg_return",
    "trade_plan_avg_drawdown",
    "admission_score",
    "admission_status",
    "admission_recommendation",
    "admission_reason",
]


def build_strategy_admission(
    strategy_evaluation: pd.DataFrame | None = None,
    parameter_search_results: pd.DataFrame | None = None,
    walk_forward_validation: pd.DataFrame | None = None,
    trade_plan_backtest_performance: pd.DataFrame | None = None,
    min_valid_count: int = 30,
    min_oos_valid_count: int = 10,
    min_trade_plan_valid_count: int = 10,
) -> pd.DataFrame:
    """Build admission recommendations from local research result tables."""
    base = _build_base_candidates(strategy_evaluation, parameter_search_results)
    if base.empty:
        return pd.DataFrame(columns=STRATEGY_ADMISSION_COLUMNS)

    base = _merge_oos(base, walk_forward_validation)
    base = _merge_trade_plan_performance(base, trade_plan_backtest_performance)

    for column in [
        "valid_count",
        "evaluation_score",
        "oos_stability_score",
        "trade_plan_valid_count",
        "trade_plan_trigger_rate",
        "trade_plan_win_rate",
        "trade_plan_avg_return",
        "trade_plan_avg_drawdown",
    ]:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")

    records = []
    for _, row in base.iterrows():
        score = _admission_score(row, min_valid_count)
        status, recommendation, reason = _admission_decision(
            row,
            score,
            min_valid_count=min_valid_count,
            min_oos_valid_count=min_oos_valid_count,
            min_trade_plan_valid_count=min_trade_plan_valid_count,
        )
        record = row.to_dict()
        record["admission_score"] = round(score, 4)
        record["admission_status"] = status
        record["admission_recommendation"] = recommendation
        record["admission_reason"] = reason
        records.append(record)

    admission = pd.DataFrame(records)
    for column in STRATEGY_ADMISSION_COLUMNS:
        if column not in admission.columns:
            admission[column] = None
    admission = admission.loc[:, STRATEGY_ADMISSION_COLUMNS]
    return admission.sort_values(
        ["admission_score", "strategy_name", "strategy_version"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_active_strategy_candidate_config(admission: pd.DataFrame) -> dict:
    """Build an observation-candidate config. This is not an auto-trading config."""
    admission = admission.copy() if admission is not None else pd.DataFrame()
    if admission.empty or "admission_recommendation" not in admission.columns:
        candidates = []
    else:
        selected = admission[admission["admission_recommendation"] == "enable_observation_candidate"].copy()
        selected = selected.sort_values("admission_score", ascending=False, na_position="last")
        candidates = [
            {
                "strategy_name": row.get("strategy_name"),
                "strategy_version": row.get("strategy_version"),
                "admission_score": row.get("admission_score"),
                "admission_status": row.get("admission_status"),
                "admission_recommendation": row.get("admission_recommendation"),
                "admission_reason": row.get("admission_reason"),
            }
            for _, row in selected.iterrows()
        ]

    return {
        "note": "This is an observation candidate config, not an auto-trading config.",
        "active_strategy_candidates": candidates,
    }


def _build_base_candidates(
    strategy_evaluation: pd.DataFrame | None,
    parameter_search_results: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = [
        _normalize_evaluation(strategy_evaluation, "manual_version"),
        _normalize_evaluation(parameter_search_results, "parameter_search"),
    ]
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    combined["strategy_version"] = combined["strategy_version"].fillna("v1")
    combined["evaluation_score"] = pd.to_numeric(combined["evaluation_score"], errors="coerce")
    combined["_score_rank"] = combined["evaluation_score"].fillna(float("-inf"))

    rows = []
    for _, group in combined.groupby(["strategy_name", "strategy_version"], dropna=False):
        sources = set(group["source"].dropna().astype(str))
        best = group.sort_values("_score_rank", ascending=False).iloc[0].copy()
        if {"manual_version", "parameter_search"} <= sources:
            best["source"] = "mixed"
        rows.append(best.drop(labels=["_score_rank"]))
    return pd.DataFrame(rows).reset_index(drop=True)


def _normalize_evaluation(df: pd.DataFrame | None, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = df.copy()
    for column in ["strategy_name", "strategy_version", "valid_count", "evaluation_score", "recommendation"]:
        if column not in normalized.columns:
            normalized[column] = None
    normalized = normalized.rename(columns={"recommendation": "evaluation_recommendation"})
    normalized["source"] = source
    return normalized[
        [
            "strategy_name",
            "strategy_version",
            "source",
            "valid_count",
            "evaluation_recommendation",
            "evaluation_score",
        ]
    ]


def _merge_oos(base: pd.DataFrame, validation: pd.DataFrame | None) -> pd.DataFrame:
    result = base.copy()
    if validation is None or validation.empty:
        result["oos_status"] = None
        result["oos_risk"] = None
        result["oos_stability_score"] = None
        return result

    oos = validation.copy()
    for column in ["strategy_name", "strategy_version", "validation_status", "overfit_risk", "stability_score"]:
        if column not in oos.columns:
            oos[column] = None
    oos["strategy_version"] = oos["strategy_version"].fillna("v1")
    oos = oos.rename(
        columns={
            "validation_status": "oos_status",
            "overfit_risk": "oos_risk",
            "stability_score": "oos_stability_score",
        }
    )
    oos = oos.drop_duplicates(subset=["strategy_name", "strategy_version"], keep="last")
    return result.merge(
        oos[["strategy_name", "strategy_version", "oos_status", "oos_risk", "oos_stability_score"]],
        on=["strategy_name", "strategy_version"],
        how="left",
    )


def _merge_trade_plan_performance(base: pd.DataFrame, performance: pd.DataFrame | None) -> pd.DataFrame:
    result = base.copy()
    if performance is None or performance.empty:
        return _ensure_trade_plan_columns(result)

    trade = _select_buy_action_performance(performance.copy())
    if trade.empty:
        LOGGER.warning(
            "trade plan performance has no buy-like action rows; strategy admission trade_plan_* metrics remain empty"
        )
        return _ensure_trade_plan_columns(result)
    rename_map = {}
    if "strategy_name" in trade.columns and "strategy_names" not in trade.columns:
        rename_map["strategy_name"] = "strategy_names"
    if "strategy_version" in trade.columns and "strategy_versions" not in trade.columns:
        rename_map["strategy_version"] = "strategy_versions"
    trade = trade.rename(columns=rename_map)
    for column in [
        "strategy_names",
        "strategy_versions",
        "plan_count",
        "triggered_count",
        "valid_count",
        "trigger_rate",
        "win_rate",
        "avg_return",
        "avg_max_drawdown",
    ]:
        if column not in trade.columns:
            trade[column] = None

    for column in ["plan_count", "triggered_count", "valid_count", "trigger_rate", "win_rate", "avg_return", "avg_max_drawdown"]:
        trade[column] = pd.to_numeric(trade[column], errors="coerce")
    valid_keys = {
        (str(row["strategy_name"]).strip(), str(row["strategy_version"] or "v1").strip())
        for _, row in base[["strategy_name", "strategy_version"]].dropna(subset=["strategy_name"]).iterrows()
    }
    trade = explode_trade_plan_strategy_dimensions(trade, valid_strategy_keys=valid_keys)
    if trade.empty:
        LOGGER.warning(
            "trade plan performance strategy_names/strategy_versions could not be mapped to admission candidates; "
            "strategy admission trade_plan_* metrics remain empty"
        )
        return _ensure_trade_plan_columns(result)
    trade = _aggregate_trade_plan_performance(trade)
    trade = trade.rename(
        columns={
            "valid_count": "trade_plan_valid_count",
            "trigger_rate": "trade_plan_trigger_rate",
            "win_rate": "trade_plan_win_rate",
            "avg_return": "trade_plan_avg_return",
            "avg_max_drawdown": "trade_plan_avg_drawdown",
        }
    )
    result = result.merge(
        trade[
            [
                "strategy_name",
                "strategy_version",
                "trade_plan_valid_count",
                "trade_plan_trigger_rate",
                "trade_plan_win_rate",
                "trade_plan_avg_return",
                "trade_plan_avg_drawdown",
            ]
        ],
        on=["strategy_name", "strategy_version"],
        how="left",
    )
    matched_count = int(result["trade_plan_win_rate"].notna().sum())
    if matched_count == 0:
        LOGGER.warning(
            "trade plan performance was parsed but no strategy_name/strategy_version keys matched admission candidates"
        )
    else:
        LOGGER.info("mapped trade plan performance to %s admission candidates", matched_count)
    return _ensure_trade_plan_columns(result)


def parse_strategy_dimension(value: object) -> list[str]:
    """Parse a strategy-name/version dimension without assuming a single value."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parsed: list[str] = []
        for item in value:
            parsed.extend(parse_strategy_dimension(item))
        return _deduplicate_dimension(parsed)
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, list):
            return parse_strategy_dimension(payload)

    parts = re.split(r"\s*[,|;+]\s*", text)
    cleaned = [part.strip().strip("\"'") for part in parts]
    return _deduplicate_dimension(
        part for part in cleaned if part and part.lower() not in {"nan", "none", "null", "<na>"}
    )


def explode_trade_plan_strategy_dimensions(
    performance: pd.DataFrame,
    valid_strategy_keys: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Expand composite performance rows to strategy_name/strategy_version rows.

    When a composite row contains more versions than names, existing admission
    candidate keys disambiguate which versions belong to each strategy.
    """
    if performance is None or performance.empty:
        return pd.DataFrame()
    records: list[dict] = []
    for _, row in performance.iterrows():
        names = parse_strategy_dimension(row.get("strategy_names", row.get("strategy_name")))
        versions = parse_strategy_dimension(row.get("strategy_versions", row.get("strategy_version")))
        pairs = _pair_strategy_dimensions(names, versions, valid_strategy_keys or set())
        for strategy_name, strategy_version in pairs:
            record = row.to_dict()
            record["strategy_name"] = strategy_name
            record["strategy_version"] = strategy_version
            records.append(record)
    return pd.DataFrame(records)


def _pair_strategy_dimensions(
    names: list[str],
    versions: list[str],
    valid_strategy_keys: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    if not names:
        return []
    versions = versions or ["v1"]
    if len(names) == 1:
        return [(names[0], version) for version in versions]

    pairs: list[tuple[str, str]] = []
    for index, name in enumerate(names):
        matched = [version for version in versions if (name, version) in valid_strategy_keys]
        if matched:
            pairs.extend((name, version) for version in matched)
        elif len(names) == len(versions):
            pairs.append((name, versions[index]))
        elif len(versions) == 1:
            pairs.append((name, versions[0]))
        elif (name, "v1") in valid_strategy_keys:
            pairs.append((name, "v1"))
    return list(dict.fromkeys(pairs))


def _deduplicate_dimension(values) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _select_buy_action_performance(performance: pd.DataFrame) -> pd.DataFrame:
    if "action" not in performance.columns:
        return performance
    actions = performance["action"].fillna("").astype(str).str.strip()
    if actions.eq("").all():
        return performance
    lowered = actions.str.lower()
    excluded = lowered.str.contains(r"仅观察|watch[_ ]?only|sell|卖出|减仓|平仓", regex=True)
    explicit_buy = lowered.str.contains(r"buy|买入|低吸|回踩|突破|支撑|建仓|加仓|long", regex=True)
    valid_count = pd.to_numeric(
        performance.get("valid_count", pd.Series(0, index=performance.index)), errors="coerce"
    ).fillna(0)
    triggered_count = pd.to_numeric(
        performance.get("triggered_count", pd.Series(0, index=performance.index)), errors="coerce"
    ).fillna(0)
    actual_buy = ~excluded & (explicit_buy | valid_count.gt(0) | triggered_count.gt(0))
    return performance.loc[actual_buy].copy()


def _aggregate_trade_plan_performance(trade: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["strategy_name", "strategy_version"]
    grouped = trade.groupby(group_columns, dropna=False)
    plan_count = grouped["plan_count"].sum(min_count=1)
    triggered_count = grouped["triggered_count"].sum(min_count=1)
    valid_count = grouped["valid_count"].sum(min_count=1)

    trigger_rate = _weighted_mean_by_group(trade, "trigger_rate", "plan_count", group_columns)
    triggered_rate_fallback = triggered_count / plan_count
    trigger_rate = trigger_rate.where(trigger_rate.notna(), triggered_rate_fallback)

    result = pd.DataFrame(
        {
            "valid_count": valid_count,
            "trigger_rate": trigger_rate,
            "win_rate": _weighted_mean_by_group(trade, "win_rate", "valid_count", group_columns),
            "avg_return": _weighted_mean_by_group(trade, "avg_return", "valid_count", group_columns),
            "avg_max_drawdown": _weighted_mean_by_group(trade, "avg_max_drawdown", "valid_count", group_columns),
        }
    )
    return result.reset_index()


def _weighted_mean_by_group(
    df: pd.DataFrame,
    value_column: str,
    weight_column: str,
    group_columns: list[str],
) -> pd.Series:
    values = pd.to_numeric(df[value_column], errors="coerce")
    weights = pd.to_numeric(df[weight_column], errors="coerce").fillna(0)
    valid = values.notna()
    weighted_values = (values * weights).where(valid)
    grouped_weighted_sum = weighted_values.groupby([df[column] for column in group_columns], dropna=False).sum(min_count=1)
    grouped_weight_sum = weights.where(valid, 0).groupby([df[column] for column in group_columns], dropna=False).sum()
    grouped_mean = values.groupby([df[column] for column in group_columns], dropna=False).mean()
    return pd.Series(
        np.where(grouped_weight_sum > 0, grouped_weighted_sum / grouped_weight_sum, grouped_mean),
        index=grouped_mean.index,
    )


def _aggregate_trade_plan(group: pd.DataFrame) -> pd.Series:
    valid_count = group["valid_count"].sum(min_count=1)
    plan_count = group["plan_count"].sum(min_count=1)
    triggered_count = group["triggered_count"].sum(min_count=1)
    trigger_rate = triggered_count / plan_count if pd.notna(plan_count) and plan_count else _weighted_mean(group, "trigger_rate", "plan_count")
    return pd.Series(
        {
            "valid_count": valid_count,
            "trigger_rate": trigger_rate,
            "win_rate": _weighted_mean(group, "win_rate", "valid_count"),
            "avg_return": _weighted_mean(group, "avg_return", "valid_count"),
            "avg_max_drawdown": _weighted_mean(group, "avg_max_drawdown", "valid_count"),
        }
    )


def _weighted_mean(group: pd.DataFrame, value_column: str, weight_column: str) -> float | None:
    values = pd.to_numeric(group[value_column], errors="coerce")
    weights = pd.to_numeric(group[weight_column], errors="coerce").fillna(0)
    valid = values.notna()
    if not valid.any():
        return None
    if weights[valid].sum() > 0:
        return float((values[valid] * weights[valid]).sum() / weights[valid].sum())
    return float(values[valid].mean())


def _ensure_trade_plan_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "trade_plan_valid_count",
        "trade_plan_trigger_rate",
        "trade_plan_win_rate",
        "trade_plan_avg_return",
        "trade_plan_avg_drawdown",
    ]:
        if column not in df.columns:
            df[column] = None
    return df


def _admission_score(row: pd.Series, min_valid_count: int) -> float:
    score = 0.0
    recommendation = row.get("evaluation_recommendation")
    if recommendation == "enable_observation":
        score += 30
    elif recommendation == "observe":
        score += 15
    elif recommendation == "continue_backtest":
        score += 5
    elif recommendation == "pause":
        score -= 50
    elif recommendation == "reduce_or_pause":
        score -= 30

    score += min(_number(row.get("evaluation_score"), 0.0), 50)

    oos_status = row.get("oos_status")
    if oos_status == "passed_oos":
        score += 30
    elif oos_status == "needs_more_observation":
        score += 10
    elif oos_status == "failed_oos":
        score -= 50
    elif oos_status == "unstable":
        score -= 25

    score += min(_number(row.get("oos_stability_score"), 0.0), 30)

    if row.get("oos_risk") == "high":
        score -= 40
    valid_count = _number(row.get("valid_count"), None)
    if valid_count is not None and valid_count < min_valid_count:
        score -= 20
    if _number(row.get("trade_plan_trigger_rate"), 0.0) >= 0.3:
        score += 10
    if _number(row.get("trade_plan_win_rate"), 0.0) >= 0.5:
        score += 10
    if _number(row.get("trade_plan_avg_return"), 0.0) > 0:
        score += 10
    avg_drawdown = _number(row.get("trade_plan_avg_drawdown"), None)
    if avg_drawdown is not None and avg_drawdown < -0.08:
        score -= 20
    return score


def _admission_decision(
    row: pd.Series,
    score: float,
    min_valid_count: int,
    min_oos_valid_count: int,
    min_trade_plan_valid_count: int,
) -> tuple[str, str, str]:
    valid_count = _number(row.get("valid_count"), 0.0)
    if valid_count < min_valid_count:
        return "insufficient_samples", "continue_research", "有效样本不足，需继续积累研究样本。"
    if row.get("oos_status") in ["failed_oos", "unstable"] or row.get("oos_risk") == "high":
        return "oos_failed", "do_not_enable", "样本外验证未通过或不稳定，不建议进入观察候选。"
    if row.get("evaluation_recommendation") in ["pause", "reduce_or_pause"]:
        return "risk_rejected", "do_not_enable", "策略评价建议暂停或降权，不建议进入观察候选。"
    if score >= 70 and row.get("oos_status") == "passed_oos":
        reason = "满足观察候选条件，可作为观察候选配置。"
        trade_valid = _number(row.get("trade_plan_valid_count"), None)
        if trade_valid is not None and trade_valid < min_trade_plan_valid_count:
            reason += " 交易规则级回测样本仍需继续积累。"
        return "qualified_for_observation", "enable_observation_candidate", reason
    if score >= 50:
        reason = "具备一定观察价值，但仍需更多验证。"
        oos_status = row.get("oos_status")
        if pd.isna(oos_status) or oos_status == "":
            reason += " 样本外验证结果尚不充分。"
        return "watchlist", "observe_more", reason
    return "research_only", "continue_research", "暂不满足观察候选条件，建议继续研究。"


def _number(value: object, default: float | None) -> float | None:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
