"""Run ParameterIterationAgent and export parameter search candidate reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.llm_client import DisabledLLMClient, get_llm_client
from src.agents.parameter_iteration_agent import run_parameter_iteration_agent
from src.agents.parameter_iteration_outputs import extract_parameter_search_space_candidate
from src.config import settings
from src.database.duckdb_store import StockAgentStore


def run_parameter_iteration_agent_pipeline(
    db_path: str | None = None,
    output_dir: str = "reports",
    report_date: str | None = None,
    export_candidate_json: bool = True,
) -> dict:
    resolved_db_path = db_path if db_path is not None else settings.DB_PATH
    resolved_date = report_date or date.today().isoformat()
    store = StockAgentStore(resolved_db_path)

    strategy_research_suggestions = _load_latest_strategy_research_suggestions(output_dir)
    strategy_evaluation = _safe_load(store.load_strategy_version_evaluation)
    parameter_results = _safe_load(store.load_parameter_search_results)
    walk_forward = _safe_load(store.load_walk_forward_validation)
    trade_plan_performance = _safe_load(store.load_trade_plan_backtest_performance)
    admission = _safe_load(store.load_strategy_admission)
    factor_diagnostics = _safe_load(store.load_factor_diagnostics)
    market_regime = _safe_load(store.load_market_regime)

    llm_client = _safe_llm_client()
    if isinstance(llm_client, DisabledLLMClient):
        markdown = _disabled_placeholder(resolved_date)
    else:
        markdown = run_parameter_iteration_agent(
            llm_client,
            strategy_research_suggestions=strategy_research_suggestions,
            strategy_evaluation=strategy_evaluation,
            parameter_search_results=parameter_results,
            walk_forward_validation=walk_forward,
            trade_plan_backtest_performance=trade_plan_performance,
            strategy_admission=admission,
            factor_diagnostics=factor_diagnostics,
            market_regime=market_regime,
        )

    output_path = Path(output_dir) / f"llm_parameter_iteration_{resolved_date}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    candidate_path: str | None = None
    requires_human_review = True
    if export_candidate_json:
        candidate = extract_parameter_search_space_candidate(markdown)
        requires_human_review = bool(candidate.get("requires_human_review", True))
        candidate_output = Path(output_dir) / f"parameter_search_space_candidate_{resolved_date}.json"
        candidate_output.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_path = str(candidate_output)

    return {
        "parameter_iteration_report_path": str(output_path),
        "parameter_search_space_candidate_path": candidate_path,
        "exported_candidate_json": bool(export_candidate_json),
        "requires_human_review": requires_human_review,
    }


def _safe_load(loader) -> pd.DataFrame:
    try:
        return loader()
    except Exception:
        return pd.DataFrame()


def _safe_llm_client():
    try:
        return get_llm_client("ParameterIterationAgent")
    except Exception:
        return DisabledLLMClient()


def _load_latest_strategy_research_suggestions(output_dir: str) -> dict:
    reports_dir = Path(output_dir)
    paths = sorted(path for path in reports_dir.glob("strategy_research_suggestions_*.json") if path.is_file()) if reports_dir.exists() else []
    if not paths:
        return {}
    try:
        payload = json.loads(paths[-1].read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _disabled_placeholder(report_date: str) -> str:
    return (
        "# LLM 参数迭代建议报告\n\n"
        f"报告日期：{report_date}\n\n"
        "LLM 当前未启用或未配置，本文件为占位报告，便于看板和报告流程测试。\n\n"
        "ParameterIterationAgent 只做参数搜索空间候选建议，不做交易决策；"
        "不直接修改正式参数配置，不启用策略，不选股，不执行交易。\n\n"
        "## 一、当前参数研究状态\n"
        "当前未调用 LLM，仅确认报告链路可用。\n\n"
        "## 二、已有参数搜索结果观察\n"
        "暂无 LLM 观察。\n\n"
        "## 三、样本外稳定性问题\n"
        "暂无 LLM 观察。\n\n"
        "## 四、建议收窄的参数\n"
        "暂无候选建议。\n\n"
        "## 五、建议扩展的参数\n"
        "暂无候选建议。\n\n"
        "## 六、建议新增的风控参数\n"
        "暂无候选建议。\n\n"
        "## 七、建议下一轮参数搜索空间\n"
        "暂无候选建议；所有参数建议必须重新回测。\n\n"
        "## 八、人工确认事项\n"
        "小样本结果不能直接用于实盘；Agent 不能直接修改正式配置文件；"
        "候选参数需要人工确认后，才允许进入正式 parameter_search_space.json。"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ParameterIterationAgent and export candidate reports.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-date", default=None)
    parser.add_argument("--no-candidate-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_parameter_iteration_agent_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        report_date=args.report_date,
        export_candidate_json=not args.no_candidate_json,
    )
    for key in [
        "parameter_iteration_report_path",
        "parameter_search_space_candidate_path",
        "exported_candidate_json",
        "requires_human_review",
    ]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
