from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.agents.llm_client import DisabledLLMClient
from src.database.duckdb_store import StockAgentStore
from src.pipeline.rebuild_strategy_admission import rebuild_strategy_admission
from src.pipeline.rerun_trade_plan_and_admission import rerun_trade_plan_and_admission
from src.pipeline.run_strategy_research_agents_readonly import run_strategy_research_agents_readonly
from src.strategy.trade_plan_generator import TRADE_PLAN_COLUMNS


RUN_ID = "test-existing-run"


def _seed_research_inputs(store: StockAgentStore) -> None:
    store.save_strategy_version_evaluation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "valid_count": [40],
                "evaluation_score": [45.0],
                "recommendation": ["enable_observation"],
                "run_id": [RUN_ID],
            }
        )
    )
    store.save_parameter_search_results(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "valid_count": [40],
                "evaluation_score": [46.0],
                "recommendation": ["enable_observation"],
                "run_id": [RUN_ID],
            }
        )
    )
    store.save_walk_forward_validation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback"],
                "strategy_version": ["v1"],
                "validation_status": ["passed_oos"],
                "overfit_risk": ["low"],
                "stability_score": [25.0],
                "run_id": [RUN_ID],
            }
        )
    )


def _seed_performance(store: StockAgentStore) -> None:
    store.save_trade_plan_backtest_performance(
        pd.DataFrame(
            {
                "strategy_names": ["trend_pullback"],
                "strategy_versions": ["v1"],
                "action": ["回踩低吸"],
                "max_holding_days": [8],
                "plan_count": [20],
                "triggered_count": [12],
                "valid_count": [12],
                "trigger_rate": [0.6],
                "win_rate": [0.5],
                "avg_return": [0.02],
                "avg_max_drawdown": [-0.03],
                "run_id": [RUN_ID],
            }
        )
    )


def test_rebuild_strategy_admission_writes_nonempty_trade_plan_metrics_and_distribution(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    reports = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    _seed_research_inputs(store)
    _seed_performance(store)

    summary = rebuild_strategy_admission(
        run_id=RUN_ID,
        db_path=str(db_path),
        output_dir=str(reports),
    )
    saved = store.load_strategy_admission(run_id=RUN_ID)

    assert summary["strategy_admission_rows"] > 0
    assert summary["trade_plan_win_rate_nonnull_rows"] > 0
    assert summary["admission_status_distribution"]
    assert saved["trade_plan_win_rate"].notna().sum() > 0
    assert (reports / f"strategy_admission_{date.today().isoformat()}.md").exists()


def test_rebuild_dry_run_does_not_replace_existing_rows(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    _seed_research_inputs(store)
    _seed_performance(store)

    summary = rebuild_strategy_admission(
        run_id=RUN_ID,
        db_path=str(db_path),
        output_dir=str(tmp_path),
        dry_run=True,
    )

    assert summary["write_status"] == "dry_run"
    assert store.load_strategy_admission(run_id=RUN_ID).empty


def test_research_rows_with_same_strategy_key_remain_isolated_by_run_id(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    first = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "valid_count": [10],
            "evaluation_score": [10.0],
            "recommendation": ["continue_backtest"],
            "run_id": ["run-one"],
        }
    )
    second = first.assign(valid_count=20, evaluation_score=20.0, run_id="run-two")

    store.save_parameter_search_results(first)
    store.save_parameter_search_results(second)

    assert store.load_parameter_search_results(run_id="run-one").loc[0, "valid_count"] == 10
    assert store.load_parameter_search_results(run_id="run-two").loc[0, "valid_count"] == 20


def test_trade_plan_grid_rerun_is_chunked_and_rebuilds_admission(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    reports = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    _seed_research_inputs(store)
    plan = {column: None for column in TRADE_PLAN_COLUMNS}
    plan.update(
        {
            "trade_date": "20260101",
            "code": "600000",
            "name": "测试股",
            "rank": 1,
            "strategy_names": "trend_pullback",
            "strategy_versions": "v1",
            "action": "回踩低吸",
            "entry_low": 9.9,
            "entry_high": 10.1,
            "stop_loss": 1.0,
            "take_profit_1": 100.0,
            "take_profit_2": 101.0,
            "run_id": RUN_ID,
        }
    )
    store.save_historical_trade_plans(pd.DataFrame([plan]))
    first_day = date(2026, 1, 2)
    bars = []
    for offset in range(35):
        trade_date = (first_day + timedelta(days=offset)).strftime("%Y%m%d")
        bars.append(
            {
                "trade_date": trade_date,
                "code": "600000",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.0 + offset / 100,
                "volume": 1000,
                "amount": 10000,
            }
        )
    store.save_daily_bars(pd.DataFrame(bars))

    summary = rerun_trade_plan_and_admission(
        run_id=RUN_ID,
        db_path=str(db_path),
        output_dir=str(reports),
        holding_days_mode="strategy_grid",
        low_memory=True,
        replace_current_run=True,
    )
    results = store.load_trade_plan_backtest_results(run_id=RUN_ID)
    performance = store.load_trade_plan_backtest_performance(run_id=RUN_ID)

    assert sorted(results["max_holding_days"].tolist()) == [5, 8, 10, 15, 20]
    assert sorted(performance["max_holding_days"].tolist()) == [5, 8, 10, 15, 20]
    assert results["holding_days"].max() > 5
    assert summary["trade_plan_win_rate_nonnull_rows"] > 0
    assert Path(summary["trade_plan_backtest_report_path"]).exists()


def test_readonly_agent_entry_warns_when_admission_metrics_are_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "stock_agent.duckdb"
    reports = tmp_path / "reports"
    store = StockAgentStore(str(db_path))
    _seed_research_inputs(store)
    monkeypatch.setattr(
        "src.pipeline.run_strategy_research_agents_readonly.get_llm_client",
        lambda _name: DisabledLLMClient(),
    )

    summary = run_strategy_research_agents_readonly(
        run_id=RUN_ID,
        db_path=str(db_path),
        output_dir=str(reports),
        report_date="2026-01-31",
    )

    assert summary["read_only"] is True
    assert summary["admission_incomplete"] is True
    for key in [
        "backtest_analysis_report_path",
        "strategy_research_report_path",
        "parameter_iteration_proposal_path",
    ]:
        assert "准入结论不完整，不建议实盘" in Path(summary[key]).read_text(encoding="utf-8")
