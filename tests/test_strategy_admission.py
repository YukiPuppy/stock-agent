import pandas as pd

from src.research.strategy_admission import build_active_strategy_candidate_config, build_strategy_admission


def _evaluation(recommendation="enable_observation", valid_count=40, score=45.0):
    return pd.DataFrame(
        {
            "strategy_name": ["trend"],
            "strategy_version": ["v1"],
            "valid_count": [valid_count],
            "evaluation_score": [score],
            "recommendation": [recommendation],
        }
    )


def _oos(status="passed_oos", risk="low", stability=25.0):
    return pd.DataFrame(
        {
            "strategy_name": ["trend"],
            "strategy_version": ["v1"],
            "validation_status": [status],
            "overfit_risk": [risk],
            "stability_score": [stability],
        }
    )


def _trade_plan():
    return pd.DataFrame(
        {
            "strategy_names": ["trend"],
            "strategy_versions": ["v1"],
            "action": ["低吸"],
            "plan_count": [20],
            "triggered_count": [8],
            "valid_count": [12],
            "trigger_rate": [0.4],
            "win_rate": [0.55],
            "avg_return": [0.02],
            "avg_max_drawdown": [-0.03],
        }
    )


def test_build_strategy_admission_merges_all_sources_and_marks_mixed():
    parameter = pd.DataFrame(
        {
            "strategy_name": ["trend"],
            "strategy_version": ["v1"],
            "valid_count": [35],
            "evaluation_score": [50.0],
            "recommendation": ["observe"],
        }
    )

    result = build_strategy_admission(_evaluation(score=40.0), parameter, _oos(), _trade_plan())

    assert len(result) == 1
    row = result.iloc[0]
    assert row["source"] == "mixed"
    assert row["evaluation_score"] == 50.0
    assert row["oos_status"] == "passed_oos"
    assert row["trade_plan_trigger_rate"] == 0.4
    assert row["admission_score"] > 0


def test_build_strategy_admission_insufficient_samples_continue_research():
    result = build_strategy_admission(_evaluation(valid_count=5), None, _oos(), _trade_plan())

    assert result.loc[0, "admission_status"] == "insufficient_samples"
    assert result.loc[0, "admission_recommendation"] == "continue_research"
    assert "有效样本不足" in result.loc[0, "admission_reason"]


def test_build_strategy_admission_failed_oos_do_not_enable():
    result = build_strategy_admission(_evaluation(), None, _oos(status="failed_oos"), _trade_plan())

    assert result.loc[0, "admission_status"] == "oos_failed"
    assert result.loc[0, "admission_recommendation"] == "do_not_enable"


def test_build_strategy_admission_pause_do_not_enable():
    result = build_strategy_admission(_evaluation(recommendation="pause"), None, _oos(), _trade_plan())

    assert result.loc[0, "admission_status"] == "risk_rejected"
    assert result.loc[0, "admission_recommendation"] == "do_not_enable"


def test_build_strategy_admission_qualified_candidate_and_config_exports_only_candidates():
    admission = build_strategy_admission(_evaluation(), None, _oos(), _trade_plan())
    rejected = build_strategy_admission(_evaluation(recommendation="pause"), None, _oos(), _trade_plan())
    combined = pd.concat([admission, rejected.assign(strategy_version="v2")], ignore_index=True)

    assert admission.loc[0, "admission_recommendation"] == "enable_observation_candidate"
    config = build_active_strategy_candidate_config(combined)

    assert config["note"] == "This is an observation candidate config, not an auto-trading config."
    assert len(config["active_strategy_candidates"]) == 1
    assert config["active_strategy_candidates"][0]["strategy_version"] == "v1"


def test_build_strategy_admission_sorts_by_admission_score():
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend", "breakout"],
            "strategy_version": ["v1", "v1"],
            "valid_count": [40, 40],
            "evaluation_score": [10.0, 45.0],
            "recommendation": ["observe", "enable_observation"],
        }
    )
    oos = pd.DataFrame(
        {
            "strategy_name": ["trend", "breakout"],
            "strategy_version": ["v1", "v1"],
            "validation_status": ["needs_more_observation", "passed_oos"],
            "overfit_risk": ["low", "low"],
            "stability_score": [5.0, 25.0],
        }
    )

    result = build_strategy_admission(evaluation, None, oos, None)

    assert result["admission_score"].tolist() == sorted(result["admission_score"], reverse=True)
    assert result.loc[0, "strategy_name"] == "breakout"
