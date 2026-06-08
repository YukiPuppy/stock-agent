import pandas as pd

from src.research.strategy_version_evaluator import (
    EVALUATION_COLUMNS,
    build_active_strategy_config,
    evaluate_strategy_versions,
)


def _performance_row(
    strategy_name="trend_pullback",
    strategy_version="v1",
    sample_count=40,
    valid_count=40,
    win_rate_3d=0.6,
    avg_return_3d=0.02,
    median_return_3d=0.01,
    avg_max_drawdown_3d=-0.03,
):
    return {
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "sample_count": sample_count,
        "valid_count": valid_count,
        "win_rate_1d": 0.5,
        "win_rate_3d": win_rate_3d,
        "win_rate_5d": 0.5,
        "avg_return_1d": 0.01,
        "avg_return_3d": avg_return_3d,
        "avg_return_5d": 0.01,
        "median_return_1d": 0.01,
        "median_return_3d": median_return_3d,
        "median_return_5d": 0.01,
        "avg_max_drawdown_1d": -0.02,
        "avg_max_drawdown_3d": avg_max_drawdown_3d,
        "avg_max_drawdown_5d": -0.03,
    }


def test_evaluate_strategy_versions_insufficient_samples():
    evaluation = evaluate_strategy_versions(pd.DataFrame([_performance_row(valid_count=5)]))

    assert evaluation.loc[0, "evaluation_status"] == "insufficient_samples"
    assert evaluation.loc[0, "risk_level"] == "unknown"
    assert evaluation.loc[0, "recommendation"] == "continue_backtest"
    assert "有效样本不足" in evaluation.loc[0, "evaluation_reason"]


def test_evaluate_strategy_versions_weak_return_and_win_rate():
    evaluation = evaluate_strategy_versions(
        pd.DataFrame([_performance_row(win_rate_3d=0.4, avg_return_3d=-0.01)])
    )

    assert evaluation.loc[0, "evaluation_status"] == "weak"
    assert evaluation.loc[0, "recommendation"] == "pause"
    assert "3日收益和胜率均未达标" in evaluation.loc[0, "evaluation_reason"]


def test_evaluate_strategy_versions_high_drawdown():
    evaluation = evaluate_strategy_versions(pd.DataFrame([_performance_row(avg_max_drawdown_3d=-0.1)]))

    assert evaluation.loc[0, "evaluation_status"] == "high_drawdown"
    assert evaluation.loc[0, "risk_level"] == "high"
    assert evaluation.loc[0, "recommendation"] == "reduce_or_pause"
    assert "平均回撤偏大" in evaluation.loc[0, "evaluation_reason"]


def test_evaluate_strategy_versions_qualified_enable_observation():
    evaluation = evaluate_strategy_versions(pd.DataFrame([_performance_row()]))

    assert evaluation.loc[0, "evaluation_status"] == "qualified"
    assert evaluation.loc[0, "risk_level"] == "low"
    assert evaluation.loc[0, "recommendation"] == "enable_observation"
    assert "满足观察启用条件" in evaluation.loc[0, "evaluation_reason"]


def test_evaluate_strategy_versions_score_and_sorting():
    low = _performance_row(
        strategy_name="low",
        strategy_version="v1",
        valid_count=40,
        win_rate_3d=0.5,
        avg_return_3d=0.0,
        median_return_3d=0.0,
        avg_max_drawdown_3d=-0.04,
    )
    high = _performance_row(
        strategy_name="high",
        strategy_version="v1",
        valid_count=200,
        win_rate_3d=0.7,
        avg_return_3d=0.03,
        median_return_3d=0.02,
        avg_max_drawdown_3d=-0.01,
    )

    evaluation = evaluate_strategy_versions(pd.DataFrame([low, high]))

    expected_high_score = round(0.7 * 40 + 0.03 * 100 + 0.02 * 50 + -0.01 * 50 + 10, 4)
    assert evaluation.loc[0, "strategy_name"] == "high"
    assert evaluation.loc[0, "evaluation_score"] == expected_high_score
    assert evaluation["evaluation_score"].tolist() == sorted(
        evaluation["evaluation_score"].tolist(),
        reverse=True,
    )


def test_evaluate_strategy_versions_empty_returns_standard_columns():
    evaluation = evaluate_strategy_versions(pd.DataFrame())

    assert evaluation.empty
    assert evaluation.columns.tolist() == EVALUATION_COLUMNS


def test_build_active_strategy_config_only_selects_enable_observation():
    evaluation = evaluate_strategy_versions(
        pd.DataFrame(
            [
                _performance_row(strategy_name="qualified", strategy_version="v1"),
                _performance_row(strategy_name="weak", strategy_version="v1", win_rate_3d=0.4, avg_return_3d=-0.01),
            ]
        )
    )

    config = build_active_strategy_config(evaluation)

    assert config == {
        "active_strategies": [
            {
                "strategy_name": "qualified",
                "strategy_version": "v1",
                "recommendation": "enable_observation",
                "evaluation_score": evaluation[evaluation["strategy_name"] == "qualified"].iloc[0][
                    "evaluation_score"
                ],
            }
        ]
    }
