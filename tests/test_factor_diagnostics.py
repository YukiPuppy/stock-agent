import pandas as pd

from src.research.factor_diagnostics import FACTOR_DIAGNOSTIC_COLUMNS, build_factor_diagnostics


def test_build_factor_diagnostics_handles_empty_daily_factors():
    result = build_factor_diagnostics(pd.DataFrame())

    assert result.empty
    assert list(result.columns) == FACTOR_DIAGNOSTIC_COLUMNS


def test_missing_factor_column_outputs_missing_column():
    result = build_factor_diagnostics(
        pd.DataFrame({"trade_date": ["2026-01-02"], "code": ["600000"], "turnover_rate": [1.0]}),
        factor_columns=["not_exists"],
    )

    row = result.iloc[0]
    assert row["diagnostic_status"] == "missing_column"
    assert row["diagnostic_message"] == "factor column not found"
    assert row["missing_rate"] == 1.0


def test_missing_rate_status_levels():
    data = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"] * 10,
            "code": [f"{index:06d}" for index in range(10)],
            "high": [1.0] + [None] * 9,
            "medium": [1.0] * 5 + [None] * 5,
            "ok_factor": [1.0] * 8 + [None] * 2,
        }
    )

    result = build_factor_diagnostics(data, factor_columns=["high", "medium", "ok_factor"]).set_index("factor_name")

    assert result.loc["high", "diagnostic_status"] == "high_missing"
    assert result.loc["medium", "diagnostic_status"] == "medium_missing"
    assert result.loc["ok_factor", "diagnostic_status"] == "ok"


def test_candidate_pool_and_trade_plan_merge_usage_means():
    daily_factors = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02", "2026-01-03"],
            "code": ["600000", "000001", "600000"],
            "turnover_rate": [1.0, 3.0, 5.0],
        }
    )
    candidate_pool = pd.DataFrame({"trade_date": ["2026-01-02"], "code": ["000001"]})
    trade_plan = pd.DataFrame({"trade_date": ["2026-01-03"], "code": ["600000"]})

    result = build_factor_diagnostics(
        daily_factors,
        candidate_pool=candidate_pool,
        trade_plan=trade_plan,
        factor_columns=["turnover_rate"],
    )

    row = result.iloc[0]
    assert row["candidate_non_null_count"] == 1
    assert row["candidate_mean"] == 3.0
    assert row["trade_plan_non_null_count"] == 1
    assert row["trade_plan_mean"] == 5.0
