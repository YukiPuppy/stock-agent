import pandas as pd
import pytest

from src.strategy.candidate_selector import CANDIDATE_COLUMNS, select_candidates, select_candidates_from_signals


def _daily_factors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
            ],
            "code": ["600000", "600000", "000001", "000002", "000003", "000004"],
            "close": [10.0, 11.0, 20.0, 8.0, 9.0, 7.0],
            "pct_chg_1d": [0.01, 0.02, 0.03, 0.096, -0.096, 0.01],
            "pct_chg_3d": [0.02, 0.03, 0.01, 0.01, 0.01, 0.01],
            "pct_chg_5d": [0.04, 0.05, 0.20, 0.10, 0.10, 0.08],
            "pct_chg_10d": [0.05, 0.06, 0.02, 0.02, 0.02, 0.03],
            "ma5": [9.0, 10.0, 19.0, 7.0, 8.0, 8.0],
            "ma10": [9.0, 10.0, 21.0, 7.0, 8.0, 8.0],
            "ma20": [9.0, 10.0, 21.0, 7.0, 8.0, 8.0],
            "volume_ratio_5": [1.0, 2.0, 4.0, 1.0, 1.0, 1.0],
            "close_position_20": [0.7, 0.8, 0.6, 0.5, 0.5, 0.5],
            "above_ma5": [True, True, True, True, True, False],
            "above_ma10": [True, True, False, True, True, False],
            "above_ma20": [True, True, False, True, True, False],
            "amount_ma5": [
                200000000.0,
                200000000.0,
                200000000.0,
                200000000.0,
                200000000.0,
                200000000.0,
            ],
        }
    )


def test_select_candidates_uses_latest_trade_date_when_missing():
    result = select_candidates(_daily_factors(), top_n=10)

    assert set(result["trade_date"]) == {"2026-01-02"}
    assert "600000" in result["code"].tolist()


def test_select_candidates_filters_by_trade_date():
    result = select_candidates(_daily_factors(), trade_date="2026-01-01")

    assert result["trade_date"].tolist() == ["2026-01-01"]
    assert result["code"].tolist() == ["600000"]


def test_select_candidates_excludes_low_amount_ma5():
    factors = _daily_factors()
    factors.loc[factors["code"] == "000001", "amount_ma5"] = 99999999.0

    result = select_candidates(factors, top_n=10)

    assert "000001" not in result["code"].tolist()


def test_select_candidates_excludes_near_limit_up_or_down():
    result = select_candidates(_daily_factors(), top_n=10)

    assert "000002" not in result["code"].tolist()
    assert "000003" not in result["code"].tolist()


def test_select_candidates_requires_above_ma5_or_above_ma10():
    result = select_candidates(_daily_factors(), top_n=10)

    assert "000004" not in result["code"].tolist()


def test_select_candidates_filters_extension_risks_by_default():
    factors = _daily_factors()
    factors["is_suspended"] = factors["code"].eq("600000")
    factors["is_limit_down_close"] = factors["code"].eq("000001")
    factors["is_limit_up_close"] = factors["code"].eq("000002")
    factors["turnover_rate"] = 1.0
    factors["circ_mv"] = 1000.0

    result = select_candidates(factors, top_n=10)

    assert "600000" not in result["code"].tolist()
    assert "000001" not in result["code"].tolist()


def test_select_candidates_keeps_limit_up_close_and_adds_risk_flag():
    factors = _daily_factors()
    factors["is_limit_up_close"] = factors["code"].eq("000001")

    result = select_candidates(factors, top_n=10)

    by_code = result.set_index("code")
    assert "000001" in by_code.index
    assert "limit_up_close" in by_code.loc["000001", "risk_flags"]


def test_select_candidates_tolerates_missing_extension_columns():
    factors = _daily_factors()

    result = select_candidates(factors, top_n=10)

    assert not result.empty
    assert set(result["code"]) == {"600000", "000001"}


def test_select_candidates_flags_missing_daily_basic_fields_without_excluding():
    factors = _daily_factors()
    factors["volume_ratio_daily_basic"] = 1.5
    factors["total_mv"] = 2000.0
    factors["circ_mv"] = 1000.0
    factors.loc[factors["code"] == "000001", ["volume_ratio_daily_basic", "total_mv", "circ_mv"]] = pd.NA

    result = select_candidates(factors, top_n=10)

    flags = result.set_index("code").loc["000001", "risk_flags"]
    assert "000001" in result["code"].tolist()
    assert "missing_daily_basic" in flags
    assert "missing_market_value" in flags
    assert "missing_volume_ratio_daily_basic" in flags


def test_select_candidates_flags_missing_circ_mv_without_market_value_filtering():
    factors = _daily_factors()
    factors["total_mv"] = 2000.0
    factors["circ_mv"] = 1000.0
    factors.loc[factors["code"] == "000001", "circ_mv"] = pd.NA

    result = select_candidates(factors, top_n=10, min_circ_mv=1500.0)

    by_code = result.set_index("code")
    assert "000001" in by_code.index
    assert "600000" not in by_code.index
    assert "missing_market_value" in by_code.loc["000001", "risk_flags"]


def test_select_candidates_flags_missing_turnover_and_limit_data():
    factors = _daily_factors()
    factors["turnover_rate"] = 2.0
    factors["up_limit"] = 11.0
    factors["down_limit"] = 9.0
    factors.loc[factors["code"] == "000001", ["turnover_rate", "up_limit"]] = pd.NA

    result = select_candidates(factors, top_n=10, min_turnover_rate=3.0)

    by_code = result.set_index("code")
    assert "000001" in by_code.index
    assert "600000" not in by_code.index
    assert "missing_turnover_rate" in by_code.loc["000001", "risk_flags"]
    assert "missing_limit_data" in by_code.loc["000001", "risk_flags"]


def test_select_candidates_generates_score_and_rank():
    result = select_candidates(_daily_factors(), top_n=10)

    assert result["code"].tolist() == ["000001", "600000"]
    assert result["rank"].tolist() == [1, 2]
    assert result.loc[0, "score"] == pytest.approx(20 + 1 + 12 + 15 + 5)
    assert result.loc[1, "score"] == pytest.approx(5 + 3 + 16 + 10 + 15)


def test_select_candidates_merges_stock_basic_name():
    stock_basic = pd.DataFrame(
        {
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "market": ["SH", "SZ"],
            "board": ["main", "main"],
        }
    )

    result = select_candidates(_daily_factors(), stock_basic=stock_basic, top_n=10)

    assert result.set_index("code").loc["600000", "name"] == "浦发银行"
    assert result.set_index("code").loc["000001", "name"] == "平安银行"


def test_select_candidates_empty_daily_factors_returns_standard_columns():
    result = select_candidates(pd.DataFrame())

    assert result.empty
    assert result.columns.tolist() == CANDIDATE_COLUMNS


def test_select_candidates_tolerates_empty_market_regime():
    result = select_candidates(_daily_factors(), top_n=10, market_regime=pd.DataFrame())

    assert not result.empty
    assert "market_high_risk" not in ",".join(result["risk_flags"].fillna("").astype(str))


def test_select_candidates_adds_market_high_risk_flag_and_score_penalty():
    baseline = select_candidates(_daily_factors(), top_n=10).set_index("code")
    result = select_candidates(
        _daily_factors(),
        top_n=10,
        market_regime=pd.DataFrame([{"trade_date": "2026-01-02", "market_regime": "weak", "risk_level": "high"}]),
    ).set_index("code")

    assert "market_high_risk" in result.loc["000001", "risk_flags"]
    assert result.loc["000001", "score"] == pytest.approx(baseline.loc["000001", "score"] - 5)


def test_select_candidates_strong_market_regime_adds_small_score_bonus():
    baseline = select_candidates(_daily_factors(), top_n=10).set_index("code")
    result = select_candidates(
        _daily_factors(),
        top_n=10,
        market_regime=pd.DataFrame([{"trade_date": "2026-01-02", "market_regime": "strong", "risk_level": "low"}]),
    ).set_index("code")

    assert result.loc["000001", "score"] == pytest.approx(baseline.loc["000001", "score"] + 3)


def test_select_candidates_from_signals_applies_industry_adjustment():
    factors = _daily_factors()
    factors["industry_strength_score"] = pd.NA
    factors["industry_strength_level"] = ""
    factors.loc[factors["code"] == "600000", ["industry_strength_score", "industry_strength_level"]] = [70, "strong"]
    factors.loc[factors["code"] == "000001", ["industry_strength_score", "industry_strength_level"]] = [10, "weak"]
    signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "000001"],
            "strategy_name": ["s", "s"],
            "strategy_version": ["v1", "v1"],
            "signal_strength": [10, 10],
            "entry_reason": ["r", "r"],
            "risk_flags": ["", ""],
        }
    )

    result = select_candidates_from_signals(signals, factors, top_n=10).set_index("code")

    assert result.loc["600000", "score"] == pytest.approx(25)
    assert "strong_industry" in result.loc["600000", "risk_flags"]
    assert result.loc["000001", "score"] == pytest.approx(15)
    assert "weak_industry" in result.loc["000001", "risk_flags"]


def test_select_candidates_from_signals_flags_missing_industry_strength():
    factors = _daily_factors()
    signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "code": ["600000"],
            "strategy_name": ["s"],
            "strategy_version": ["v1"],
            "signal_strength": [10],
            "entry_reason": ["r"],
            "risk_flags": [""],
        }
    )

    result = select_candidates_from_signals(signals, factors, top_n=10)

    assert "missing_industry_strength" in result.loc[0, "risk_flags"]


def test_select_candidates_from_signals_aggregates_multiple_strategies():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02", "2026-01-02"],
            "code": ["600000", "600000", "000001"],
            "strategy_name": ["trend_pullback", "breakout_volume", "support_rebound"],
            "signal_strength": [20.0, 30.0, 15.0],
            "entry_reason": ["reason_a", "reason_b", "reason_c"],
            "risk_flags": ["near_20d_high", "near_20d_high,extended_position", ""],
        }
    )
    stock_basic = pd.DataFrame(
        {
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "market": ["SH", "SZ"],
            "board": ["main", "main"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals=strategy_signals,
        daily_factors=_daily_factors(),
        stock_basic=stock_basic,
        top_n=10,
        min_amount_ma5=100000000.0,
    )

    first = result.iloc[0]
    assert first["code"] == "600000"
    assert first["strategy_names"] == "trend_pullback,breakout_volume"
    assert first["signal_count"] == 2
    assert first["max_signal_strength"] == 30.0
    assert first["total_signal_strength"] == 50.0
    assert "near_20d_high" in first["risk_flags"]
    assert "extended_position" in first["risk_flags"]


def test_select_candidates_applies_moneyflow_score_and_flags():
    factors = _daily_factors()
    factors["moneyflow_score"] = 0.0
    factors["moneyflow_risk_flags"] = ""
    factors.loc[factors["code"] == "600000", "moneyflow_score"] = 30.0
    factors.loc[factors["code"] == "600000", "moneyflow_risk_flags"] = "strong_main_inflow"

    result = select_candidates(factors, top_n=10).set_index("code")

    assert result.loc["600000", "score"] > result.loc["000001", "score"]
    assert "strong_main_inflow" in result.loc["600000", "risk_flags"]


def test_select_candidates_from_signals_adds_strong_main_outflow_flag():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "signal_strength": [20.0],
            "entry_reason": ["reason"],
            "risk_flags": [""],
        }
    )
    factors = _daily_factors()
    factors["moneyflow_score"] = 0.0
    factors["moneyflow_risk_flags"] = ""
    factors.loc[factors["code"] == "600000", "moneyflow_score"] = -30.0
    factors.loc[factors["code"] == "600000", "moneyflow_risk_flags"] = "main_outflow,strong_main_outflow"

    result = select_candidates_from_signals(strategy_signals, factors, top_n=10)

    assert "strong_main_outflow" in result.loc[0, "risk_flags"]


def test_enable_observation_weight_is_higher_than_observe():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "000001"],
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v2", "v1"],
            "signal_strength": [10.0, 10.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v2", "v1"],
            "recommendation": ["enable_observation", "observe"],
            "risk_level": ["low", "low"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals,
        _daily_factors(),
        strategy_evaluation=evaluation,
        top_n=10,
    )

    by_code = result.set_index("code")
    assert by_code.loc["600000", "avg_strategy_weight"] == pytest.approx(1.2)
    assert by_code.loc["000001", "avg_strategy_weight"] == pytest.approx(1.0)
    assert result["code"].tolist()[0] == "600000"


def test_pause_signal_does_not_enter_candidate_pool():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "signal_strength": [100.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "recommendation": ["pause"],
            "risk_level": ["low"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals,
        _daily_factors(),
        strategy_evaluation=evaluation,
    )

    assert result.empty


def test_reduce_or_pause_signal_is_significantly_down_weighted():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "000001"],
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v1"],
            "signal_strength": [20.0, 20.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v1"],
            "recommendation": ["reduce_or_pause", "observe"],
            "risk_level": ["low", "low"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals,
        _daily_factors(),
        strategy_evaluation=evaluation,
        top_n=10,
    )

    by_code = result.set_index("code")
    assert by_code.loc["600000", "total_weighted_signal_strength"] == pytest.approx(10.0)
    assert by_code.loc["000001", "total_weighted_signal_strength"] == pytest.approx(20.0)
    assert by_code.loc["600000", "score"] == pytest.approx(20.0)
    assert by_code.loc["000001", "score"] == pytest.approx(30.0)


def test_high_risk_level_further_reduces_weight():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "000001"],
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v1"],
            "signal_strength": [20.0, 20.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v1"],
            "recommendation": ["observe", "observe"],
            "risk_level": ["high", "low"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals,
        _daily_factors(),
        strategy_evaluation=evaluation,
        top_n=10,
    )

    by_code = result.set_index("code")
    assert by_code.loc["600000", "avg_strategy_weight"] == pytest.approx(0.6)
    assert by_code.loc["000001", "avg_strategy_weight"] == pytest.approx(1.0)


def test_multiple_signals_score_uses_weighted_signal_strength():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "code": ["600000", "600000"],
            "strategy_name": ["trend_pullback", "breakout_volume"],
            "strategy_version": ["v1", "v1"],
            "signal_strength": [20.0, 30.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "breakout_volume"],
            "strategy_version": ["v1", "v1"],
            "recommendation": ["observe", "reduce_or_pause"],
            "risk_level": ["low", "low"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals,
        _daily_factors(),
        strategy_evaluation=evaluation,
    )

    first = result.iloc[0]
    assert first["total_signal_strength"] == pytest.approx(50.0)
    assert first["total_weighted_signal_strength"] == pytest.approx(35.0)
    assert first["score"] == pytest.approx(55.0)


def test_select_candidates_from_signals_without_evaluation_still_works():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "signal_strength": [20.0],
        }
    )

    result = select_candidates_from_signals(strategy_signals, _daily_factors())

    assert result["code"].tolist() == ["600000"]
    assert result.loc[0, "strategy_versions"] == "v1"
    assert result.loc[0, "total_weighted_signal_strength"] == pytest.approx(20.0)
    assert result.loc[0, "avg_strategy_weight"] == pytest.approx(1.0)


def test_missing_strategy_version_defaults_to_v1_for_evaluation_merge():
    strategy_signals = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "signal_strength": [10.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback"],
            "strategy_version": ["v1"],
            "recommendation": ["enable_observation"],
            "risk_level": ["low"],
        }
    )

    result = select_candidates_from_signals(
        strategy_signals,
        _daily_factors(),
        strategy_evaluation=evaluation,
    )

    assert result.loc[0, "strategy_versions"] == "v1"
    assert result.loc[0, "avg_strategy_weight"] == pytest.approx(1.2)
    assert "enable_observation" in result.loc[0, "recommendations"]
