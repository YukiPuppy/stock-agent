import pandas as pd
import pytest

from src.strategy.candidate_selector import CANDIDATE_COLUMNS, select_candidates


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
