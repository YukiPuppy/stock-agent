import pandas as pd

from src.factors.moneyflow_factors import build_moneyflow_factors


def _moneyflow(main_inflow: bool = True) -> pd.DataFrame:
    if main_inflow:
        lg_buy, lg_sell, elg_buy, elg_sell, net = 100.0, 20.0, 50.0, 10.0, 120.0
    else:
        lg_buy, lg_sell, elg_buy, elg_sell, net = 20.0, 100.0, 10.0, 50.0, -120.0
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-02"],
            "code": ["000001"],
            "buy_sm_vol": [10.0],
            "buy_sm_amount": [20.0],
            "sell_sm_vol": [5.0],
            "sell_sm_amount": [10.0],
            "buy_md_amount": [30.0],
            "buy_lg_vol": [10.0],
            "buy_lg_amount": [lg_buy],
            "sell_lg_vol": [4.0],
            "sell_lg_amount": [lg_sell],
            "buy_elg_vol": [5.0],
            "buy_elg_amount": [elg_buy],
            "sell_elg_vol": [1.0],
            "sell_elg_amount": [elg_sell],
            "net_mf_vol": [10.0 if main_inflow else -10.0],
            "net_mf_amount": [net],
        }
    )


def test_build_moneyflow_factors_calculates_main_net_amount_and_score():
    result = build_moneyflow_factors(_moneyflow(main_inflow=True))

    assert result.loc[0, "main_net_amount"] == 120.0
    assert result.loc[0, "big_net_amount"] == 80.0
    assert result.loc[0, "moneyflow_score"] > 0
    assert "strong_main_inflow" in result.loc[0, "moneyflow_risk_flags"]


def test_build_moneyflow_factors_marks_main_outflow():
    result = build_moneyflow_factors(_moneyflow(main_inflow=False))

    assert result.loc[0, "main_net_amount"] == -120.0
    assert result.loc[0, "moneyflow_score"] < 0
    assert "main_outflow" in result.loc[0, "moneyflow_risk_flags"]


def test_build_moneyflow_factors_handles_missing_columns():
    result = build_moneyflow_factors(pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"]}))

    assert len(result) == 1
    assert "missing_moneyflow" in result.loc[0, "moneyflow_risk_flags"]
