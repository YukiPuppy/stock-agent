import pandas as pd
import pytest

from src.database.duckdb_store import DAILY_FACTOR_COLUMNS, StockAgentStore
from src.pipeline.build_daily_factors import build_daily_factors, enrich_daily_factors_with_extension_data


def test_build_daily_factors_reads_bars_computes_and_saves(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260101", "20260102"],
                "code": ["600000", "600000"],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 11.0],
                "volume": [1000.0, 2000.0],
                "amount": [10000.0, 22000.0],
            }
        )
    )

    result = build_daily_factors(db_path=str(db_path))
    saved = store.load_daily_factors()

    assert len(result) == 2
    assert result.loc[1, "pct_chg_1d"] == pytest.approx(0.1)
    pd.testing.assert_frame_equal(saved, result)


def test_build_daily_factors_handles_empty_daily_bars(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.init_tables()

    result = build_daily_factors(db_path=str(db_path))
    saved = store.load_daily_factors()

    assert result.empty
    assert result.columns.tolist() == DAILY_FACTOR_COLUMNS
    assert saved.empty


def test_build_daily_factors_merges_tushare_extension_data(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-02", "2025-01-02"],
                "code": ["000001", "000002", "000003"],
                "open": [10.0, 9.0, 10.0],
                "high": [11.0, 9.0, 10.0],
                "low": [10.0, 8.0, 9.0],
                "close": [10.99, 9.0, 9.0],
                "volume": [1000.0, 1000.0, 1000.0],
                "amount": [10000.0, 9000.0, 9000.0],
            }
        )
    )
    store.save_daily_basic(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-02", "2025-01-02"],
                "code": ["000001", "000002", "000003"],
                "turnover_rate": [1.0, 2.0, 3.0],
                "volume_ratio": [1.5, 1.6, 1.7],
                "pe_ttm": [10.0, 11.0, 12.0],
                "pb": [1.0, 1.1, 1.2],
                "total_mv": [1000.0, 2000.0, 3000.0],
                "circ_mv": [900.0, 1800.0, 2700.0],
            }
        )
    )
    store.save_stock_limits(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-02", "2025-01-02"],
                "code": ["000001", "000002", "000003"],
                "up_limit": [11.0, 9.9, 11.0],
                "down_limit": [9.0, 8.1, 9.0],
            }
        )
    )
    store.save_suspend_daily(
        pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000002"], "suspend_type": ["S"]})
    )

    result = build_daily_factors(db_path=str(db_path)).set_index("code")

    assert result.loc["000001", "volume_ratio_daily_basic"] == 1.5
    assert result.loc["000001", "total_mv"] == 1000.0
    assert result.loc["000001", "circ_mv"] == 900.0
    assert bool(result.loc["000002", "is_suspended"]) is True
    assert bool(result.loc["000001", "is_limit_up_close"]) is True
    assert bool(result.loc["000003", "is_limit_down_close"]) is True


def test_enrich_daily_factors_with_extension_data_handles_direct_inputs():
    factors = pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"], "close": [10.0]})
    result = enrich_daily_factors_with_extension_data(
        factors,
        daily_basic=pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"], "circ_mv": [100.0]}),
        stock_limits=pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"], "up_limit": [11.0], "down_limit": [9.0]}),
        suspend_daily=pd.DataFrame({"trade_date": ["2025-01-02"], "code": ["000001"], "suspend_type": ["R"]}),
    )

    assert result.loc[0, "circ_mv"] == 100.0
    assert result.loc[0, "limit_up_distance"] == pytest.approx(0.1)
    assert bool(result.loc[0, "is_suspended"]) is False


def test_build_daily_factors_merges_moneyflow_factors(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
            }
        )
    )
    store.save_moneyflow_factors(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "main_net_amount": [120.0],
                "main_net_amount_ratio": [0.2],
                "moneyflow_score": [30.0],
                "moneyflow_risk_flags": ["strong_main_inflow"],
            }
        )
    )

    result = build_daily_factors(db_path=str(db_path))

    assert result.loc[0, "moneyflow_score"] == 30.0
    assert result.loc[0, "main_net_amount"] == 120.0


def test_build_daily_factors_merges_industry_strength(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"
    store = StockAgentStore(str(db_path))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "code": ["000001"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
            }
        )
    )
    store.save_stock_industry_map(
        pd.DataFrame(
            [{"code": "000001", "name": "A", "industry_name": "银行", "industry_code": "801780.SI", "industry_level": "L1"}]
        )
    )
    store.save_industry_strength(
        pd.DataFrame(
            [
                {
                    "trade_date": "2025-01-02",
                    "industry_code": "801780.SI",
                    "industry_name": "银行",
                    "industry_strength_score": 65,
                    "industry_strength_level": "strong",
                    "industry_return_3d": 0.02,
                    "industry_return_5d": 0.03,
                    "industry_amount_ratio_5": 1.3,
                    "industry_risk_flags": "strong_industry",
                }
            ]
        )
    )

    result = build_daily_factors(db_path=str(db_path))

    assert result.loc[0, "industry_code"] == "801780.SI"
    assert result.loc[0, "industry_name"] == "银行"
    assert result.loc[0, "industry_strength_score"] == 65
    assert result.loc[0, "industry_strength_level"] == "strong"
