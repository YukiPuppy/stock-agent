import duckdb
import pandas as pd

from src.database.duckdb_store import StockAgentStore


def test_init_tables_creates_tables(tmp_path):
    db_path = tmp_path / "stock_agent.duckdb"

    store = StockAgentStore(str(db_path))
    store.init_tables()

    with duckdb.connect(str(db_path)) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }

    assert {"stock_basic", "daily_bars", "daily_factors", "candidate_pool", "trade_plan"} <= tables


def test_save_and_load_stock_basic(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "market": ["SH", "SZ"],
            "board": ["main", "main"],
            "list_status": ["L", "L"],
        }
    )

    store.save_stock_basic(df)
    result = store.load_stock_basic()

    assert result.to_dict("records") == [
        {
            "code": "000001",
            "name": "平安银行",
            "market": "SZ",
            "board": "main",
            "list_status": "L",
        },
        {
            "code": "600000",
            "name": "浦发银行",
            "market": "SH",
            "board": "main",
            "list_status": "L",
        },
    ]


def test_stock_basic_duplicate_code_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_stock_basic(
        pd.DataFrame(
            {
                "code": ["600000"],
                "name": ["旧名称"],
                "market": ["SH"],
                "board": ["main"],
                "list_status": ["L"],
            }
        )
    )

    store.save_stock_basic(
        pd.DataFrame(
            {
                "code": ["600000"],
                "name": ["新名称"],
                "market": ["SH"],
                "board": ["main"],
                "list_status": ["D"],
            }
        )
    )

    result = store.load_stock_basic()

    assert len(result) == 1
    assert result.loc[0, "name"] == "新名称"
    assert result.loc[0, "list_status"] == "D"


def test_save_and_load_daily_bars(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [1000.0, 2000.0],
            "amount": [10000.0, 40000.0],
        }
    )

    store.save_daily_bars(df)
    result = store.load_daily_bars()

    assert result.to_dict("records") == [
        {
            "trade_date": "20260102",
            "code": "000001",
            "open": 20.0,
            "high": 21.0,
            "low": 19.0,
            "close": 20.5,
            "volume": 2000.0,
            "amount": 40000.0,
        },
        {
            "trade_date": "20260102",
            "code": "600000",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
            "amount": 10000.0,
        },
    ]


def test_daily_bars_duplicate_trade_date_and_code_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "volume": [1000.0],
                "amount": [10000.0],
            }
        )
    )

    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260102"],
                "code": ["600000"],
                "open": [12.0],
                "high": [13.0],
                "low": [11.0],
                "close": [12.5],
                "volume": [3000.0],
                "amount": [36000.0],
            }
        )
    )

    result = store.load_daily_bars()

    assert len(result) == 1
    assert result.loc[0, "open"] == 12.0
    assert result.loc[0, "close"] == 12.5
    assert result.loc[0, "amount"] == 36000.0


def test_load_daily_bars_filters_by_date_range(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_daily_bars(
        pd.DataFrame(
            {
                "trade_date": ["20260101", "20260102", "20260103"],
                "code": ["600000", "600000", "600000"],
                "open": [10.0, 11.0, 12.0],
                "high": [10.0, 11.0, 12.0],
                "low": [10.0, 11.0, 12.0],
                "close": [10.0, 11.0, 12.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "amount": [10000.0, 12100.0, 14400.0],
            }
        )
    )

    result = store.load_daily_bars(start_date="20260102", end_date="20260103")

    assert result["trade_date"].tolist() == ["20260102", "20260103"]


def test_save_and_load_daily_factors(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "close": [10.5, 20.5],
            "pct_chg_1d": [0.05, 0.025],
            "pct_chg_3d": [None, None],
            "pct_chg_5d": [None, None],
            "pct_chg_10d": [None, None],
            "ma5": [10.0, 20.0],
            "ma10": [10.0, 20.0],
            "ma20": [10.0, 20.0],
            "volume_ma5": [1000.0, 2000.0],
            "amount_ma5": [10000.0, 40000.0],
            "volume_ratio_5": [1.0, 1.0],
            "high_20": [11.0, 21.0],
            "low_20": [9.0, 19.0],
            "close_position_20": [0.75, 0.75],
            "above_ma5": [True, True],
            "above_ma10": [True, True],
            "above_ma20": [True, True],
        }
    )

    store.save_daily_factors(df)
    result = store.load_daily_factors()

    assert result["code"].tolist() == ["000001", "600000"]
    assert result.loc[0, "close"] == 20.5
    assert result.loc[1, "pct_chg_1d"] == 0.05
    assert result["above_ma5"].tolist() == [True, True]

    filtered = store.load_daily_factors(trade_date="20260102")
    assert len(filtered) == 2


def test_daily_factors_duplicate_trade_date_and_code_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["20260102"],
            "code": ["600000"],
            "close": [10.5],
            "pct_chg_1d": [0.05],
            "pct_chg_3d": [None],
            "pct_chg_5d": [None],
            "pct_chg_10d": [None],
            "ma5": [10.0],
            "ma10": [10.0],
            "ma20": [10.0],
            "volume_ma5": [1000.0],
            "amount_ma5": [10000.0],
            "volume_ratio_5": [1.0],
            "high_20": [11.0],
            "low_20": [9.0],
            "close_position_20": [0.75],
            "above_ma5": [True],
            "above_ma10": [True],
            "above_ma20": [True],
        }
    )
    store.save_daily_factors(base)

    updated = base.copy()
    updated.loc[0, "close"] = 12.5
    updated.loc[0, "pct_chg_1d"] = 0.25
    updated.loc[0, "above_ma20"] = False
    store.save_daily_factors(updated)

    result = store.load_daily_factors()

    assert len(result) == 1
    assert result.loc[0, "close"] == 12.5
    assert result.loc[0, "pct_chg_1d"] == 0.25
    assert result.loc[0, "above_ma20"] == False


def test_save_and_load_candidate_pool(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "close": [10.5, 20.5],
            "pct_chg_1d": [0.05, 0.025],
            "pct_chg_3d": [0.06, 0.03],
            "pct_chg_5d": [0.07, 0.04],
            "pct_chg_10d": [0.08, 0.05],
            "volume_ratio_5": [1.5, 2.0],
            "close_position_20": [0.75, 0.65],
            "above_ma5": [True, True],
            "above_ma10": [True, False],
            "above_ma20": [True, False],
            "amount_ma5": [200000000.0, 300000000.0],
            "score": [44.5, 29.5],
            "rank": [1, 2],
            "reason": ["趋势较强", "趋势较强"],
        }
    )

    store.save_candidate_pool(df)
    result = store.load_candidate_pool()

    assert result["code"].tolist() == ["600000", "000001"]
    assert result.loc[0, "rank"] == 1
    assert result.loc[1, "name"] == "平安银行"

    filtered = store.load_candidate_pool(trade_date="20260102")
    assert len(filtered) == 2


def test_candidate_pool_duplicate_trade_date_and_code_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["20260102"],
            "code": ["600000"],
            "name": ["旧名称"],
            "close": [10.5],
            "pct_chg_1d": [0.05],
            "pct_chg_3d": [0.06],
            "pct_chg_5d": [0.07],
            "pct_chg_10d": [0.08],
            "volume_ratio_5": [1.5],
            "close_position_20": [0.75],
            "above_ma5": [True],
            "above_ma10": [True],
            "above_ma20": [True],
            "amount_ma5": [200000000.0],
            "score": [44.5],
            "rank": [1],
            "reason": ["旧理由"],
        }
    )
    store.save_candidate_pool(base)

    updated = base.copy()
    updated.loc[0, "name"] = "新名称"
    updated.loc[0, "score"] = 55.0
    updated.loc[0, "reason"] = "新理由"
    store.save_candidate_pool(updated)

    result = store.load_candidate_pool()

    assert len(result) == 1
    assert result.loc[0, "name"] == "新名称"
    assert result.loc[0, "score"] == 55.0
    assert result.loc[0, "reason"] == "新理由"


def test_save_and_load_trade_plan(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "rank": [1, 2],
            "close": [10.5, 20.5],
            "strategy_type": ["trend_pullback", "watch_only"],
            "action": ["回踩低吸", "仅观察"],
            "entry_low": [10.24, None],
            "entry_high": [10.45, None],
            "position_low": [0.10, 0.0],
            "position_high": [0.20, 0.0],
            "stop_loss": [9.97, None],
            "take_profit_1": [10.92, None],
            "take_profit_2": [11.34, None],
            "invalid_condition": ["计划失效条件", "仅观察，不主动买入；等待新的量价确认。"],
            "t_plus_1_risk": ["T+1 风险", "T+1 风险"],
            "plan_reason": ["趋势较强", "条件不足"],
        }
    )

    store.save_trade_plan(df)
    result = store.load_trade_plan()

    assert result["code"].tolist() == ["600000", "000001"]
    assert result.loc[0, "strategy_type"] == "trend_pullback"
    assert result.loc[1, "position_high"] == 0.0

    filtered = store.load_trade_plan(trade_date="20260102")
    assert len(filtered) == 2


def test_trade_plan_duplicate_trade_date_and_code_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["20260102"],
            "code": ["600000"],
            "name": ["旧名称"],
            "rank": [1],
            "close": [10.5],
            "strategy_type": ["watch_only"],
            "action": ["仅观察"],
            "entry_low": [None],
            "entry_high": [None],
            "position_low": [0.0],
            "position_high": [0.0],
            "stop_loss": [None],
            "take_profit_1": [None],
            "take_profit_2": [None],
            "invalid_condition": ["旧条件"],
            "t_plus_1_risk": ["T+1 风险"],
            "plan_reason": ["旧理由"],
        }
    )
    store.save_trade_plan(base)

    updated = base.copy()
    updated.loc[0, "name"] = "新名称"
    updated.loc[0, "strategy_type"] = "trend_pullback"
    updated.loc[0, "action"] = "回踩低吸"
    updated.loc[0, "entry_low"] = 10.24
    updated.loc[0, "position_high"] = 0.20
    updated.loc[0, "plan_reason"] = "新理由"
    store.save_trade_plan(updated)

    result = store.load_trade_plan()

    assert len(result) == 1
    assert result.loc[0, "name"] == "新名称"
    assert result.loc[0, "strategy_type"] == "trend_pullback"
    assert result.loc[0, "position_high"] == 0.20
    assert result.loc[0, "plan_reason"] == "新理由"
