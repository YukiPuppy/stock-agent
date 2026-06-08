import pandas as pd
import duckdb

from src.database.duckdb_store import DATA_QUALITY_REPORT_COLUMNS, PROVIDER_COMPARE_RESULT_COLUMNS, StockAgentStore


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

    assert {
        "stock_basic",
        "daily_bars",
        "trade_calendar",
        "daily_basic",
        "stock_limits",
        "suspend_daily",
        "index_daily",
        "limit_list_daily",
        "moneyflow",
        "moneyflow_factors",
        "market_regime",
        "sw_industry_classification",
        "sw_daily",
        "stock_industry_map",
        "industry_strength",
        "data_quality_report",
        "provider_compare_result",
        "daily_factors",
        "factor_diagnostics",
        "candidate_pool",
        "strategy_signals",
        "backtest_results",
        "strategy_performance",
        "parameter_search_results",
        "parameter_search_performance",
        "parameter_search_backtest_results",
        "walk_forward_validation",
        "strategy_admission",
        "trade_plan",
        "historical_trade_plans",
        "trade_plan_backtest_results",
        "trade_plan_backtest_performance",
        "actual_trades",
        "execution_review",
        "actual_trade_performance",
        "daily_review",
        "period_review",
        "positions",
        "position_review",
    } <= tables


def test_save_and_load_tushare_extension_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))

    store.save_trade_calendar(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-02", "exchange": "SSE", "is_open": 1, "pretrade_date": "2024-12-31"},
                {"trade_date": "2025-01-02", "exchange": "SSE", "is_open": 0, "pretrade_date": "2024-12-31"},
            ]
        )
    )
    store.save_daily_basic(
        pd.DataFrame(
            [
                {
                    "trade_date": "2025-01-02",
                    "code": "000001",
                    "turnover_rate": 1.2,
                    "volume_ratio": 1.5,
                    "total_mv": 200.0,
                    "circ_mv": 100.0,
                },
                {
                    "trade_date": "2025-01-02",
                    "code": "000001",
                    "turnover_rate": 1.3,
                    "volume_ratio": 1.6,
                    "total_mv": 201.0,
                    "circ_mv": 101.0,
                },
            ]
        )
    )
    store.save_stock_limits(
        pd.DataFrame(
            {"trade_date": ["2025-01-02"], "code": ["000001"], "pre_close": [10.0], "up_limit": [11.0], "down_limit": [9.0]}
        )
    )
    store.save_suspend_daily(
        pd.DataFrame(
            {"trade_date": ["2025-01-02"], "code": ["000001"], "suspend_type": ["S"], "suspend_timing": ["09:30"]}
        )
    )

    assert store.load_trade_calendar().loc[0, "is_open"] == 0
    daily_basic = store.load_daily_basic("2025-01-02")
    assert daily_basic.loc[0, "turnover_rate"] == 1.3
    assert daily_basic.loc[0, "volume_ratio"] == 1.6
    assert daily_basic.loc[0, "total_mv"] == 201.0
    assert daily_basic.loc[0, "circ_mv"] == 101.0
    assert store.load_stock_limits("2025-01-02").loc[0, "up_limit"] == 11.0
    assert store.load_suspend_daily("2025-01-02").loc[0, "suspend_type"] == "S"


def test_save_and_load_market_environment_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))

    store.save_index_daily(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-02", "index_code": "000001.SH", "close": 3100.0, "pct_chg": 0.5},
                {"trade_date": "2025-01-02", "index_code": "000001.SH", "close": 3110.0, "pct_chg": 0.6},
            ]
        )
    )
    store.save_limit_list_daily(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-02", "code": "000001", "name": "A", "limit_type": "U", "open_times": 0},
                {"trade_date": "2025-01-02", "code": "000001", "name": "A", "limit_type": "U", "open_times": 2},
            ]
        )
    )
    store.save_market_regime(
        pd.DataFrame(
            [
                {
                    "trade_date": "2025-01-02",
                    "sh_close": 3110.0,
                    "sh_pct_chg": 0.6,
                    "sh_above_ma5": True,
                    "sh_above_ma10": True,
                    "sh_above_ma20": True,
                    "index_trend_score": 50,
                    "limit_up_count": 80,
                    "limit_down_count": 1,
                    "break_board_count": 2,
                    "sentiment_score": 40,
                    "market_regime": "strong",
                    "risk_level": "low",
                    "regime_reason": "强势",
                }
            ]
        )
    )

    index_daily = store.load_index_daily(index_code="000001.SH")
    limit_list = store.load_limit_list_daily(trade_date="2025-01-02")
    regime = store.load_market_regime(trade_date="2025-01-02")

    assert len(index_daily) == 1
    assert index_daily.loc[0, "close"] == 3110.0
    assert len(limit_list) == 1
    assert limit_list.loc[0, "open_times"] == 2
    assert regime.loc[0, "market_regime"] == "strong"


def test_save_empty_tushare_extension_tables_is_noop(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))

    store.save_trade_calendar(pd.DataFrame())
    store.save_daily_basic(pd.DataFrame())
    store.save_stock_limits(pd.DataFrame())
    store.save_suspend_daily(pd.DataFrame())

    assert store.load_trade_calendar().empty
    assert store.load_daily_basic().empty
    assert store.load_stock_limits().empty
    assert store.load_suspend_daily().empty


def test_save_and_load_moneyflow_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_moneyflow(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-02", "code": "000001", "buy_lg_amount": 100.0, "sell_lg_amount": 20.0},
                {"trade_date": "2025-01-02", "code": "000001", "buy_lg_amount": 101.0, "sell_lg_amount": 20.0},
            ]
        )
    )
    store.save_moneyflow_factors(
        pd.DataFrame(
            [
                {
                    "trade_date": "2025-01-02",
                    "code": "000001",
                    "main_net_amount": 81.0,
                    "moneyflow_score": 25.0,
                    "moneyflow_risk_flags": "strong_main_inflow",
                }
            ]
        )
    )

    moneyflow = store.load_moneyflow("2025-01-02")
    factors = store.load_moneyflow_factors("2025-01-02")

    assert len(moneyflow) == 1
    assert moneyflow.loc[0, "buy_lg_amount"] == 101.0
    assert factors.loc[0, "moneyflow_score"] == 25.0
    assert factors.loc[0, "moneyflow_risk_flags"] == "strong_main_inflow"


def test_save_and_load_industry_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_sw_industry_classification(
        pd.DataFrame(
            [
                {"industry_code": "801010.SI", "industry_name": "旧", "level": "L1", "src": "SW2021"},
                {"industry_code": "801010.SI", "industry_name": "农林牧渔", "level": "L1", "src": "SW2021"},
            ]
        )
    )
    store.save_sw_daily(
        pd.DataFrame(
            [
                {"trade_date": "2025-01-02", "industry_code": "801010.SI", "close": 1000.0},
                {"trade_date": "2025-01-02", "industry_code": "801010.SI", "close": 1001.0},
            ]
        )
    )
    store.save_stock_industry_map(
        pd.DataFrame(
            [{"code": "000001", "name": "A", "industry_name": "农林牧渔", "industry_code": "801010.SI"}]
        )
    )
    store.save_industry_strength(
        pd.DataFrame(
            [
                {
                    "trade_date": "2025-01-02",
                    "industry_code": "801010.SI",
                    "industry_strength_score": 65,
                    "industry_strength_level": "strong",
                }
            ]
        )
    )

    assert store.load_sw_industry_classification("L1").loc[0, "industry_name"] == "农林牧渔"
    assert store.load_sw_daily("2025-01-02").loc[0, "close"] == 1001.0
    assert store.load_stock_industry_map().loc[0, "industry_code"] == "801010.SI"
    assert store.load_industry_strength("2025-01-02").loc[0, "industry_strength_level"] == "strong"


def test_save_and_load_data_quality_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_data_quality_report(
        pd.DataFrame(
            [
                {"check_name": "empty_data", "status": "error", "issue_count": 1, "message": "old"},
                {"check_name": "empty_data", "status": "ok", "issue_count": 0, "message": "new"},
            ]
        )
    )
    store.save_provider_compare_result(
        pd.DataFrame(
            [
                {
                    "trade_date": "20260102",
                    "code": "600000",
                    "field": "close",
                    "left_value": 10.0,
                    "right_value": 10.5,
                    "relative_diff": 0.04,
                    "status": "warning",
                    "message": "old",
                },
                {
                    "trade_date": "20260102",
                    "code": "600000",
                    "field": "close",
                    "left_value": 10.0,
                    "right_value": 10.6,
                    "relative_diff": 0.05,
                    "status": "warning",
                    "message": "new",
                },
            ]
        )
    )

    quality = store.load_data_quality_report()
    compare = store.load_provider_compare_result()

    assert len(quality) == 1
    assert quality.loc[0, "message"] == "new"
    assert len(compare) == 1
    assert compare.loc[0, "right_value"] == 10.6


def test_save_and_load_data_unit_metadata(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))

    store.save_data_unit_metadata(
        {
            "OFFICIAL_DATA_PROVIDER": "tushare",
            "DATA_UNIT_VERSION": "daily_bars_v2_tushare_units",
        }
    )

    result = store.load_data_unit_metadata()
    metadata = result.set_index("key")["value"].to_dict()
    assert metadata["OFFICIAL_DATA_PROVIDER"] == "tushare"
    assert metadata["DATA_UNIT_VERSION"] == "daily_bars_v2_tushare_units"
    assert "updated_at" in result.columns
    assert result["updated_at"].notna().all()


def test_save_empty_provider_compare_result_clears_old_rows(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_provider_compare_result(
        pd.DataFrame(
            [
                {
                    "trade_date": "20260102",
                    "code": "600000",
                    "field": "volume",
                    "left_value": 100.0,
                    "right_value": 10000.0,
                    "relative_diff": 0.99,
                    "status": "warning",
                    "message": "old volume mismatch",
                },
                {
                    "trade_date": "20260102",
                    "code": "600000",
                    "field": "amount",
                    "left_value": 100.0,
                    "right_value": 100000.0,
                    "relative_diff": 0.999,
                    "status": "warning",
                    "message": "old amount mismatch",
                },
            ]
        )
    )

    store.save_provider_compare_result(pd.DataFrame(columns=PROVIDER_COMPARE_RESULT_COLUMNS))

    loaded = store.load_provider_compare_result()
    assert loaded.empty
    assert list(loaded.columns) == PROVIDER_COMPARE_RESULT_COLUMNS


def test_save_empty_data_quality_report_clears_old_rows(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_data_quality_report(
        pd.DataFrame(
            [
                {"check_name": "missing_dates", "status": "warning", "issue_count": 2, "message": "old warning"},
                {"check_name": "empty_data", "status": "error", "issue_count": 1, "message": "old error"},
            ]
        )
    )

    store.save_data_quality_report(pd.DataFrame(columns=DATA_QUALITY_REPORT_COLUMNS))

    loaded = store.load_data_quality_report()
    assert loaded.empty
    assert list(loaded.columns) == DATA_QUALITY_REPORT_COLUMNS


def test_save_and_load_factor_diagnostics(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_factor_diagnostics(
        pd.DataFrame(
            [
                {
                    "factor_name": "turnover_rate",
                    "total_count": 1,
                    "non_null_count": 1,
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "mean": 1.0,
                    "diagnostic_status": "ok",
                    "diagnostic_message": "old",
                },
                {
                    "factor_name": "turnover_rate",
                    "total_count": 2,
                    "non_null_count": 2,
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "mean": 2.0,
                    "diagnostic_status": "ok",
                    "diagnostic_message": "new",
                },
            ]
        )
    )
    store.save_factor_diagnostics(pd.DataFrame())

    loaded = store.load_factor_diagnostics()

    assert len(loaded) == 1
    assert loaded.loc[0, "factor_name"] == "turnover_rate"
    assert loaded.loc[0, "mean"] == 2.0


def test_save_and_load_trade_plan_backtest_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_historical_trade_plans(
        pd.DataFrame(
            {
                "trade_date": ["2026-01-01", "2026-01-01"],
                "code": ["600000", "600000"],
                "name": ["旧名称", "新名称"],
                "rank": [1, 1],
                "strategy_names": ["trend", "trend"],
                "strategy_versions": ["v1", "v1"],
                "action": ["回踩低吸", "回踩低吸"],
            }
        )
    )
    store.save_trade_plan_backtest_results(
        pd.DataFrame(
            {
                "plan_date": ["2026-01-01", "2026-01-01"],
                "code": ["600000", "600000"],
                "name": ["旧名称", "新名称"],
                "action": ["回踩低吸", "回踩低吸"],
                "strategy_names": ["trend", "trend"],
                "strategy_versions": ["v1", "v1"],
                "entry_price": [10.0, 10.1],
                "is_triggered": [True, True],
                "is_valid": [True, True],
                "return_pct": [0.01, 0.02],
            }
        )
    )
    store.save_trade_plan_backtest_performance(
        pd.DataFrame(
            {
                "strategy_names": ["trend", "trend"],
                "strategy_versions": ["v1", "v1"],
                "action": ["回踩低吸", "回踩低吸"],
                "plan_count": [1, 2],
                "triggered_count": [1, 2],
                "valid_count": [1, 2],
            }
        )
    )

    historical = store.load_historical_trade_plans()
    results = store.load_trade_plan_backtest_results()
    performance = store.load_trade_plan_backtest_performance()

    assert len(historical) == 1
    assert historical.loc[0, "name"] == "新名称"
    assert len(results) == 1
    assert results.loc[0, "entry_price"] == 10.1
    assert len(performance) == 1
    assert performance.loc[0, "plan_count"] == 2


def test_save_and_load_parameter_search_tables(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_parameter_search_results(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback", "trend_pullback"],
                "strategy_version": ["search_001", "search_001"],
                "sample_count": [1, 2],
                "valid_count": [1, 2],
                "win_rate_3d": [0.5, 0.6],
                "avg_return_3d": [0.01, 0.02],
                "median_return_3d": [0.01, 0.02],
                "avg_max_drawdown_3d": [-0.02, -0.01],
                "evaluation_score": [10.0, 20.0],
                "evaluation_status": ["neutral", "qualified"],
                "risk_level": ["medium", "low"],
                "recommendation": ["observe", "enable_observation"],
                "evaluation_reason": ["old", "new"],
            }
        )
    )
    store.save_parameter_search_performance(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback", "trend_pullback"],
                "strategy_version": ["search_001", "search_001"],
                "sample_count": [1, 2],
                "valid_count": [1, 2],
            }
        )
    )
    store.save_parameter_search_backtest_results(
        pd.DataFrame(
            {
                "signal_date": ["20260101", "20260101"],
                "code": ["600000", "600000"],
                "strategy_name": ["trend_pullback", "trend_pullback"],
                "strategy_version": ["search_001", "search_001"],
                "signal_strength": [1.0, 2.0],
                "is_valid": [False, True],
            }
        )
    )

    results = store.load_parameter_search_results()
    performance = store.load_parameter_search_performance()
    backtest = store.load_parameter_search_backtest_results()

    assert len(results) == 1
    assert results.loc[0, "evaluation_score"] == 20.0
    assert len(performance) == 1
    assert performance.loc[0, "sample_count"] == 2
    assert len(backtest) == 1
    assert backtest.loc[0, "signal_strength"] == 2.0


def test_save_and_load_walk_forward_validation(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_walk_forward_validation(
        pd.DataFrame(
            {
                "strategy_name": ["trend_pullback", "trend_pullback"],
                "strategy_version": ["search_001", "search_001"],
                "train_valid_count": [20, 30],
                "train_win_rate_3d": [0.55, 0.60],
                "train_avg_return_3d": [0.01, 0.02],
                "train_avg_drawdown_3d": [-0.02, -0.01],
                "validation_valid_count": [10, 12],
                "validation_win_rate_3d": [0.50, 0.58],
                "validation_avg_return_3d": [0.0, 0.01],
                "validation_avg_drawdown_3d": [-0.03, -0.02],
                "return_decay": [-0.01, -0.01],
                "win_rate_decay": [-0.05, -0.02],
                "drawdown_worsening": [-0.01, -0.01],
                "stability_score": [10.0, 20.0],
                "overfit_risk": ["medium", "low"],
                "validation_status": ["unstable", "passed_oos"],
                "validation_reason": ["old", "new"],
            }
        )
    )

    result = store.load_walk_forward_validation()

    assert len(result) == 1
    assert result.loc[0, "stability_score"] == 20.0
    assert result.loc[0, "validation_reason"] == "new"


def test_save_and_load_strategy_admission(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_strategy_admission(
        pd.DataFrame(
            {
                "strategy_name": ["trend", "trend"],
                "strategy_version": ["v1", "v1"],
                "source": ["manual_version", "mixed"],
                "valid_count": [30, 40],
                "evaluation_recommendation": ["observe", "enable_observation"],
                "evaluation_score": [20.0, 45.0],
                "oos_status": ["needs_more_observation", "passed_oos"],
                "oos_risk": ["medium", "low"],
                "oos_stability_score": [10.0, 25.0],
                "admission_score": [50.0, 100.0],
                "admission_status": ["watchlist", "qualified_for_observation"],
                "admission_recommendation": ["observe_more", "enable_observation_candidate"],
                "admission_reason": ["old", "new"],
            }
        )
    )

    result = store.load_strategy_admission()

    assert len(result) == 1
    assert result.loc[0, "source"] == "mixed"
    assert result.loc[0, "admission_reason"] == "new"


def test_save_and_load_period_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_period_review(
        pd.DataFrame(
            {
                "start_date": ["2025-01-01", "2025-02-01"],
                "end_date": ["2025-01-31", "2025-02-28"],
                "trading_days": [2, 1],
                "actual_trade_count": [3, 1],
                "buy_count": [2, 1],
                "sell_count": [1, 0],
                "follow_plan_count": [1, 1],
                "off_plan_count": [1, 0],
                "deviation_count": [1, 0],
                "chase_count": [1, 0],
                "over_position_count": [0, 0],
                "bought_watch_only_count": [0, 0],
                "avg_execution_score": [80.0, 100.0],
                "valid_performance_count": [2, 1],
                "avg_return_1d": [0.01, 0.02],
                "avg_return_3d": [0.03, 0.04],
                "avg_return_5d": [0.05, 0.06],
                "plan_trade_avg_return_3d": [0.02, 0.04],
                "off_plan_avg_return_3d": [-0.01, None],
                "chase_avg_return_3d": [-0.02, None],
                "over_position_avg_return_3d": [None, None],
                "best_trade_code": ["600000", "000001"],
                "worst_trade_code": ["000001", "000001"],
                "main_issues": ["存在计划外交易", "未发现明显执行偏差。"],
                "period_summary": ["周期总结", "周期总结"],
                "next_period_suggestion": ["减少计划外交易", "继续保持"],
            }
        )
    )
    store.save_period_review(
        pd.DataFrame(
            {
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "actual_trade_count": [4],
                "period_summary": ["更新后的周期总结"],
                "next_period_suggestion": ["继续观察"],
            }
        )
    )

    result = store.load_period_review("2025-01-01", "2025-01-31")

    assert len(result) == 1
    assert result.loc[0, "actual_trade_count"] == 4
    assert result.loc[0, "period_summary"] == "更新后的周期总结"


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


def test_save_and_load_positions(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_positions(
        pd.DataFrame(
            {
                "as_of_date": ["2025-01-10", "2025-01-10"],
                "code": ["600000", "600000"],
                "name": ["旧名称", "浦发银行"],
                "holding_volume": [50, 100],
                "available_volume": [50, 100],
                "frozen_volume": [0, 0],
                "cost_amount": [500.0, 1000.0],
                "cost_price": [10.0, 10.0],
                "latest_price": [10.5, 10.5],
                "market_value": [525.0, 1050.0],
                "floating_pnl": [25.0, 50.0],
                "floating_pnl_pct": [0.05, 0.05],
                "position_ratio": [0.1, 0.1],
                "first_buy_date": ["2025-01-09", "2025-01-09"],
                "latest_trade_date": ["2025-01-09", "2025-01-09"],
                "strategy_name": ["trend", "trend"],
                "plan_rank": [1, 1],
                "t_plus_1_status": ["sellable", "sellable"],
                "position_status": ["profit_watch", "profit_watch"],
            }
        )
    )

    result = store.load_positions(as_of_date="2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "code"] == "600000"
    assert result.loc[0, "holding_volume"] == 100


def test_save_and_load_position_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_position_review(
        pd.DataFrame(
            {
                "as_of_date": ["2025-01-10"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "holding_volume": [100],
                "available_volume": [0],
                "frozen_volume": [100],
                "cost_amount": [1000.0],
                "cost_price": [10.0],
                "latest_price": [9.0],
                "market_value": [900.0],
                "floating_pnl": [-100.0],
                "floating_pnl_pct": [-0.1],
                "position_ratio": [0.1],
                "first_buy_date": ["2025-01-10"],
                "latest_trade_date": ["2025-01-10"],
                "strategy_name": ["trend"],
                "plan_rank": [1],
                "t_plus_1_status": ["not_sellable_today"],
                "position_status": ["loss_warning"],
                "planned_stop_loss": [9.5],
                "planned_take_profit_1": [11.0],
                "planned_take_profit_2": [12.0],
                "position_risk_level": ["high"],
                "position_flags": ["below_stop_loss,t_plus_1_locked"],
                "position_comment": ["当前价格低于或接近计划止损价"],
                "next_action_hint": ["受 T+1 限制，需次日优先处理风险"],
            }
        )
    )

    result = store.load_position_review(as_of_date="2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "position_risk_level"] == "high"
    assert "below_stop_loss" in result.loc[0, "position_flags"]


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


def test_save_and_load_strategy_signals(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "strategy_name": ["trend_pullback", "support_rebound"],
            "signal_strength": [32.0, 10.0],
            "entry_reason": ["趋势", "支撑"],
            "risk_flags": ["near_20d_high", ""],
        }
    )

    store.save_strategy_signals(df)
    result = store.load_strategy_signals()

    assert result["code"].tolist() == ["600000", "000001"]
    assert result.loc[0, "strategy_name"] == "trend_pullback"
    assert result.loc[0, "strategy_version"] == "v1"
    assert result.loc[0, "risk_flags"] == "near_20d_high"

    filtered = store.load_strategy_signals(trade_date="20260102")
    assert len(filtered) == 2


def test_strategy_signals_duplicate_key_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["20260102"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "signal_strength": [20.0],
            "entry_reason": ["旧理由"],
            "risk_flags": [""],
        }
    )
    store.save_strategy_signals(base)

    updated = base.copy()
    updated.loc[0, "signal_strength"] = 35.0
    updated.loc[0, "entry_reason"] = "新理由"
    updated.loc[0, "risk_flags"] = "near_20d_high"
    store.save_strategy_signals(updated)

    result = store.load_strategy_signals()

    assert len(result) == 1
    assert result.loc[0, "signal_strength"] == 35.0
    assert result.loc[0, "entry_reason"] == "新理由"
    assert result.loc[0, "risk_flags"] == "near_20d_high"


def test_strategy_signals_version_is_part_of_duplicate_key(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "600000"],
            "strategy_name": ["trend_pullback", "trend_pullback"],
            "strategy_version": ["v1", "v2"],
            "signal_strength": [20.0, 30.0],
            "entry_reason": ["v1", "v2"],
            "risk_flags": ["", ""],
        }
    )

    store.save_strategy_signals(df)
    result = store.load_strategy_signals()

    assert len(result) == 2
    assert set(result["strategy_version"]) == {"v1", "v2"}


def test_save_and_load_daily_review_overwrites_trade_date(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "actual_trade_count": [1],
            "buy_count": [1],
            "sell_count": [0],
            "planned_trade_count": [1],
            "matched_plan_count": [1],
            "off_plan_count": [0],
            "follow_plan_count": [1],
            "deviation_count": [0],
            "chase_count": [0],
            "over_position_count": [0],
            "bought_watch_only_count": [0],
            "execution_score": [100],
            "main_issues": ["未发现明显执行偏差"],
            "review_summary": ["执行良好"],
            "next_action_suggestion": ["继续保持"],
        }
    )
    store.save_daily_review(base)

    updated = base.copy()
    updated.loc[0, "actual_trade_count"] = 2
    updated.loc[0, "execution_score"] = 80
    updated.loc[0, "main_issues"] = "存在计划外交易"
    store.save_daily_review(updated)

    result = store.load_daily_review("2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "actual_trade_count"] == 2
    assert result.loc[0, "execution_score"] == 80
    assert result.loc[0, "main_issues"] == "存在计划外交易"


def test_save_and_load_actual_trade_performance_overwrites_duplicate_key(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "trade_time": ["10:00:00"],
            "code": ["600000"],
            "name": ["浦发银行"],
            "side": ["buy"],
            "entry_price": [10.0],
            "entry_volume": [100],
            "entry_amount": [1000.0],
            "position_ratio": [0.1],
            "strategy_name": ["trend_pullback"],
            "plan_rank": [1],
            "plan_match_status": ["matched"],
            "execution_status": ["follow_plan"],
            "execution_flags": ["price_in_range"],
            "return_1d": [0.01],
            "return_3d": [0.02],
            "return_5d": [0.03],
            "max_drawdown_1d": [-0.01],
            "max_drawdown_3d": [-0.02],
            "max_drawdown_5d": [-0.03],
            "max_favorable_1d": [0.02],
            "max_favorable_3d": [0.04],
            "max_favorable_5d": [0.05],
            "is_valid": [True],
            "invalid_reason": [""],
            "performance_comment": ["短期表现较稳"],
        }
    )
    store.save_actual_trade_performance(base)

    updated = base.copy()
    updated.loc[0, "return_3d"] = 0.08
    updated.loc[0, "performance_comment"] = "更新"
    store.save_actual_trade_performance(updated)

    result = store.load_actual_trade_performance("2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "return_3d"] == 0.08
    assert result.loc[0, "performance_comment"] == "更新"


def test_save_and_load_backtest_results(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "signal_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "strategy_name": ["trend_pullback", "support_rebound"],
            "signal_strength": [32.0, 10.0],
            "entry_date": ["20260105", "20260105"],
            "entry_open": [10.0, 20.0],
            "exit_date_1d": ["20260106", "20260106"],
            "exit_close_1d": [10.5, 19.0],
            "return_1d": [0.05, -0.05],
            "exit_date_3d": ["20260108", "20260108"],
            "exit_close_3d": [11.0, 18.0],
            "return_3d": [0.10, -0.10],
            "exit_date_5d": ["20260112", "20260112"],
            "exit_close_5d": [12.0, 17.0],
            "return_5d": [0.20, -0.15],
            "max_drawdown_1d": [-0.02, -0.08],
            "max_drawdown_3d": [-0.03, -0.10],
            "max_drawdown_5d": [-0.04, -0.12],
            "is_valid": [True, True],
            "invalid_reason": ["", ""],
        }
    )

    store.save_backtest_results(df)
    result = store.load_backtest_results()

    assert result["code"].tolist() == ["000001", "600000"]
    assert result.loc[0, "strategy_name"] == "support_rebound"
    assert result.loc[0, "strategy_version"] == "v1"
    assert result.loc[1, "return_5d"] == 0.20

    filtered = store.load_backtest_results(strategy_name="trend_pullback")
    assert len(filtered) == 1
    assert filtered.loc[0, "code"] == "600000"


def test_backtest_results_duplicate_key_is_overwritten(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "signal_date": ["20260102"],
            "code": ["600000"],
            "strategy_name": ["trend_pullback"],
            "signal_strength": [32.0],
            "entry_date": ["20260105"],
            "entry_open": [10.0],
            "return_1d": [0.05],
            "return_3d": [0.10],
            "return_5d": [0.20],
            "max_drawdown_1d": [-0.02],
            "max_drawdown_3d": [-0.03],
            "max_drawdown_5d": [-0.04],
            "is_valid": [True],
            "invalid_reason": [""],
        }
    )
    store.save_backtest_results(base)

    updated = base.copy()
    updated.loc[0, "entry_open"] = 11.0
    updated.loc[0, "return_1d"] = -0.01
    updated.loc[0, "invalid_reason"] = "updated"
    store.save_backtest_results(updated)

    result = store.load_backtest_results()

    assert len(result) == 1
    assert result.loc[0, "entry_open"] == 11.0
    assert result.loc[0, "return_1d"] == -0.01
    assert result.loc[0, "invalid_reason"] == "updated"


def test_save_and_load_strategy_performance(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "support_rebound"],
            "strategy_version": ["v1", "v1"],
            "sample_count": [3, 2],
            "valid_count": [2, 1],
            "win_rate_1d": [0.5, 1.0],
            "win_rate_3d": [0.5, 0.0],
            "win_rate_5d": [1.0, 0.0],
            "avg_return_1d": [0.02, 0.03],
            "avg_return_3d": [0.03, -0.01],
            "avg_return_5d": [0.04, 0.00],
            "median_return_1d": [0.02, 0.03],
            "median_return_3d": [0.03, -0.01],
            "median_return_5d": [0.04, 0.00],
            "avg_max_drawdown_1d": [-0.02, -0.01],
            "avg_max_drawdown_3d": [-0.03, -0.02],
            "avg_max_drawdown_5d": [-0.04, -0.03],
        }
    )

    store.save_strategy_performance(df)
    result = store.load_strategy_performance()

    assert result["strategy_name"].tolist() == ["support_rebound", "trend_pullback"]
    assert result["strategy_version"].tolist() == ["v1", "v1"]
    assert result.loc[0, "valid_count"] == 1
    assert result.loc[1, "avg_return_5d"] == 0.04

    updated = df.iloc[[0]].copy()
    updated.loc[0, "valid_count"] = 3
    updated.loc[0, "win_rate_1d"] = 2 / 3
    store.save_strategy_performance(updated)

    overwritten = store.load_strategy_performance()
    trend = overwritten[overwritten["strategy_name"] == "trend_pullback"].iloc[0]
    assert trend["valid_count"] == 3
    assert trend["win_rate_1d"] == 2 / 3


def test_save_and_load_strategy_version_performance(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "trend_pullback"],
            "strategy_version": ["v1", "v2"],
            "sample_count": [3, 4],
            "valid_count": [2, 3],
            "win_rate_1d": [0.5, 2 / 3],
            "win_rate_3d": [0.5, 2 / 3],
            "win_rate_5d": [1.0, 2 / 3],
            "avg_return_1d": [0.02, 0.03],
            "avg_return_3d": [0.03, 0.04],
            "avg_return_5d": [0.04, 0.05],
            "median_return_1d": [0.02, 0.03],
            "median_return_3d": [0.03, 0.04],
            "median_return_5d": [0.04, 0.05],
            "avg_max_drawdown_1d": [-0.02, -0.01],
            "avg_max_drawdown_3d": [-0.03, -0.02],
            "avg_max_drawdown_5d": [-0.04, -0.03],
        }
    )

    store.save_strategy_version_performance(df)
    result = store.load_strategy_version_performance()

    assert result["strategy_version"].tolist() == ["v1", "v2"]
    assert result.loc[1, "valid_count"] == 3

    updated = df.iloc[[1]].copy()
    updated.loc[1, "valid_count"] = 4
    store.save_strategy_version_performance(updated)

    overwritten = store.load_strategy_version_performance()
    v2 = overwritten[overwritten["strategy_version"] == "v2"].iloc[0]
    assert len(overwritten) == 2
    assert v2["valid_count"] == 4


def test_save_and_load_strategy_version_evaluation(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "strategy_name": ["trend_pullback", "trend_pullback"],
            "strategy_version": ["v1", "v2"],
            "sample_count": [40, 50],
            "valid_count": [35, 45],
            "win_rate_3d": [0.6, 0.4],
            "avg_return_3d": [0.02, -0.01],
            "median_return_3d": [0.01, -0.01],
            "avg_max_drawdown_3d": [-0.03, -0.02],
            "evaluation_score": [25.25, 12.0],
            "evaluation_status": ["qualified", "weak"],
            "risk_level": ["low", "medium"],
            "recommendation": ["enable_observation", "pause"],
            "evaluation_reason": ["满足观察启用条件。", "3日收益和胜率均未达标。"],
        }
    )

    store.save_strategy_version_evaluation(df)
    result = store.load_strategy_version_evaluation()

    assert result["strategy_version"].tolist() == ["v1", "v2"]
    assert result.loc[0, "recommendation"] == "enable_observation"

    updated = df.iloc[[1]].copy()
    updated.loc[1, "evaluation_score"] = 30.0
    updated.loc[1, "recommendation"] = "observe"
    store.save_strategy_version_evaluation(updated)

    overwritten = store.load_strategy_version_evaluation()
    v2 = overwritten[overwritten["strategy_version"] == "v2"].iloc[0]
    assert len(overwritten) == 2
    assert v2["evaluation_score"] == 30.0
    assert v2["recommendation"] == "observe"


def test_save_and_load_trade_plan(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    df = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "code": ["600000", "000001"],
            "name": ["浦发银行", "平安银行"],
            "rank": [1, 2],
            "strategy_names": ["trend_pullback", "watch_only"],
            "strategy_versions": ["v1", "v2"],
            "active_signal_count": [1, 0],
            "avg_strategy_weight": [1.2, 0.8],
            "recommendations": ["enable_observation", "pause"],
            "risk_flags": ["", "weak_signal"],
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
    assert result.loc[0, "strategy_versions"] == "v1"
    assert result.loc[0, "recommendations"] == "enable_observation"
    assert result.loc[1, "risk_flags"] == "weak_signal"
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


def test_save_and_load_actual_trades(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_actual_trades(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["09:45:00"],
                "code": ["1"],
                "name": ["平安银行"],
                "side": ["BUY"],
                "price": [10.0],
                "volume": [100],
                "position_ratio": [0.1],
            }
        )
    )

    result = store.load_actual_trades(trade_date="2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "side"] == "buy"
    assert result.loc[0, "amount"] == 1000.0


def test_actual_trades_duplicate_rule_overwrites(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    base = pd.DataFrame(
        {
            "trade_date": ["2025-01-10"],
            "trade_time": ["09:45:00"],
            "code": ["600000"],
            "side": ["buy"],
            "price": [10.0],
            "volume": [100],
            "note": ["old"],
        }
    )
    store.save_actual_trades(base)

    updated = base.copy()
    updated.loc[0, "note"] = "new"
    store.save_actual_trades(updated)

    result = store.load_actual_trades()

    assert len(result) == 1
    assert result.loc[0, "note"] == "new"


def test_save_and_load_execution_review(tmp_path):
    store = StockAgentStore(str(tmp_path / "stock_agent.duckdb"))
    store.save_execution_review(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-10"],
                "trade_time": ["09:45:00"],
                "code": ["600000"],
                "name": ["浦发银行"],
                "side": ["buy"],
                "actual_price": [10.5],
                "actual_volume": [100],
                "actual_amount": [1050.0],
                "position_ratio": [0.15],
                "plan_rank": [1],
                "planned_action": ["回踩低吸"],
                "entry_low": [10.0],
                "entry_high": [11.0],
                "position_low": [0.1],
                "position_high": [0.2],
                "execution_status": ["follow_plan"],
                "execution_flags": ["price_in_range"],
            }
        )
    )

    result = store.load_execution_review(trade_date="2025-01-10")

    assert len(result) == 1
    assert result.loc[0, "execution_status"] == "follow_plan"
    assert result.loc[0, "execution_flags"] == "price_in_range"
